"""
feature_engineering.py

Milestone 3 of the Prediction Market Alpha Engine:
    "Turn raw market snapshots into features a probability model can use."

This module reads the time series in `market_snapshots` (built by
src/data/polymarket_api.py, one row per market per ingestion run) and derives
predictive features per market per timestamp: price momentum, rolling
volatility, volume/liquidity change, spread, and time-to-resolution. Results
are written to a `market_features` table, matching the schema style from the
project plan.

Design notes:
    - Polymarket's `yes_price` is a PROBABILITY (bounded 0-1), not a regular
      asset price. This matters for momentum features specifically: a naive
      percent-return (`pct_change()`) blows up near the boundaries — a move
      from 1% to 2% shows as "+100%" while carrying almost no real
      information, whereas the same-size move from 50% to 51% is much more
      meaningful. This module computes TWO momentum measures per window:
        - `price_change_*`: raw probability-point change (e.g. +0.02).
          Intuitive, always well-behaved, use this for most purposes.
        - `logit_change_*`: change in log-odds, log(p/(1-p)). This is the
          statistically principled way to measure movement in a bounded
          probability — equal logit moves represent equal "evidence" shifts
          regardless of whether you're near 0.5 or near an extreme. Prefer
          this as a model input if you want a feature that behaves
          consistently across the whole probability range.
    - Snapshots don't arrive on perfectly even intervals (ingestion runs on a
      best-effort hourly cron), so all windowed calculations use precise
      time-based lookback (see `_lookback_value`), not row-count windows.
      This is correct regardless of whether your ingestion cadence is
      hourly, every 15 minutes, or spotty.
    - With few historical snapshots per market, most windowed features will
      legitimately be NaN (e.g. you can't compute 24h volatility from data
      that's only 2 hours old) — that's correct behavior, not a bug. As your
      scheduled ingestion (Milestone 2) accumulates more history, more rows
      will have real values.
    - This module never talks to Polymarket directly — it only reads/writes
      Postgres, keeping ingestion and feature engineering cleanly separated
      per the project's layered architecture.

Usage:
    python -m src.features.feature_engineering --to-postgres   # full run
    python -m src.features.feature_engineering --market-id 703258  # one market
    from src.features.feature_engineering import compute_features
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("feature_engineering")

# Columns this module produces, in the order they're written to Postgres.
FEATURE_COLUMNS = [
    "price_change_1h", "price_change_24h", "price_change_7d",
    "logit_change_1h", "logit_change_24h", "logit_change_7d",
    "volatility_1h", "volatility_24h", "volatility_7d",
    "volume_change_24h", "liquidity_change_24h",
    "spread_pct",
    "hours_to_resolution", "days_to_resolution",
    "distance_from_50",
]

CREATE_MARKET_FEATURES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_features (
    feature_id           BIGSERIAL PRIMARY KEY,
    market_id             TEXT NOT NULL REFERENCES markets(market_id),
    "timestamp"           TIMESTAMPTZ NOT NULL,
    price_change_1h       DOUBLE PRECISION,
    price_change_24h      DOUBLE PRECISION,
    price_change_7d       DOUBLE PRECISION,
    logit_change_1h       DOUBLE PRECISION,
    logit_change_24h      DOUBLE PRECISION,
    logit_change_7d       DOUBLE PRECISION,
    volatility_1h         DOUBLE PRECISION,
    volatility_24h        DOUBLE PRECISION,
    volatility_7d         DOUBLE PRECISION,
    volume_change_24h     DOUBLE PRECISION,
    liquidity_change_24h  DOUBLE PRECISION,
    spread_pct            DOUBLE PRECISION,
    hours_to_resolution   DOUBLE PRECISION,
    days_to_resolution    DOUBLE PRECISION,
    distance_from_50      DOUBLE PRECISION,
    UNIQUE (market_id, "timestamp")
);

CREATE INDEX IF NOT EXISTS idx_features_market_time
    ON market_features (market_id, "timestamp");
"""


def _safe_logit(p: pd.Series, eps: float = 1e-4) -> pd.Series:
    """log(p / (1-p)), clipped away from exactly 0 or 1 (which are common in
    prediction markets near resolution and would otherwise produce +/-inf).
    """
    p_clipped = p.clip(lower=eps, upper=1 - eps)
    return np.log(p_clipped / (1 - p_clipped))


def _lookback_value(series: pd.Series, window: str) -> pd.Series:
    """For each timestamp t, the value from the most recent snapshot at or
    before (t - window). Returns NaN if no such snapshot exists yet — i.e.
    if the market's history doesn't go back far enough for this window.

    This is deliberately stricter than pandas' built-in `.rolling(window)`,
    which silently uses whatever data IS available even if the window isn't
    fully populated (e.g. it would treat "the earliest price we've ever
    seen" as if it were "the price 7 days ago" when the market is only 2
    days old). That's a real methodology error for this kind of feature —
    better to honestly report NaN than to mislabel a short-history value as
    a long-window one.
    """
    if series.empty:
        return series.copy()

    idx_times = series.index.values.astype("datetime64[ns]")
    window_td = pd.Timedelta(window).to_timedelta64()
    target_times = idx_times - window_td

    # For each target time, find the position of the last snapshot at or
    # before it (searchsorted assumes idx_times is sorted ascending, which
    # the caller guarantees).
    pos = np.searchsorted(idx_times, target_times, side="right") - 1
    valid = pos >= 0

    values = series.to_numpy(dtype="float64")
    result = np.full(len(series), np.nan)
    result[valid] = values[pos[valid]]
    return pd.Series(result, index=series.index)


def compute_features(
    snapshots: pd.DataFrame,
    resolution_date: Optional[datetime] = None,
    as_of: Optional[datetime] = None,
) -> pd.DataFrame:
    """Compute features for every row in a single market's snapshot history.

    Parameters
    ----------
    snapshots : DataFrame with columns ['timestamp', 'yes_price', 'volume',
        'liquidity', 'spread'], one row per historical snapshot for ONE
        market. Does not need to be pre-sorted or have a unique index.
    resolution_date : the market's scheduled resolution date/time, used to
        compute hours/days_to_resolution. Pass None to skip those columns.
    as_of : reserved for future use (deterministic "now" in tests).

    Returns
    -------
    A DataFrame with the same number of rows as `snapshots`, with all
    FEATURE_COLUMNS added. Feature values that can't yet be computed
    (insufficient history for that window) are NaN — this is expected,
    not an error.
    """
    if snapshots.empty:
        return pd.DataFrame(columns=["timestamp"] + FEATURE_COLUMNS)

    df = snapshots.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    df = df.set_index("timestamp", drop=False)

    price = df["yes_price"].astype(float)
    logit_price = _safe_logit(price)
    returns = price.diff()  # simple period-over-period change, used for volatility

    for label, window in (("1h", "1h"), ("24h", "24h"), ("7d", "7d")):
        baseline_price = _lookback_value(price, window)
        df[f"price_change_{label}"] = price - baseline_price

        baseline_logit = _lookback_value(logit_price, window)
        df[f"logit_change_{label}"] = logit_price - baseline_logit

        df[f"volatility_{label}"] = returns.rolling(window).std()

    if "volume" in df.columns:
        volume = df["volume"].astype(float)
        baseline_volume = _lookback_value(volume, "24h")
        df["volume_change_24h"] = np.where(
            baseline_volume > 0, (volume - baseline_volume) / baseline_volume, np.nan
        )
    else:
        df["volume_change_24h"] = np.nan

    if "liquidity" in df.columns:
        liquidity = df["liquidity"].astype(float)
        baseline_liquidity = _lookback_value(liquidity, "24h")
        df["liquidity_change_24h"] = np.where(
            baseline_liquidity > 0,
            (liquidity - baseline_liquidity) / baseline_liquidity, np.nan
        )
    else:
        df["liquidity_change_24h"] = np.nan

    if "spread" in df.columns:
        # spread as a fraction of the market price itself (how wide the
        # market is relative to where it's trading) rather than a raw
        # dollar/probability-point spread, so it's comparable across markets
        # at very different price levels.
        df["spread_pct"] = np.where(
            price > 0, df["spread"].astype(float) / price, np.nan
        )
    else:
        df["spread_pct"] = np.nan

    if resolution_date is not None:
        resolution_ts = pd.Timestamp(resolution_date)
        if resolution_ts.tzinfo is None:
            resolution_ts = resolution_ts.tz_localize("UTC")
        delta = (resolution_ts - df["timestamp"])
        df["hours_to_resolution"] = delta.dt.total_seconds() / 3600
        df["days_to_resolution"] = delta.dt.total_seconds() / 86400
    else:
        df["hours_to_resolution"] = np.nan
        df["days_to_resolution"] = np.nan

    df["distance_from_50"] = (price - 0.5).abs()

    result = df[["timestamp"] + FEATURE_COLUMNS].reset_index(drop=True)
    # Replace inf/-inf (can arise from division edge cases) with NaN so they
    # serialize cleanly to Postgres NULLs rather than erroring.
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


# ---------------------------------------------------------------------------
# Postgres integration
# ---------------------------------------------------------------------------

def _get_pg_connection(dsn: Optional[str] = None):
    """Reuses the same connection helper as the ingestion module, so
    connection-handling logic (env var lookup, error message) lives in one
    place. Imported lazily to avoid a hard dependency for pure feature-math
    use (e.g. unit tests) that never touch Postgres.
    """
    from src.data.polymarket_api import get_pg_connection
    return get_pg_connection(dsn)


def ensure_features_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_MARKET_FEATURES_TABLE_SQL)
    conn.commit()


def fetch_market_ids(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT market_id FROM market_snapshots;")
        return [row[0] for row in cur.fetchall()]


def fetch_market_history(conn, market_id: str) -> tuple[pd.DataFrame, Optional[datetime]]:
    """Returns (snapshot history DataFrame, resolution_date) for one market."""
    query = """
        SELECT s.timestamp, s.yes_price, s.volume, s.liquidity, s.spread,
               m.resolution_date
        FROM market_snapshots s
        JOIN markets m USING (market_id)
        WHERE s.market_id = %s
        ORDER BY s.timestamp ASC;
    """
    df = pd.read_sql(query, conn, params=(market_id,))
    resolution_date = df["resolution_date"].iloc[0] if not df.empty else None
    return df.drop(columns=["resolution_date"]), resolution_date


def save_features_to_postgres(market_id: str, features: pd.DataFrame, conn,
                               batch_size: int = 500) -> int:
    """Upserts computed feature rows for one market. Returns rows written."""
    from psycopg2.extras import execute_values

    if features.empty:
        return 0

    rows = [
        (
            market_id, row.timestamp,
            *(None if pd.isna(getattr(row, col)) else float(getattr(row, col))
              for col in FEATURE_COLUMNS),
        )
        for row in features.itertuples(index=False)
    ]

    with conn.cursor() as cur:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            execute_values(
                cur,
                f"""
                INSERT INTO market_features
                    (market_id, "timestamp", {", ".join(FEATURE_COLUMNS)})
                VALUES %s
                ON CONFLICT (market_id, "timestamp") DO UPDATE SET
                    {", ".join(f"{c} = EXCLUDED.{c}" for c in FEATURE_COLUMNS)}
                """,
                batch,
            )
    conn.commit()
    return len(rows)


def run_feature_pipeline(dsn: Optional[str] = None,
                          market_id: Optional[str] = None) -> int:
    """Computes and stores features for one market, or all markets that have
    snapshot history if `market_id` is None. Returns total feature rows written.
    """
    conn = _get_pg_connection(dsn)
    total_written = 0
    try:
        ensure_features_table(conn)
        market_ids = [market_id] if market_id else fetch_market_ids(conn)
        logger.info("Computing features for %d market(s)...", len(market_ids))

        for i, mid in enumerate(market_ids, start=1):
            history, resolution_date = fetch_market_history(conn, mid)
            if history.empty:
                continue
            features = compute_features(history, resolution_date=resolution_date)
            written = save_features_to_postgres(mid, features, conn)
            total_written += written
            if i % 500 == 0 or i == len(market_ids):
                logger.info("Processed %d/%d markets (%d feature rows written so far)...",
                            i, len(market_ids), total_written)

        logger.info("Done. Wrote %d total feature rows across %d markets.",
                    total_written, len(market_ids))
        return total_written
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute prediction-market features from ingested snapshots."
    )
    parser.add_argument("--market-id", type=str, default=None,
                         help="Only compute features for this market (default: all)")
    args = parser.parse_args()

    total = run_feature_pipeline(market_id=args.market_id)
    print(f"\nWrote {total} feature rows to market_features.")


if __name__ == "__main__":
    main()
