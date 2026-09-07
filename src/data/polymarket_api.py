"""
polymarket_api.py

Milestone 1 of the Prediction Market Alpha Engine:
    "I can programmatically pull prediction-market data and save it."

This module wraps Polymarket's public Gamma API (market discovery / metadata)
and gives you two things:
    1. A thin, dependency-light client for fetching markets (with pagination
       and retries) and normalizing the messy raw response into clean records.
    2. Storage helpers to persist those records as timestamped raw JSON
       snapshots, and optionally into PostgreSQL (matching the
       `market_snapshots`-style schema from the project spec).

Design notes (things that will bite you if you don't handle them):
    - Gamma is public and needs no API key for market discovery endpoints.
    - Base URL: https://gamma-api.polymarket.com
    - `outcomePrices`, `outcomes`, and `clobTokenIds` come back from the API
      as JSON-*encoded strings* (e.g. '["0.42", "0.58"]'), not real arrays.
      They must be parsed before use.
    - Numeric fields (volume, liquidity, prices) frequently arrive as strings.
    - Rate limits on Gamma are roughly 4,000 req/10s, but be a good citizen:
      this client paginates with a small delay and retries with backoff.
    - Treat this ingestion layer as a boundary: nothing downstream (features,
      models, backtester) should talk to Polymarket directly. Everything goes
      through here, so if the API changes, you only fix it in one place.

Usage:
    python -m src.data.polymarket_api                # quick smoke test / CLI
    from src.data.polymarket_api import PolymarketClient
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("polymarket_api")

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

# Where raw snapshots land by default. Matches the data/raw/ folder from the
# recommended project structure.
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ---------------------------------------------------------------------------
# Normalized record
# ---------------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    """A cleaned, typed snapshot of a single market at fetch time.

    This is intentionally a *subset* of everything Gamma returns — the fields
    called out in the project spec (market id, name, YES/NO prices, volume,
    liquidity, resolution date, status) — plus a few identifiers you'll need
    later (event_id, slug, condition_id / clob token ids) to join against
    the CLOB API for order-book / historical price data.

    Everything else from the raw response is preserved in `raw` so you never
    lose information, even if this dataclass doesn't surface it yet.
    """

    market_id: str
    event_id: Optional[str]
    slug: Optional[str]
    question: str
    category: Optional[str]
    yes_price: Optional[float]
    no_price: Optional[float]
    volume: Optional[float]
    liquidity: Optional[float]
    spread: Optional[float]
    created_at: Optional[str]
    resolution_date: Optional[str]
    active: bool
    closed: bool
    status: str
    clob_token_ids: Optional[list]
    fetched_at: str
    raw: dict

    def to_dict(self, include_raw: bool = True) -> dict:
        d = asdict(self)
        if not include_raw:
            d.pop("raw", None)
        return d


# ---------------------------------------------------------------------------
# Helpers for Gamma's "JSON-encoded string" fields
# ---------------------------------------------------------------------------

def _parse_json_field(value: Any) -> Any:
    """Gamma often returns arrays as JSON-encoded strings, e.g. '["Yes","No"]'.
    Parse them; if it's already a list/dict (or None), pass through unchanged.
    """
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _derive_status(active: bool, closed: bool) -> str:
    if closed:
        return "closed"
    if active:
        return "active"
    return "inactive"


def normalize_market(raw_market: dict) -> MarketSnapshot:
    """Convert one raw Gamma /markets record into a clean MarketSnapshot."""

    outcomes = _parse_json_field(raw_market.get("outcomes")) or []
    outcome_prices = _parse_json_field(raw_market.get("outcomePrices")) or []
    clob_token_ids = _parse_json_field(raw_market.get("clobTokenIds"))

    yes_price = no_price = None
    if outcomes and outcome_prices and len(outcomes) == len(outcome_prices):
        price_by_outcome = {
            str(o).strip().lower(): _to_float(p)
            for o, p in zip(outcomes, outcome_prices)
        }
        yes_price = price_by_outcome.get("yes")
        no_price = price_by_outcome.get("no")
    elif outcome_prices:
        # Fallback: binary market with exactly two prices in [YES, NO] order.
        if len(outcome_prices) >= 1:
            yes_price = _to_float(outcome_prices[0])
        if len(outcome_prices) >= 2:
            no_price = _to_float(outcome_prices[1])

    bid = _to_float(raw_market.get("bestBid"))
    ask = _to_float(raw_market.get("bestAsk"))
    spread = (ask - bid) if (bid is not None and ask is not None) else _to_float(
        raw_market.get("spread")
    )

    active = bool(raw_market.get("active", False))
    closed = bool(raw_market.get("closed", False))

    return MarketSnapshot(
        market_id=str(raw_market.get("id")),
        event_id=str(raw_market.get("eventId")) if raw_market.get("eventId") else None,
        slug=raw_market.get("slug"),
        question=raw_market.get("question") or raw_market.get("title") or "",
        category=raw_market.get("category"),
        yes_price=yes_price,
        no_price=no_price,
        volume=_to_float(raw_market.get("volume")),
        liquidity=_to_float(raw_market.get("liquidity")),
        spread=spread,
        created_at=raw_market.get("createdAt") or raw_market.get("startDate"),
        resolution_date=raw_market.get("endDate"),
        active=active,
        closed=closed,
        status=_derive_status(active, closed),
        clob_token_ids=clob_token_ids,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        raw=raw_market,
    )


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class PolymarketAPIError(RuntimeError):
    """Raised when the Gamma API can't be reached or returns something we
    can't parse, after retries are exhausted."""


class PolymarketClient:
    """Minimal client for Polymarket's public Gamma API.

    Only covers market/event discovery (no auth needed). Trading, order-book,
    and historical-price data live on the CLOB / Data APIs respectively and
    are out of scope for this module by design — keep ingestion concerns
    separated, per the project architecture.
    """

    def __init__(
        self,
        base_url: str = GAMMA_BASE_URL,
        timeout: float = 15.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
        request_delay: float = 0.2,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.request_delay = request_delay
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    wait = self.backoff_seconds * attempt
                    logger.warning("Rate limited (429). Backing off %.1fs...", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                wait = self.backoff_seconds * attempt
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    url, attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        raise PolymarketAPIError(f"Failed GET {url} after {self.max_retries} attempts: {last_exc}")

    def fetch_markets_page(
        self,
        limit: int = 100,
        after_cursor: Optional[str] = None,
        closed: Optional[bool] = False,
        end_date_min: Optional[str] = None,
        order: Optional[str] = None,
        ascending: Optional[bool] = None,
        liquidity_num_min: Optional[float] = None,
        volume_num_min: Optional[float] = None,
    ) -> tuple[list[dict], Optional[str]]:
        """Fetch a single page of raw market records from GET /markets/keyset.

        Polymarket migrated Gamma's /markets endpoint to cursor-based
        ("keyset") pagination. `offset` is no longer accepted beyond the
        first couple thousand records and returns a 422 if you try to page
        past that point — you must use `after_cursor` from the previous
        response instead.

        There's also no `active` query filter on this endpoint. `active`
        markets are a subset of non-`closed` ones, but Polymarket's non-closed
        backlog is enormous — tens of thousands of short-lived recurring
        markets (hourly crypto price bets, daily sports lines, etc.) are all
        technically "active" at any given moment. Filtering by `end_date_min`
        alone still leaves a huge set. `liquidity_num_min` / `volume_num_min`
        are what actually narrow this down to markets worth analyzing.

        Returns (markets, next_cursor). `next_cursor` is None on the last page.
        """
        params: dict[str, Any] = {"limit": limit}
        if closed is not None:
            params["closed"] = str(closed).lower()
        if after_cursor:
            params["after_cursor"] = after_cursor
        if end_date_min:
            params["end_date_min"] = end_date_min
        if order:
            params["order"] = order
        if ascending is not None:
            params["ascending"] = str(ascending).lower()
        if liquidity_num_min is not None:
            params["liquidity_num_min"] = liquidity_num_min
        if volume_num_min is not None:
            params["volume_num_min"] = volume_num_min

        data = self._get("/markets/keyset", params=params)

        if not isinstance(data, dict):
            raise PolymarketAPIError(f"Unexpected /markets/keyset response shape: {type(data)}")

        markets = data.get("markets", [])
        next_cursor = data.get("next_cursor")
        return markets, next_cursor

    def fetch_all_markets(
        self,
        active: Optional[bool] = True,
        closed: Optional[bool] = False,
        page_size: int = 100,
        max_pages: Optional[int] = None,
        end_date_min: Optional[str] = "now",
        hard_page_cap: int = 100,
        liquidity_num_min: Optional[float] = None,
        volume_num_min: Optional[float] = None,
    ) -> list[dict]:
        """Paginate through GET /markets/keyset via cursor until the last page
        (indicated by a missing `next_cursor`), `max_pages` is hit, or the
        `hard_page_cap` safety limit is hit (protects against an unexpectedly
        enormous result set or an API pagination bug looping forever).

        `end_date_min="now"` (the default) filters server-side to markets
        that haven't already resolved. On its own this still leaves tens of
        thousands of short-lived recurring markets, so `liquidity_num_min`
        and/or `volume_num_min` are the filters that actually get you down
        to a research-worthy set — pass at least one of these for any real
        pull. Results are ordered by volume (descending) so if you do hit a
        page cap, you keep the most active markets rather than an arbitrary
        slice.

        `active` is applied as a client-side filter on top of the above.
        """
        if end_date_min == "now":
            end_date_min = datetime.now(timezone.utc).isoformat()

        all_markets: list[dict] = []
        cursor: Optional[str] = None
        page_num = 0

        while True:
            page_num += 1
            if page_num > hard_page_cap:
                logger.warning(
                    "Hit hard_page_cap=%d pages (%d records so far) — stopping early. "
                    "Pass a smaller max_pages, or narrow further with "
                    "liquidity_num_min/volume_num_min if you expected fewer results.",
                    hard_page_cap, len(all_markets),
                )
                break

            logger.info("Fetching markets page %d (cursor=%s, limit=%d, total so far=%d)...",
                        page_num, cursor, page_size, len(all_markets))
            page, next_cursor = self.fetch_markets_page(
                limit=page_size, after_cursor=cursor, closed=closed,
                end_date_min=end_date_min, order="volumeNum", ascending=False,
                liquidity_num_min=liquidity_num_min, volume_num_min=volume_num_min,
            )
            if not page:
                break

            all_markets.extend(page)

            if max_pages is not None and page_num >= max_pages:
                break
            if not next_cursor or next_cursor == cursor:
                # Last page, or the API handed back the same cursor twice —
                # treat that as "no more progress" rather than looping forever.
                break

            cursor = next_cursor
            time.sleep(self.request_delay)

        if active is True:
            all_markets = [m for m in all_markets if m.get("active") is True]
        elif active is False:
            all_markets = [m for m in all_markets if m.get("active") is False]

        logger.info("Fetched %d raw market records total (after active filter).",
                    len(all_markets))
        return all_markets

    def get_market(self, market_id: str) -> MarketSnapshot:
        """Fetch and normalize a single market by id."""
        data = self._get(f"/markets/{market_id}")
        raw_market = data["data"] if isinstance(data, dict) and "data" in data else data
        return normalize_market(raw_market)

    def get_markets(
        self,
        active: Optional[bool] = True,
        closed: Optional[bool] = False,
        page_size: int = 100,
        max_pages: Optional[int] = None,
        end_date_min: Optional[str] = "now",
        hard_page_cap: int = 100,
        liquidity_num_min: Optional[float] = None,
        volume_num_min: Optional[float] = None,
    ) -> list[MarketSnapshot]:
        """Fetch markets and return them as normalized MarketSnapshot objects."""
        raw_markets = self.fetch_all_markets(
            active=active, closed=closed, page_size=page_size, max_pages=max_pages,
            end_date_min=end_date_min, hard_page_cap=hard_page_cap,
            liquidity_num_min=liquidity_num_min, volume_num_min=volume_num_min,
        )
        snapshots = []
        for rm in raw_markets:
            try:
                snapshots.append(normalize_market(rm))
            except Exception as exc:  # noqa: BLE001 - log and skip malformed rows
                logger.warning("Skipping malformed market record (id=%s): %s",
                               rm.get("id"), exc)
        return snapshots


# ---------------------------------------------------------------------------
# Storage: raw JSON snapshots
# ---------------------------------------------------------------------------

def save_raw_json(
    markets: list[MarketSnapshot],
    output_dir: Path = DEFAULT_RAW_DIR,
    filename_prefix: str = "markets_snapshot",
) -> Path:
    """Save a list of MarketSnapshots as a single timestamped JSON file.

    This is your point-in-time archive: every ingestion run writes a new file
    rather than overwriting the last one, so you can always reconstruct
    "what did the market look like at time T" for backtesting later.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"{filename_prefix}_{timestamp}.json"

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(markets),
        "markets": [m.to_dict(include_raw=True) for m in markets],
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("Saved %d markets to %s", len(markets), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Storage: PostgreSQL (optional)
# ---------------------------------------------------------------------------

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS markets (
    market_id       TEXT PRIMARY KEY,
    event_id        TEXT,
    slug            TEXT,
    question        TEXT NOT NULL,
    category        TEXT,
    created_at      TIMESTAMPTZ,
    resolution_date TIMESTAMPTZ,
    status          TEXT
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id  BIGSERIAL PRIMARY KEY,
    market_id    TEXT NOT NULL REFERENCES markets(market_id),
    "timestamp"  TIMESTAMPTZ NOT NULL,
    yes_price    DOUBLE PRECISION,
    no_price     DOUBLE PRECISION,
    volume       DOUBLE PRECISION,
    liquidity    DOUBLE PRECISION,
    spread       DOUBLE PRECISION,
    active       BOOLEAN,
    closed       BOOLEAN,
    UNIQUE (market_id, "timestamp")
);

CREATE INDEX IF NOT EXISTS idx_snapshots_market_time
    ON market_snapshots (market_id, "timestamp");
"""


def get_pg_connection(dsn: Optional[str] = None):
    """Create a psycopg2 connection.

    dsn defaults to the DATABASE_URL environment variable, e.g.:
        postgresql://user:password@localhost:5432/alpha_engine

    Imports psycopg2 lazily so this module works fine (for JSON-only use)
    even if psycopg2 isn't installed yet.
    """
    import psycopg2  # noqa: WPS433 - intentional lazy import

    dsn = dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        raise ValueError(
            "No Postgres DSN provided. Pass dsn=... or set the DATABASE_URL "
            "environment variable, e.g. postgresql://user:pass@localhost:5432/alpha_engine"
        )
    return psycopg2.connect(dsn)


def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLES_SQL)
    conn.commit()


def save_to_postgres(markets: list[MarketSnapshot], dsn: Optional[str] = None) -> None:
    """Upsert market metadata and insert a fresh snapshot row for each market.

    Uses ON CONFLICT so re-running ingestion is safe (idempotent for
    `markets`; `market_snapshots` gets one new row per run per market, which
    is exactly the time series you want for backtesting).
    """
    conn = get_pg_connection(dsn)
    try:
        ensure_tables(conn)
        with conn.cursor() as cur:
            for m in markets:
                cur.execute(
                    """
                    INSERT INTO markets (market_id, event_id, slug, question,
                                          category, created_at, resolution_date, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (market_id) DO UPDATE SET
                        event_id = EXCLUDED.event_id,
                        slug = EXCLUDED.slug,
                        question = EXCLUDED.question,
                        category = EXCLUDED.category,
                        resolution_date = EXCLUDED.resolution_date,
                        status = EXCLUDED.status
                    """,
                    (
                        m.market_id, m.event_id, m.slug, m.question, m.category,
                        m.created_at, m.resolution_date, m.status,
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO market_snapshots
                        (market_id, "timestamp", yes_price, no_price, volume,
                         liquidity, spread, active, closed)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (market_id, "timestamp") DO NOTHING
                    """,
                    (
                        m.market_id, m.fetched_at, m.yes_price, m.no_price,
                        m.volume, m.liquidity, m.spread, m.active, m.closed,
                    ),
                )
        conn.commit()
        logger.info("Upserted %d markets and inserted %d snapshot rows into Postgres.",
                    len(markets), len(markets))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI entry point — the actual "milestone" script
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Polymarket markets and save them locally."
    )
    parser.add_argument("--active-only", action="store_true", default=True,
                         help="Only fetch active markets (default: True)")
    parser.add_argument("--include-closed", action="store_true",
                         help="Also include closed markets")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=None,
                         help="Cap the number of pages fetched (useful for testing)")
    parser.add_argument("--hard-page-cap", type=int, default=100,
                         help="Safety limit on total pages fetched, regardless of "
                              "--max-pages, to avoid an accidental runaway fetch "
                              "(default: 100, i.e. up to 10,000 raw records)")
    parser.add_argument("--full-history", action="store_true",
                         help="Disable the end_date_min filter and pull the ENTIRE "
                              "historical market catalog (tens of thousands of "
                              "records, many already resolved). Slow — off by default.")
    parser.add_argument("--min-liquidity", type=float, default=1000.0,
                         help="Only fetch markets with at least this much liquidity "
                              "(default: 1000). Polymarket has tens of thousands of "
                              "near-zero-liquidity markets that aren't worth analyzing "
                              "— this is the main filter keeping your pull sane. Pass "
                              "0 to disable.")
    parser.add_argument("--min-volume", type=float, default=None,
                         help="Only fetch markets with at least this much total volume.")
    parser.add_argument("--to-postgres", action="store_true",
                         help="Also write to Postgres (requires DATABASE_URL env var)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RAW_DIR))
    args = parser.parse_args()

    client = PolymarketClient()

    active_filter = None if args.include_closed else True
    closed_filter = None if args.include_closed else False
    end_date_min = None if args.full_history else "now"
    liquidity_num_min = None if args.min_liquidity == 0 else args.min_liquidity

    markets = client.get_markets(
        active=active_filter,
        closed=closed_filter,
        page_size=args.page_size,
        max_pages=args.max_pages,
        end_date_min=end_date_min,
        hard_page_cap=args.hard_page_cap,
        liquidity_num_min=liquidity_num_min,
        volume_num_min=args.min_volume,
    )

    if not markets:
        logger.warning("No markets fetched. Check API availability / filters.")
        return

    out_path = save_raw_json(markets, output_dir=Path(args.output_dir))

    if args.to_postgres:
        save_to_postgres(markets)

    # Quick sanity summary so you can eyeball the milestone worked.
    sample = markets[0]
    print("\n--- Ingestion summary ---")
    print(f"Markets fetched : {len(markets)}")
    print(f"Saved to        : {out_path}")
    print(f"Sample market   : {sample.question!r}")
    print(f"  market_id     : {sample.market_id}")
    print(f"  yes/no price  : {sample.yes_price} / {sample.no_price}")
    print(f"  volume        : {sample.volume}")
    print(f"  liquidity     : {sample.liquidity}")
    print(f"  resolution    : {sample.resolution_date}")
    print(f"  status        : {sample.status}")


if __name__ == "__main__":
    main()
