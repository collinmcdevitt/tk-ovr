"""Offline tests for compute_features() using synthetic snapshot histories.
Doesn't touch Postgres or the network — pure pandas/numpy logic checks.

Run from the repo root: python3 tests/test_feature_engineering.py
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.feature_engineering import compute_features, _safe_logit  # noqa: E402


def make_synthetic_history(n_hours=48, start_price=0.5, seed=42):
    """Hourly snapshots over n_hours, price doing a small random walk in
    (0.05, 0.95), plus volume/liquidity/spread that also drift a bit."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=i) for i in range(n_hours)]

    price = start_price
    prices, volumes, liquidities, spreads = [], [], [], []
    volume, liquidity = 10000.0, 5000.0

    for _ in range(n_hours):
        price = float(np.clip(price + rng.normal(0, 0.01), 0.02, 0.98))
        volume = max(100.0, volume + rng.normal(0, 200))
        liquidity = max(100.0, liquidity + rng.normal(0, 100))
        prices.append(price)
        volumes.append(volume)
        liquidities.append(liquidity)
        spreads.append(abs(rng.normal(0.01, 0.005)))

    return pd.DataFrame({
        "timestamp": timestamps,
        "yes_price": prices,
        "volume": volumes,
        "liquidity": liquidities,
        "spread": spreads,
    })


def test_basic_shape_and_no_crash():
    history = make_synthetic_history(n_hours=48)
    resolution = datetime(2026, 1, 10, tzinfo=timezone.utc)
    features = compute_features(history, resolution_date=resolution)

    assert len(features) == len(history), "should produce one feature row per snapshot"
    expected_cols = {
        "price_change_1h", "price_change_24h", "price_change_7d",
        "logit_change_1h", "logit_change_24h", "logit_change_7d",
        "volatility_1h", "volatility_24h", "volatility_7d",
        "volume_change_24h", "liquidity_change_24h", "spread_pct",
        "hours_to_resolution", "days_to_resolution", "distance_from_50",
    }
    assert expected_cols.issubset(set(features.columns)), "missing expected feature columns"
    print("✓ compute_features runs without crashing and produces all expected columns")


def test_early_rows_are_nan_for_long_windows():
    """With only 48 hourly rows, 7-day (168h) windows can never have a full
    lookback — every row should be NaN for the 7d features. This confirms
    we're not fabricating values when there isn't enough history."""
    history = make_synthetic_history(n_hours=48)
    features = compute_features(history)
    assert features["price_change_7d"].isna().all(), (
        "7d features should all be NaN with only 48h of history — "
        "getting real values here would mean we're using rows from a window "
        "that couldn't actually exist yet"
    )
    print("✓ Insufficient-history windows correctly produce NaN rather than fabricated values")


def test_price_change_and_logit_change_are_consistent():
    """Manually verify one row's price_change_1h and logit_change_1h against
    a hand-built two-point series, so we know the math itself is right."""
    timestamps = [
        datetime(2026, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
    ]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "yes_price": [0.40, 0.50],
        "volume": [1000.0, 1200.0],
        "liquidity": [500.0, 550.0],
        "spread": [0.01, 0.01],
    })
    features = compute_features(df)
    second_row = features.iloc[1]

    assert abs(second_row["price_change_1h"] - 0.10) < 1e-9, (
        f"expected price_change_1h ≈ 0.10, got {second_row['price_change_1h']}"
    )

    expected_logit_change = _safe_logit(pd.Series([0.50])).iloc[0] - _safe_logit(pd.Series([0.40])).iloc[0]
    assert abs(second_row["logit_change_1h"] - expected_logit_change) < 1e-9, (
        f"expected logit_change_1h ≈ {expected_logit_change}, got {second_row['logit_change_1h']}"
    )
    print(f"✓ price_change_1h and logit_change_1h match hand-computed values "
          f"(price_change={second_row['price_change_1h']:.4f}, "
          f"logit_change={second_row['logit_change_1h']:.4f})")


def test_logit_handles_extreme_probabilities_without_inf():
    """Prices very close to 0 or 1 are common near market resolution.
    Naive log(p/(1-p)) would blow up to +/-inf there — confirm we clip
    instead of producing infinities that would break a downstream model."""
    df = pd.DataFrame({
        "timestamp": [
            datetime(2026, 1, 1, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        ],
        "yes_price": [0.0001, 0.9999],
        "volume": [1000.0, 1000.0],
        "liquidity": [500.0, 500.0],
        "spread": [0.01, 0.01],
    })
    features = compute_features(df)
    assert np.isfinite(features["logit_change_1h"].iloc[1]), (
        "logit_change should stay finite even at extreme probabilities near 0/1"
    )
    print("✓ Extreme probabilities near 0/1 produce finite logit values (no inf/-inf)")


def test_distance_from_50_and_time_to_resolution():
    history = make_synthetic_history(n_hours=5, start_price=0.5, seed=1)
    resolution = history["timestamp"].iloc[-1] + timedelta(hours=10)
    features = compute_features(history, resolution_date=resolution)

    last_row = features.iloc[-1]
    assert abs(last_row["hours_to_resolution"] - 10.0) < 1e-6
    assert abs(last_row["days_to_resolution"] - 10.0 / 24) < 1e-6

    expected_distance = abs(history["yes_price"].iloc[0] - 0.5)
    assert abs(features["distance_from_50"].iloc[0] - expected_distance) < 1e-9
    print("✓ distance_from_50 and hours/days_to_resolution compute correctly")


def test_empty_history_does_not_crash():
    empty = pd.DataFrame(columns=["timestamp", "yes_price", "volume", "liquidity", "spread"])
    result = compute_features(empty)
    assert result.empty
    print("✓ Empty snapshot history handled gracefully (no crash)")


if __name__ == "__main__":
    test_basic_shape_and_no_crash()
    test_early_rows_are_nan_for_long_windows()
    test_price_change_and_logit_change_are_consistent()
    test_logit_handles_extreme_probabilities_without_inf()
    test_distance_from_50_and_time_to_resolution()
    test_empty_history_does_not_crash()
    print("\nAll offline feature engineering checks passed.")
