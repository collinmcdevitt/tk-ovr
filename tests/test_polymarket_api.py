"""Offline sanity check for normalize_market() using realistic mock Gamma
payloads (i.e. outcomes/outcomePrices as JSON-encoded strings, string
numerics, missing fields). Doesn't hit the network at all.

Run from the repo root: python3 tests/test_polymarket_api.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))

from polymarket_api import normalize_market, save_raw_json  # noqa: E402

MOCK_MARKET_GOOD = {
    "id": "540817",
    "eventId": "12345",
    "slug": "will-fed-cut-rates-by-dec-31",
    "question": "Will the Fed cut rates by December 31?",
    "category": "Economics",
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0.62", "0.38"]',
    "volume": "184230.55",
    "liquidity": "42110.20",
    "bestBid": "0.61",
    "bestAsk": "0.63",
    "createdAt": "2026-01-15T00:00:00Z",
    "endDate": "2026-12-31T12:00:00Z",
    "active": True,
    "closed": False,
    "clobTokenIds": '["0xabc123", "0xdef456"]',
}

MOCK_MARKET_MISSING_FIELDS = {
    "id": "999999",
    "eventId": None,
    "slug": "some-obscure-market",
    "question": "Will X happen?",
    "category": None,
    "outcomes": None,
    "outcomePrices": None,
    "volume": None,
    "liquidity": "",
    "bestBid": None,
    "bestAsk": None,
    "createdAt": None,
    "endDate": None,
    "active": False,
    "closed": True,
    "clobTokenIds": None,
}


def main():
    good = normalize_market(MOCK_MARKET_GOOD)
    assert good.market_id == "540817"
    assert good.yes_price == 0.62
    assert good.no_price == 0.38
    assert good.volume == 184230.55
    assert good.liquidity == 42110.20
    assert round(good.spread, 2) == 0.02
    assert good.status == "active"
    assert good.clob_token_ids == ["0xabc123", "0xdef456"]
    print("✓ Well-formed market parsed correctly")
    print(f"  {good.question} -> YES={good.yes_price}, NO={good.no_price}, "
          f"vol={good.volume}, liq={good.liquidity}, spread={good.spread}")

    missing = normalize_market(MOCK_MARKET_MISSING_FIELDS)
    assert missing.market_id == "999999"
    assert missing.yes_price is None
    assert missing.no_price is None
    assert missing.liquidity is None
    assert missing.status == "closed"
    print("✓ Market with missing/null fields handled without crashing")
    print(f"  {missing.question} -> status={missing.status}, "
          f"yes_price={missing.yes_price}")

    # End-to-end: normalize -> save as JSON, same as the real pipeline would.
    out_path = save_raw_json([good, missing], filename_prefix="test_snapshot")
    print(f"✓ Saved test snapshot to {out_path}")

    print("\nAll offline checks passed.")


if __name__ == "__main__":
    main()
