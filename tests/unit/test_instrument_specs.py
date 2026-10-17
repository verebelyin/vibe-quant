"""Instrument specs must match Binance USDT-M exchangeInfo (fixture fetched 2026-10-17)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from vibe_quant.data.catalog import INSTRUMENT_CONFIGS, create_instrument

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "binance_exchange_info_usdtm.json"
_INFO = {s["symbol"]: {f["filterType"]: f for f in s["filters"]} for s in json.loads(_FIXTURE.read_text())["symbols"]}


_HISTORICAL_TICK = {"SOLUSDT": Decimal("0.001"), "NEARUSDT": Decimal("0.0001")}


def test_fixture_covers_all_configured_symbols() -> None:
    assert set(_INFO) == set(INSTRUMENT_CONFIGS)


@pytest.mark.parametrize("symbol", sorted(INSTRUMENT_CONFIGS))
def test_spec_matches_exchange_info(symbol: str) -> None:
    f = _INFO[symbol]
    inst = create_instrument(symbol)
    tick = Decimal(f["PRICE_FILTER"]["tickSize"])
    # Deliberate deviation: SOL's tick was 0.001 for most of 2022-2025 and the archived closes
    # carry the 3rd decimal; today's 0.01 would destroy real history (chief decision, yul7u.18).
    # The bug being fixed is the SOL size step (was 1), not the tick.
    expected_tick = _HISTORICAL_TICK.get(symbol, tick)
    assert Decimal(str(inst.price_increment)) == expected_tick
    assert Decimal(str(inst.size_increment)) == Decimal(f["LOT_SIZE"]["stepSize"])
    assert Decimal(str(inst.min_quantity)) == Decimal(f["LOT_SIZE"]["minQty"])
    assert Decimal(str(inst.min_quantity)) == Decimal(f["MARKET_LOT_SIZE"]["minQty"])
    assert inst.min_notional is not None
    assert inst.min_notional.as_decimal() == Decimal(f["MIN_NOTIONAL"]["notional"])


def test_sol_half_unit_not_rounded_to_zero() -> None:
    qty = create_instrument("SOLUSDT").make_qty(0.5, round_down=True)
    assert float(qty) == 0.5


def test_near_history_tick_keeps_fourth_decimal() -> None:
    """R3: NEAR's tick was 0.0001 until ~2022-02, so a 0.6175 close must survive.

    With today's 0.001 tick NT would round the same input to 0.618 (MEASURED probe).
    """
    inst = create_instrument("NEARUSDT")
    assert float(inst.make_price(0.6175)) == 0.6175
    assert float(inst.make_price(0.61754)) == 0.6175


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("SUIUSDT", 2.7), ("ZECUSDT", 2.777), ("NEARUSDT", 2), ("HYPEUSDT", 2.77)],
)
def test_new_coin_lot_rounding(symbol: str, expected: float) -> None:
    """R4 / R4b: round_down to each new coin's size increment."""
    assert float(create_instrument(symbol).make_qty(2.777, round_down=True)) == expected


def test_near_sub_unit_lot_rounds_to_zero() -> None:
    """R4: NEAR's step is 1, so a sub-1 lot rounds to zero; templates skip the entry."""
    with pytest.raises(ValueError, match="rounded to zero due to size increment 1"):
        create_instrument("NEARUSDT").make_qty(0.4, round_down=True)


def test_new_coin_min_notional_edges() -> None:
    """R5: min_notional is 5 for all four; templates.py:782 skips strictly below it."""
    for symbol in ("SUIUSDT", "ZECUSDT", "NEARUSDT", "HYPEUSDT"):
        inst = create_instrument(symbol)
        assert inst.min_notional is not None
        assert inst.min_notional.as_decimal() == Decimal(5)
    qty = create_instrument("NEARUSDT").make_qty(2, round_down=True)
    assert float(qty) * 2.5 == 5.0  # notional 5.0, not < 5 -> entered
    assert float(qty) * 1.8 < 5  # notional 3.6 -> skipped


def test_write_instrument_replaces_changed_and_skips_identical(tmp_path: Path) -> None:
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.objects import Money, Quantity

    from vibe_quant.data.catalog import CatalogManager

    new = create_instrument("SOLUSDT")
    old = type(new).from_dict({**type(new).to_dict(new), "size_increment": "1", "size_precision": 0,
                               "min_quantity": None})
    mgr = CatalogManager(tmp_path)
    mgr.write_instrument(old)
    assert mgr.get_instruments()[0].size_increment == Quantity.from_str("1")

    mgr.write_instrument(new)
    got = CatalogManager(tmp_path).get_instruments()
    assert len(got) == 1
    assert got[0].size_increment == Quantity.from_str("0.01")
    assert got[0].min_quantity == Quantity.from_str("0.01")
    assert got[0].min_notional == Money(5, got[0].quote_currency)
    assert got[0].id == InstrumentId.from_str("SOLUSDT-PERP.BINANCE")

    # Identical rewrite must not touch the file (no churn).
    f = next((tmp_path / "data" / "crypto_perpetual" / "SOLUSDT-PERP.BINANCE").glob("*.parquet"))
    mtime = f.stat().st_mtime_ns
    mgr.write_instrument(create_instrument("SOLUSDT"))
    assert f.stat().st_mtime_ns == mtime
