"""Instrument specs must match Binance USDT-M exchangeInfo (fixture fetched 2026-10-10)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from vibe_quant.data.catalog import INSTRUMENT_CONFIGS, create_instrument

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "binance_exchange_info_usdtm.json"
_INFO = {s["symbol"]: {f["filterType"]: f for f in s["filters"]} for s in json.loads(_FIXTURE.read_text())["symbols"]}


_HISTORICAL_TICK = {"SOLUSDT": Decimal("0.001")}


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
