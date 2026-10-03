"""Position sizers honour instrument lot rules (audit MEDIUM, risk/sizing.py)."""

from __future__ import annotations

from decimal import Decimal

from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from vibe_quant.risk.sizing import (
    ATRConfig,
    ATRSizer,
    FixedFractionalConfig,
    FixedFractionalSizer,
)


def _coarse(min_notional: Money | None = None) -> CryptoPerpetual:
    btc = TestInstrumentProvider.btcusdt_perp_binance()
    return CryptoPerpetual(
        instrument_id=btc.id,
        raw_symbol=btc.raw_symbol,
        base_currency=btc.base_currency,
        quote_currency=btc.quote_currency,
        settlement_currency=btc.settlement_currency,
        is_inverse=False,
        price_precision=1,
        size_precision=3,
        price_increment=Price.from_str("0.1"),
        size_increment=Quantity.from_str("0.005"),
        max_quantity=None,
        min_quantity=Quantity.from_str("0.005"),
        max_notional=None,
        min_notional=min_notional,
        max_price=None,
        min_price=None,
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.025"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0004"),
        ts_event=0,
        ts_init=0,
    )


FF = FixedFractionalSizer(
    FixedFractionalConfig(
        max_leverage=Decimal(20), max_position_pct=Decimal("1"), risk_per_trade=Decimal("0.02")
    )
)


def test_size_rounds_down_to_size_increment() -> None:
    # risk 200 / stop distance 1626 = 0.12300... -> precision 0.123 -> step 0.005 -> 0.120
    q = FF.calculate_size(Decimal(10000), _coarse(), Decimal(50000), stop_price=Decimal(48374))
    assert q == Quantity.from_str("0.120")


def test_below_min_quantity_means_no_trade() -> None:
    # equity 20: risk 0.4 / 10000 = 0.00004 -> 0 lots (venue minimum 0.005)
    q = FF.calculate_size(Decimal(20), _coarse(), Decimal(50000), stop_price=Decimal(40000))
    assert q == Quantity.zero(precision=3)


def test_below_min_notional_means_no_trade() -> None:
    # 0.005 BTC * 50000 = 250 USDT < 300 USDT minimum notional
    inst = _coarse(min_notional=Money(300, _coarse().quote_currency))
    sizer = FixedFractionalSizer(
        FixedFractionalConfig(
            max_leverage=Decimal(20), max_position_pct=Decimal("1"), risk_per_trade=Decimal("0.01")
        )
    )
    # risk 2.5 / 500 = 0.005 BTC -> notional 250 < 300 -> 0
    q = sizer.calculate_size(Decimal(250), inst, Decimal(50000), stop_price=Decimal(49500))
    assert q == Quantity.zero(precision=3)


def test_atr_sizer_also_respects_increment() -> None:
    sizer = ATRSizer(
        ATRConfig(
            max_leverage=Decimal(20),
            max_position_pct=Decimal("1"),
            risk_per_trade=Decimal("0.02"),
            atr_multiplier=Decimal("2"),
        )
    )
    # 200 / (2 * 813) = 0.12300 -> 0.120
    q = sizer.calculate_size(Decimal(10000), _coarse(), Decimal(50000), atr=Decimal(813))
    assert q == Quantity.from_str("0.120")
