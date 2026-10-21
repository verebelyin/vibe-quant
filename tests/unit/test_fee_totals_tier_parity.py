"""Guard: NT nets commissions per currency, so both fee tiers read one number.

Bead vibe-quant-neuvu.3 read the two tiers' fee extraction as a contradiction:
screening sums ``abs(c)`` per commission while validation takes ``abs(sum(c))``,
so a position mixing a taker charge with a maker rebate could differ. It cannot:
NautilusTrader's ``Position.commissions()`` returns ONE ``Money`` per currency
(it nets every fill; nautilus_trader position.pyx ~:578-580, :873), and a USDT-M
linear perp has one currency. R1 pins that on a real ``Position`` driven through
both real extractors.

The formulas under guard (quoted, never re-typed):

* ``vibe_quant/screening/nt_runner.py``::

      total_fees += sum(abs(float(c)) for c in pos.commissions())

* ``vibe_quant/validation/extraction.py``::

      pos_fees = sum(float(c) for c in pos.commissions())

Both report ``|net|`` as a positive fee, so a negative (rebate) rate would
silently flip the sign: R2 is the tripwire for the day a rebate tier ships.
Every shipped rate is >= 0 today, which is why R3 documents the sign flip as
unreachable. Fees are reporting-only (``fitness._fee_drag_fields``: "never
scored").
"""

from __future__ import annotations

import time
from dataclasses import fields
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import LiquiditySide, OrderSide
from nautilus_trader.model.identifiers import PositionId
from nautilus_trader.model.objects import Money
from nautilus_trader.model.position import Position
from nautilus_trader.test_kit.stubs.events import TestEventStubs
from nautilus_trader.test_kit.stubs.execution import TestExecStubs

from vibe_quant.data.catalog import INSTRUMENT_CONFIGS, create_instrument
from vibe_quant.ethereal.instruments import EtherealInstrumentConfig
from vibe_quant.ethereal.venue import ETHEREAL_MAKER_FEE, ETHEREAL_TAKER_FEE
from vibe_quant.fees import BINANCE_USDTM_TIERS, FeeSchedule
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.validation.extraction import extract_results

if TYPE_CHECKING:
    from nautilus_trader.model.events import OrderFilled
    from nautilus_trader.model.instruments import CryptoPerpetual

_NO_ARCHIVE = "/nonexistent/vq_test_funding_archive.db"
RATE_MESSAGE = (
    "a negative (rebate) rate makes both tiers report |net| as a positive fee: "
    "decide the sign rule first (vibe-quant-neuvu.3)"
)


def _fill(
    instrument: CryptoPerpetual,
    side: OrderSide,
    price: str,
    commission: float,
    liquidity_side: LiquiditySide,
) -> OrderFilled:
    """One full 1.000 BTCUSDT fill, priced in USDT, with an explicit commission."""
    order = TestExecStubs.market_order(
        instrument=instrument,
        order_side=side,
        quantity=instrument.make_qty(Decimal("1.000")),
    )
    order.apply(TestEventStubs.order_submitted(order=order))
    order.apply(TestEventStubs.order_accepted(order=order))
    filled: OrderFilled = TestEventStubs.order_filled(
        order=order,
        instrument=instrument,
        last_px=instrument.make_price(Decimal(price)),
        liquidity_side=liquidity_side,
        commission=Money(commission, USDT),
        position_id=PositionId("P-1"),
    )
    return filled


def _position(entry_commission: float, exit_commission: float) -> Position:
    """A closed 1.000 BTCUSDT long: BUY @40000.0 TAKER, SELL @40100.0 MAKER."""
    instrument = create_instrument("BTCUSDT")
    position = Position(
        instrument=instrument,
        fill=_fill(instrument, OrderSide.BUY, "40000.0", entry_commission, LiquiditySide.TAKER),
    )
    position.apply(
        _fill(instrument, OrderSide.SELL, "40100.0", exit_commission, LiquiditySide.MAKER)
    )
    assert position.is_closed
    return position


def _engine(position: Position) -> SimpleNamespace:
    """An engine cache exposing the one position both extractors walk."""
    cache = SimpleNamespace(
        positions=lambda: [position],
        position_snapshots=list,
        bars=list,
        accounts=list,
    )
    return SimpleNamespace(kernel=SimpleNamespace(cache=cache))


def _bt_result() -> SimpleNamespace:
    return SimpleNamespace(
        total_positions=1,
        elapsed_time=0.0,
        stats_pnls={"USDT": {"PnL% (total)": 1.0, "Win Rate": 0.5}},
        stats_returns={"Profit Factor": 3.335, "Sharpe Ratio (252 days)": 1.0},
    )


def _venue_config() -> SimpleNamespace:
    # prob_slippage > 0 disables post-fill SPEC slippage, so fees stay the only cost.
    return SimpleNamespace(
        default_leverage=Decimal("10"),
        starting_balance_usdt=100_000,
        fill_config=SimpleNamespace(impact_coefficient=0.1, prob_slippage=1.0),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0005"),
    )


def _screening_total_fees(position: Position) -> float:
    """Run the real screening extractor: `sum(abs(c))` over ``commissions()``."""
    runner = NTScreeningRunner(
        dsl_dict={},
        symbols=["BTCUSDT"],
        start_date="",
        end_date="",
        funding_archive_path=_NO_ARCHIVE,
    )
    metrics = runner._extract_metrics(
        {}, _bt_result(), _engine(position), time.time(), starting_balance=1000.0
    )
    return float(metrics.total_fees)


def _validation_total_fees(position: Position) -> float:
    """Run the real validation extractor: `abs(sum(c))` over ``commissions()``."""
    result = extract_results(1, "s", _bt_result(), _engine(position), _venue_config())
    return float(result.total_fees)


def test_nt_nets_taker_and_rebate_into_one_commission() -> None:
    """+20 taker / -4 maker is ONE Money(16.0, USDT), and both tiers report 16.0.

    ``Position.commissions()`` nets every fill into one Money per currency, so
    screening's ``sum(abs(c))`` and validation's ``abs(sum(c))`` cannot disagree
    on a single-currency USDT-M perp. This is a guard: green on current code; a
    mutant that drops the netting (two USDT entries) makes the length assert fail.
    """
    position = _position(entry_commission=20.0, exit_commission=-4.0)

    commissions = position.commissions()
    assert len(commissions) == 1
    assert float(commissions[0]) == 16.0

    assert _screening_total_fees(position) == 16.0
    assert _validation_total_fees(position) == 16.0


def _shipped_fee_rates() -> list[tuple[str, Decimal]]:
    rates: list[tuple[str, Decimal]] = []
    for tier in BINANCE_USDTM_TIERS:
        for bnb_discount in (False, True):
            schedule = FeeSchedule(tier=tier, bnb_discount=bnb_discount)
            rates.append((f"fees:{tier}:maker:bnb={bnb_discount}", schedule.maker_rate))
            rates.append((f"fees:{tier}:taker:bnb={bnb_discount}", schedule.taker_rate))
    for symbol, config in INSTRUMENT_CONFIGS.items():
        rates.append((f"catalog:{symbol}:maker_fee", cast("Decimal", config["maker_fee"])))
        rates.append((f"catalog:{symbol}:taker_fee", cast("Decimal", config["taker_fee"])))
    rates.append(("ethereal:ETHEREAL_MAKER_FEE", ETHEREAL_MAKER_FEE))
    rates.append(("ethereal:ETHEREAL_TAKER_FEE", ETHEREAL_TAKER_FEE))
    for field in fields(EtherealInstrumentConfig):
        if field.name in ("maker_fee", "taker_fee"):
            rates.append(
                (
                    f"EtherealInstrumentConfig.default:{field.name}",
                    cast("Decimal", field.default),
                )
            )
    return rates


def test_every_shipped_fee_rate_is_non_negative() -> None:
    """The tripwire: a rebate rate makes both tiers report |net| as a positive fee."""
    rates = _shipped_fee_rates()
    assert {label.split(":")[0] for label, _ in rates} == {
        "fees",
        "catalog",
        "ethereal",
        "EtherealInstrumentConfig.default",
    }
    for _, rate in rates:
        assert rate >= 0, RATE_MESSAGE


def test_net_rebate_position_shows_the_known_sign_flip() -> None:
    """Documented, NOT fixed: a net rebate is reported as a positive fee.

    Fills +2 taker / -8 maker net to Money(-6.0, USDT); both tiers report 6.0,
    losing the sign. Unreachable while R2 holds (every shipped rate >= 0), so
    this pins today's behaviour instead of changing it.
    """
    position = _position(entry_commission=2.0, exit_commission=-8.0)

    commissions = position.commissions()
    assert commissions == [Money(-6.0, USDT)]
    assert float(commissions[0]) == -6.0

    assert _screening_total_fees(position) == 6.0
    assert _validation_total_fees(position) == 6.0
