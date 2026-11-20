"""Cross-account liquidation overlay of the carry book (x0axf.14 T6).

Contract rows R5 (``liquidation_mode`` range), R39 (cross, no hit: the scanner's
notes), R40 (position views built from a run), R41 (a bust raises
``CarryLiquidated`` before the daily ``E(B) <= 0`` check), R44 (horizon, fill-minute
wick, interior gap, simultaneous extremes) and R46 (scanner data errors become
``CarryDataError`` with the original text) of design x0axf.14 §3.

The runs are hand-built (``CarryFill``/``FundingPayment``) on a continuous 1m MARK
fixture, so every bust number is hand-checkable: E0 10,000, a 2.0 ETH short at 2,500
(notional 5,000), a MARK high of 7,750 (3.1x) -> margin balance ``9,998 - 10,500 =
-502`` against maintenance ``15,500 * 0.004 = 62``. Both executors are covered by the
CLI test (``test_carry_cli::test_liquidated_run_recorded``).
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vibe_quant.data.catalog import create_instrument
from vibe_quant.data.mark_bars import MarkSeries
from vibe_quant.portfolio.ledger import CarryFill, CarryRun, FundingPayment
from vibe_quant.portfolio.liquidation import carry_liquidation, carry_views, require_tier_symbols
from vibe_quant.portfolio.metrics import carry_metrics, weekly_sharpe_only
from vibe_quant.portfolio.sizing import Decision, InstrumentSpec
from vibe_quant.portfolio.spec import (
    CarryDataError,
    CarryEquityTooSmall,
    CarryLiquidated,
    CarrySpec,
)
from vibe_quant.portfolio.ticks import CarryInputs
from vibe_quant.validation.funding import FundingCalculator

if TYPE_CHECKING:
    from collections.abc import Mapping

_MIN = 60 * 10**9
_HOUR = 60 * _MIN
_DAY = 24 * _HOUR
_WEEK = 7 * _DAY
_START = 1_736_121_600_000_000_000  # 2025-01-06 00:00 UTC, a Monday
_END = _START + 2 * _WEEK
_D0 = _START + 5 * _MIN
_FILL_TS = _D0 + 59_999_000_000  # the fill tick: ts_init of the bar opening at D0
_BTC = "BTCUSDT"
_ETH = "ETHUSDT"
_PRICE = {_BTC: 50_000.0, _ETH: 2_500.0}
_SPIKE = _D0 + _DAY  # a bar open one day after the decision
_NOTES_BASE: dict[str, Any] = {
    "mode": "cross",
    "mark_source": "1m_mark",
    "tier_table_version": "2026-10-17",
    "clearance_rule": "forfeit_remaining_margin",
    "leverage": 10.0,
    "total_forfeited": 0.0,
    "total_realized_pnl": 0.0,
    "total_insurance_deficit": 0.0,
    "path_divergence": False,
    "dropped": 0,
    "fill_rule": "lp_or_open_if_gapped",
    "gap_rule": "carry_forward_last_close_max_60m",
}


def _spec(mode: str = "cross", **kw: Any) -> CarrySpec:
    return CarrySpec(
        symbols=(_BTC, _ETH), start_ns=_START, end_ns=_END, liquidation_mode=mode, **kw
    )


def _marks(
    price: float,
    bars: Mapping[int, Mapping[str, float]] | None = None,
    missing: range | tuple[int, ...] = (),
) -> MarkSeries:
    """A continuous 1m MARK series from ``START - 8 h`` to ``END + 1 h`` (opens past the end)."""
    ts = np.arange(_START - 8 * _HOUR, _END + _HOUR, _MIN, dtype=np.int64)
    columns = {name: np.full(ts.shape, price, dtype=np.float64) for name in MarkSeries._fields[1:]}
    for at, values in (bars or {}).items():
        for name, value in values.items():
            columns[name][(at - int(ts[0])) // _MIN] = value
    keep = ~np.isin(ts, np.asarray(missing, dtype=np.int64))
    return MarkSeries(ts=ts[keep], **{name: column[keep] for name, column in columns.items()})


def _fill(symbol: str, ts: int, qty: str, price: float, fee: float = 0.0) -> CarryFill:
    step = InstrumentSpec.from_instrument(create_instrument(symbol)).step
    steps = int(Decimal(qty) / step)
    return CarryFill(symbol, ts, steps, float(Decimal(steps) * step), price, fee, False)


def _inputs(
    spec: CarrySpec,
    marks: Mapping[str, MarkSeries],
    boundaries: Mapping[str, Mapping[int, float]] | None = None,
) -> CarryInputs:
    flat = {
        symbol: {b: _PRICE[symbol] for b in range(spec.start_ns, spec.end_ns + _DAY, _DAY)}
        for symbol in spec.symbols
    }
    return CarryInputs(
        spec=spec,
        decisions=(_D0, _D0 + _WEEK),
        instruments={s: create_instrument(s) for s in spec.symbols},
        specs={s: InstrumentSpec.from_instrument(create_instrument(s)) for s in spec.symbols},
        ticks={},
        settlements={},
        marks=dict(marks),
        calc=FundingCalculator(),
        boundary_prices={s: dict(v) for s, v in (boundaries or flat).items()},
    )


def _run(
    spec: CarrySpec, fills: list[CarryFill], payments: list[FundingPayment] | None = None
) -> CarryRun:
    decision = Decision(
        ts=_D0,
        equity=spec.initial_balance,
        signals={},
        excluded={},
        weights={},
        prices={},
        target_steps={},
        orders=(),
        band_skips=0,
        min_notional_skips=0,
        rounding_drift=0.0,
    )
    return CarryRun(
        spec=spec,
        decisions=(decision,),
        fills=tuple(fills),
        payments=tuple(payments or ()),
        final_steps={},
        engine="reference",
    )


def _short_eth(spike: Mapping[int, Mapping[str, float]] | None = None, **kw: Any) -> tuple[CarryRun, CarryInputs]:
    """E0 10,000; short 2.0 ETH at 2,500 (5,000 notional, fee 2.0) at the fill tick."""
    spec = _spec(**kw)
    run = _run(spec, [_fill(_ETH, _FILL_TS, "-2.0", 2_500.0, fee=2.0)])
    marks = {_BTC: _marks(_PRICE[_BTC]), _ETH: _marks(_PRICE[_ETH], spike)}
    return run, _inputs(spec, marks)


def _pair(
    eth: Mapping[int, Mapping[str, float]] | None = None,
    btc: Mapping[int, Mapping[str, float]] | None = None,
    eth_missing: tuple[int, ...] = (),
) -> tuple[CarryRun, CarryInputs]:
    """Long 0.1 BTC at 50,000 and short 2.0 ETH at 2,500, no fees (wallet 10,000)."""
    spec = _spec()
    run = _run(
        spec,
        [_fill(_BTC, _FILL_TS, "0.1", 50_000.0), _fill(_ETH, _FILL_TS, "-2.0", 2_500.0)],
    )
    marks = {_BTC: _marks(_PRICE[_BTC], btc), _ETH: _marks(_PRICE[_ETH], eth, eth_missing)}
    return run, _inputs(spec, marks)


# --- R5: the liquidation_mode field ---------------------------------------------------


@pytest.mark.parametrize("bad", ["isolated", "Cross", None, ""])
def test_liquidation_mode_range(bad: object) -> None:
    """R5: only 'off'/'cross'; cross adds its key to the hash, off keeps the default hash."""
    with pytest.raises(ValueError, match=r"liquidation_mode must be one of \('off', 'cross'\)"):
        _spec(mode=bad)  # type: ignore[arg-type]
    default = CarrySpec(symbols=(_BTC, _ETH), start_ns=_START, end_ns=_END)
    off = _spec(mode="off")
    cross = _spec()
    assert off.to_json() == default.to_json()
    assert "liquidation_mode" not in off.to_json()
    assert off.spec_hash() == default.spec_hash()
    assert cross.to_json() == {**default.to_json(), "liquidation_mode": "cross"}
    assert cross.spec_hash() != default.spec_hash()


# --- R39 / R40: notes and views ---------------------------------------------------------


def test_cross_no_hit_notes() -> None:
    """R39: no hit -> the scanner's notes (count 0, bust_ts None, tier 2026-10-17)."""
    run, inputs = _short_eth()
    expected = {**_NOTES_BASE, "count": 0, "pct_trades": 0.0, "bust_ts": None, "events": []}

    assert carry_liquidation(run, inputs) == expected
    assert carry_metrics(run, inputs).summary["liquidation"] == expected
    off_run, off_inputs = _short_eth(mode="off")
    assert carry_metrics(off_run, off_inputs).summary["liquidation"] == "not_modelled"


class _Spy:
    """Records the scanner call and returns an empty cross result."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[Any], dict[str, Any], dict[str, Any]]] = []

    def __call__(self, views: list[Any], marks: dict[str, Any], **kw: Any) -> Any:
        from vibe_quant.liquidation.scan import LiquidationResult

        self.calls.append((views, marks, kw))
        return LiquidationResult(mode="cross", leverage=10.0, tier_table_version="t")


def test_views_from_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """R40: one view per symbol, events in ``run.fills`` order with cash -fee, funding
    (ts, -amount), the spec's balance and leverage 10, MARK bars opening before the end."""
    spy = _Spy()
    monkeypatch.setattr("vibe_quant.liquidation.scan.scan_liquidations", spy)
    t1, t2, t3 = _FILL_TS, _FILL_TS + _HOUR, _D0 + 8 * _HOUR
    spec = _spec(initial_balance=12_345.0)
    run = _run(
        spec,
        [
            _fill(_ETH, t1, "-2.0", 2_500.0, fee=2.5),
            _fill(_BTC, t1, "0.1", 50_000.0, fee=2.5),
            _fill(_ETH, t2, "1.5", 2_400.0, fee=1.8),
        ],
        [FundingPayment(_BTC, t3, 0.25, False), FundingPayment(_ETH, t3, -1.5, False)],
    )
    marks = {_BTC: _marks(_PRICE[_BTC]), _ETH: _marks(_PRICE[_ETH])}
    carry_liquidation(run, _inputs(spec, marks))

    ((views, sliced, kw),) = spy.calls
    assert kw["mode"] == "cross"
    assert kw["starting_balance"] == 12_345.0
    assert kw["leverage"] == 10.0
    assert [(v.key, v.symbol, v.instrument_id) for v in views] == [
        (_BTC, _BTC, "BTCUSDT-PERP.BINANCE"),
        (_ETH, _ETH, "ETHUSDT-PERP.BINANCE"),
    ]
    assert [tuple(e) for e in views[0].events] == [(t1, 0.1, 50_000.0, -2.5)]
    assert [tuple(e) for e in views[1].events] == [(t1, -2.0, 2_500.0, -2.5), (t2, 1.5, 2_400.0, -1.8)]
    assert list(views[0].funding) == [(t3, -0.25)]
    assert list(views[1].funding) == [(t3, 1.5)]
    for symbol in (_BTC, _ETH):
        assert int(sliced[symbol].ts[-1]) == _END - _MIN  # the bar opening at END is cut
        assert len(sliced[symbol].open) == len(sliced[symbol].ts)

    # a second, unrelated run: other balance, one symbol traded, no funding
    spy.calls.clear()
    other = _spec(initial_balance=7_000.0)
    carry_liquidation(
        _run(other, [_fill(_BTC, t2, "-0.2", 48_000.0, fee=4.8)]), _inputs(other, marks)
    )
    ((views, _sliced, kw),) = spy.calls
    assert kw["starting_balance"] == 7_000.0
    assert [tuple(e) for e in views[0].events] == [(t2, -0.2, 48_000.0, -4.8)]
    assert list(views[1].events) == []
    assert list(views[0].funding) == list(views[1].funding) == []


# --- R41: a bust raises before the E(B) <= 0 check ---------------------------------------


def test_cross_bust_raises() -> None:
    """R41: MARK high 3.1x entry on one minute busts the account: ``CarryLiquidated`` at the
    bar open, legs (ETHUSDT,), raised before the daily equity walk would see E(B) = -502."""
    eth_after = {b: 2_500.0 if b < _START + 2 * _DAY else 7_750.0 for b in range(_START, _END + _DAY, _DAY)}
    boundaries = {_BTC: {}, _ETH: eth_after}
    spike = {_SPIKE: {"high": 7_750.0}}

    off_run, _ = _short_eth(spike, mode="off")
    off_inputs = _inputs(off_run.spec, {_ETH: _marks(2_500.0, spike)}, boundaries)
    with pytest.raises(CarryEquityTooSmall) as ruin:
        carry_metrics(off_run, off_inputs)
    assert type(ruin.value) is CarryEquityTooSmall  # off: the daily walk's ruin, at 00:00
    assert (ruin.value.symbol, ruin.value.decision_ns) == ("<book>", _START + 2 * _DAY)

    run, _ = _short_eth(spike)
    inputs = _inputs(run.spec, {_ETH: _marks(2_500.0, spike)}, boundaries)
    for compute in (carry_metrics, weekly_sharpe_only):
        with pytest.raises(CarryLiquidated) as bust:
            compute(run, inputs)  # type: ignore[operator]
        error = bust.value
        assert error.decision_ns == _SPIKE
        assert error.symbols == (_ETH,)
        assert error.margin_balance == -502.0
        assert error.maint_margin == 62.0
        assert str(error) == (
            f"account bust (cross) at {_SPIKE}: legs ETHUSDT margin balance -502.0 "
            "<= maintenance 62.0"
        )

    # 2.9x is not a bust: MB 9998 - 9500 = 498 > MM 58
    calm, calm_inputs = _short_eth({_SPIKE: {"high": 7_250.0}})
    assert carry_liquidation(calm, calm_inputs)["count"] == 0


# --- R44: horizon, fill-minute wick, interior gap, simultaneous extremes -----------------


def test_liquidation_horizon_gap_and_extrema() -> None:
    """R44: the bar opening at END is outside the run, END - 60 s inside; a wick in the
    fill minute hits at the fill ts; a missing interior minute is valued at the previous
    close; cross charges both legs' adverse extremes of one minute together."""
    run, inputs = _short_eth({_END: {"high": 7_750.0}})
    assert carry_liquidation(run, inputs)["bust_ts"] is None
    run, inputs = _short_eth({_END - _MIN: {"high": 7_750.0}})
    with pytest.raises(CarryLiquidated) as edge:
        carry_liquidation(run, inputs)
    assert edge.value.decision_ns == _END - _MIN

    run, inputs = _short_eth({_D0: {"high": 7_750.0}})  # the bar the fill tick closes
    with pytest.raises(CarryLiquidated) as wick:
        carry_liquidation(run, inputs)
    assert wick.value.decision_ns == _FILL_TS

    # each extreme alone survives: ETH high 6,000 -> MB 3,000; BTC low 20,000 -> MB 7,000
    assert carry_liquidation(*_pair(eth={_SPIKE: {"high": 6_000.0}}))["count"] == 0
    assert carry_liquidation(*_pair(btc={_SPIKE: {"low": 20_000.0}}))["count"] == 0
    # together in one minute: MB 10,000 - 7,000 - 3,000 = 0 <= MM 8 + 48
    with pytest.raises(CarryLiquidated) as both:
        carry_liquidation(
            *_pair(eth={_SPIKE: {"high": 6_000.0}}, btc={_SPIKE: {"low": 20_000.0}})
        )
    assert both.value.decision_ns == _SPIKE
    assert both.value.symbols == (_BTC, _ETH)
    assert (both.value.margin_balance, both.value.maint_margin) == (0.0, 56.0)

    # ETH's bar at the spike minute is missing: it carries the previous bar's close 6,000
    before = _SPIKE - _MIN
    carried = {before: {"high": 6_000.0, "close": 6_000.0}}
    with pytest.raises(CarryLiquidated) as gap:
        carry_liquidation(
            *_pair(eth=carried, btc={_SPIKE: {"low": 20_000.0}}, eth_missing=(_SPIKE,))
        )
    assert gap.value.decision_ns == _SPIKE
    # same gap, previous close back at 2,500: valued at 2,500, no bust and no data error
    recovered = {before: {"high": 6_000.0}}
    notes = carry_liquidation(
        *_pair(eth=recovered, btc={_SPIKE: {"low": 20_000.0}}, eth_missing=(_SPIKE,))
    )
    assert notes["count"] == 0


# --- R46: scanner data errors keep their text --------------------------------------------


def test_liquidation_data_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """R46: a missing series, a > 60 min MARK hole, an out-of-tier mark (even after an
    earlier bust in the same segment) and a symbol without a tier table are
    ``CarryDataError`` with the scanner's text, never a bust."""
    run, inputs = _short_eth()
    with pytest.raises(CarryDataError, match=r"^ETHUSDT: no mark-price series for position ETHUSDT$"):
        carry_liquidation(run, _inputs(run.spec, {_BTC: _marks(_PRICE[_BTC])}))

    hole = tuple(range(_SPIKE, _SPIKE + 61 * _MIN, _MIN))  # bars 62 min apart
    with pytest.raises(CarryDataError, match=r"^ETHUSDT: mark gap of more than 60 minutes"):
        carry_liquidation(run, _inputs(run.spec, {_ETH: _marks(2_500.0, None, hole)}))
    short_hole = tuple(range(_SPIKE, _SPIKE + 59 * _MIN, _MIN))  # bars 60 min apart: allowed
    assert carry_liquidation(run, _inputs(run.spec, {_ETH: _marks(2_500.0, None, short_hole)}))["count"] == 0

    # the bust at the spike is masked by a later out-of-tier mark in the same segment
    masked = {_SPIKE: {"high": 7_750.0}, _SPIKE + _HOUR: {"high": 1e9}}
    run, inputs = _short_eth(masked)
    with pytest.raises(CarryDataError, match=r"^ETHUSDT: notional 2000000000\.0 >= last cap 1200000000\.0$") as tier:
        carry_liquidation(run, inputs)
    assert not isinstance(tier.value, CarryEquityTooSmall)

    from vibe_quant.liquidation.tiers import TierTables, load_tier_tables

    shipped = load_tier_tables()
    no_eth = TierTables(version=shipped.version, fetched_at="", source="", note="")
    no_eth.update({s: t for s, t in shipped.items() if s != _ETH})
    monkeypatch.setattr("vibe_quant.liquidation.tiers.load_tier_tables", lambda: no_eth)
    with pytest.raises(CarryDataError, match=r"^ETHUSDT$"):
        carry_liquidation(*_short_eth())
    with pytest.raises(CarryDataError, match=r"^no liquidation tier table for ETHUSDT in 2026-10-17$"):
        require_tier_symbols((_BTC, _ETH))
    require_tier_symbols((_BTC,))


def test_views_cover_every_spec_symbol() -> None:
    """R40: a symbol that never trades still gets its (empty) view."""
    spec = _spec()
    views = carry_views(_run(spec, [_fill(_ETH, _FILL_TS, "-1.0", 2_500.0)]))
    assert [(v.symbol, len(v.events)) for v in views] == [(_BTC, 0), (_ETH, 1)]
