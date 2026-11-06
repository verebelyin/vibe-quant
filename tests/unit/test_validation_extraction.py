"""Tests for validation extraction module."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.fake_engine import FakePosition, fake_engine, fake_slippage, fake_venue
from tests.unit.test_screening_funding_exit_timing import interval_archive
from vibe_quant.db.state_manager import StateManager
from vibe_quant.modeled_costs import (
    CostLedger,
    CostModel,
    MarketStatsSeries,
    ModeledCostIntegrityError,
    cost_key,
)
from vibe_quant.validation.extraction import (
    _compute_max_drawdown_from_trades,
    compute_extended_metrics,
    extract_trades,
    load_bar_stats_windows,
)
from vibe_quant.validation.fill_model import SlippageEstimator
from vibe_quant.validation.funding import FundingAccrual
from vibe_quant.validation.results import TradeRecord, ValidationResult
from vibe_quant.validation.runner import ValidationRunner, build_slippage_model


def _make_trade(entry_time: str, exit_time: str, net_pnl: float = 10.0) -> TradeRecord:
    return TradeRecord(
        symbol="BTCUSDT-PERP",
        direction="LONG",
        leverage=10,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=40000.0,
        exit_price=40100.0,
        quantity=0.1,
        net_pnl=net_pnl,
        gross_pnl=net_pnl + 2.0,
        roi_percent=(net_pnl / (40000.0 * 0.1)) * 100.0,
    )


def test_trades_sorted_before_cagr_computation() -> None:
    """CAGR must use chronological first/last trade, not insertion order."""
    # Trades intentionally out of chronological order
    t_middle = _make_trade("2025-03-01T00:00:00+00:00", "2025-03-02T00:00:00+00:00")
    t_first = _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00")
    t_last = _make_trade("2025-06-01T00:00:00+00:00", "2025-06-02T00:00:00+00:00")

    result = ValidationResult(
        total_return=0.10,
        trades=[t_middle, t_first, t_last],
    )
    result.total_trades = len(result.trades)
    # Sort as extract_trades does
    result.trades.sort(key=lambda t: t.entry_time)

    compute_extended_metrics(result)

    # After sort, first trade should be Jan, last should be Jun
    assert result.trades[0].entry_time == "2025-01-01T00:00:00+00:00"
    assert result.trades[-1].entry_time == "2025-06-01T00:00:00+00:00"
    assert result.cagr != 0.0

    # Now compute with WRONG order (no sort) to show the difference
    result_bad = ValidationResult(
        total_return=0.10,
        trades=[t_middle, t_first, t_last],  # unsorted
    )
    result_bad.total_trades = len(result_bad.trades)
    compute_extended_metrics(result_bad)

    # Bad order: uses Mar as first, Jun as last => shorter period => higher CAGR
    # Correct order: Jan to Jun => longer period => lower CAGR
    assert result.cagr != result_bad.cagr, "Sorted vs unsorted trades should produce different CAGR"


def test_trades_sorted_single_trade_no_error() -> None:
    """Single trade should not break sorting or CAGR."""
    t = _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00")
    result = ValidationResult(total_return=0.05, trades=[t])
    result.total_trades = 1
    result.trades.sort(key=lambda t: t.entry_time)
    compute_extended_metrics(result)
    assert result.cagr != 0.0


def test_trades_already_sorted_unchanged() -> None:
    """Already-sorted trades should produce identical result after sort."""
    t1 = _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00")
    t2 = _make_trade("2025-02-01T00:00:00+00:00", "2025-02-02T00:00:00+00:00")
    t3 = _make_trade("2025-03-01T00:00:00+00:00", "2025-03-02T00:00:00+00:00")

    result = ValidationResult(total_return=0.10, trades=[t1, t2, t3])
    result.total_trades = 3
    result.trades.sort(key=lambda t: t.entry_time)
    compute_extended_metrics(result)

    assert result.trades[0].entry_time == "2025-01-01T00:00:00+00:00"
    assert result.trades[-1].entry_time == "2025-03-01T00:00:00+00:00"
    assert result.cagr != 0.0


def test_trade_duration_garbled_times_warn_without_changing_avg(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unparseable entry/exit pair is warned about and only that trade is dropped."""
    bad = _make_trade("not-a-date", "also-not-a-date")
    good = _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00")
    result = ValidationResult(total_return=0.10, trades=[bad, good])
    result.total_trades = len(result.trades)

    with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.extraction"):
        compute_extended_metrics(
            result, run_start_date="2024-01-01", run_end_date="2024-12-31"
        )

    # Only the good trade contributes: exactly one day.
    assert result.avg_trade_duration_hours == pytest.approx(24.0)
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "not-a-date" in m and "also-not-a-date" in m for m in messages
    ), messages


def test_cagr_unparseable_trade_span_warns_and_leaves_cagr_at_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A trade span that cannot be parsed warns and leaves CAGR at its default."""
    result = ValidationResult(
        total_return=0.10, trades=[_make_trade("nope", "also-nope")]
    )
    result.total_trades = 1

    with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.extraction"):
        compute_extended_metrics(result)

    assert result.cagr == 0.0
    messages = [record.getMessage() for record in caplog.records]
    assert any("CAGR not computed" in m for m in messages), messages


def test_cagr_unparseable_run_window_warns_and_falls_back_to_trade_span(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run window that cannot be parsed warns (naming the dates) and uses the span."""
    with_bad_window = ValidationResult(
        total_return=0.10,
        trades=[_make_trade("2024-03-01T00:00:00+00:00", "2024-04-01T00:00:00+00:00")],
    )
    with_bad_window.total_trades = 1
    no_window = ValidationResult(
        total_return=0.10,
        trades=[_make_trade("2024-03-01T00:00:00+00:00", "2024-04-01T00:00:00+00:00")],
    )
    no_window.total_trades = 1

    with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.extraction"):
        compute_extended_metrics(
            with_bad_window, run_start_date="2024-01-01", run_end_date="garbage"
        )
    compute_extended_metrics(no_window)

    assert with_bad_window.cagr != 0.0
    # The failed window falls back to the trade span, exactly as when omitted.
    assert with_bad_window.cagr == pytest.approx(no_window.cagr)
    messages = [record.getMessage() for record in caplog.records]
    assert any("2024-01-01" in m and "garbage" in m for m in messages), messages


class TestComputeMaxDrawdownFromTrades:
    """Tests for equity-curve max drawdown fallback (NT 1.222+ compat)."""

    def test_all_winning_trades_zero_drawdown(self) -> None:
        """No drawdown when all trades win."""
        trades = [
            _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00", net_pnl=100.0),
            _make_trade("2025-01-03T00:00:00+00:00", "2025-01-04T00:00:00+00:00", net_pnl=200.0),
        ]
        dd = _compute_max_drawdown_from_trades(trades, 10000.0)
        assert dd == 0.0

    def test_single_losing_trade(self) -> None:
        """Single loss from starting balance."""
        trades = [
            _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00", net_pnl=-1000.0),
        ]
        dd = _compute_max_drawdown_from_trades(trades, 10000.0)
        assert abs(dd - 0.10) < 1e-9  # 10% drawdown

    def test_recovery_drawdown(self) -> None:
        """Win then loss then win: drawdown from peak."""
        trades = [
            _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00", net_pnl=5000.0),
            _make_trade("2025-01-03T00:00:00+00:00", "2025-01-04T00:00:00+00:00", net_pnl=-3000.0),
            _make_trade("2025-01-05T00:00:00+00:00", "2025-01-06T00:00:00+00:00", net_pnl=1000.0),
        ]
        # Starting 10000 -> 15000 (peak) -> 12000 -> 13000
        # Max DD = (15000-12000)/15000 = 0.20
        dd = _compute_max_drawdown_from_trades(trades, 10000.0)
        assert abs(dd - 0.20) < 1e-9

    def test_empty_trades(self) -> None:
        """Empty trades should return 0."""
        dd = _compute_max_drawdown_from_trades([], 10000.0)
        assert dd == 0.0

    def test_zero_starting_balance(self) -> None:
        """Zero starting balance should return 0."""
        trades = [
            _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00", net_pnl=100.0),
        ]
        dd = _compute_max_drawdown_from_trades(trades, 0.0)
        assert dd == 0.0


# ---------------------------------------------------------------------------
# SPEC slippage: causal stats from the catalog's primary LAST bars (D1 / R15)
# ---------------------------------------------------------------------------

BTC = "BTCUSDT-PERP.BINANCE"
_HOUR_NS = 3_600 * 1_000_000_000
_BASE_NS = 1_735_689_600_000 * 1_000_000  # 2025-01-01T00:00Z


def _write_bar_catalog(root: Any, series: dict[str, list[tuple[int, float, float]]]) -> None:
    """A real catalog of LAST bars: ``tf -> [(ts_init, close, volume)]`` for BTCUSDT."""
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.objects import Price, Quantity

    from vibe_quant.data.catalog import CatalogManager, create_instrument, get_bar_type

    manager = CatalogManager(root)
    manager.write_instrument(create_instrument("BTCUSDT"))
    for timeframe, rows in series.items():
        bar_type = get_bar_type("BTCUSDT", timeframe)
        manager.write_bars(
            [
                Bar(
                    bar_type,
                    price := Price.from_str(f"{close:.1f}"),
                    price,
                    price,
                    price,
                    Quantity.from_str(f"{volume:.1f}"),
                    ts,
                    ts,
                )
                for ts, close, volume in rows
            ]
        )


def _venue(prob_slippage: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        fill_config=SimpleNamespace(impact_coefficient=0.1, prob_slippage=prob_slippage)
    )


def _btc_position(qty: float = 10.0) -> FakePosition:
    return FakePosition(
        id="P-1",
        instrument_id=BTC,
        ts_opened=_BASE_NS,
        ts_closed=_BASE_NS + _HOUR_NS,
        events=[],
        realized_pnl=0.0,
        avg_px_open=100.0,
        avg_px_close=101.0,
        peak_qty=qty,
        commission_list=[],
    )


def _four_hour_catalog(tmp_path: Any) -> Any:
    """4h volumes 1000/2000/3000/4000 plus a 1m decoy series (volume 5.0)."""
    root = tmp_path / "cat"
    _write_bar_catalog(
        root,
        {
            "4h": [
                (_BASE_NS + i * 4 * _HOUR_NS, 100.0 + i, 1000.0 * (i + 1)) for i in range(4)
            ],
            "1m": [(_BASE_NS + i * 60_000_000_000, 100.0, 5.0) for i in range(10)],
        },
    )
    return root


def test_runner_slippage_model_reads_the_primary_last_bars_before_the_fill(
    tmp_path: Any,
) -> None:
    """The runner's model uses the PRIMARY LAST series and excludes the fill bar."""
    root = _four_hour_catalog(tmp_path)
    model = build_slippage_model(
        _venue(), "4h", [BTC], root, _BASE_NS, _BASE_NS + 4 * 4 * _HOUR_NS
    )

    assert model.estimator is not None
    fill_ts = _BASE_NS + 2 * 4 * _HOUR_NS  # bars 0 and 1 are strictly before it
    expected = SlippageEstimator(impact_coefficient=0.1).estimate_cost(
        entry_price=100.0,
        order_size=10.0,
        avg_volume=(1000.0 + 2000.0) / 2.0,  # the 1m decoy (5.0) is not the series
        volatility=0.02,  # one return before the fill: the default
        spread=0.0001,
    )
    assert model.leg(_btc_position(), 100.0, fill_ts) == expected


def test_runner_slippage_model_falls_back_without_catalog_bars(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing primary series is observable -- the WARNING names instrument and bar."""
    with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.runner"):
        model = build_slippage_model(
            _venue(), "4h", [BTC], tmp_path / "missing", _BASE_NS, _BASE_NS + _HOUR_NS
        )
    assert model.stats == {}
    assert model.leg(_btc_position(), 100.0, _BASE_NS) == SlippageEstimator(
        impact_coefficient=0.1
    ).estimate_cost(
        entry_price=100.0,
        order_size=10.0,
        avg_volume=1000.0,
        volatility=0.02,
        spread=0.0001,
    )
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    text = record.getMessage()
    assert BTC in text
    assert f"{BTC}-4-HOUR-LAST-EXTERNAL" in text
    assert record.exc_info is None


def test_runner_slippage_model_warns_for_an_empty_primary_series(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Files that exist but cover no run row are missing too (empty loaded series)."""
    root = tmp_path / "cat"
    _write_bar_catalog(root, {"4h": [(_BASE_NS - 10 * _HOUR_NS, 100.0, 1000.0)]})

    with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.runner"):
        model = build_slippage_model(
            _venue(), "4h", [BTC], root, _BASE_NS, _BASE_NS + _HOUR_NS
        )

    assert model.stats == {}
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert BTC in record.getMessage()
    assert f"{BTC}-4-HOUR-LAST-EXTERNAL" in record.getMessage()


def test_runner_slippage_model_is_off_when_the_engine_models_slippage(tmp_path: Any) -> None:
    root = _four_hour_catalog(tmp_path)
    model = build_slippage_model(
        _venue(prob_slippage=0.5), "4h", [BTC], root, _BASE_NS, _BASE_NS + 4 * 4 * _HOUR_NS
    )
    assert model.estimator is None
    assert model.leg(_btc_position(), 100.0, _BASE_NS + 2 * 4 * _HOUR_NS) == 0.0


def test_load_bar_stats_windows_reads_close_and_volume(tmp_path: Any) -> None:
    root = _four_hour_catalog(tmp_path)
    bar_type = f"{BTC}-4-HOUR-LAST-EXTERNAL"

    loaded = load_bar_stats_windows(
        root, bar_type, [(_BASE_NS, _BASE_NS + 5 * _HOUR_NS)]
    )

    assert loaded is not None
    ts, close, volume = loaded
    assert ts.tolist() == [_BASE_NS, _BASE_NS + 4 * _HOUR_NS]
    assert close.tolist() == [100.0, 101.0]
    assert volume.tolist() == [1000.0, 2000.0]
    assert load_bar_stats_windows(tmp_path, bar_type, [(0, 1)]) is None


# ---------------------------------------------------------------------------
# Slippage + the sizing ledger in extract_trades (R21-R23, amendment note 3)
# ---------------------------------------------------------------------------


class _FundingThreeSettlements:
    """Three 8h settlements at 0.0003 on the position's entry notional (paid by a long)."""

    def accrue(
        self,
        *,
        instrument_id: str,
        direction: str,
        entry_notional: float,
        entry_ns: int,
        exit_ns: int | None,
    ) -> FundingAccrual:
        return FundingAccrual(total=3 * 0.0003 * entry_notional, payments=[])


def _cost_model() -> CostModel:
    return CostModel(
        _FundingThreeSettlements(),  # type: ignore[arg-type]
        frozenset(),
        "1m",
        60_000_000_000,
        None,
        slippage=fake_slippage(),
    )


def _closed_btc() -> FakePosition:
    return FakePosition.closed(
        "B-1",
        BTC,
        entry_ns=_BASE_NS,
        exit_ns=_BASE_NS + _HOUR_NS,
        entry_price=1000.0,
        exit_price=1010.0,
    )


def test_extract_trades_reconciles_with_cost_ledger() -> None:
    """A finalized value off by 1e-9 raises with the exact disagree text."""
    pos = _closed_btc()
    engine = fake_engine([pos])
    model = _cost_model()
    ledger = CostLedger(model, SimpleNamespace())
    ledger.on_position_closed(pos)

    result = ValidationResult(starting_balance=100_000.0)
    extract_trades(
        result,
        engine,
        fake_venue(),
        funding_calculator=model.funding,  # type: ignore[arg-type]
        slippage=model.slippage,
        cost_ledger=ledger,
    )
    assert len(result.trades) == 1
    assert result.trades[0].slippage_cost == ledger.finalized[next(iter(ledger.finalized))] - (
        3 * 0.0003 * 1000.0
    )

    (key,) = ledger.finalized
    ledger.finalized[key] += 1e-9
    with pytest.raises(ModeledCostIntegrityError) as err:
        extract_trades(
            ValidationResult(starting_balance=100_000.0),
            engine,
            fake_venue(),
            funding_calculator=model.funding,  # type: ignore[arg-type]
            slippage=model.slippage,
            cost_ledger=ledger,
        )
    assert str(err.value).startswith(
        f"modeled costs: sizing ledger and trade ledger disagree for {key}: sizing="
    )
    assert " ledger=" in str(err.value)


def test_cost_ledger_exit_leg_reads_the_bars_before_the_exit_instant() -> None:
    """The exit leg is the EXIT instant's stats (the entry instant's would differ).

    The same series the post-hoc ledger reads: bars before the entry average 1000, bars
    before the exit average 1500, so a leg charged at the wrong instant cannot reconcile.
    """
    series = MarketStatsSeries.from_bars(
        [_BASE_NS, _BASE_NS + 4 * _HOUR_NS, _BASE_NS + 8 * _HOUR_NS],
        [100.0, 101.0, 102.0],
        [1000.0, 2000.0, 3000.0],
    )
    model = CostModel(
        _FundingThreeSettlements(),  # type: ignore[arg-type]
        frozenset(),
        "1m",
        60_000_000_000,
        None,
        slippage=fake_slippage(stats={BTC: series}),
    )
    pos = FakePosition.closed(
        "B-2",
        BTC,
        entry_ns=_BASE_NS + 4 * _HOUR_NS,
        exit_ns=_BASE_NS + 8 * _HOUR_NS,
        entry_price=1000.0,
        exit_price=1010.0,
    )
    ledger = CostLedger(model, SimpleNamespace())
    ledger.on_position_closed(pos)

    estimator = SlippageEstimator(impact_coefficient=0.1)
    entry_leg = estimator.estimate_cost(1000.0, 1.0, 1000.0, 0.02, 0.0001)
    exit_leg = estimator.estimate_cost(1010.0, 1.0, 1500.0, 0.02, 0.0001)
    assert exit_leg != entry_leg
    (key,) = ledger.finalized
    assert ledger.finalized[key] == pytest.approx(3 * 0.0003 * 1000.0 + entry_leg + exit_leg)

    extract_trades(
        ValidationResult(starting_balance=100_000.0),
        fake_engine([pos]),
        fake_venue(),
        funding_calculator=model.funding,  # type: ignore[arg-type]
        slippage=model.slippage,
        cost_ledger=ledger,
    )


def test_extract_trades_requires_a_slippage_model() -> None:
    """No silent zero-slippage default."""
    with pytest.raises(TypeError):
        extract_trades(ValidationResult(), fake_engine([]), fake_venue())  # type: ignore[call-arg]


def test_extract_trades_no_positions_raises_for_a_finalized_ledger_key() -> None:
    """Note 3: the no-position early return must not hide a ledger close."""
    model = _cost_model()
    ledger = CostLedger(model, SimpleNamespace())
    ledger.on_position_closed(_closed_btc())

    with pytest.raises(ModeledCostIntegrityError, match="no closed trade for the finalized"):
        extract_trades(
            ValidationResult(),
            fake_engine([]),
            fake_venue(),
            slippage=model.slippage,
            cost_ledger=ledger,
        )


def test_extract_trades_no_positions_raises_after_a_swallowed_cache_read() -> None:
    """Note 3: split_positions swallowing a cache-read error must not hide it either."""

    def broken() -> list[Any]:
        raise RuntimeError("cache read failed")

    engine = SimpleNamespace(
        kernel=SimpleNamespace(
            cache=SimpleNamespace(positions=broken, position_snapshots=list)
        )
    )
    model = _cost_model()
    ledger = CostLedger(model, SimpleNamespace())
    ledger.on_position_closed(_closed_btc())

    with pytest.raises(ModeledCostIntegrityError, match="no closed trade for the finalized"):
        extract_trades(
            ValidationResult(),
            engine,
            fake_venue(),
            slippage=model.slippage,
            cost_ledger=ledger,
        )


def test_other_symbols_open_funding_counts_at_sizing_but_its_slippage_waits_for_exit() -> None:
    """Ledger timing is kept -- an open position pays funding to now, no slippage yet."""
    eth = "ETHUSDT-PERP.BINANCE"
    model = _cost_model()
    ledger = CostLedger(model, SimpleNamespace())
    btc = _closed_btc()
    eth_open = FakePosition.open(
        "E-1", eth, entry_ns=_BASE_NS, entry_price=100.0, qty=5.0
    )
    ledger.on_position_closed(btc)

    now = _BASE_NS + 25 * _HOUR_NS
    paid = ledger.paid_through(now, [eth_open])

    estimator = SlippageEstimator(impact_coefficient=0.1)
    legs = estimator.estimate_cost(1000.0, 1.0, 1000.0, 0.02, 0.0001) + (
        estimator.estimate_cost(1010.0, 1.0, 1000.0, 0.02, 0.0001)
    )
    expected = 3 * 0.0003 * 1000.0 + legs + 3 * 0.0003 * 500.0
    assert paid == pytest.approx(expected)
    # the open ETH position's own entry slippage would be charged only once it closes
    assert abs(paid - (expected + estimator.estimate_cost(100.0, 5.0, 1000.0, 0.02, 0.0001))) > 1e-9


_T3_DAY_MS = 1_704_067_200_000  # 2024-01-01T00:00Z
_ETH = "ETHUSDT-PERP.BINANCE"
_T3_RATE = 0.0001
_SETTLE_NS = 8 * _HOUR_NS
_T3_DSL: dict[str, Any] = {
    "name": "costed_t3",
    "timeframe": "1h",
    "indicators": {"ema": {"type": "EMA", "period": 3}},
    "entry_conditions": {"long": ["ema > 0"]},
    "stop_loss": {"type": "fixed_pct", "percent": 50.0},
    "take_profit": {"type": "fixed_pct", "percent": 10.0},
}


def _two_symbol_catalog(root: Any, n_min: int) -> Any:
    """BTC flat at 100 and ETH jumping 100 -> 120 at Jan-2 18:00 (1m volume 1.0 each).

    The 1h LAST bars (the PRIMARY series) aggregate to volume 60.0; the 1m detail series
    is the decoy. Both are written so the runner's detail resolution loads 1m.
    """
    from vibe_quant.data.catalog import (
        CatalogManager,
        aggregate_bars,
        create_instrument,
        get_bar_type,
        klines_to_bars,
    )

    manager = CatalogManager(root)
    for symbol, price_at in (
        ("BTCUSDT", lambda _i: 100.0),
        ("ETHUSDT", lambda i: 100.0 if i < 42 * 60 else 120.0),
    ):
        instrument = create_instrument(symbol)
        manager.write_instrument(instrument)
        klines = [
            {
                "open_time": _T3_DAY_MS + i * 60_000,
                "open": price_at(i),
                "high": price_at(i),
                "low": price_at(i),
                "close": price_at(i),
                "volume": 1.0,
                "close_time": _T3_DAY_MS + i * 60_000 + 59_999,
            }
            for i in range(n_min)
        ]
        bars_1m = klines_to_bars(
            klines,
            instrument.id,
            get_bar_type(symbol, "1m"),
            instrument.size_precision,
            instrument.price_precision,
        )
        manager.write_bars(bars_1m)
        manager.write_bars(
            aggregate_bars(
                bars_1m,
                get_bar_type(symbol, "1h"),
                60,
                instrument.size_precision,
                instrument.price_precision,
            )
        )
    return root


def test_real_validation_run_sizes_on_costs_with_another_symbol_open(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/ amendment note 4 on the REAL engine (production ``_run_backtest`` + extraction).

    BTC holds one flat position throughout; ETH takes a 10% profit when it jumps to 120 and
    re-enters on the next 1h close. That entry is sized on the costed balance: NT's balance
    minus the ETH close's funding + its TWO slippage legs minus BTC's funding to the sizing
    instant, and NOT BTC's still-unclosed entry slippage. The legs read the PRIMARY 1h bars
    (volume 60.0) strictly before each fill, never the 1m decoy (volume 1.0); the post-hoc
    extraction reconciles with the sizing ledger exactly.
    """
    import vibe_quant.validation.extraction as ex

    catalog = _two_symbol_catalog(tmp_path / "cat", 5 * 1440)
    rows = [(_T3_DAY_MS + t * 3_600_000, 8) for t in range(-24, 24 * 14, 8)]
    archive = interval_archive(
        tmp_path / "funding.db", {"BTCUSDT": rows, "ETHUSDT": rows}, rate=_T3_RATE
    )

    db = tmp_path / "s.db"
    state = StateManager(db)
    strategy_id = state.create_strategy(name="costed_t3", dsl_config=_T3_DSL)
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="validation",
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-06",
        parameters={"max_position_pct": 1.0, "risk_per_trade": 0.5},
    )
    state.close()

    captured: dict[str, Any] = {}
    paid_calls: list[tuple[int, float, float]] = []
    reconciled: list[Any] = []
    orig_extract = ValidationRunner._extract_results
    orig_paid = CostLedger.paid_through
    orig_reconcile = ex.reconcile

    def spy_extract(self: ValidationRunner, **kwargs: Any) -> Any:
        cache = kwargs["engine"].kernel.cache
        captured["ledger"] = kwargs["cost_ledger"]
        captured["positions"] = [*cache.positions(), *cache.position_snapshots()]
        return orig_extract(self, **kwargs)

    def spy_paid(self: CostLedger, now_ns: int, open_positions: Any) -> float:
        value = orig_paid(self, now_ns, open_positions)
        from nautilus_trader.model.currencies import USDT

        # The balance sizing reads right before this call (same synchronous read).
        balance = float(self.cache.accounts()[0].balance_total(USDT))
        paid_calls.append((int(now_ns), value, balance))
        return value

    def spy_reconcile(ledger: Any, costed: Any) -> None:
        reconciled.append((ledger, costed))
        orig_reconcile(ledger, costed)

    monkeypatch.setattr(ValidationRunner, "_extract_results", spy_extract)
    monkeypatch.setattr(CostLedger, "paid_through", spy_paid)
    monkeypatch.setattr(ex, "reconcile", spy_reconcile)

    runner = ValidationRunner(
        db_path=db,
        logs_path=tmp_path / "logs",
        catalog_path=catalog,
        funding_archive_path=archive,
    )
    try:
        result = runner.run(run_id=run_id)
    finally:
        runner.close()

    by_key = {cost_key(pos): pos for pos in captured["positions"]}
    (eth_pos,) = [p for p in by_key.values() if str(p.instrument_id) == _ETH and p.is_closed]
    (btc_pos,) = [p for p in by_key.values() if str(p.instrument_id) == BTC and p.is_open]
    ledger = captured["ledger"]

    # The still-open second ETH entry, as the post-hoc trade record sees it.
    second = next(
        t for t in result.trades if t.symbol == _ETH and t.exit_reason == "end_of_data"
    )
    filled_at = datetime.fromisoformat(second.entry_time)
    second_entry_ns = int(filled_at.timestamp()) * 1_000_000_000 + filled_at.microsecond * 1_000
    sizing_ts, paid_at_sizing, balance_at = max(
        (call for call in paid_calls if call[0] < second_entry_ns), key=lambda c: c[0]
    )
    assert sizing_ts % _HOUR_NS == 3_599_999_000_000  # a 1h bar close, not a fill tick

    # The closed ETH trade: funding to its exit plus both legs, from the PRIMARY prefix.
    estimator = SlippageEstimator(impact_coefficient=0.1)
    eth_qty = float(eth_pos.peak_qty)
    entry_leg = estimator.estimate_cost(
        float(eth_pos.avg_px_open), eth_qty, 60.0, 0.02, 0.0001
    )
    exit_leg = estimator.estimate_cost(
        float(eth_pos.avg_px_close), eth_qty, 60.0, 0.02, 0.0001
    )
    assert exit_leg != entry_leg
    # The 1m decoy (volume 1.0) is not the series the legs read.
    assert estimator.estimate_cost(
        float(eth_pos.avg_px_open), eth_qty, 1.0, 0.02, 0.0001
    ) != entry_leg
    eth_settlements = int(eth_pos.ts_closed) // _SETTLE_NS - int(eth_pos.ts_opened) // _SETTLE_NS
    eth_funding = eth_settlements * _T3_RATE * eth_qty * float(eth_pos.avg_px_open)
    eth_key = cost_key(eth_pos)
    assert ledger.finalized[eth_key] == pytest.approx(eth_funding + entry_leg + exit_leg)
    assert not any("BTCUSDT" in key for key in ledger.finalized)  # its legs wait for its exit
    eth_trade = next(
        t for t in result.trades if t.symbol == _ETH and t.exit_reason != "end_of_data"
    )
    assert eth_trade.slippage_cost == pytest.approx(entry_leg + exit_leg)
    assert eth_trade.funding_fees == pytest.approx(eth_funding)

    # Sizing at the second entry: the closed legs + BTC's open funding, no BTC slippage.
    btc_funding = (
        sizing_ts // _SETTLE_NS - int(btc_pos.ts_opened) // _SETTLE_NS
    ) * _T3_RATE * float(btc_pos.peak_qty) * float(btc_pos.avg_px_open)
    expected_paid = ledger.finalized[eth_key] + btc_funding
    assert paid_at_sizing == pytest.approx(expected_paid, abs=1e-9)
    btc_entry_leg = estimator.estimate_cost(
        float(btc_pos.avg_px_open), float(btc_pos.peak_qty), 60.0, 0.02, 0.0001
    )
    assert abs(paid_at_sizing - (expected_paid + btc_entry_leg)) > 1e-3

    expected_qty = math.floor((balance_at - paid_at_sizing) / 120.0 * 1000 + 1e-9) / 1000
    assert second.quantity == expected_qty
    # Teeth: the two legs and the other symbol's funding each move at least one size tick.
    assert second.quantity < math.floor(
        (balance_at - (paid_at_sizing - (entry_leg + exit_leg))) / 120.0 * 1000 + 1e-9
    ) / 1000
    assert second.quantity < math.floor(
        (balance_at - (paid_at_sizing - btc_funding)) / 120.0 * 1000 + 1e-9
    ) / 1000

    # The post-hoc ledger reconciled with the sizing ledger (it would have raised otherwise).
    assert reconciled and reconciled[-1][0] is ledger


def test_extract_results_forwards_the_slippage_model_and_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import vibe_quant.validation.extraction as ex

    runner = ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")
    seen: dict[str, Any] = {}

    def fake_extract(*args: Any, **kwargs: Any) -> str:
        seen.update(kwargs)
        return "forwarded"

    monkeypatch.setattr(ex, "extract_results", fake_extract)
    model = fake_slippage()
    ledger = object()
    try:
        assert (
            runner._extract_results(  # noqa: SLF001
                1, "s", None, None, None, slippage=model, cost_ledger=ledger  # type: ignore[arg-type]
            )
            == "forwarded"
        )
    finally:
        runner.close()
    assert seen["slippage"] is model
    assert seen["cost_ledger"] is ledger


def test_market_twin_reconcile_failure_is_reported_and_the_primary_result_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The twin's handler records ``ModeledCostIntegrityError: ...``; the run survives."""
    runner = ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")

    def boom(_self: Any, **_kwargs: Any) -> None:
        raise ModeledCostIntegrityError("modeled costs: sizing ledger and trade ledger disagree for X@1")

    monkeypatch.setattr(ValidationRunner, "_run_backtest", boom)
    try:
        note = runner._run_market_twin(  # noqa: SLF001
            run_id=1,
            strategy_name="s",
            dsl=None,  # type: ignore[arg-type]
            venue_config=None,  # type: ignore[arg-type]
            run_config={},
            writer=None,  # type: ignore[arg-type]
            detail_timeframe=None,
            launch=None,  # type: ignore[arg-type]
        )
    finally:
        runner.close()
    assert note == {
        "error": "ModeledCostIntegrityError: modeled costs: sizing ledger and trade ledger "
        "disagree for X@1"
    }


class TestSlippageModelSelection:
    """The extractor charges the legs of the model it was handed."""

    @staticmethod
    def _make_engine_with_closed_position() -> Any:
        return fake_engine([_btc_position(qty=0.1)])

    def test_extract_trades_skips_post_fill_slippage_when_engine_slippage_enabled(self) -> None:
        result = ValidationResult(starting_balance=100_000.0)
        extract_trades(
            result,
            self._make_engine_with_closed_position(),
            fake_venue(slippage=1.0),
            slippage=fake_slippage(1.0),
            cost_ledger=None,
        )

        assert len(result.trades) == 1
        assert result.trades[0].slippage_cost == 0.0
        assert result.total_slippage == 0.0

    def test_extract_trades_applies_post_fill_slippage_when_engine_slippage_disabled(self) -> None:
        result = ValidationResult(starting_balance=100_000.0)
        extract_trades(
            result,
            self._make_engine_with_closed_position(),
            fake_venue(slippage=0.0),
            slippage=fake_slippage(0.0),
            cost_ledger=None,
        )

        assert len(result.trades) == 1
        assert result.trades[0].slippage_cost > 0.0
        assert result.total_slippage > 0.0

