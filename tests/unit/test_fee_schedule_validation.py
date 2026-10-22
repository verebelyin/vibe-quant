"""BNB-discounted fee schedule plumbed through validation runs (bd vibe-quant-2gyk1.4, D3-3).

The schedule is venue-level: ``bnb_discount`` / ``fee_tier`` in the run
parameters build a :class:`FeeSchedule` on the validation venue, the
open-position exit-fee estimate uses its taker rate, and the schedule is
recorded in the run notes. The default keeps every pre-existing number
bit-identical.
"""

from __future__ import annotations

import json
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel, LatencyModel, MakerTakerFeeModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.db.state_manager import StateManager
from vibe_quant.fees import FeeSchedule
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.validation.extraction import extract_results
from vibe_quant.validation.results import ValidationResult
from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError
from vibe_quant.validation.venue import create_venue_config_for_validation

IID = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
BT = get_bar_type("BTCUSDT", "1m")
_MIN_NS = 60_000_000_000

BNB = FeeSchedule(tier="VIP0", bnb_discount=True)

_DSL_CONFIG: dict[str, object] = {
    "name": "test_strategy",
    "description": "Test strategy for validation",
    "version": 1,
    "timeframe": "5m",
    "additional_timeframes": [],
    "indicators": {"rsi": {"type": "RSI", "period": 14, "source": "close"}},
    "entry_conditions": {"long": ["rsi < 30"], "short": ["rsi > 70"]},
    "exit_conditions": {"long": ["rsi > 50"], "short": ["rsi < 50"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}


@pytest.fixture(autouse=True)
def _no_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the run() path offline: no data-window clamping or detail probing."""
    monkeypatch.setattr(
        ValidationRunner, "_clamp_run_window", lambda self, run_config, tf: (run_config, None)
    )
    monkeypatch.setattr(
        ValidationRunner, "_resolve_detail_timeframe", lambda self, rc, tf, ov: None
    )


@pytest.fixture
def temp_db() -> Path:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Path(f.name)


@pytest.fixture
def temp_logs() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _runner(temp_db: Path, temp_logs: Path) -> ValidationRunner:
    return ValidationRunner(db_path=temp_db, logs_path=temp_logs)


# ---------------------------------------------------------------------------
# Venue schedule from run parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_detail_data", [False, True])
def test_validation_params_bnb_discount_sets_venue_schedule(
    temp_db: Path, temp_logs: Path, has_detail_data: bool
) -> None:
    """The schedule survives both venue paths (skipped latency and detail data)."""
    runner = _runner(temp_db, temp_logs)
    try:
        cfg, _launch = runner._create_venue_config(
            {"parameters": {"bnb_discount": True}},
            None,
            timeframe="1m",
            has_detail_data=has_detail_data,
        )
        assert cfg.taker_fee == Decimal("0.00045")
        assert cfg.maker_fee == Decimal("0.00018")
        assert cfg.fee_schedule == BNB
        assert not cfg.fee_schedule.is_default

        # fee_tier is a first-class run parameter too (VIP0 is the only tier).
        tier_cfg, _launch = runner._create_venue_config(
            {"parameters": {"fee_tier": "VIP0"}},
            None,
            timeframe="1m",
            has_detail_data=has_detail_data,
        )
        assert tier_cfg.fee_schedule.is_default

        default_cfg, _launch = runner._create_venue_config(
            {"parameters": {}}, None, timeframe="1m", has_detail_data=has_detail_data
        )
        assert default_cfg.fee_schedule.is_default
        assert default_cfg.taker_fee == Decimal("0.0005")
    finally:
        runner.close()


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        ({"fee_tier": "VIP9"}, "VIP9"),
        ({"bnb_discount": "true"}, "bnb_discount"),
        ({"bnb_discount": 1}, "bnb_discount"),
        ({"bnb_discount": "yes"}, "bnb_discount"),
        ({"bnb_discount": None}, "bnb_discount"),
    ],
)
def test_validation_bad_fee_param_fails_run(
    temp_db: Path, temp_logs: Path, parameters: dict[str, object], match: str
) -> None:
    """A bad fee parameter fails the run with a clear message, never defaults."""
    state = StateManager(temp_db)
    strategy_id = state.create_strategy(name="bnb_bad_fee", dsl_config=_DSL_CONFIG)
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-31",
        parameters=parameters,
    )
    state.close()

    runner = _runner(temp_db, temp_logs)
    try:
        with pytest.raises(ValidationRunnerError, match=match):
            runner.run(run_id=run_id)
    finally:
        runner.close()

    check = StateManager(temp_db)
    try:
        row = check.get_backtest_run(run_id)
        assert row is not None
        assert row["status"] == "failed"
        assert match in str(row["error_message"])
    finally:
        check.close()


def test_validation_notes_record_fee_schedule(temp_db: Path, temp_logs: Path) -> None:
    state = StateManager(temp_db)
    strategy_id = state.create_strategy(name="bnb_notes", dsl_config=_DSL_CONFIG)
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-31",
        parameters={"bnb_discount": True},
    )
    state.close()

    runner = _runner(temp_db, temp_logs)
    runner._run_backtest = lambda **kwargs: ValidationResult(  # type: ignore[assignment]
        run_id=kwargs["run_id"], strategy_name=kwargs["strategy_name"], total_trades=1
    )
    try:
        result = runner.run(run_id=run_id)
    finally:
        runner.close()

    assert result.notes["fee_schedule"] == {"fee_tier": "VIP0", "bnb_discount": True}

    # The launched run must reach its FINAL status with the fee schedule stored —
    # a "running"/201 state alone proves nothing.
    check = StateManager(temp_db)
    try:
        row = check.get_backtest_run(run_id)
        assert row is not None
        assert row["status"] == "completed"
        stored = check.get_result_notes_raw(run_id)
    finally:
        check.close()
    assert stored is not None
    assert json.loads(stored)["fee_schedule"] == {"fee_tier": "VIP0", "bnb_discount": True}


def test_fee_params_not_forwarded_to_strategy_config(temp_db: Path, temp_logs: Path) -> None:
    runner = _runner(temp_db, temp_logs)
    try:
        params = runner._build_strategy_params(
            {"parameters": {"rsi_period": 14, "bnb_discount": True, "fee_tier": "VIP0"}}
        )
    finally:
        runner.close()
    assert params == {"rsi_period": 14}


# ---------------------------------------------------------------------------
# extract_trades forwards the schedule's taker rate to the open-position mark
# ---------------------------------------------------------------------------


class _Cfg(StrategyConfig, frozen=True):
    bar_type: BarType


class _EnterAndFlattenOnStop(Strategy):
    """Buys on the first bar; flattens in on_stop like the compiled DSL strategies."""

    def __init__(self, config: _Cfg) -> None:
        super().__init__(config)
        self._done = False

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if self._done:
            return
        self._done = True
        inst = self.cache.instrument(IID)
        self.submit_order(
            self.order_factory.market(
                IID, OrderSide.BUY, inst.make_qty(Decimal("0.050")), time_in_force=TimeInForce.IOC
            )
        )

    def on_stop(self) -> None:
        self.cancel_all_orders(IID)
        self.close_all_positions(IID)


def _bar(p: float, minute: int) -> Bar:
    return Bar(
        bar_type=BT,
        open=Price.from_str(f"{p:.1f}"),
        high=Price.from_str(f"{p + 5:.1f}"),
        low=Price.from_str(f"{p - 5:.1f}"),
        close=Price.from_str(f"{p:.1f}"),
        volume=Quantity.from_str("100.000"),
        ts_event=minute * _MIN_NS,
        ts_init=(minute + 1) * _MIN_NS - 1_000_000,
    )


@pytest.fixture
def open_at_end_engine() -> BacktestEngine:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
        latency_model=LatencyModel(base_latency_nanos=60_000_000),
        fee_model=MakerTakerFeeModel(),
        bar_execution=True,
        use_position_ids=True,
        use_reduce_only=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    engine.add_data([_bar(p, i) for i, p in enumerate([10000, 10100, 10200, 10300, 10400])])
    engine.add_strategy(_EnterAndFlattenOnStop(_Cfg(bar_type=BT)))
    engine.run()
    yield engine
    engine.reset()
    engine.dispose()


def test_open_mark_uses_run_schedule_taker(open_at_end_engine: BacktestEngine) -> None:
    """A non-default schedule's taker rate prices the mark, not the instrument rate."""
    from types import SimpleNamespace

    positions = list(open_at_end_engine.cache.positions())
    assert len(positions) == 1 and positions[0].is_open  # the flatten never filled
    entry = float(positions[0].avg_px_open)  # latency: next bar close, 10100
    last_close = 10400.0

    default_venue = create_venue_config_for_validation(latency_preset=None)
    bnb_venue = create_venue_config_for_validation(latency_preset=None)
    # engine slippage "on" disables post-fill SPEC slippage: same costs as screening.
    for venue in (default_venue, bnb_venue):
        venue.fill_config = SimpleNamespace(impact_coefficient=0.1, prob_slippage=1.0)  # type: ignore[assignment]
    bnb_venue.fee_schedule = BNB

    def extract(venue: object) -> ValidationResult:
        return extract_results(
            1,
            "x",
            open_at_end_engine.get_result(),
            open_at_end_engine,
            venue,  # type: ignore[arg-type]
            primary_timeframe="1m",
            execution_timeframe="1m",
        )

    default_result = extract(default_venue)
    bnb_result = extract(bnb_venue)

    entry_fee = 0.05 * entry * 0.0005
    assert default_result.total_fees == pytest.approx(entry_fee + 0.05 * last_close * 0.0005)
    assert bnb_result.total_fees == pytest.approx(entry_fee + 0.05 * last_close * 0.00045)
    assert bnb_result.total_fees < default_result.total_fees
