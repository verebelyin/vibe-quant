"""BNB fee schedule + post-only refusal plumbing for screening and discovery (D3-2).

The fee schedule is venue-level (bd vibe-quant-2gyk1.4): ``FeeSchedule`` must
survive the pickling of ``NTBacktestFn`` into worker processes, reach every
``NTScreeningRunner`` it builds including the holdout/cross-window factories,
be readable from the screening pipeline's run parameters, and be recorded in
discovery run notes.  ``entry_order != "market"`` (post-only maker entries) is
validation-only and must fail closed everywhere screening/discovery could run
it (screening maker-fill parity is unproven, bead vibe-quant-7eek0).
"""

from __future__ import annotations

import json
import pickle
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.fees import FeeSchedule
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.liquidation_settings import LiquidationSettings

if TYPE_CHECKING:
    from pathlib import Path

    from vibe_quant.dsl.schema import StrategyDSL

BNB = FeeSchedule(tier="VIP0", bnb_discount=True)
_POST_ONLY = "post_only"
# Every spelling that is not exactly "market" must be refused (fail closed).
_NON_MARKET = ["post_only", "POST_ONLY", "limit", "", "Market", "None"]

_MINIMAL_YAML = """
name: plumbing_probe
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""


# ---------------------------------------------------------------------------
# Runner / worker plumbing
# ---------------------------------------------------------------------------


def test_bnb_flag_reaches_worker_venue() -> None:
    """A BNB NTBacktestFn survives pickling and builds a 0.00045-taker runner."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    fn = NTBacktestFn(
        ["BTCUSDT"], "4h", "2024-01-01", "2024-02-01", fee_schedule=BNB
    )
    loaded = pickle.loads(pickle.dumps(fn))

    assert loaded.fee_schedule.taker_rate == Decimal("0.00045")

    runner = loaded._make_runner(
        {"name": "worker_probe", "timeframe": "4h"},
        "2024-01-01",
        "2024-02-01",
    )
    assert runner._venue_config().fee_schedule.taker_rate == Decimal("0.00045")


def test_screening_runner_default_schedule_unchanged() -> None:
    """No fee_schedule -> default VIP0 schedule on NT's MakerTakerFeeModel path."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.validation.venue import create_backtest_venue_config

    runner = NTScreeningRunner({}, ["BTCUSDT"], "2024-01-01", "2024-02-01")
    venue = runner._venue_config()
    assert venue.fee_schedule == FeeSchedule()
    assert venue.fee_schedule.is_default
    assert venue.taker_fee == Decimal("0.0005")
    assert venue.maker_fee == Decimal("0.0002")

    built = create_backtest_venue_config(venue)
    assert built.fee_model.fee_model_path == (
        "nautilus_trader.backtest.models:MakerTakerFeeModel"
    )

    # The worker factory's default is the same schedule.
    fn = NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-02-01")
    assert fn.fee_schedule == FeeSchedule()


def test_screening_refuses_post_only() -> None:
    """A post-only screening runner fails closed before any engine exists."""
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    with pytest.raises(ValueError, match="validation-only"):
        NTScreeningRunner(
            {},
            ["BTCUSDT"],
            "2024-01-01",
            "2024-02-01",
            launch=LaunchSettings(entry_order=_POST_ONLY),
        )


def test_discovery_refuses_post_only() -> None:
    """The discovery CLI factory and NTBacktestFn both refuse post-only."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.discovery.run import _make_nt_backtest_fn

    with pytest.raises(ValueError, match="validation-only"):
        NTBacktestFn(
            ["BTCUSDT"], "4h", "2024-01-01", "2024-02-01", entry_order=_POST_ONLY
        )

    with pytest.raises(ValueError, match="validation-only"):
        _make_nt_backtest_fn(
            ["BTCUSDT"],
            "4h",
            "2024-01-01",
            "2024-02-01",
            fee_schedule=FeeSchedule(),
            entry_order=_POST_ONLY,
            liquidation=LiquidationSettings(),
        )


# ---------------------------------------------------------------------------
# Discovery CLI + notes
# ---------------------------------------------------------------------------


def test_cli_bnb_discount_flag() -> None:
    """--bnb-discount opts in; --fee-tier defaults to VIP0 and BNB is off by default."""
    from vibe_quant.discovery.run_spec import build_parser

    parser = build_parser()
    opted_in = parser.parse_args(["--run-id", "1", "--bnb-discount"])
    default = parser.parse_args(["--run-id", "1"])

    assert opted_in.bnb_discount is True
    assert opted_in.fee_tier == "VIP0"
    assert default.bnb_discount is False
    assert default.fee_tier == "VIP0"

    assert FeeSchedule(
        tier=opted_in.fee_tier, bnb_discount=opted_in.bnb_discount
    ).taker_rate == Decimal("0.00045")
    assert FeeSchedule(
        tier=default.fee_tier, bnb_discount=default.bnb_discount
    ) == FeeSchedule()


def test_discovery_notes_record_fee_schedule() -> None:
    """Discovery run notes carry the effective schedule (default or BNB)."""
    from vibe_quant.discovery.pipeline import DiscoveryResult
    from vibe_quant.discovery.record import provenance_notes

    def notes_for(schedule: FeeSchedule) -> dict[str, object]:
        result = DiscoveryResult(
            generations=[],
            top_strategies=[],
            total_candidates_evaluated=0,
            converged=False,
            convergence_generation=None,
        )
        return provenance_notes(
            result,
            use_mock=True,
            eval_windows_count=1,
            eval_windows=None,
            symbol_agg="portfolio",
            split_ratio=0.0,
            direction=None,
            cross_window_months=[],
            cross_window_min_sharpe=0.5,
            wfa_oos_step_days=0,
            wfa_min_consistency=0.75,
            num_seeds=1,
            seed=42,
            bootstrap_min_sharpe=0.0,
            holdout_min_sharpe=0.0,
            symbols=["BTCUSDT"],
            fee_schedule=schedule,
        )

    assert notes_for(BNB)["fee_schedule"] == {
        "fee_tier": "VIP0",
        "bnb_discount": True,
    }
    assert notes_for(FeeSchedule())["fee_schedule"] == {
        "fee_tier": "VIP0",
        "bnb_discount": False,
    }


# ---------------------------------------------------------------------------
# Screening pipeline run parameters
# ---------------------------------------------------------------------------


def _minimal_dsl() -> StrategyDSL:
    from vibe_quant.dsl import parse_strategy_string

    return parse_strategy_string(_MINIMAL_YAML)


def test_screening_pipeline_reads_bnb_param(monkeypatch: Any) -> None:
    """create_screening_pipeline derives the venue schedule from run parameters."""
    from vibe_quant.screening import pipeline as pipeline_mod
    from vibe_quant.screening.pipeline import create_screening_pipeline

    monkeypatch.setattr(
        pipeline_mod, "resolve_parent_fill_ticks", lambda *a, **k: None
    )

    default = create_screening_pipeline(_minimal_dsl(), symbols=["BTCUSDT"])
    assert default._runner._venue_config().fee_schedule == FeeSchedule()

    bnb = create_screening_pipeline(
        _minimal_dsl(),
        symbols=["BTCUSDT"],
        parameters={"bnb_discount": True},
    )
    assert (
        bnb._runner._venue_config().fee_schedule.taker_rate == Decimal("0.00045")
    )


def test_screening_pipeline_refuses_post_only_param() -> None:
    """A post-only run parameter is refused by the pipeline factory (mock or not)."""
    from vibe_quant.screening.pipeline import create_screening_pipeline

    with pytest.raises(ValueError, match="validation-only"):
        create_screening_pipeline(
            _minimal_dsl(),
            use_mock=True,
            parameters={"entry_order": _POST_ONLY},
        )


# ---------------------------------------------------------------------------
# Fail-closed guard: only the literal "market" passes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry_order", _NON_MARKET)
def test_every_non_market_entry_is_refused(entry_order: str) -> None:
    """The guard is `!= "market"`, not `== "post_only"`: case/limit/empty all fail."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.screening.pipeline import create_screening_pipeline

    with pytest.raises(ValueError, match="validation-only"):
        NTScreeningRunner(
            {},
            ["BTCUSDT"],
            "2024-01-01",
            "2024-02-01",
            launch=LaunchSettings(entry_order=entry_order),
        )
    with pytest.raises(ValueError, match="validation-only"):
        NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-02-01", entry_order=entry_order)
    with pytest.raises(ValueError, match="validation-only"):
        create_screening_pipeline(
            _minimal_dsl(), use_mock=True, parameters={"entry_order": entry_order}
        )


def test_market_entry_is_accepted() -> None:
    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-02-01", entry_order="market")


# ---------------------------------------------------------------------------
# Live discovery CLI: BNB reaches EVERY runner the run builds
# ---------------------------------------------------------------------------


def _discovery_run(db_path: Path) -> int:
    from vibe_quant.db.state_manager import StateManager

    state = StateManager(db_path)
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-01-01",
        end_date="2025-04-01",
        parameters={},
    )
    state.close()
    return run_id


def _run_live_discovery_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str
) -> tuple[list[Any], dict[str, Any]]:
    """Run discovery main() on the NON-mock path with the engine stubbed out.

    Returns every NTBacktestFn constructed (train, holdout, cross-window / WFA
    factory, full-range headline, continuous-train efficiency) and the notes.
    """
    import vibe_quant.discovery.run as discovery_run
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import __main__ as cli
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )
    from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult, HoldoutResult

    built: list[Any] = []
    real_init = NTBacktestFn.__init__

    def recording_init(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        built.append(self)

    metrics: dict[str, float | int] = {
        "sharpe_ratio": 1.0, "total_return": 0.05, "max_drawdown": 0.05,
        "profit_factor": 1.5, "total_trades": 100,
    }  # fmt: skip
    monkeypatch.setattr(NTBacktestFn, "__init__", recording_init)
    monkeypatch.setattr(NTBacktestFn, "__call__", lambda self, chrom: dict(metrics))
    monkeypatch.setattr(discovery_run, "_check_data_available", lambda syms: True)

    chrom = StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        stop_loss_pct=2.0, take_profit_pct=4.0, direction=Direction.SHORT,
    )  # fmt: skip
    fit = FitnessResult(
        sharpe_ratio=1.5, max_drawdown=0.05, profit_factor=1.8, total_trades=100,
        total_return=0.10, complexity_penalty=0.0, overtrade_penalty=0.0,
        sl_tp_penalty=0.0, raw_score=1.0, adjusted_score=1.0,
        passed_filters=True, filter_results={},
    )  # fmt: skip
    holdout = HoldoutResult(
        sharpe_ratio=0.8, max_drawdown=0.03, profit_factor=1.2, total_trades=30, total_return=0.02
    )  # fmt: skip

    def fake_run(self: DiscoveryPipeline) -> DiscoveryResult:
        # Exercise the cross-window / WFA factory exactly as the gates do.
        assert self._backtest_fn_factory is not None
        self._backtest_fn_factory("2025-01-10", "2025-01-20")
        return DiscoveryResult(
            generations=[], top_strategies=[(chrom, fit)],
            total_candidates_evaluated=10, converged=True, convergence_generation=1,
            holdout_results=[holdout],
            train_dates=("2025-01-01", "2025-03-01"),
            holdout_dates=("2025-03-01", "2025-04-01"),
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    db = tmp_path / "live.db"
    run_id = _discovery_run(db)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--run-id", str(run_id), "--population-size", "6",
            "--max-generations", "1", "--elite-count", "1", "--symbols", "BTCUSDT",
            "--timeframe", "1h", "--start-date", "2025-01-01", "--end-date", "2025-04-01",
            "--eval-windows", "2", "--cross-window-months", "1", "--db", str(db), *extra,
        ],
    )  # fmt: skip
    assert cli.main() == 0

    state = StateManager(db)
    run = state.get_backtest_run(run_id)
    result = state.get_backtest_result(run_id)
    state.close()
    assert run is not None and run["status"] == "completed"
    assert result is not None
    return built, json.loads(result["notes"])


def test_live_cli_bnb_reaches_every_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train, holdout, cross/WFA factory, headline and efficiency runners all carry BNB."""
    built, notes = _run_live_discovery_cli(tmp_path, monkeypatch, "--bnb-discount")

    # train GA fn, holdout, factory call, full-range headline, train-range efficiency
    spans = {(fn.start_date, fn.end_date) for fn in built}
    assert ("2025-01-10", "2025-01-20") in spans  # cross-window / WFA factory
    assert ("2025-01-01", "2025-04-01") in spans  # full-range headline
    assert ("2025-01-01", "2025-03-01") in spans  # continuous-train efficiency
    assert len(built) >= 5, spans
    assert [fn.fee_schedule for fn in built] == [BNB] * len(built)
    assert {fn.entry_order for fn in built} == {"market"}
    assert notes["fee_schedule"] == {"fee_tier": "VIP0", "bnb_discount": True}


def test_live_cli_default_keeps_vip0_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, notes = _run_live_discovery_cli(tmp_path, monkeypatch)

    assert len(built) >= 5
    assert [fn.fee_schedule for fn in built] == [FeeSchedule()] * len(built)
    assert notes["fee_schedule"] == {"fee_tier": "VIP0", "bnb_discount": False}


def test_discovery_nt_fn_factory_requires_explicit_fee_knobs() -> None:
    """A new call site cannot forget the schedule: the knobs have no defaults."""
    from vibe_quant.discovery.run import _make_nt_backtest_fn

    with pytest.raises(TypeError):
        _make_nt_backtest_fn(  # type: ignore[call-arg]
            symbols=["BTCUSDT"], timeframe="4h", start_date="2024-01-01", end_date="2024-02-01"
        )


# ---------------------------------------------------------------------------
# Open-position mark uses the run's schedule (screening _extract_metrics)
# ---------------------------------------------------------------------------


def test_screening_open_mark_uses_run_schedule_taker() -> None:
    """A BNB screening run charges the BNB taker rate on the end-of-run flatten estimate."""
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.backtest.models import FillModel, LatencyModel, MakerTakerFeeModel
    from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
    from nautilus_trader.model.identifiers import InstrumentId, Venue
    from nautilus_trader.model.objects import Currency, Money, Price, Quantity
    from nautilus_trader.trading.strategy import Strategy, StrategyConfig

    from vibe_quant.data.catalog import create_instrument, get_bar_type
    from vibe_quant.nt_compat import retain_log_guard
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    iid = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    bt = get_bar_type("BTCUSDT", "1m")
    minute_ns = 60_000_000_000

    class Cfg(StrategyConfig, frozen=True):
        bar_type: BarType

    class BuyOnce(Strategy):
        def __init__(self, config: Cfg) -> None:
            super().__init__(config)
            self._done = False

        def on_start(self) -> None:
            self.subscribe_bars(self.config.bar_type)

        def on_bar(self, bar: Bar) -> None:
            if self._done:
                return
            self._done = True
            inst = self.cache.instrument(iid)
            self.submit_order(
                self.order_factory.market(
                    iid, OrderSide.BUY, inst.make_qty(Decimal("0.050")),
                    time_in_force=TimeInForce.IOC,
                )
            )

    def make_bar(price: float, minute: int) -> Bar:
        return Bar(
            bar_type=bt,
            open=Price.from_str(f"{price:.1f}"),
            high=Price.from_str(f"{price + 5:.1f}"),
            low=Price.from_str(f"{price - 5:.1f}"),
            close=Price.from_str(f"{price:.1f}"),
            volume=Quantity.from_str("100.000"),
            ts_event=minute * minute_ns,
            ts_init=(minute + 1) * minute_ns - 1_000_000,
        )

    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    try:
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
        engine.add_data([make_bar(p, i) for i, p in enumerate([10000, 10100, 10200, 10300, 10400])])
        engine.add_strategy(BuyOnce(Cfg(bar_type=bt)))
        engine.run()

        positions = list(engine.cache.positions())
        assert len(positions) == 1 and positions[0].is_open  # never flattened
        entry_fee = 0.05 * float(positions[0].avg_px_open) * 0.0005
        last_close = 10400.0

        def total_fees(schedule: FeeSchedule) -> float:
            runner = NTScreeningRunner(
                {},
                ["BTCUSDT"],
                "1970-01-01",
                "1970-01-02",
                launch=LaunchSettings(fee_schedule=schedule),
            )
            runner._all_timeframes = {"1m"}
            metrics = runner._extract_metrics({}, engine.get_result(), engine, 0.0)
            return metrics.total_fees

        assert total_fees(FeeSchedule()) == pytest.approx(entry_fee + 0.05 * last_close * 0.0005)
        assert total_fees(BNB) == pytest.approx(entry_fee + 0.05 * last_close * 0.00045)
    finally:
        engine.reset()
        engine.dispose()


# ---------------------------------------------------------------------------
# Terminal outcomes: real (tiny) runs on the catalog reach `completed` at BNB fees
# ---------------------------------------------------------------------------

_RSI_STRATEGY: dict[str, object] = {
    "name": "plumbing_terminal",
    "timeframe": "4h",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi < 45"]},
    "exit_conditions": {"long": ["rsi > 55"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 3.0},
}


def _run_screening_cli(
    db: Path, strategy_id: int, parameters: dict[str, object]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Launch one real screening run through the CLI entrypoint; return (run, metrics)."""
    import argparse

    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.screening.__main__ import cmd_run

    state = StateManager(db)
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2024-03-01",
        parameters=parameters,
    )
    state.close()
    assert cmd_run(argparse.Namespace(run_id=run_id, db=str(db))) == 0

    state = StateManager(db)
    try:
        run = state.get_backtest_run(run_id)
        row = state.conn.execute(
            "SELECT total_trades, total_fees, total_return FROM sweep_results WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    finally:
        state.close()
    assert run is not None
    assert row is not None
    return run, {"trades": row[0], "fees": row[1], "return": row[2]}


def test_screening_cli_run_completes_and_charges_bnb_fees(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """parameters.bnb_discount -> pipeline -> runner -> a completed run with 0.9x fees."""
    from vibe_quant.db.state_manager import StateManager

    db = tmp_path / "screening.db"
    state = StateManager(db)
    strategy_id = state.create_strategy(name="plumbing_terminal", dsl_config=_RSI_STRATEGY)
    state.close()

    default_run, default = _run_screening_cli(db, strategy_id, {})
    default_out = capsys.readouterr().out
    bnb_run, bnb = _run_screening_cli(db, strategy_id, {"bnb_discount": True})
    bnb_out = capsys.readouterr().out

    assert default_run["status"] == bnb_run["status"] == "completed"
    assert default["trades"] == bnb["trades"] > 0
    assert default["fees"] > 0
    assert bnb["fees"] == pytest.approx(0.9 * default["fees"], rel=1e-6)
    assert bnb["return"] > default["return"]
    # The schedule is logged (no fee_schedule key lands in backtest_runs.parameters).
    assert "'bnb_discount': True" in bnb_out
    assert "'bnb_discount': False" in default_out
    assert "fee_schedule" not in bnb_run["parameters"]


def test_discovery_cli_tiny_real_run_completes_and_records_bnb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real (NT, no mocks) tiny discovery run completes with BNB in notes and log."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import __main__ as cli

    db = tmp_path / "discovery.db"
    state = StateManager(db)
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2024-03-01",
        parameters={"bnb_discount": True},
    )
    state.close()
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--run-id", str(run_id), "--population-size", "4",
            "--max-generations", "1", "--elite-count", "1", "--symbols", "BTCUSDT",
            "--timeframe", "4h", "--start-date", "2024-01-01", "--end-date", "2024-03-01",
            "--eval-windows", "1", "--train-test-split", "0", "--max-workers", "-1",
            "--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate", "--seed", "7",
            "--bnb-discount", "--db", str(db),
        ],
    )  # fmt: skip
    assert cli.main() == 0

    state = StateManager(db)
    run = state.get_backtest_run(run_id)
    result = state.get_backtest_result(run_id)
    state.close()
    assert run is not None and run["status"] == "completed"
    assert result is not None
    assert json.loads(result["notes"])["fee_schedule"] == {
        "fee_tier": "VIP0",
        "bnb_discount": True,
    }
