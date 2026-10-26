"""Discovery liquidation plumbing (bead vibe-quant-2gyk1.1.9, task C1).

Covers the discovery side of the liquidation contract:

- ``--liquidation-mode`` / ``--liquidation-leverage`` CLI flags (E9; the alias
  is the deprecated spelling of ``--leverage``, vibe-quant-6wq6w) and the
  refresh of the run's :class:`LiquidationSettings` (a ``LiquidationConfigError``
  fails the run).
- C0: ``_make_nt_backtest_fn(..., liquidation=)`` is a REQUIRED keyword, and an
  AST walk proves every ``NTBacktestFn(`` / ``_make_nt_backtest_fn(`` call in
  ``vibe_quant/`` passes it.
- D2: ``notes["liquidation"]`` is written for champion AND zero-champion runs,
  and absent when the mode is off.
- R1/R4/R5: report-only ``liquidation_count`` / ``liquidation_forfeited`` on the
  FitnessResult, champion and rejection rows, the holdout payload and
  ``full_range_liquidation_count``; nothing when the mode is off.
- I9/I10: a :class:`LiquidationIntegrityError` fails the discovery run (final DB
  status ``failed`` + the ``liquidation_mode='isolated': MarkDataMissing:``
  message) on the parallel train eval, holdout, cross-window, WFA, full-range and
  train_metrics paths.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.discovery import fitness as fitness_mod
from vibe_quant.discovery.backtest_fn import NTBacktestFn, full_range_headline
from vibe_quant.discovery.fitness import FitnessResult, evaluate_population
from vibe_quant.discovery.gate_chain import GateChain, GateEvidence, rejection_record
from vibe_quant.discovery.pipeline import DiscoveryResult, HoldoutResult
from vibe_quant.errors import DataUnavailableError, LiquidationIntegrityError
from vibe_quant.fees import FeeSchedule
from vibe_quant.liquidation_settings import LiquidationSettings

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe_quant.discovery.operators import StrategyChromosome

ISOLATED_20 = LiquidationSettings(mode="isolated", leverage=20.0)


class MarkDataMissing(Exception):
    """Shaped like the real MARK-bar failure the integrity wrap names."""


def _integrity() -> LiquidationIntegrityError:
    return LiquidationIntegrityError.wrap(MarkDataMissing("no marks"), "isolated")


def _chrom() -> StrategyChromosome:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.SHORT,
    )


def _fit(**over: Any) -> FitnessResult:
    base: dict[str, Any] = {
        "sharpe_ratio": 1.5, "max_drawdown": 0.05, "profit_factor": 1.8,
        "total_trades": 100, "total_return": 0.10, "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0, "sl_tp_penalty": 0.0, "raw_score": 1.0,
        "adjusted_score": 1.0, "passed_filters": True, "filter_results": {},
    }  # fmt: skip
    base.update(over)
    return FitnessResult(**base)


def _result(
    *, top: list[tuple[StrategyChromosome, FitnessResult]] | None = None, **over: Any
) -> DiscoveryResult:
    base: dict[str, Any] = {
        "generations": [], "top_strategies": top or [],
        "total_candidates_evaluated": 10, "converged": True, "convergence_generation": 1,
    }  # fmt: skip
    base.update(over)
    return DiscoveryResult(**base)


# ---------------------------------------------------------------------------
# CLI harness
# ---------------------------------------------------------------------------


def _create_run(db: Path, **over: Any) -> int:
    from vibe_quant.db.state_manager import StateManager

    params: dict[str, Any] = {
        "strategy_id": None, "run_mode": "discovery", "symbols": ["BTCUSDT"],
        "timeframe": "4h", "start_date": "2024-01-01", "end_date": "2025-01-01",
        "parameters": {},
    }  # fmt: skip
    params.update(over)
    state = StateManager(db)
    run_id = state.create_backtest_run(**params)
    state.close()
    return run_id


def _run_cli(
    db: Path,
    run_id: int,
    args: list[str],
    monkeypatch: pytest.MonkeyPatch,
    run_fn: Callable[[Any], DiscoveryResult],
    *,
    nt_call: Callable[..., dict[str, float | int]] | None = None,
    record_built: list[NTBacktestFn] | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
    """Run discovery ``main()`` with the pipeline's ``run`` replaced.

    ``run_fn`` receives the real pipeline instance (so it can drive the real
    gates) and returns the DiscoveryResult. Returns ``(rc, run, notes)``.
    """
    import vibe_quant.discovery.run as discovery_run
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import __main__ as cli
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    monkeypatch.setattr(discovery_run, "_check_data_available", lambda syms: True)
    if nt_call is not None:
        monkeypatch.setattr(NTBacktestFn, "__call__", nt_call)
    if record_built is not None:
        real_init = NTBacktestFn.__init__

        def recording_init(self: Any, *a: Any, **k: Any) -> None:
            real_init(self, *a, **k)
            record_built.append(self)

        monkeypatch.setattr(NTBacktestFn, "__init__", recording_init)
    monkeypatch.setattr(DiscoveryPipeline, "run", lambda self: run_fn(self))
    monkeypatch.setattr(
        "sys.argv", ["prog", "--run-id", str(run_id), "--db", str(db), *args]
    )
    rc = cli.main()

    state = StateManager(db)
    run = state.get_backtest_run(run_id)
    result = state.get_backtest_result(run_id)
    state.close()
    notes = json.loads(result["notes"]) if result and result.get("notes") else None
    assert run is not None
    return rc, run, notes


# ---------------------------------------------------------------------------
# E9 -- CLI flags
# ---------------------------------------------------------------------------


def test_bogus_liquidation_mode_exits_2() -> None:
    from vibe_quant.discovery.run_spec import build_parser

    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--run-id", "1", "--liquidation-mode", "bogus"])
    assert exc.value.code == 2


def test_liquidation_flag_defaults() -> None:
    from vibe_quant.discovery.run_spec import build_parser

    args = build_parser().parse_args(["--run-id", "1"])
    assert args.liquidation_mode == "off"
    # Both spellings default to None (no value given), never to 10 (C6/R4).
    assert args.leverage is None
    assert args.liquidation_leverage is None
    primary = build_parser().parse_args(["--run-id", "1", "--leverage", "3"])
    assert primary.leverage == 3.0
    assert primary.liquidation_leverage is None
    accepted = build_parser().parse_args(
        ["--run-id", "1", "--liquidation-mode", "cross", "--liquidation-leverage", "125"]
    )
    assert accepted.liquidation_mode == "cross"
    assert accepted.liquidation_leverage == 125.0
    assert accepted.leverage is None


def test_invalid_leverage_fails_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """leverage 0 with the mode on -> LiquidationConfigError fails the run (E9)."""
    db = tmp_path / "rt.db"
    run_id = _create_run(db)
    rc, run, _ = _run_cli(
        db, run_id,
        ["--liquidation-mode", "isolated", "--liquidation-leverage", "0"],
        monkeypatch, run_fn=lambda self: _result(),
    )  # fmt: skip
    assert rc == 1
    assert run["status"] == "failed"
    assert "LiquidationConfigError" in run["error_message"]
    assert "leverage must be a whole number in [1, 125], got 0.0" in run["error_message"]


# ---------------------------------------------------------------------------
# C0 -- required keyword + AST safety net
# ---------------------------------------------------------------------------


def test_factory_requires_liquidation_keyword() -> None:
    from vibe_quant.discovery.run import _make_nt_backtest_fn

    with pytest.raises(TypeError):
        _make_nt_backtest_fn(
            symbols=["BTCUSDT"], timeframe="4h", start_date="2024-01-01",
            end_date="2024-02-01", fee_schedule=FeeSchedule(), entry_order="market",
        )  # type: ignore[call-arg]  # fmt: skip


def test_every_construction_passes_liquidation() -> None:
    """AST walk: every NTBacktestFn(/ _make_nt_backtest_fn( call passes liquidation=."""
    import vibe_quant

    root = Path(vibe_quant.__file__).parent
    missing: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in {"NTBacktestFn", "_make_nt_backtest_fn"}:
                continue
            if not any(kw.arg == "liquidation" for kw in node.keywords):
                missing.append(f"{path.relative_to(root)}:{node.lineno}")
    assert missing == []


def test_settings_reach_every_built_fn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train GA, holdout, cross-window/WFA factory, full-range and train_metrics
    fns the CLI builds all carry the run's LiquidationSettings (V1/C0)."""
    db = tmp_path / "reach.db"
    run_id = _create_run(db, start_date="2024-01-01", end_date="2025-01-01")
    chrom = _chrom()
    fit = _fit()
    holdout = HoldoutResult(
        sharpe_ratio=0.8, max_drawdown=0.03, profit_factor=1.2,
        total_trades=30, total_return=0.02,
    )  # fmt: skip

    def run_fn(self: Any) -> DiscoveryResult:
        assert self._backtest_fn_factory is not None
        self._backtest_fn_factory("2024-02-01", "2024-03-01")
        return _result(
            top=[(chrom, fit)], holdout_results=[holdout],
            train_dates=("2024-01-01", "2024-11-01"),
            holdout_dates=("2024-11-01", "2025-01-01"),
        )

    built: list[NTBacktestFn] = []
    rc, _, notes = _run_cli(
        db, run_id,
        [
            "--start-date", "2024-01-01", "--end-date", "2025-01-01",
            "--liquidation-mode", "isolated", "--liquidation-leverage", "20",
            "--eval-windows", "2", "--cross-window-months", "1",
            "--train-test-split", "0.8",
        ],
        monkeypatch, run_fn,
        nt_call=lambda self, chrom: {
            "sharpe_ratio": 1.0, "total_return": 0.05, "max_drawdown": 0.05,
            "profit_factor": 1.5, "total_trades": 100,
        },  # fmt: skip
        record_built=built,
    )
    assert rc == 0
    assert built
    assert all(fn.liquidation == ISOLATED_20 for fn in built)
    spans = {(fn.start_date, fn.end_date) for fn in built}
    assert ("2024-02-01", "2024-03-01") in spans  # cross-window / WFA factory


# ---------------------------------------------------------------------------
# D2 -- notes["liquidation"]
# ---------------------------------------------------------------------------


def _notes_for(liquidation: LiquidationSettings | None) -> dict[str, object]:
    from vibe_quant.discovery import record

    kwargs: dict[str, Any] = dict(
        use_mock=True, eval_windows_count=1, eval_windows=None, symbol_agg="portfolio",
        split_ratio=0.0, direction=None, cross_window_months=[],
        cross_window_min_sharpe=0.5, wfa_oos_step_days=0, wfa_min_consistency=0.75,
        num_seeds=1, seed=42, bootstrap_min_sharpe=0.0, holdout_min_sharpe=0.0,
        symbols=["BTCUSDT"],
    )  # fmt: skip
    if liquidation is not None:
        kwargs["liquidation"] = liquidation
    return record.provenance_notes(_result(), **kwargs)


def test_notes_liquidation_on_and_off() -> None:
    on = _notes_for(ISOLATED_20)
    assert on["liquidation"] == {"mode": "isolated", "leverage": 20.0}
    assert "liquidation" not in _notes_for(LiquidationSettings())
    assert "liquidation" not in _notes_for(None)


def test_zero_champion_run_records_liquidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run with no champion still records notes.liquidation + row counts (D2/R1)."""
    db = tmp_path / "zero.db"
    run_id = _create_run(db)
    fit = _fit(liquidation_count=4, liquidation_forfeited=1.5)

    def run_fn(self: Any) -> DiscoveryResult:
        rejection = rejection_record(
            _chrom(), fit, "guardrails", ["nope"], timeframe=self.config.timeframe
        )
        return _result(top=[], guardrail_rejections=[rejection])

    rc, run, notes = _run_cli(
        db, run_id,
        [
            "--liquidation-mode", "isolated", "--liquidation-leverage", "20",
            "--eval-windows", "1", "--train-test-split", "0",
        ],
        monkeypatch, run_fn,
    )  # fmt: skip
    assert rc == 0
    assert run["status"] == "completed"
    assert notes is not None
    assert notes["liquidation"] == {"mode": "isolated", "leverage": 20.0}
    rejection = notes["guardrail_rejections"][0]
    assert rejection["liquidation_count"] == 4
    assert rejection["liquidation_forfeited"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# R1/R4/R5 -- report-only counts
# ---------------------------------------------------------------------------


def test_mode_off_keeps_rows_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mode off: no liquidation* keys on champion/holdout/full-range or a rejection (R4)."""
    db = tmp_path / "off.db"
    run_id = _create_run(db)
    chrom = _chrom()
    fit = _fit()  # liquidation_count / liquidation_forfeited stay None
    assert fit.liquidation_count is None and fit.liquidation_forfeited is None
    holdout = HoldoutResult(
        sharpe_ratio=0.8, max_drawdown=0.03, profit_factor=1.2,
        total_trades=30, total_return=0.02,
    )  # fmt: skip

    def run_fn(self: Any) -> DiscoveryResult:
        rejection = rejection_record(
            _chrom(), fit, "guardrails", ["nope"], timeframe=self.config.timeframe
        )
        return _result(
            top=[(chrom, fit)], holdout_results=[holdout],
            train_dates=("2024-01-01", "2024-11-01"),
            holdout_dates=("2024-11-01", "2025-01-01"),
            guardrail_rejections=[rejection],
        )

    rc, run, notes = _run_cli(
        db, run_id,
        ["--eval-windows", "1", "--train-test-split", "0"],
        monkeypatch, run_fn,
    )  # fmt: skip
    assert rc == 0
    assert run["status"] == "completed"
    assert notes is not None
    assert "liquidation" not in notes
    entry = notes["top_strategies"][0]
    assert "liquidation_count" not in entry
    assert "liquidation_forfeited" not in entry
    assert "liquidation_count" not in entry["holdout"]
    assert "full_range_liquidation_count" not in entry
    rejection = notes["guardrail_rejections"][0]
    assert "liquidation_count" not in rejection
    assert "liquidation_forfeited" not in rejection


def test_mode_on_entries_carry_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Champion + holdout + full-range rows carry the counts; off rows do not (R1/R5)."""
    db = tmp_path / "on.db"
    run_id = _create_run(db)
    chrom = _chrom()
    fit = _fit(liquidation_count=3, liquidation_forfeited=1.0)
    holdout = HoldoutResult(
        sharpe_ratio=0.8, max_drawdown=0.03, profit_factor=1.2,
        total_trades=30, total_return=0.02, liquidation_count=2,
    )  # fmt: skip

    def run_fn(self: Any) -> DiscoveryResult:
        return _result(
            top=[(chrom, fit)], holdout_results=[holdout],
            train_dates=("2024-01-01", "2024-11-01"),
            holdout_dates=("2024-11-01", "2025-01-01"),
        )

    def nt_call(self: Any, chrom: Any) -> dict[str, float | int]:
        return {
            "sharpe_ratio": 1.0, "total_return": 0.05, "max_drawdown": 0.05,
            "profit_factor": 1.5, "total_trades": 100,
            "liquidation_count": 9, "liquidation_forfeited": 2.0,
        }  # fmt: skip

    rc, run, notes = _run_cli(
        db, run_id,
        [
            "--liquidation-mode", "isolated", "--liquidation-leverage", "20",
            "--eval-windows", "2", "--train-test-split", "0",
        ],
        monkeypatch, run_fn, nt_call=nt_call,
    )  # fmt: skip
    assert rc == 0
    assert run["status"] == "completed"
    assert notes is not None
    entry = notes["top_strategies"][0]
    assert entry["liquidation_count"] == 3
    assert entry["liquidation_forfeited"] == pytest.approx(1.0)
    assert entry["holdout"]["liquidation_count"] == 2
    assert entry["full_range_liquidation_count"] == 9


def test_fitness_result_reads_counts_from_bt_dict() -> None:
    """_evaluate_single mirrors fee_drag: counts come from the bt dict, else None (R1/R4)."""
    bt_on: dict[str, float | int] = {
        "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 100, "total_return": 0.05,
        "liquidation_count": 7, "liquidation_forfeited": 3.0,
    }  # fmt: skip
    bt_off: dict[str, float | int] = {
        "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 100, "total_return": 0.05,
    }  # fmt: skip
    chrom = _chrom()
    on = evaluate_population([chrom], lambda c: bt_on, max_workers=None)[0]
    off = evaluate_population([chrom], lambda c: bt_off, max_workers=None)[0]

    assert on.liquidation_count == 7
    assert on.liquidation_forfeited == pytest.approx(3.0)
    assert off.liquidation_count is None and off.liquidation_forfeited is None
    # No score term, no gate: the same formula over the same metrics.
    assert on.adjusted_score == off.adjusted_score


def test_full_range_headline_carries_count() -> None:
    head = full_range_headline(
        {
            "sharpe_ratio": 1.0, "total_trades": 10, "max_drawdown": 0.1,
            "profit_factor": 1.5, "total_return": 0.05, "liquidation_count": 4,
        },  # fmt: skip
        fallback_sharpe=0.0, fallback_trades=0, fallback_max_dd=1.0,
        fallback_pf=0.0, fallback_return=0.0,
    )  # fmt: skip
    assert head["full_range_liquidation_count"] == 4

    off = full_range_headline(
        {
            "sharpe_ratio": 1.0, "total_trades": 10, "max_drawdown": 0.1,
            "profit_factor": 1.5, "total_return": 0.05,
        },  # fmt: skip
        fallback_sharpe=0.0, fallback_trades=0, fallback_max_dd=1.0,
        fallback_pf=0.0, fallback_return=0.0,
    )  # fmt: skip
    assert "full_range_liquidation_count" not in off


# ---------------------------------------------------------------------------
# I9 -- parallel train eval transport
# ---------------------------------------------------------------------------


def test_sequential_train_eval_integrity_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fn = NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-06-01", liquidation=ISOLATED_20)

    def boom(self: Any, chrom: Any) -> dict[str, float | int]:
        raise _integrity()

    monkeypatch.setattr(NTBacktestFn, "__call__", boom)
    with pytest.raises(
        DataUnavailableError, match="liquidation_mode='isolated': MarkDataMissing:"
    ):
        evaluate_population([_chrom()], fn, max_workers=None)


def test_parallel_train_eval_integrity_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worst-split pool transport re-raises the integrity error, not -inf (I9)."""
    from concurrent.futures import ThreadPoolExecutor

    fn = NTBacktestFn(
        ["A", "B"], "4h", "2024-01-01", "2024-06-01",
        symbol_agg="worst", liquidation=ISOLATED_20,
    )  # fmt: skip
    assert fn.splits_by_symbol

    def boom(self: Any, chrom: Any, symbols: Any = None) -> dict[str, float | int]:
        raise _integrity()

    monkeypatch.setattr(NTBacktestFn, "_eval_symbols", boom)
    monkeypatch.setattr(fitness_mod, "_parallel_broken", False)
    with ThreadPoolExecutor(max_workers=2) as ex, pytest.raises(
        DataUnavailableError, match="liquidation_mode='isolated': MarkDataMissing:"
    ):
        evaluate_population([_chrom(), _chrom()], fn, max_workers=2, executor=ex)


# ---------------------------------------------------------------------------
# I10 -- a LiquidationIntegrityError fails the run on every gate path
# ---------------------------------------------------------------------------


def _assert_failed_run(run: dict[str, Any]) -> None:
    assert run["status"] == "failed"
    assert "liquidation_mode='isolated': MarkDataMissing:" in run["error_message"]


def test_holdout_integrity_error_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "holdout.db"
    run_id = _create_run(db)

    def run_fn(self: Any) -> DiscoveryResult:
        def boom(chrom: Any) -> dict[str, float | int]:
            raise _integrity()

        chain = GateChain(
            self.config,
            GateEvidence(self._backtest_fn, self._dsr_requires_returns, holdout_backtest_fn=boom),
        )
        chain.evaluate_holdout([(_chrom(), _fit())])
        raise AssertionError("evaluate_holdout must re-raise the integrity error")

    rc, run, _ = _run_cli(
        db, run_id,
        ["--liquidation-mode", "isolated", "--liquidation-leverage", "20",
         "--train-test-split", "0.8", "--eval-windows", "1"],
        monkeypatch, run_fn,
    )  # fmt: skip
    assert rc == 1
    _assert_failed_run(run)


def test_cross_window_integrity_error_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "xwin.db"
    run_id = _create_run(db)

    def run_fn(self: Any) -> DiscoveryResult:
        def factory(s: str, e: str) -> Callable[..., dict[str, float | int]]:
            def boom(chrom: Any) -> dict[str, float | int]:
                raise _integrity()

            return boom

        chain = GateChain(
            self.config,
            GateEvidence(
                self._backtest_fn, self._dsr_requires_returns, backtest_fn_factory=factory
            ),
        )
        chain.evaluate_cross_windows([(_chrom(), _fit())])
        raise AssertionError("evaluate_cross_windows must re-raise")

    rc, run, _ = _run_cli(
        db, run_id,
        ["--liquidation-mode", "isolated", "--liquidation-leverage", "20",
         "--cross-window-months", "1", "--eval-windows", "1"],
        monkeypatch, run_fn,
    )  # fmt: skip
    assert rc == 1
    _assert_failed_run(run)


def test_wfa_integrity_error_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "wfa.db"
    run_id = _create_run(db)

    def run_fn(self: Any) -> DiscoveryResult:
        def factory(s: str, e: str) -> Callable[..., dict[str, float | int]]:
            def boom(chrom: Any) -> dict[str, float | int]:
                raise _integrity()

            return boom

        chain = GateChain(
            self.config,
            GateEvidence(
                self._backtest_fn, self._dsr_requires_returns, backtest_fn_factory=factory
            ),
        )
        chain.evaluate_wfa_rolling(
            [(_chrom(), _fit())], self.config.start_date, self.config.end_date
        )
        raise AssertionError("evaluate_wfa_rolling must re-raise")

    rc, run, _ = _run_cli(
        db, run_id,
        ["--liquidation-mode", "isolated", "--liquidation-leverage", "20",
         "--wfa-oos-step-days", "30", "--eval-windows", "1"],
        monkeypatch, run_fn,
    )  # fmt: skip
    assert rc == 1
    _assert_failed_run(run)


def test_full_range_integrity_error_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "full.db"
    run_id = _create_run(db)
    chrom, fit = _chrom(), _fit()

    def run_fn(self: Any) -> DiscoveryResult:
        return _result(top=[(chrom, fit)])

    def nt_call(self: Any, chrom: Any) -> dict[str, float | int]:
        raise _integrity()

    rc, run, _ = _run_cli(
        db, run_id,
        ["--liquidation-mode", "isolated", "--liquidation-leverage", "20",
         "--eval-windows", "2", "--train-test-split", "0"],
        monkeypatch, run_fn, nt_call=nt_call,
    )  # fmt: skip
    assert rc == 1
    _assert_failed_run(run)


def test_train_metrics_integrity_error_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "train.db"
    run_id = _create_run(db)
    chrom, fit = _chrom(), _fit()
    holdout = HoldoutResult(
        sharpe_ratio=0.8, max_drawdown=0.03, profit_factor=1.2,
        total_trades=30, total_return=0.02,
    )  # fmt: skip

    def run_fn(self: Any) -> DiscoveryResult:
        return _result(
            top=[(chrom, fit)], holdout_results=[holdout],
            train_dates=("2024-01-01", "2024-11-01"),
            holdout_dates=("2024-11-01", "2025-01-01"),
        )

    def nt_call(self: Any, chrom: Any) -> dict[str, float | int]:
        # The full-range headline fn spans args.start_date..args.end_date; the
        # continuous-train efficiency fn spans the train range only.
        if self.end_date != "2025-01-01":
            raise _integrity()
        return {
            "sharpe_ratio": 1.0, "total_return": 0.05, "max_drawdown": 0.05,
            "profit_factor": 1.5, "total_trades": 100,
        }  # fmt: skip

    rc, run, _ = _run_cli(
        db, run_id,
        ["--liquidation-mode", "isolated", "--liquidation-leverage", "20",
         "--eval-windows", "2", "--train-test-split", "0.8"],
        monkeypatch, run_fn, nt_call=nt_call,
    )  # fmt: skip
    assert rc == 1
    _assert_failed_run(run)
