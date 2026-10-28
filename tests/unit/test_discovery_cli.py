"""Tests for discovery CLI entrypoint.

Outcome tests only: what a run persists (the backtest_results row, notes fields the
API/replay/promote/DSR readers consume) and which exit code/status a failure class
ends with. Print and log text, parser plumbing and call-order pins live elsewhere or
not at all.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_run_discovery import drive_run_discovery
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import run as discovery_run
from vibe_quant.discovery.__main__ import main
from vibe_quant.discovery.run_spec import DiscoveryRunSpec, build_parser

if TYPE_CHECKING:
    from pathlib import Path


def _create_discovery_run(db_path: Path) -> int:
    state = StateManager(db_path)
    strategy_id = state.create_strategy(
        name="__test_discovery_cli__",
        dsl_config={"type": "discovery"},
        description="test strategy for discovery CLI",
        strategy_type="discovery",
    )
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-01-01",
        end_date="2025-02-01",
        parameters={},
    )
    state.close()
    return run_id


def test_main_runs_and_persists_result(tmp_path: Path, monkeypatch) -> None:
    """Running discovery CLI should complete run and persist summary metrics."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-id",
            str(run_id),
            "--population-size",
            "6",
            "--max-generations",
            "2",
            "--mutation-rate",
            "0.1",
            "--elite-count",
            "1",
            "--symbols",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-02-01",
            "--db",
            str(db_path),
            "--mock",
        ],
    )

    assert main() == 0

    state = StateManager(db_path)
    run = state.get_backtest_run(run_id)
    result = state.get_backtest_result(run_id)
    state.close()

    assert run is not None
    assert run["status"] == "completed"
    assert result is not None
    assert result["total_trades"] >= 0


def test_multi_seed_runs(tmp_path: Path, monkeypatch) -> None:
    """Multi-seed discovery should run N seeds and aggregate results."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-id",
            str(run_id),
            "--population-size",
            "6",
            "--max-generations",
            "2",
            "--elite-count",
            "1",
            "--symbols",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-02-01",
            "--num-seeds",
            "3",
            "--db",
            str(db_path),
            "--mock",
        ],
    )

    assert main() == 0

    state = StateManager(db_path)
    run = state.get_backtest_run(run_id)
    result = state.get_backtest_result(run_id)
    state.close()

    assert run is not None
    assert run["status"] == "completed"
    assert result is not None

    import json

    notes = json.loads(result["notes"])
    assert notes.get("num_seeds") == 3


def test_cross_window_metadata_persisted_to_notes(tmp_path: Path, monkeypatch) -> None:
    """Discovery notes should retain direction and cross-window config."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-id",
            str(run_id),
            "--population-size",
            "6",
            "--max-generations",
            "2",
            "--elite-count",
            "1",
            "--symbols",
            "BTCUSDT",
            "--timeframe",
            "1m",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-03-01",
            "--direction",
            "short",
            "--cross-window-months=1",
            "--cross-window-min-sharpe",
            "0.8",
            "--db",
            str(db_path),
            "--mock",
        ],
    )

    assert main() == 0

    state = StateManager(db_path)
    result = state.get_backtest_result(run_id)
    state.close()

    assert result is not None

    import json

    notes = json.loads(result["notes"])
    assert notes["direction"] == "short"
    assert notes["cross_window_months"] == [1]
    assert notes["cross_window_min_sharpe"] == 0.8


def test_no_viable_strategies_completes_cleanly(tmp_path: Path, monkeypatch) -> None:
    """When all candidates fail hard guardrails, run should exit 0 with a
    structured summary — not crash with RuntimeError. Regression test for
    vibe-quant-97oh.
    """
    from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult

    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    def fake_run(self) -> DiscoveryResult:
        return DiscoveryResult(
            generations=[],
            top_strategies=[],
            total_candidates_evaluated=42,
            converged=False,
            convergence_generation=None,
        )

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)

    argv = [
        "--run-id", str(run_id),
        "--population-size", "6",
        "--max-generations", "2",
        "--elite-count", "1",
        "--symbols", "BTCUSDT",
        "--timeframe", "1h",
        "--start-date", "2025-01-01",
        "--end-date", "2025-02-01",
        "--db", str(db_path),
        "--mock", "--max-workers", "-1",
    ]  # fmt: skip

    outcome = drive_run_discovery(db_path, run_id, argv, discovery_run.MockBacktests())
    assert outcome.exit_code == 0

    state = StateManager(db_path)
    run = state.get_backtest_run(run_id)
    result = state.get_backtest_result(run_id)
    state.close()

    assert run is not None
    assert run["status"] == "completed"
    assert result is not None

    import json

    notes = json.loads(result["notes"])
    assert notes["outcome"] == "no_viable_strategies"
    assert notes["evaluated"] == 42
    assert notes["top_strategies"] == []
    assert "reason" in notes


@pytest.mark.parametrize(
    ("timeframe", "extra", "expected"),
    [("1m", [], 0.5), ("1h", ["--bootstrap-min-sharpe", "1.5"], 1.5)],
    ids=["1m_default", "explicit_override"],
)
def test_bootstrap_floor_persisted_to_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeframe: str,
    extra: list[str],
    expected: float,
) -> None:
    """The resolved floor is what the run records (and later gates read)."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--run-id", str(run_id),
            "--population-size", "6", "--max-generations", "2",
            "--elite-count", "1", "--symbols", "BTCUSDT", "--timeframe", timeframe,
            "--start-date", "2025-01-01", "--end-date", "2025-02-01",
            "--db", str(db_path), "--mock", "--max-workers", "-1", *extra,
        ],
    )  # fmt: skip
    assert main() == 0
    assert _run_notes(db_path, run_id)["bootstrap_min_sharpe"] == expected


def test_walk_forward_efficiency_persisted_to_column(tmp_path: Path, monkeypatch) -> None:
    """Walk-forward efficiency = holdout return/day over train return/day (Pardo),
    length-normalized (vibe-quant-e70tl.10 / bd-2ur2).
    """
    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )
    from vibe_quant.discovery.pipeline import (
        DiscoveryPipeline,
        DiscoveryResult,
        HoldoutResult,
    )

    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    chrom = StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        stop_loss_pct=2.0, take_profit_pct=4.0, direction=Direction.SHORT,
    )
    fit = FitnessResult(
        sharpe_ratio=1.5, max_drawdown=0.05, profit_factor=1.8,
        total_trades=100, total_return=0.10,
        complexity_penalty=0.0, overtrade_penalty=0.0, sl_tp_penalty=0.0,
        raw_score=1.0, adjusted_score=1.0,
        passed_filters=True, filter_results={},
    )
    holdout = HoldoutResult(sharpe_ratio=0.8, max_drawdown=0.03, profit_factor=1.2,
                            total_trades=30, total_return=0.02)

    def fake_run(self) -> DiscoveryResult:
        return DiscoveryResult(
            generations=[], top_strategies=[(chrom, fit)],
            total_candidates_evaluated=10, converged=True, convergence_generation=1,
            holdout_results=[holdout],
            train_dates=("2025-01-01", "2025-01-25"),
            holdout_dates=("2025-01-25", "2025-02-01"),
        )

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)

    argv = [
        "--run-id", str(run_id),
        "--population-size", "6", "--max-generations", "2", "--elite-count", "1",
        "--symbols", "BTCUSDT", "--timeframe", "1h",
        "--start-date", "2025-01-01", "--end-date", "2025-02-01",
        "--eval-windows", "1",
        "--db", str(db_path), "--mock", "--max-workers", "-1",
    ]

    outcome = drive_run_discovery(db_path, run_id, argv, discovery_run.MockBacktests())
    assert outcome.exit_code == 0

    state = StateManager(db_path)
    result = state.get_backtest_result(run_id)
    state.close()
    assert result is not None
    # (0.02 / 7 days) / (0.10 / 24 days) = 0.48 / 0.70
    assert result["walk_forward_efficiency"] == pytest.approx(0.48 / 0.70, abs=1e-12)


def test_walk_forward_efficiency_formula() -> None:
    """Stationary edge -> 1.0 whatever the period lengths; IS<=0 -> undefined."""
    from vibe_quant.discovery.run import walk_forward_efficiency

    # 0.05%/day in both periods: 270d IS (13.5%) vs 90d OOS (4.5%)
    assert walk_forward_efficiency(
        oos_return=0.045, oos_days=90, is_return=0.135, is_days=270,
    ) == pytest.approx(1.0, abs=1e-12)
    # Losing in-sample, winning out-of-sample is NOT "max efficiency"
    assert walk_forward_efficiency(
        oos_return=0.05, oos_days=90, is_return=-0.10, is_days=270,
    ) is None
    assert walk_forward_efficiency(
        oos_return=0.05, oos_days=None, is_return=0.10, is_days=270,
    ) is None


def test_walk_forward_efficiency_none_when_no_wfa(tmp_path: Path, monkeypatch) -> None:
    """WFA column stays NULL when WFA did not run."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--run-id", str(run_id),
            "--population-size", "6", "--max-generations", "2",
            "--mutation-rate", "0.1", "--elite-count", "1",
            "--symbols", "BTCUSDT", "--timeframe", "1h",
            "--start-date", "2025-01-01", "--end-date", "2025-02-01",
            "--db", str(db_path), "--mock", "--max-workers", "-1",
        ],
    )
    assert main() == 0

    state = StateManager(db_path)
    result = state.get_backtest_result(run_id)
    state.close()
    assert result is not None
    assert result["walk_forward_efficiency"] is None


def test_multi_seed_preserves_validation_metadata_per_strategy(tmp_path: Path, monkeypatch) -> None:
    """Merged winners should keep their own holdout metrics, not another seed's."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-id",
            str(run_id),
            "--population-size",
            "6",
            "--max-generations",
            "2",
            "--elite-count",
            "1",
            "--max-workers",
            "-1",
            "--symbols",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-03-01",
            "--train-test-split",
            "0.5",
            "--num-seeds",
            "3",
            # mock Sharpes over 1 month aren't DSR-significant; this test is
            # about holdout metadata alignment, not DSR
            "--no-dsr",
            "--db",
            str(db_path),
            "--mock", "--max-workers", "-1",
        ],
    )

    assert main() == 0

    state = StateManager(db_path)
    result = state.get_backtest_result(run_id)
    state.close()

    assert result is not None

    import json

    notes = json.loads(result["notes"])
    strategies = notes["top_strategies"]
    assert strategies
    for strategy in strategies:
        holdout = strategy.get("holdout")
        assert holdout is not None
        assert holdout["sharpe"] == strategy["sharpe"]
        assert holdout["trades"] == strategy["trades"]
        assert holdout["return_pct"] == strategy["return_pct"]


def test_guardrail_flags_default_to_enabled() -> None:
    """Parser should default to bootstrap CI + DSR enabled, matching prior hardcoded behavior."""
    args = build_parser().parse_args(["--run-id", "1"])
    assert args.require_bootstrap_ci is True
    assert args.require_dsr is True
    # bootstrap_min_sharpe defaults to None at parse time; resolved to
    # 0.5 (1m) or 1.0 (else) in main() based on --timeframe.
    assert args.bootstrap_min_sharpe is None
    assert args.bootstrap_ci_level == 0.95


def test_guardrail_flags_relax_and_disable() -> None:
    """--no-bootstrap-ci / --no-dsr / --bootstrap-min-sharpe should flow through the parser."""
    args = build_parser().parse_args(
        [
            "--run-id",
            "1",
            "--no-bootstrap-ci",
            "--no-dsr",
            "--bootstrap-min-sharpe",
            "0.25",
            "--bootstrap-ci-level",
            "0.9",
        ]
    )
    assert args.require_bootstrap_ci is False
    assert args.require_dsr is False
    assert args.bootstrap_min_sharpe == 0.25
    assert args.bootstrap_ci_level == 0.9


def test_mock_backtest_is_picklable_by_module() -> None:
    """_mock_backtest must live in an importable module so workers can unpickle it."""
    from vibe_quant.discovery.mock_backtest import mock_backtest as _mock_backtest

    assert _mock_backtest.__module__ == "vibe_quant.discovery.mock_backtest"
    assert pickle.loads(pickle.dumps(_mock_backtest)) is _mock_backtest


def _seed_argv(db_path: Path, run_id: int, *extra: str) -> list[str]:
    return [
        "prog", "--run-id", str(run_id),
        "--population-size", "6", "--max-generations", "2", "--elite-count", "1",
        "--symbols", "BTCUSDT", "--timeframe", "1h",
        "--start-date", "2025-01-01", "--end-date", "2025-02-01",
        "--db", str(db_path), "--mock", "--max-workers", "-1", *extra,
    ]  # fmt: skip


def _run_notes(db_path: Path, run_id: int) -> dict[str, Any]:
    import json

    state = StateManager(db_path)
    result = state.get_backtest_result(run_id)
    state.close()
    assert result is not None
    notes: dict[str, Any] = json.loads(result["notes"])
    return notes


def test_seed_persisted_in_notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    monkeypatch.setattr("sys.argv", _seed_argv(db_path, run_id, "--seed", "123"))
    assert main() == 0
    assert _run_notes(db_path, run_id)["seed"] == 123


_TIMING_KEYS = ("time", "elapsed", "duration", "timestamp", "_at")


def _strip_timing(obj: Any) -> Any:
    """Drop wall-clock keys (execution_time, *_at, ...) recursively."""
    if isinstance(obj, dict):
        return {
            k: _strip_timing(v)
            for k, v in obj.items()
            if not any(t in str(k).lower() for t in _TIMING_KEYS)
        }
    if isinstance(obj, list):
        return [_strip_timing(v) for v in obj]
    return obj


def _notes_for_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: str, seed: int) -> str:
    import json

    db_path = tmp_path / f"state_{tag}.db"
    run_id = _create_discovery_run(db_path)
    monkeypatch.setattr(
        "sys.argv", _seed_argv(
            db_path, run_id, "--seed", str(seed), "--max-workers", "-1",
            "--no-dsr", "--no-bootstrap-ci",
        )  # fmt: skip
    )
    assert main() == 0
    return json.dumps(_strip_timing(_run_notes(db_path, run_id)), sort_keys=True)


def test_same_seed_same_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    a = _notes_for_seed(tmp_path, monkeypatch, "a", 7)
    b = _notes_for_seed(tmp_path, monkeypatch, "b", 7)
    c = _notes_for_seed(tmp_path, monkeypatch, "c", 8)
    assert json.loads(a)["top_strategies"], "toothless compare: no champions produced"
    assert a == b
    assert a != c


def _always_raises(chromosome: object) -> dict[str, float]:
    raise RuntimeError("catalog exploded")


def test_main_marks_run_failed_when_every_evaluation_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    monkeypatch.setattr("vibe_quant.discovery.run._mock_backtest", _always_raises)
    monkeypatch.setattr(
        "sys.argv", _seed_argv(db_path, run_id, "--seed", "1", "--max-workers", "-1")
    )
    assert main() != 0
    state = StateManager(db_path)
    row = state.conn.execute(
        "SELECT status, error_message FROM backtest_runs WHERE id = ?", (run_id,)
    ).fetchone()
    state.close()
    assert row[0] == "failed"
    assert "DiscoveryEvaluationError" in row[1]


def _recording_seed(seen: list[int], real_seed: Any) -> Any:
    def _seed(s: int | None = None) -> None:
        assert s is not None
        seen.append(s)
        real_seed(s)

    return _seed


def test_multi_seed_seed_autodrawn_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import random

    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    seen: list[int] = []
    real_seed = random.seed
    monkeypatch.setattr(random, "seed", _recording_seed(seen, real_seed))
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, *a, **k: 1234)
    monkeypatch.setattr("sys.argv", _seed_argv(db_path, run_id, "--num-seeds", "2"))
    assert main() == 0
    # seed i = base + i * 7919, and the BASE is what the run records.
    assert seen[:2] == [1234, 9153]
    assert _run_notes(db_path, run_id)["seed"] == 1234


def test_multi_seed_explicit_seed_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import random

    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    seen: list[int] = []
    real_seed = random.seed
    monkeypatch.setattr(random, "seed", _recording_seed(seen, real_seed))
    monkeypatch.setattr("sys.argv", _seed_argv(db_path, run_id, "--num-seeds", "2", "--seed", "5"))
    assert main() == 0
    assert seen[:2] == [5, 7924]
    assert _run_notes(db_path, run_id)["seed"] == 5


@pytest.mark.parametrize(
    ("start", "end", "split", "n_eval", "expected"),
    [
        # default split: 80% of the range, one continuous train window
        (
            "2024-01-01", "2026-02-24", 0.8, 1,
            ("2024-01-01", "2025-09-20", "2025-09-20", "2026-02-24", None),
        ),
        # a 5-month range split 25/75 into 3 eval windows
        (
            "2025-01-01", "2025-06-01", 0.25, 3,
            (
                "2025-01-01", "2025-02-07", "2025-02-07", "2025-06-01",
                [("2025-01-01", "2025-01-13"), ("2025-01-13", "2025-01-25"),
                 ("2025-01-25", "2025-02-07")],
            ),
        ),
        # odd day count across a leap year
        (
            "2024-02-03", "2025-01-30", 0.75, 2,
            (
                "2024-02-03", "2024-10-31", "2024-10-31", "2025-01-30",
                [("2024-02-03", "2024-06-17"), ("2024-06-17", "2024-10-31")],
            ),
        ),
        # split 0 (no holdout) keeps the whole range as train
        (
            "2023-05-17", "2025-11-18", 0.0, 4,
            (
                "2023-05-17", "2025-11-18", None, None,
                [("2023-05-17", "2024-01-01"), ("2024-01-01", "2024-08-17"),
                 ("2024-08-17", "2025-04-03"), ("2025-04-03", "2025-11-18")],
            ),
        ),
        # eval_windows 1 is a single window = no sub-window split
        (
            "2025-01-01", "2025-06-01", 0.25, 1,
            ("2025-01-01", "2025-02-07", "2025-02-07", "2025-06-01", None),
        ),
    ],
)  # fmt: skip
def test_plan_windows_boundaries(
    start: str,
    end: str,
    split: float,
    n_eval: int,
    expected: tuple[str, str, str | None, str | None, list[tuple[str, str]] | None],
) -> None:
    """The spec's plan is the literal split the run (and the API precheck) uses."""
    args = build_parser().parse_args(
        [
            "--run-id", "0", "--start-date", start, "--end-date", end,
            "--train-test-split", str(split), "--eval-windows", str(n_eval),
        ]
    )
    plan = DiscoveryRunSpec.from_args(args).plan()
    assert (
        plan.train_start, plan.train_end, plan.holdout_start, plan.holdout_end, plan.eval_windows
    ) == expected


def test_bad_window_config_fails_run_fast(tmp_path: Path, monkeypatch) -> None:
    """--cross-window-months 6 on a 3-month range: run failed, DiscoveryConfigError."""
    import time

    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--run-id", str(run_id), "--population-size", "6",
            "--max-generations", "2", "--elite-count", "1", "--timeframe", "1h",
            "--start-date", "2025-01-01", "--end-date", "2025-04-01",
            "--cross-window-months", "6", "--mock", "--max-workers", "-1", "--db", str(db_path),
        ],
    )
    t0 = time.perf_counter()
    assert main() == 1
    assert time.perf_counter() - t0 < 10
    state = StateManager(db_path)
    run = state.get_backtest_run(run_id)
    state.close()
    assert run is not None
    assert run["status"] == "failed"
    assert run["error_message"].startswith("DiscoveryConfigError:")


def test_stale_progress_file_never_shown_for_a_new_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A progress file left by another DB's run with the same id must not survive
    into this run: the API would show its generation/best fitness as ours."""
    import json

    from vibe_quant.utils import log_dir

    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(tmp_path / "logs"))  # xdist workers share one
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    stale = log_dir() / f"discovery_{run_id}_progress.json"
    stale.write_text(json.dumps({"generation": 1, "max_generations": 3, "best_fitness": 0.10225}))
    # Fails in generation 1, before the pipeline writes any progress of its own.
    monkeypatch.setattr("vibe_quant.discovery.run._mock_backtest", _always_raises)
    monkeypatch.setattr(
        "sys.argv", _seed_argv(db_path, run_id, "--seed", "1", "--max-workers", "-1")
    )
    assert main() != 0
    assert not stale.exists()


def _create_seed_run(db_path: Path) -> int:
    """A bare discovery run (no strategy row) to act as a --seed-from-run source."""
    state = StateManager(db_path)
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-01-01",
        end_date="2025-02-01",
        parameters={},
    )
    state.close()
    return run_id


def _seed_chromosome() -> Any:
    """A minimally valid chromosome that round-trips through the genome serializer."""
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
        take_profit_pct=3.0,
        direction=Direction.LONG,
    )


def test_seed_from_run_chromosome_reaches_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm-start: the persisted champion reaches DiscoveryPipeline seeded, intact.

    Drives main() -- the path that calls load_seed_chromosomes before building the
    pipeline. A --seed-from-run whose run carries one persisted top_strategies
    chromosome must hand that exact chromosome to DiscoveryPipeline's
    seed_chromosomes argument: the same object the loader returned (same uid) with
    its serialized genome unchanged. A None result there (the round-1 gap) raises
    the warm-start ValueError before any pipeline is built.
    """
    import json

    import vibe_quant.discovery.run as discovery_run
    from vibe_quant.discovery.genome import chromosome_to_serializable

    db_path = tmp_path / "state.db"
    seed_run_id = _create_seed_run(db_path)
    seeded = _seed_chromosome()
    state = StateManager(db_path)
    state.save_backtest_result(
        seed_run_id,
        {
            "notes": json.dumps(
                {"top_strategies": [{"chromosome": chromosome_to_serializable(seeded)}]}
            )
        },
    )
    state.close()

    run_id = _create_discovery_run(db_path)

    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(tmp_path / "logs"))  # keep progress files out of the repo

    loaded: list[Any] = []
    real_loader = discovery_run.load_seed_chromosomes

    def spy_loader(state: StateManager, run_id: int) -> Any:  # noqa: F811
        chromosomes = real_loader(state, run_id)
        loaded.append(chromosomes)
        return chromosomes

    monkeypatch.setattr(discovery_run, "load_seed_chromosomes", spy_loader)

    captured: dict[str, Any] = {}
    real_pipeline = discovery_run.DiscoveryPipeline

    def spy_pipeline(*args: Any, **kwargs: Any) -> Any:
        captured["seed_chromosomes"] = kwargs.get("seed_chromosomes")
        return real_pipeline(*args, **kwargs)

    monkeypatch.setattr(discovery_run, "DiscoveryPipeline", spy_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        _seed_argv(db_path, run_id, "--seed-from-run", str(seed_run_id), "--max-workers", "-1"),
    )

    assert main() == 0

    assert loaded and loaded[0] is not None
    assert captured["seed_chromosomes"] is loaded[0]  # the loader's own list, not a copy
    got = captured["seed_chromosomes"][0]
    assert got.uid == loaded[0][0].uid  # the exact chromosome, by uid
    assert chromosome_to_serializable(got) == chromosome_to_serializable(seeded)  # ... and genes


def test_seed_from_run_without_champions_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --seed-from-run with no loadable champion fails the run, not silently.

    The CLI keeps the runtime backstop (the API refuses such a launch earlier): a
    run row created directly must end failed, with the exact ValueError text, and
    no DiscoveryPipeline may be built.
    """
    import vibe_quant.discovery.run as discovery_run

    db_path = tmp_path / "state.db"
    seed_run_id = _create_seed_run(db_path)  # no result row -> no champions
    run_id = _create_discovery_run(db_path)

    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(tmp_path / "logs"))  # keep progress files out of the repo

    def _no_pipeline(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no pipeline may be built for an unhonorable warm-start")

    monkeypatch.setattr(discovery_run, "DiscoveryPipeline", _no_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        _seed_argv(db_path, run_id, "--seed-from-run", str(seed_run_id), "--max-workers", "-1"),
    )

    assert main() == 1

    state = StateManager(db_path)
    run = state.get_backtest_run(run_id)
    state.close()
    assert run is not None
    assert run["status"] == "failed"
    assert run["error_message"] == (
        f"ValueError: warm-start run {seed_run_id} has no persisted champions to seed from "
        "(guardrails may have rejected all candidates); launch without seed_run_id instead"
    )
