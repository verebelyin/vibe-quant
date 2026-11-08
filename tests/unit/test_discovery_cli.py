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
    structured summary — not crash with RuntimeError.

    regression: vibe-quant-97oh
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
    length-normalized.

    regression: vibe-quant-e70tl.10
    regression: vibe-quant-2ur2
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


# ---------------------------------------------------------------------------
# --port: evolve a TradingView port's own parameters
# ---------------------------------------------------------------------------


def _port_doc(name: str = "dual_ema", **extra: Any) -> dict[str, Any]:
    """Dual EMA cross, 2 free params (fast, slow) and one fixed (exit_pct)."""
    doc: dict[str, Any] = {
        "name": name,
        "params": {
            "fast": {"kind": "int", "low": 5, "high": 20, "step": 1, "default": 10},
            "slow": {"kind": "int", "low": 30, "high": 60, "step": 5, "default": 40},
        },
        "constraints": ["fast < slow"],
        "dsl": {
            "name": name,
            "timeframe": "1h",
            "indicators": {
                "fast_ema": {"type": "EMA", "period": "{{fast}}"},
                "slow_ema": {"type": "EMA", "period": "{{slow}}"},
            },
            "entry_conditions": {"long": ["fast_ema crosses_above slow_ema"], "short": []},
            "exit_conditions": {"long": ["fast_ema crosses_below slow_ema"], "short": []},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 6.0},
        },
    }
    doc.update(extra)
    return doc


def _write_port(tmp_path: Path, doc: dict[str, Any] | None = None, folder: str = "port") -> Path:
    import yaml

    doc = doc or _port_doc()
    directory = tmp_path / folder
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "port.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return directory


def _port_argv(run_id: int, db_path: Path, port: Path | str, *extra: str) -> list[str]:
    return [
        "--run-id", str(run_id),
        "--population-size", "6", "--max-generations", "2", "--elite-count", "1",
        "--symbols", "BTCUSDT", "--timeframe", "1h",
        "--start-date", "2025-01-01", "--end-date", "2025-02-01",
        "--eval-windows", "1", "--train-test-split", "0",
        "--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate",
        "--max-workers", "-1", "--seed", "5",
        "--db", str(db_path), "--port", str(port), *extra,
    ]  # fmt: skip


class _PortBacktests:
    """Real-shaped adapter whose backtest is a fake that scores a port genome by its values."""

    mock = False

    def __init__(self, seen: list[dict[str, object]] | None = None) -> None:
        self.seen = seen if seen is not None else []

    def data_available(self, symbols: list[str]) -> bool:
        return True

    def cell_fns(self, **_kwargs: Any) -> Any:
        from vibe_quant.discovery.pooling import _CellFns

        seen = self.seen

        def backtest(genome: Any) -> dict[str, float | int]:
            seen.append(dict(genome.values))
            gap = float(genome.values["slow"]) - float(genome.values["fast"])
            return {"sharpe_ratio": 1.0 + gap / 100.0, "max_drawdown": 0.1, "profit_factor": 1.5,
                    "total_trades": 200, "total_return": 0.3}  # fmt: skip

        return _CellFns(backtest)

    def nt_fn(self, **_kwargs: Any) -> Any:
        raise RuntimeError("the fake adapter runs no NT backtest")


def _result_notes(db_path: Path, run_id: int) -> dict[str, Any]:
    import json

    state = StateManager(db_path)
    try:
        result = state.get_backtest_result(run_id)
    finally:
        state.close()
    assert result is not None
    return json.loads(result["notes"])  # type: ignore[no-any-return]


@pytest.mark.parametrize("as_file", [False, True])
def test_port_run_completes_and_records_the_port(tmp_path: Path, as_file: bool) -> None:
    """A port folder or its port.yaml runs the pipeline; notes carry the port identity."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    folder = _write_port(tmp_path)
    port_arg = folder / "port.yaml" if as_file else folder
    seen: list[dict[str, object]] = []

    outcome = drive_run_discovery(
        db_path, run_id, _port_argv(run_id, db_path, port_arg), _PortBacktests(seen)
    )

    assert outcome.exit_code == 0, outcome.message
    state = StateManager(db_path)
    row = state.get_backtest_run(run_id)
    state.close()
    assert row is not None and row["status"] == "completed"
    notes = _result_notes(db_path, run_id)
    import hashlib

    port_note = notes["port"]
    assert port_note["name"] == "dual_ema"
    assert port_note["path"] == str(folder / "port.yaml")
    assert port_note["sha256"] == hashlib.sha256((folder / "port.yaml").read_bytes()).hexdigest()
    assert port_note["params"] == [
        {"name": "fast", "kind": "int", "low": 5, "high": 20, "step": 1, "choices": [], "default": 10},
        {"name": "slow", "kind": "int", "low": 30, "high": 60, "step": 5, "choices": [], "default": 40},
    ]
    champion = notes["top_strategies"][0]
    assert champion["chromosome"]["kind"] == "param"
    assert champion["dsl"]["indicators"]["fast_ema"]["type"] == "EMA"
    assert {"fast": 10, "slow": 40} in seen  # the defaults genome is always in generation 0
    assert all(set(values) == {"fast", "slow"} for values in seen)


def test_legacy_run_has_no_port_note(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    argv = [
        "--run-id", str(run_id), "--population-size", "6", "--max-generations", "2",
        "--symbols", "BTCUSDT", "--timeframe", "1h",
        "--start-date", "2025-01-01", "--end-date", "2025-02-01",
        "--db", str(db_path), "--mock", "--max-workers", "-1",
    ]  # fmt: skip
    assert drive_run_discovery(db_path, run_id, argv, discovery_run.MockBacktests()).exit_code == 0
    assert "port" not in _result_notes(db_path, run_id)


def _main_exit(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["prog", *argv])
    with pytest.raises(SystemExit) as raised:
        main()
    assert isinstance(raised.value.code, int)
    return raised.value.code


def _refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    port: Path | str,
    *extra: str,
) -> str:
    """Launch through ``main()`` and return stderr; the DB must never have been opened."""
    db_path = tmp_path / "refused.db"
    assert _main_exit(monkeypatch, _port_argv(1, db_path, port, *extra)) == 2
    assert not db_path.exists(), "a refused launch must not touch the state DB"
    return capsys.readouterr().err


def test_port_error_exits_2_with_the_message_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    missing = tmp_path / "no_such_port"
    assert "cannot read port file" in _refused(monkeypatch, capsys, tmp_path, missing)
    bad = _write_port(tmp_path, {**_port_doc(), "constraints": ["fast < nope"]}, folder="bad")
    assert "nope" in _refused(monkeypatch, capsys, tmp_path, bad)


@pytest.mark.parametrize(
    ("extra", "flag"),
    [
        (["--indicator-pool", "RSI,SMA"], "--indicator-pool"),
        (["--direction", "long"], "--direction"),
        (["--mock"], "--mock"),
        (["--evaluator", "fast"], "--evaluator fast"),
        (["--num-seeds", "2"], "--num-seeds > 1"),
        (
            ["--bankroll-modes", "off", "--bankroll-leverages", "5", "--bankroll-max-margin-pcts", "50"],
            "the --bankroll-* grid",
        ),
    ],
)
def test_port_excludes_flags_that_shape_the_legacy_chromosome(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    extra: list[str],
    flag: str,
) -> None:
    err = _refused(monkeypatch, capsys, tmp_path, _write_port(tmp_path), *extra)
    assert f"--port excludes {flag}" in err


def test_port_timeframe_guard_refuses_a_timeframe_the_port_does_not_allow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """backtest_fn overwrites the rendered timeframe, so the launch check is the only guard."""
    port = _write_port(tmp_path, _port_doc(allowed_timeframes=["1h", "4h"]))
    err = _refused(monkeypatch, capsys, tmp_path, port, "--timeframe", "15m")
    assert "dual_ema does not support timeframe 15m" in err


@pytest.mark.parametrize("timeframe", ["1h", "4h"])
def test_port_timeframe_guard_accepts_an_allowed_timeframe(tmp_path: Path, timeframe: str) -> None:
    from vibe_quant.discovery.run_spec import check_launch_port

    port = _write_port(tmp_path, _port_doc(allowed_timeframes=["1h", "4h"]))
    parser = build_parser()
    args = parser.parse_args(_port_argv(1, tmp_path / "x.db", port, "--timeframe", timeframe))
    check_launch_port(parser, args)  # returns: no SystemExit


@pytest.mark.parametrize("where", ["dsl_timeframe", "additional_timeframes", "indicator_timeframe"])
def test_port_refuses_a_param_that_changes_a_timeframe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, where: str
) -> None:
    """The loaded timeframe set is a launch invariant (it sizes bar checks and fill ticks)."""
    doc = _port_doc()
    doc["params"]["tf"] = {"kind": "choice", "choices": ["15m", "5m"], "default": "15m"}
    dsl = doc["dsl"]
    if where == "dsl_timeframe":
        dsl["timeframe"] = "{{tf}}"
    elif where == "additional_timeframes":
        dsl["additional_timeframes"] = ["{{tf}}"]
    else:
        dsl["additional_timeframes"] = ["15m", "5m"]
        dsl["indicators"]["slow_ema"]["timeframe"] = "{{tf}}"
    err = _refused(monkeypatch, capsys, tmp_path, _write_port(tmp_path, doc))
    assert "port parameters may not change timeframes" in err


def test_a_direct_port_run_refuses_a_timeframe_param_and_fails_the_run(tmp_path: Path) -> None:
    """run_discovery guards a caller that skips the launch check: exit 2, run row failed."""
    doc = _port_doc()
    doc["params"]["tf"] = {"kind": "choice", "choices": ["15m", "5m"], "default": "15m"}
    doc["dsl"]["additional_timeframes"] = ["{{tf}}"]
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    outcome = drive_run_discovery(
        db_path, run_id, _port_argv(run_id, db_path, _write_port(tmp_path, doc)), _PortBacktests()
    )
    assert outcome.exit_code == 2
    assert "port parameters may not change timeframes" in outcome.message
    state = StateManager(db_path)
    row = state.get_backtest_run(run_id)
    state.close()
    assert row is not None and row["status"] == "failed"


def test_port_accepts_a_fixed_finer_additional_timeframe(tmp_path: Path) -> None:
    from vibe_quant.discovery.run_spec import check_launch_port

    doc = _port_doc()
    doc["dsl"]["additional_timeframes"] = ["15m"]
    doc["dsl"]["indicators"]["slow_ema"]["timeframe"] = "15m"
    parser = build_parser()
    args = parser.parse_args(_port_argv(1, tmp_path / "x.db", _write_port(tmp_path, doc)))
    check_launch_port(parser, args)  # returns: no SystemExit


def test_port_with_no_free_param_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    doc = _port_doc()
    doc["params"] = {
        "fast": {"kind": "int", "low": 10, "high": 10, "step": 1, "default": 10},
        "slow": {"kind": "int", "low": 40, "high": 40, "step": 1, "default": 40},
    }
    err = _refused(monkeypatch, capsys, tmp_path, _write_port(tmp_path, doc))
    assert "has no free parameters" in err


def test_legacy_launch_never_reads_port_checks(tmp_path: Path) -> None:
    from vibe_quant.discovery.run_spec import check_launch_port

    parser = build_parser()
    args = parser.parse_args(["--run-id", "1", "--direction", "long", "--indicator-pool", "RSI"])
    check_launch_port(parser, args)  # no --port: nothing to refuse


def _new_run(db_path: Path) -> int:
    """Another discovery run row in an existing state DB (no strategy row, so repeatable)."""
    state = StateManager(db_path)
    try:
        return state.create_backtest_run(
            None, "discovery", ["BTCUSDT"], "1h", "2025-01-01", "2025-02-01", {}
        )
    finally:
        state.close()


def _finished_run_notes(db_path: Path, notes: dict[str, Any]) -> int:
    from vibe_quant.db.result_notes import dump_notes

    run_id = _new_run(db_path)
    state = StateManager(db_path)
    state.save_backtest_result(run_id, {
        "total_return": 0.1, "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 100, "skewness": 0.0, "kurtosis": 3.0, "execution_time_seconds": 1.0,
        "notes": dump_notes(notes),
    })  # fmt: skip
    state.close()
    return run_id


def test_warm_start_from_a_port_run_seeds_generation_zero(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    folder = _write_port(tmp_path)
    first = _new_run(db_path)
    assert drive_run_discovery(db_path, first, _port_argv(first, db_path, folder), _PortBacktests()).exit_code == 0
    champion = _result_notes(db_path, first)["top_strategies"][0]["chromosome"]["values"]

    second = _new_run(db_path)
    seen: list[dict[str, object]] = []
    outcome = drive_run_discovery(
        db_path, second,
        _port_argv(second, db_path, folder, "--seed-from-run", str(first), "--population-size", "3"),
        _PortBacktests(seen),
    )
    assert outcome.exit_code == 0, outcome.message
    assert seen[0] == champion  # the stored champion is the first genome evaluated


def test_warm_start_refusals_exit_2_with_the_message(tmp_path: Path) -> None:
    """Legacy seeds, another port and a changed port file all refuse the launch."""
    from vibe_quant.discovery.genome import chromosome_to_serializable
    from vibe_quant.discovery.operators import Direction, _random_chromosome

    db_path = tmp_path / "state.db"
    folder = _write_port(tmp_path)
    legacy = _finished_run_notes(
        db_path,
        {"top_strategies": [{"chromosome": chromosome_to_serializable(_random_chromosome(Direction.LONG))}]},
    )
    first = _new_run(db_path)
    assert drive_run_discovery(db_path, first, _port_argv(first, db_path, folder), _PortBacktests()).exit_code == 0

    # a legacy seed run into a port run
    run_id = _new_run(db_path)
    outcome = drive_run_discovery(
        db_path, run_id, _port_argv(run_id, db_path, folder, "--seed-from-run", str(legacy)), _PortBacktests()
    )
    assert outcome.exit_code == 2
    assert "seed run holds legacy chromosomes; a run of port 'dual_ema' needs param genomes" in outcome.message
    state = StateManager(db_path)
    row = state.get_backtest_run(run_id)
    state.close()
    assert row is not None and row["status"] == "failed"

    # another port revision (same name, edited file) and another port name
    edited = _write_port(tmp_path, {**_port_doc(), "constraints": ["fast <= slow"]}, folder="edited")
    run_id = _new_run(db_path)
    outcome = drive_run_discovery(
        db_path, run_id, _port_argv(run_id, db_path, edited, "--seed-from-run", str(first)), _PortBacktests()
    )
    assert outcome.exit_code == 2
    assert "produced by a different revision of port 'dual_ema'" in outcome.message
    renamed = _write_port(tmp_path, _port_doc("other_port"), folder="renamed")
    run_id = _new_run(db_path)
    outcome = drive_run_discovery(
        db_path, run_id, _port_argv(run_id, db_path, renamed, "--seed-from-run", str(first)), _PortBacktests()
    )
    assert outcome.exit_code == 2
    assert "belongs to port 'dual_ema', not 'other_port'" in outcome.message
