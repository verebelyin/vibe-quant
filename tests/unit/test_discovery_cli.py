"""Tests for discovery CLI entrypoint."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.db.state_manager import StateManager
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


def test_build_parser_requires_run_id() -> None:
    """CLI parser should require run id."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-id",
            "42",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-02-01",
        ]
    )
    assert args.run_id == 42


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

    # Force the "no viable strategies" path: patch Pipeline.run to return
    # a result with empty top_strategies (what happens when every top-K
    # candidate fails hard guardrails).
    def fake_run(self) -> DiscoveryResult:
        return DiscoveryResult(
            generations=[],
            top_strategies=[],
            total_candidates_evaluated=42,
            converged=False,
            convergence_generation=None,
        )

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)

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
            "--db",
            str(db_path),
            "--mock", "--max-workers", "-1",
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
    assert notes["outcome"] == "no_viable_strategies"
    assert notes["evaluated"] == 42
    assert notes["top_strategies"] == []
    assert "reason" in notes


def test_1m_default_bootstrap_min_sharpe_is_0_5(tmp_path: Path, monkeypatch, caplog) -> None:
    """1m timeframe should default bootstrap_min_sharpe to 0.5, logged at INFO."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--run-id", str(run_id),
            "--population-size", "6", "--max-generations", "2",
            "--mutation-rate", "0.1", "--elite-count", "1",
            "--symbols", "BTCUSDT", "--timeframe", "1m",
            "--start-date", "2025-01-01", "--end-date", "2025-02-01",
            "--db", str(db_path), "--mock", "--max-workers", "-1",
        ],
    )
    with caplog.at_level("INFO", logger="vibe_quant.discovery.__main__"):
        assert main() == 0

    assert any(
        "Bootstrap min t-stat default: 0.5" in r.message and "timeframe=1m" in r.message
        for r in caplog.records
    ), f"expected 1m default=0.5, got: {[r.message for r in caplog.records]}"


def test_non_1m_default_bootstrap_min_sharpe_is_1_0(tmp_path: Path, monkeypatch, caplog) -> None:
    """Non-1m timeframe should default bootstrap_min_sharpe to 1.0."""
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
    with caplog.at_level("INFO", logger="vibe_quant.discovery.__main__"):
        assert main() == 0

    assert any(
        "Bootstrap min t-stat default: 1.0" in r.message and "timeframe=1h" in r.message
        for r in caplog.records
    )


def test_4h_default_bootstrap_min_sharpe_is_0_0(tmp_path: Path, monkeypatch, caplog) -> None:
    """4h defaults to 0.0 — low trade counts make the CI lower bound
    structurally < 1.0 for any strategy (vibe-quant-gds1c)."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--run-id", str(run_id),
            "--population-size", "6", "--max-generations", "2",
            "--mutation-rate", "0.1", "--elite-count", "1",
            "--symbols", "BTCUSDT", "--timeframe", "4h",
            "--start-date", "2025-01-01", "--end-date", "2025-02-01",
            "--db", str(db_path), "--mock", "--max-workers", "-1",
        ],
    )
    with caplog.at_level("INFO", logger="vibe_quant.discovery.__main__"):
        assert main() == 0

    assert any(
        "Bootstrap min t-stat default: 0.0" in r.message and "timeframe=4h" in r.message
        for r in caplog.records
    )


def test_default_bootstrap_floor_by_timeframe() -> None:
    from vibe_quant.discovery.run_spec import _default_bootstrap_min_sharpe

    assert _default_bootstrap_min_sharpe("1m") == 0.5
    assert _default_bootstrap_min_sharpe("4h") == 0.0
    assert _default_bootstrap_min_sharpe("1d") == 0.0
    assert _default_bootstrap_min_sharpe("1h") == 1.0
    assert _default_bootstrap_min_sharpe("15m") == 1.0


def test_explicit_bootstrap_min_sharpe_overrides_default(tmp_path: Path, monkeypatch, caplog) -> None:
    """Explicit --bootstrap-min-sharpe should override the timeframe-aware default."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--run-id", str(run_id),
            "--population-size", "6", "--max-generations", "2",
            "--mutation-rate", "0.1", "--elite-count", "1",
            "--symbols", "BTCUSDT", "--timeframe", "1m",
            "--start-date", "2025-01-01", "--end-date", "2025-02-01",
            "--bootstrap-min-sharpe", "1.5",
            "--db", str(db_path), "--mock", "--max-workers", "-1",
        ],
    )
    with caplog.at_level("INFO", logger="vibe_quant.discovery.__main__"):
        assert main() == 0

    # Should NOT log the default-resolution line when user provided value.
    assert not any(
        "Bootstrap min t-stat default:" in r.message
        for r in caplog.records
    )


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
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--run-id", str(run_id),
            "--population-size", "6", "--max-generations", "2", "--elite-count", "1",
            "--symbols", "BTCUSDT", "--timeframe", "1h",
            "--start-date", "2025-01-01", "--end-date", "2025-02-01",
            "--eval-windows", "1",
            "--db", str(db_path), "--mock", "--max-workers", "-1",
        ],
    )

    assert main() == 0

    state = StateManager(db_path)
    result = state.get_backtest_result(run_id)
    state.close()
    assert result is not None
    # (0.02 / 7 days) / (0.10 / 24 days) = 0.48 / 0.70
    assert result["walk_forward_efficiency"] == pytest.approx(0.48 / 0.70, abs=1e-12)


def test_walk_forward_efficiency_formula() -> None:
    """Stationary edge -> 1.0 whatever the period lengths; IS<=0 -> undefined."""
    from vibe_quant.discovery.__main__ import walk_forward_efficiency

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


def test_guardrail_flags_propagate_to_pipeline_config(tmp_path: Path, monkeypatch) -> None:
    """CLI guardrail flags should reach DiscoveryPipeline via DiscoveryConfig."""
    from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline, DiscoveryResult

    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    captured: dict[str, DiscoveryConfig] = {}

    original_init = DiscoveryPipeline.__init__

    def capturing_init(self, config: DiscoveryConfig, **kwargs) -> None:  # type: ignore[no-untyped-def]
        captured["config"] = config
        original_init(self, config, **kwargs)

    def fake_run(self) -> DiscoveryResult:
        return DiscoveryResult(
            generations=[],
            top_strategies=[],
            total_candidates_evaluated=0,
            converged=False,
            convergence_generation=None,
        )

    monkeypatch.setattr(DiscoveryPipeline, "__init__", capturing_init)
    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-id",
            str(run_id),
            "--population-size",
            "4",
            "--max-generations",
            "2",
            "--elite-count",
            "1",
            "--timeframe",
            "1h",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-02-01",
            "--no-bootstrap-ci",
            "--no-dsr",
            "--bootstrap-min-sharpe",
            "0.3",
            "--bootstrap-ci-level",
            "0.8",
            "--db",
            str(db_path),
            "--mock", "--max-workers", "-1",
        ],
    )

    assert main() == 0

    cfg = captured["config"]
    assert cfg.require_bootstrap_ci is False
    assert cfg.require_dsr is False
    assert cfg.bootstrap_min_sharpe == 0.3
    assert cfg.bootstrap_ci_level == 0.8


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


def test_seed_autogenerated_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    monkeypatch.setattr("sys.argv", _seed_argv(db_path, run_id))
    assert main() == 0
    seed = _run_notes(db_path, run_id)["seed"]
    assert isinstance(seed, int)


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
    monkeypatch.setattr("vibe_quant.discovery.__main__._mock_backtest", _always_raises)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging
    import random

    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    seen: list[int] = []
    real_seed = random.seed
    monkeypatch.setattr(random, "seed", _recording_seed(seen, real_seed))
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda self, *a, **k: 1234)
    monkeypatch.setattr("sys.argv", _seed_argv(db_path, run_id, "--num-seeds", "2"))
    caplog.set_level(logging.INFO, logger="vibe_quant.discovery.__main__")
    assert main() == 0
    assert seen[:2] == [1234, 9153]
    assert _run_notes(db_path, run_id)["seed"] == 1234
    assert "Discovery seed: 1234" in caplog.text


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


def test_multi_seed_seed_differs_across_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hostile probe: two launches without --seed must not repeat the same base."""
    db_a = tmp_path / "a.db"
    run_a = _create_discovery_run(db_a)
    monkeypatch.setattr("sys.argv", _seed_argv(db_a, run_a, "--num-seeds", "2"))
    assert main() == 0
    db_b = tmp_path / "b.db"
    run_b = _create_discovery_run(db_b)
    monkeypatch.setattr("sys.argv", _seed_argv(db_b, run_b, "--num-seeds", "2"))
    assert main() == 0
    assert _run_notes(db_a, run_a)["seed"] != _run_notes(db_b, run_b)["seed"]


@pytest.mark.parametrize(("flag", "expected"), [("-1", None), ("0", 0), ("3", 3)])
def test_max_workers_negative_means_sequential(flag: str, expected: int | None) -> None:
    """``--max-workers -1`` resolves to None (in-process); the speed-up of this file relies on it."""
    spec = DiscoveryRunSpec.from_argv(["--max-workers", flag])
    assert spec.worker_count() == expected


def test_seed_help_has_no_fixed_default() -> None:
    import re

    help_text = build_parser().format_help()
    seed_help = help_text.split("\n  --seed ", 1)[1].split("\n  --", 1)[0]
    assert re.search(r"default\s+base", seed_help) is None
    assert "42" not in seed_help


def test_multi_seed_explicit_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import random

    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    seen: list[int] = []
    real_seed = random.seed
    monkeypatch.setattr(random, "seed", _recording_seed(seen, real_seed))
    monkeypatch.setattr(
        "sys.argv", _seed_argv(db_path, run_id, "--num-seeds", "2", "--seed", "100")
    )
    assert main() == 0
    assert seen[:2] == [100, 8019]
    assert _run_notes(db_path, run_id)["seed"] == 100


def _legacy_windows(
    start: str, end: str, split: float, n_eval: int
) -> tuple[str, str, str | None, str | None, list[tuple[str, str]] | None]:
    """The inline split main() used before plan_from_args (reference)."""
    from vibe_quant.utils import split_date_range, split_into_windows

    train_start, train_end = start, end
    holdout_start: str | None = None
    holdout_end: str | None = None
    if split > 0:
        train_start, train_end, holdout_start, holdout_end = split_date_range(start, end, split)
    n = max(1, n_eval)
    eval_windows = split_into_windows(train_start, train_end, n) if n >= 2 else None
    return train_start, train_end, holdout_start, holdout_end, eval_windows


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-01-01", "2026-02-24"),
        ("2025-01-01", "2025-06-01"),
        ("2024-02-03", "2025-01-30"),  # odd day count, leap year
        ("2023-05-17", "2025-11-18"),
    ],
)
@pytest.mark.parametrize("split", [0.8, 0.25, 0.75, 0])
@pytest.mark.parametrize("n_eval", [1, 2, 3, 4])
def test_plan_windows_matches_legacy_split(start: str, end: str, split: float, n_eval: int) -> None:
    args = build_parser().parse_args(
        [
            "--run-id", "0", "--start-date", start, "--end-date", end,
            "--train-test-split", str(split), "--eval-windows", str(n_eval),
        ]
    )
    plan = DiscoveryRunSpec.from_args(args).plan()
    expected = _legacy_windows(start, end, split, n_eval)
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


def test_main_derives_windows_via_the_spec(tmp_path: Path, monkeypatch) -> None:
    """main() must plan through the spec (a bypass would be masked by the pipeline gate)."""

    class _Sentinel(Exception):
        pass

    def _boom(_self: object) -> None:
        raise _Sentinel("spec.plan reached")

    monkeypatch.setattr(DiscoveryRunSpec, "plan", _boom)
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--run-id", str(run_id), "--population-size", "6",
            "--max-generations", "2", "--elite-count", "1", "--timeframe", "1h",
            "--start-date", "2025-01-01", "--end-date", "2025-02-01",
            "--mock", "--max-workers", "-1", "--db", str(db_path),
        ],
    )
    assert main() == 1
    state = StateManager(db_path)
    run = state.get_backtest_run(run_id)
    state.close()
    assert run is not None
    assert run["status"] == "failed"
    assert "spec.plan reached" in run["error_message"]


def test_cross_window_months_equals_form_parses_negative() -> None:
    from vibe_quant.discovery.config_check import parse_cross_window_months

    args = build_parser().parse_args(["--run-id", "0", "--cross-window-months=-15,3"])
    assert parse_cross_window_months(args.cross_window_months) == [-15, 3]


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
    monkeypatch.setattr("vibe_quant.discovery.__main__._mock_backtest", _always_raises)
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

    from vibe_quant.discovery import __main__ as cli
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
    real_loader = cli.load_seed_chromosomes

    def spy_loader(state: StateManager, run_id: int) -> Any:  # noqa: F811
        chromosomes = real_loader(state, run_id)
        loaded.append(chromosomes)
        return chromosomes

    monkeypatch.setattr(cli, "load_seed_chromosomes", spy_loader)

    captured: dict[str, Any] = {}
    real_pipeline = cli.DiscoveryPipeline

    def spy_pipeline(*args: Any, **kwargs: Any) -> Any:
        captured["seed_chromosomes"] = kwargs.get("seed_chromosomes")
        return real_pipeline(*args, **kwargs)

    monkeypatch.setattr(cli, "DiscoveryPipeline", spy_pipeline)
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
    from vibe_quant.discovery import __main__ as cli

    db_path = tmp_path / "state.db"
    seed_run_id = _create_seed_run(db_path)  # no result row -> no champions
    run_id = _create_discovery_run(db_path)

    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(tmp_path / "logs"))  # keep progress files out of the repo

    def _no_pipeline(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no pipeline may be built for an unhonorable warm-start")

    monkeypatch.setattr(cli, "DiscoveryPipeline", _no_pipeline)
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
