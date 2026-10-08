"""Tests for discovery CLI entrypoint."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.__main__ import build_parser, main

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
            "2025-02-01",
            "--direction",
            "short",
            "--cross-window-months=-1",
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
    assert notes["cross_window_months"] == [-1]
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
            "--db", str(db_path), "--mock",
        ],
    )
    with caplog.at_level("INFO", logger="vibe_quant.discovery.__main__"):
        assert main() == 0

    assert any(
        "Bootstrap min Sharpe default: 0.5" in r.message and "timeframe=1m" in r.message
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
            "--db", str(db_path), "--mock",
        ],
    )
    with caplog.at_level("INFO", logger="vibe_quant.discovery.__main__"):
        assert main() == 0

    assert any(
        "Bootstrap min Sharpe default: 1.0" in r.message and "timeframe=1h" in r.message
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
            "--db", str(db_path), "--mock",
        ],
    )
    with caplog.at_level("INFO", logger="vibe_quant.discovery.__main__"):
        assert main() == 0

    assert any(
        "Bootstrap min Sharpe default: 0.0" in r.message and "timeframe=4h" in r.message
        for r in caplog.records
    )


def test_default_bootstrap_floor_by_timeframe() -> None:
    from vibe_quant.discovery.__main__ import _default_bootstrap_min_sharpe

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
            "--db", str(db_path), "--mock",
        ],
    )
    with caplog.at_level("INFO", logger="vibe_quant.discovery.__main__"):
        assert main() == 0

    # Should NOT log the default-resolution line when user provided value.
    assert not any(
        "Bootstrap min Sharpe default:" in r.message
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
            "--db", str(db_path), "--mock",
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
            "--db", str(db_path), "--mock",
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
            "--mock",
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
        "--db", str(db_path), "--mock", *extra,
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


def test_multi_seed_default_base_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import random

    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    seen: list[int] = []
    real_seed = random.seed
    monkeypatch.setattr(random, "seed", _recording_seed(seen, real_seed))
    monkeypatch.setattr("sys.argv", _seed_argv(db_path, run_id, "--num-seeds", "2"))
    assert main() == 0
    assert seen[:2] == [42, 7961]


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
