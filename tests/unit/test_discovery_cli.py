"""Tests for discovery CLI entrypoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    assert args.bootstrap_min_sharpe == 1.0
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
