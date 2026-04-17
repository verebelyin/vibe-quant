"""Tests for the regime-cross discovery campaign orchestrator."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TCH003
from unittest.mock import patch

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.campaign import (
    RUN_MODE_DISCOVERY,
    RUN_MODE_OOS,
    OOSWindow,
    RegimeCrossConfig,
    TrainWindow,
    build_matrix_report,
    extract_top_chromosomes,
    plan_campaign,
    plan_oos_validations,
    run_campaign,
)


@pytest.fixture()
def state(tmp_path: Path) -> StateManager:
    sm = StateManager(db_path=tmp_path / "campaign_test.db")
    # Force schema init
    _ = sm.conn
    yield sm
    sm.close()


@pytest.fixture()
def minimal_config() -> RegimeCrossConfig:
    return RegimeCrossConfig(
        name="test_campaign",
        symbol="BTCUSDT",
        timeframe="1h",
        train_windows=[
            TrainWindow(label="bear_a", start="2025-01-01", end="2025-06-01", direction="short"),
            TrainWindow(label="bull_b", start="2025-06-01", end="2025-12-01", direction="long"),
        ],
        oos_windows=[
            OOSWindow(label="oos_x", start="2024-01-01", end="2024-06-01"),
        ],
        population_size=20,
        max_generations=5,
        top_k=2,
    )


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_from_yaml_parses_minimal_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        """
name: t
symbol: BTCUSDT
timeframe: 1h
train_windows:
  - {label: w1, start: 2025-01-01, end: 2025-06-01, direction: short}
oos_windows:
  - {label: o1, start: 2024-01-01, end: 2024-06-01}
"""
    )
    cfg = RegimeCrossConfig.from_yaml(cfg_path)
    assert cfg.name == "t"
    assert len(cfg.train_windows) == 1
    assert cfg.train_windows[0].direction == "short"
    assert cfg.oos_windows[0].label == "o1"
    # Defaults when omitted
    assert cfg.population_size == 100
    assert cfg.top_k == 3


def test_from_yaml_rejects_missing_fields(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("name: t\n")
    with pytest.raises(ValueError, match="missing required fields"):
        RegimeCrossConfig.from_yaml(cfg_path)


def test_from_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    cfg_path = tmp_path / "list.yaml"
    cfg_path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        RegimeCrossConfig.from_yaml(cfg_path)


# ---------------------------------------------------------------------------
# plan_campaign
# ---------------------------------------------------------------------------


def test_plan_campaign_creates_one_run_per_train_window(
    state: StateManager, minimal_config: RegimeCrossConfig
) -> None:
    plan = plan_campaign(minimal_config, state)
    assert len(plan.discovery_runs) == 2
    for run_id in plan.discovery_runs.values():
        row = state.conn.execute(
            "SELECT run_mode, parameters FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row[0] == RUN_MODE_DISCOVERY
        params = json.loads(row[1])
        assert params["campaign_id"] == plan.campaign_id
        assert "window_label" in params
        assert "direction" in params


def test_plan_campaign_tags_campaign_id_consistently(
    state: StateManager, minimal_config: RegimeCrossConfig
) -> None:
    plan = plan_campaign(minimal_config, state)
    for run_id in plan.discovery_runs.values():
        params = json.loads(
            state.conn.execute(
                "SELECT parameters FROM backtest_runs WHERE id = ?", (run_id,)
            ).fetchone()[0]
        )
        assert params["campaign_id"] == plan.campaign_id


# ---------------------------------------------------------------------------
# extract_top_chromosomes
# ---------------------------------------------------------------------------


def _seed_discovery_result(
    state: StateManager, run_id: int, top_strategies: list[dict[str, object]]
) -> None:
    notes = {"type": "discovery", "top_strategies": top_strategies}
    state.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)",
        (run_id, json.dumps(notes)),
    )
    state.conn.commit()


def test_extract_top_chromosomes_returns_top_k(state: StateManager) -> None:
    # Create a dummy backtest_runs row we can FK to
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode=RUN_MODE_DISCOVERY,
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-01-01",
        end_date="2025-06-01",
        parameters={},
    )
    _seed_discovery_result(
        state, run_id,
        [
            {"dsl": {"name": "s0"}, "score": 0.9},
            {"dsl": {"name": "s1"}, "score": 0.8},
            {"dsl": {"name": "s2"}, "score": 0.7},
            {"dsl": {"name": "s3"}, "score": 0.6},
        ],
    )
    tops = extract_top_chromosomes(run_id, state, top_k=2)
    assert len(tops) == 2
    assert tops[0]["dsl"]["name"] == "s0"


def test_extract_top_chromosomes_missing_notes_returns_empty(state: StateManager) -> None:
    run_id = state.create_backtest_run(
        strategy_id=None, run_mode=RUN_MODE_DISCOVERY, symbols=["BTCUSDT"],
        timeframe="1h", start_date="x", end_date="y", parameters={},
    )
    assert extract_top_chromosomes(run_id, state) == []


def test_extract_top_chromosomes_malformed_notes_returns_empty(state: StateManager) -> None:
    run_id = state.create_backtest_run(
        strategy_id=None, run_mode=RUN_MODE_DISCOVERY, symbols=["BTCUSDT"],
        timeframe="1h", start_date="x", end_date="y", parameters={},
    )
    state.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)",
        (run_id, "not json {"),
    )
    state.conn.commit()
    assert extract_top_chromosomes(run_id, state) == []


# ---------------------------------------------------------------------------
# plan_oos_validations
# ---------------------------------------------------------------------------


def _valid_dsl(name: str) -> dict[str, object]:
    """Minimal DSL that survives StrategyDSL validation + create_strategy."""
    return {
        "name": name,
        "timeframe": "1h",
        "indicators": {"rsi_1": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi_1 < 30"]},
        "exit_conditions": {"long": ["rsi_1 > 70"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


def test_plan_oos_validations_creates_row_per_champion_per_window(
    state: StateManager, minimal_config: RegimeCrossConfig
) -> None:
    plan = plan_campaign(minimal_config, state)
    # Seed top-2 champions for each discovery run
    for label, run_id in plan.discovery_runs.items():
        _seed_discovery_result(
            state, run_id,
            [
                {"dsl": _valid_dsl(f"{label}_top0"), "score": 0.9},
                {"dsl": _valid_dsl(f"{label}_top1"), "score": 0.8},
            ],
        )

    plan_oos_validations(plan, state)

    # 2 train windows × 2 top × 1 OOS window = 4 OOS runs
    assert len(plan.oos_runs) == 4
    for key, run_id in plan.oos_runs.items():
        row = state.conn.execute(
            "SELECT run_mode, strategy_id, parameters FROM backtest_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert row[0] == RUN_MODE_OOS
        assert row[1] is not None  # strategy_id filled in
        params = json.loads(row[2])
        assert params["campaign_id"] == plan.campaign_id
        assert params["source_window"] == key[0]
        assert params["champion_idx"] == key[1]
        assert params["oos_window"] == key[2]


def test_plan_oos_validations_idempotent(
    state: StateManager, minimal_config: RegimeCrossConfig
) -> None:
    """Re-calling plan_oos_validations must not duplicate rows."""
    plan = plan_campaign(minimal_config, state)
    for run_id in plan.discovery_runs.values():
        _seed_discovery_result(
            state, run_id, [{"dsl": _valid_dsl(f"s_{run_id}"), "score": 0.9}]
        )

    plan_oos_validations(plan, state)
    first_count = len(plan.oos_runs)
    plan_oos_validations(plan, state)
    assert len(plan.oos_runs) == first_count

    total_oos_rows = state.conn.execute(
        "SELECT COUNT(*) FROM backtest_runs WHERE run_mode = ?", (RUN_MODE_OOS,)
    ).fetchone()[0]
    assert total_oos_rows == first_count


# ---------------------------------------------------------------------------
# run_campaign — subprocess calls mocked
# ---------------------------------------------------------------------------


def test_run_campaign_skips_completed_runs(
    state: StateManager, minimal_config: RegimeCrossConfig
) -> None:
    plan = plan_campaign(minimal_config, state)
    # Pre-mark both discoveries completed + seed champions so OOS gets planned
    for label, run_id in plan.discovery_runs.items():
        state.update_backtest_run_status(run_id, "completed")
        _seed_discovery_result(
            state, run_id, [{"dsl": _valid_dsl(f"{label}_s"), "score": 0.9}]
        )

    with patch("vibe_quant.discovery.campaign.subprocess.run") as m_run:
        m_run.return_value.returncode = 0
        run_campaign(plan, state, skip_existing=True)
        # Discoveries all skipped → only OOS subprocess calls (2 × 1)
        assert m_run.call_count == 2
        for call in m_run.call_args_list:
            cmd = call.args[0]
            assert "vibe_quant.validation" in cmd


def test_run_campaign_runs_discoveries_when_not_completed(
    state: StateManager, minimal_config: RegimeCrossConfig
) -> None:
    plan = plan_campaign(minimal_config, state)

    def _fake_run(cmd, **_kwargs):
        # First time through we're running discoveries. Mark run completed
        # and seed results so the OOS phase can plan + run.
        run_id = int(cmd[cmd.index("--run-id") + 1])
        if "vibe_quant.discovery" in cmd:
            state.update_backtest_run_status(run_id, "completed")
            _seed_discovery_result(
                state, run_id,
                [{"dsl": _valid_dsl(f"s_{run_id}"), "score": 0.9}],
            )
        from unittest.mock import MagicMock
        res = MagicMock()
        res.returncode = 0
        return res

    with patch("vibe_quant.discovery.campaign.subprocess.run", side_effect=_fake_run) as m_run:
        run_campaign(plan, state, skip_existing=True)
        # 2 discoveries + 2 OOS validations
        assert m_run.call_count == 4


# ---------------------------------------------------------------------------
# build_matrix_report
# ---------------------------------------------------------------------------


def test_build_matrix_report_collects_oos_cells(
    state: StateManager, minimal_config: RegimeCrossConfig
) -> None:
    plan = plan_campaign(minimal_config, state)
    for run_id in plan.discovery_runs.values():
        _seed_discovery_result(
            state, run_id, [{"dsl": _valid_dsl(f"s_{run_id}"), "score": 0.9}]
        )
    plan_oos_validations(plan, state)

    # Seed results for each OOS row
    for i, run_id in enumerate(plan.oos_runs.values()):
        state.conn.execute(
            "INSERT INTO backtest_results (run_id, sharpe_ratio, total_return, "
            "max_drawdown, total_trades) VALUES (?, ?, ?, ?, ?)",
            (run_id, 1.5 + i * 0.1, 0.05 + i * 0.01, 0.03, 50 + i),
        )
    state.conn.commit()

    report = build_matrix_report(plan.campaign_id, state)
    assert len(report.cells) == len(plan.oos_runs)
    assert report.campaign_name == minimal_config.name
    for c in report.cells:
        assert c.sharpe is not None
        assert c.total_trades is not None

    text = report.as_text()
    assert "Campaign test_campaign" in text
    assert "bear_a" in text or "bull_b" in text


def test_build_matrix_report_empty_campaign() -> None:
    from vibe_quant.discovery.campaign import MatrixReport

    report = MatrixReport(campaign_id="deadbeef", campaign_name="none")
    text = report.as_text()
    assert "no results yet" in text


def test_build_matrix_report_isolates_campaigns(
    state: StateManager, minimal_config: RegimeCrossConfig
) -> None:
    """A second campaign's cells must not bleed into the first's report."""
    plan_a = plan_campaign(minimal_config, state)
    plan_b = plan_campaign(minimal_config, state)
    for plan in (plan_a, plan_b):
        for run_id in plan.discovery_runs.values():
            _seed_discovery_result(
                state, run_id, [{"dsl": _valid_dsl(f"s_{run_id}"), "score": 0.9}]
            )
        plan_oos_validations(plan, state)

    report_a = build_matrix_report(plan_a.campaign_id, state)
    report_b = build_matrix_report(plan_b.campaign_id, state)
    # Each campaign has 2 train × 1 top × 1 OOS = 2 cells
    assert len(report_a.cells) == 2
    assert len(report_b.cells) == 2
    a_ids = {c.oos_run_id for c in report_a.cells}
    b_ids = {c.oos_run_id for c in report_b.cells}
    assert a_ids.isdisjoint(b_ids)
