"""Validation-vs-screening consistency flags (vibe-quant-o11tp)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.consistency import (
    ScreeningReference,
    assess_consistency,
    find_screening_reference,
)

if TYPE_CHECKING:
    from pathlib import Path


def _ref(sharpe: float, trades: int) -> ScreeningReference:
    return ScreeningReference(sharpe=sharpe, trades=trades, source="screening_run:1")


class TestAssessConsistency:
    def test_sign_flip_flags_collapse(self) -> None:
        """Batch 41 shape: screen 5.40 → val -2.78 is a collapse."""
        report = assess_consistency(_ref(5.40, 54), val_sharpe=-2.78, val_trades=65)
        assert report.is_flagged
        assert any("sign flip" in f for f in report.flags)

    def test_below_half_screening_flags_collapse(self) -> None:
        report = assess_consistency(_ref(3.0, 100), val_sharpe=1.0, val_trades=100)
        assert any("validation-collapse" in f for f in report.flags)

    def test_clean_champion_not_flagged(self) -> None:
        """Batch 40 shape: screen 3.70 → val 3.66 stays clean."""
        report = assess_consistency(_ref(3.70, 100), val_sharpe=3.66, val_trades=105)
        assert not report.is_flagged

    def test_trade_divergence_is_distinct_flag(self) -> None:
        """236 shape: 54 → 65 trades (+20%) raises the divergence warning."""
        report = assess_consistency(_ref(5.40, 54), val_sharpe=5.0, val_trades=65)
        assert len(report.flags) == 1
        assert "trade-count-divergence" in report.flags[0]

    def test_negative_screening_sharpe_never_collapse(self) -> None:
        """A bad screening champion getting worse is not a *collapse* signal."""
        report = assess_consistency(_ref(-0.5, 100), val_sharpe=-1.5, val_trades=100)
        assert not any("collapse" in f for f in report.flags)

    def test_to_dict_round_trip(self) -> None:
        report = assess_consistency(_ref(5.0, 50), val_sharpe=-1.0, val_trades=50)
        d = report.to_dict()
        assert d["screen_sharpe"] == 5.0
        assert d["val_sharpe"] == -1.0
        assert d["flags"] and json.dumps(d)


class TestFindScreeningReference:
    def test_prefers_standalone_screening_run(self, tmp_path: Path) -> None:
        state = StateManager(db_path=tmp_path / "t.db")
        sid = state.create_strategy("s1", {"name": "s1"})
        run_id = state.create_backtest_run(
            strategy_id=sid,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2025-01-01",
            parameters={},
        )
        state.update_backtest_run_status(run_id, "completed")
        state.conn.execute(
            "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades)"
            " VALUES (?, '{}', 2.5, 80)",
            (run_id,),
        )
        state.conn.commit()

        ref = find_screening_reference(state, sid, "s1")
        assert ref is not None
        assert ref.sharpe == 2.5
        assert ref.trades == 80
        assert ref.source == f"screening_run:{run_id}"
        state.close()

    def test_falls_back_to_discovery_notes(self, tmp_path: Path) -> None:
        state = StateManager(db_path=tmp_path / "t.db")
        sid = state.create_strategy("genome_abc123", {"name": "genome_abc123"})
        disc_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="discovery",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2025-01-01",
            parameters={},
        )
        state.update_backtest_run_status(disc_id, "completed")
        notes = json.dumps(
            {
                "top_strategies": [
                    {
                        "dsl": {"name": "genome_abc123"},
                        "sharpe": 1.8,
                        "trades": 64,
                    }
                ]
            }
        )
        state.save_backtest_result(disc_id, {"sharpe_ratio": 1.8, "notes": notes})

        ref = find_screening_reference(state, sid, "genome_abc123")
        assert ref is not None
        assert ref.sharpe == 1.8
        assert ref.trades == 64
        assert ref.source == f"discovery_run:{disc_id}"
        state.close()

    def test_no_reference_returns_none(self, tmp_path: Path) -> None:
        state = StateManager(db_path=tmp_path / "t.db")
        sid = state.create_strategy("lonely", {"name": "lonely"})
        assert find_screening_reference(state, sid, "lonely") is None
        state.close()
