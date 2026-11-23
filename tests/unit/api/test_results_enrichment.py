"""Unit tests for results router notes-JSON enrichment helper."""

from __future__ import annotations

import json

from vibe_quant.api.routers.results import _enrich_result_with_notes
from vibe_quant.api.schemas.result import BacktestResultResponse


def _base_row() -> dict[str, object]:
    return {
        "id": 1,
        "run_id": 99,
        "notes": None,
        "cross_window_results": None,
        "wfa_sharpe_consistency": None,
        "bootstrap_sharpe_lower": None,
        "bootstrap_sharpe_upper": None,
        "bootstrap_ci_level": None,
        "cross_regime_results": None,
        "random_short_baseline_pct": None,
    }


def test_enrich_missing_notes_returns_row_unchanged() -> None:
    row = _base_row()
    enriched = _enrich_result_with_notes(row)
    assert enriched["cross_window_results"] is None
    assert enriched["wfa_sharpe_consistency"] is None
    assert enriched["bootstrap_sharpe_lower"] is None


def test_enrich_bad_json_returns_row_unchanged() -> None:
    row = _base_row()
    row["notes"] = "not valid json"
    enriched = _enrich_result_with_notes(row)
    assert enriched["cross_window_results"] is None


def test_enrich_populates_cross_window_from_top_strategy() -> None:
    row = _base_row()
    row["notes"] = json.dumps(
        {
            "type": "discovery",
            "cross_window_months": [1, 2],
            "cross_window_min_sharpe": 0.5,
            "top_strategies": [
                {
                    "cross_window": {
                        "passed": True,
                        "windows": [
                            {
                                "sharpe": 1.2,
                                "return_pct": 0.04,
                                "max_dd": 0.02,
                                "trades": 50,
                            },
                            {
                                "sharpe": 0.3,
                                "return_pct": 0.01,
                                "max_dd": 0.03,
                                "trades": 45,
                            },
                        ],
                    },
                    "wfa": {"sharpe_consistency": 0.82},
                }
            ],
        }
    )
    enriched = _enrich_result_with_notes(row)
    assert enriched["wfa_sharpe_consistency"] == 0.82
    cw = enriched["cross_window_results"]
    assert isinstance(cw, list) and len(cw) == 2
    # windows[0] is the original in-sample window; windows[1] is shifted +1mo
    assert cw[0]["offset"] == 0
    assert cw[0]["in_sample"] is True
    assert cw[0]["sharpe"] == 1.2
    assert cw[0]["passed"] is True
    assert cw[1]["offset"] == 1
    assert cw[1]["in_sample"] is False
    assert cw[1]["passed"] is False  # 0.3 < 0.5


def test_enrich_populates_bootstrap_ci_when_present() -> None:
    row = _base_row()
    row["notes"] = json.dumps(
        {
            "bootstrap_ci": {"lower": 0.7, "upper": 2.1, "level": 0.95},
            "top_strategies": [{}],
        }
    )
    enriched = _enrich_result_with_notes(row)
    assert enriched["bootstrap_sharpe_lower"] == 0.7
    assert enriched["bootstrap_sharpe_upper"] == 2.1
    assert enriched["bootstrap_ci_level"] == 0.95


def _response_row(**over: object) -> dict[str, object]:
    """A row with every required ``BacktestResultResponse`` field (all null)."""
    return {
        "id": 1,
        "run_id": 99,
        "total_return": None,
        "cagr": None,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "calmar_ratio": None,
        "max_drawdown": None,
        "max_drawdown_duration_days": None,
        "volatility_annual": None,
        "total_trades": None,
        "winning_trades": None,
        "losing_trades": None,
        "win_rate": None,
        "profit_factor": None,
        "avg_win": None,
        "avg_loss": None,
        "largest_win": None,
        "largest_loss": None,
        "avg_trade_duration_hours": None,
        "max_consecutive_wins": None,
        "max_consecutive_losses": None,
        "total_fees": None,
        "total_funding": None,
        "total_slippage": None,
        "deflated_sharpe": None,
        "walk_forward_efficiency": None,
        "purged_kfold_mean_sharpe": None,
        "execution_time_seconds": None,
        "starting_balance": None,
        "notes": None,
        "created_at": None,
        **over,
    }


def _windows(min_sharpe_key: dict[str, object]) -> str:
    return json.dumps(
        {
            "type": "discovery",
            "cross_window_months": [1],
            **min_sharpe_key,
            "top_strategies": [
                {
                    "cross_window": {
                        "passed": True,
                        "windows": [
                            {"sharpe": 1.2, "return_pct": 0.04},
                            {"sharpe": 0.6, "return_pct": 0.01},
                        ],
                    }
                }
            ],
        }
    )


def test_enrich_exposes_the_cross_window_limit_the_gate_used() -> None:
    """R8: an old run's stored limit is surfaced raw, and its display twin is × √(365/252).

    The enrichment forwards the SAME float ``passed`` compares against (stored √252);
    the response's computed twin is the only 365-day number.
    """
    row = _response_row(notes=_windows({"cross_window_min_sharpe": 0.5}))
    enriched = _enrich_result_with_notes(row)
    assert enriched["cross_window_min_sharpe"] == 0.5

    body = BacktestResultResponse(**enriched)
    assert body.cross_window_min_sharpe == 0.5
    assert body.cross_window_min_sharpe_display == 0.6017500931476244


def test_enrich_cross_window_limit_defaults_to_the_gate_default() -> None:
    """R8: notes without the key use 0.5, the same default ``passed`` reads."""
    row = _response_row(notes=_windows({}))
    enriched = _enrich_result_with_notes(row)
    assert enriched["cross_window_min_sharpe"] == 0.5
    assert BacktestResultResponse(**enriched).cross_window_min_sharpe_display == 0.6017500931476244


def test_enrich_cross_window_limit_is_absent_without_windows() -> None:
    """R8: no cross-window windows -> neither the raw limit nor its twin."""
    row = _response_row(notes=json.dumps({"type": "discovery", "top_strategies": [{}]}))
    enriched = _enrich_result_with_notes(row)
    assert enriched.get("cross_window_min_sharpe") is None
    assert BacktestResultResponse(**enriched).cross_window_min_sharpe_display is None
