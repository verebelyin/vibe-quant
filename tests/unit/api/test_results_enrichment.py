"""Unit tests for results router notes-JSON enrichment helper."""

from __future__ import annotations

import json

from vibe_quant.api.routers.results import _enrich_result_with_notes


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
    assert cw[0]["offset"] == 1
    assert cw[0]["sharpe"] == 1.2
    assert cw[0]["passed"] is True
    assert cw[1]["offset"] == 2
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
