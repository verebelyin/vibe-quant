"""Multi-seed discovery notes -> overfitting MTM DSR recheck (bd vibe-quant-0z4ud).

The pooled multi-seed champion entry (built by the REAL producer, not a
hand-written record) must persist a DSR record whose ``input`` is the pooled
mark-to-market series, and ``OverfittingPipeline._mtm_columns_from_notes`` must
read that record back as exactly the Sharpe/moments/T the recheck then tests --
so dropping or renaming the record in the producer turns this test red.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_dsr_cumulative_n import _chrom, _fit, _mock_cfg, _series
from vibe_quant.discovery import __main__ as cli
from vibe_quant.discovery import record
from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult
from vibe_quant.overfitting.pipeline import OverfittingPipeline

if TYPE_CHECKING:
    from pathlib import Path

MTM_INPUT = "pooled_mtm_daily_returns"
MTM_COLUMNS = ("mtm_sharpe_daily", "mtm_skewness", "mtm_kurtosis", "mtm_days")


def _pooled_multi_seed_result(monkeypatch: pytest.MonkeyPatch, mtm: Any) -> Any:
    """Real ``_run_multi_seed`` where every seed returns one champion carrying ``mtm``.

    ``DiscoveryPipeline.run`` is faked (the real GA is irrelevant here); the pooled
    champion re-check -- and so the DSR record -- is the real production code.
    """
    champ = (_chrom(), _fit(mtm_daily_returns=mtm))

    def fake_run(self: Any) -> DiscoveryResult:
        return DiscoveryResult(
            generations=[], top_strategies=[champ], total_candidates_evaluated=400,
            converged=False, convergence_generation=None,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    return cli._run_multi_seed(
        3, _mock_cfg(prior_trials_fn=lambda s, e: 30), lambda c: {}, "/tmp/vq_eb_notes.json"
    )


def _champion_entry_notes(result: Any) -> dict[str, Any]:
    """The notes payload main() persists: run notes + the champion entry.

    The champion entry is built by ``_champion_entry`` with the pooled result's
    own ``dsr_records`` -- the exact production line, so a dropped/renamed record
    propagates into these notes.
    """
    chrom, fitness = result.top_strategies[0]
    notes = record.provenance_notes(
        result,
        use_mock=False,
        eval_windows_count=1,
        eval_windows=None,
        symbol_agg="portfolio",
        split_ratio=0.0,
        direction=None,
        cross_window_months=[],
        cross_window_min_sharpe=0.0,
        wfa_oos_step_days=0,
        wfa_min_consistency=0.0,
        num_seeds=3,
        seed=42,
        bootstrap_min_sharpe=0.0,
        holdout_min_sharpe=0.0,
        symbols=["BTCUSDT"],
    )
    notes["top_strategies"] = [
        cli._champion_entry(
            chrom, fitness, timeframe="4h",
            dsr=result.dsr_records.get(chrom.uid),
            buy_hold=result.buy_hold_records.get(chrom.uid),
        )
    ]
    return notes


def test_multi_seed_champion_notes_round_trip_to_mtm_recheck_columns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pooled champion's persisted record IS the recheck's MTM input: the
    ``input`` marker survives and all four mtm_* columns come back non-None."""
    mtm = _series(5.0)
    result = _pooled_multi_seed_result(monkeypatch, mtm)
    notes_json = json.dumps(_champion_entry_notes(result))

    record = json.loads(notes_json)["top_strategies"][0]["dsr"]
    assert record["input"] == MTM_INPUT
    assert record["observations"] == len(mtm)  # T = the pooled MTM series length

    cols = OverfittingPipeline(tmp_path / "recheck.db")._mtm_columns_from_notes(notes_json)
    assert all(cols[c] is not None for c in MTM_COLUMNS)  # read, not failed closed
    assert cols["mtm_sharpe_daily"] == pytest.approx(record["sharpe_daily"], rel=1e-12)
    assert cols["mtm_skewness"] == pytest.approx(record["skewness"], rel=1e-12)
    assert cols["mtm_kurtosis"] == pytest.approx(record["kurtosis"], rel=1e-12)
    assert cols["mtm_days"] == int(record["observations"])
