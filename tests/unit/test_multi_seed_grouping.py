"""Multi-seed grouping: behaviourally identical genomes share one group
(bd vibe-quant-mxg8g).

``_run_multi_seed`` places a candidate into an existing group when it is
structurally close to the group representative (Gower distance below
``min_diversity_distance``) OR when its behavioural fingerprint equals the
representative's (non-None). Two seeds returning structurally distant genomes
with identical trade behaviour therefore occupy ONE group / one top-K slot,
instead of wasting a slot on a behaviourally redundant strategy.

These tests drive the REAL ``_run_multi_seed`` (only ``DiscoveryPipeline.run``
is faked to return a chosen per-seed champion) and read the group count from the
"Merged: ... (N groups)" summary it logs.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from tests.unit.test_dsr_cumulative_n import _chrom, _fit, _mock_cfg, _series
from vibe_quant.discovery import pooling
from vibe_quant.discovery.distance import chromosome_distance
from vibe_quant.discovery.operators import ConditionType, Direction, StrategyGene
from vibe_quant.discovery.pipeline import (
    DiscoveryPipeline,
    DiscoveryResult,
    _behaviour_fingerprint,
)

if TYPE_CHECKING:
    from pytest import LogCaptureFixture, MonkeyPatch

_MIN_DIST = _mock_cfg().min_diversity_distance
_MERGED = re.compile(r"\((\d+) groups\)")


def _distant_chrom() -> Any:
    """A genome structurally far from ``_chrom()`` (different indicator + direction)."""
    gene = StrategyGene("CCI", {"period": 14.0}, ConditionType.LT, 30.0, None)
    exit_gene = StrategyGene("CCI", {"period": 14.0}, ConditionType.GT, 70.0, None)
    return _chrom(entry_genes=[gene], exit_genes=[exit_gene], direction=Direction.SHORT)


def _groups_logged(caplog: LogCaptureFixture) -> int:
    for record in caplog.records:
        match = _MERGED.search(record.getMessage())
        if match:
            return int(match.group(1))
    raise AssertionError("no 'Merged: ... (N groups)' log line found")


def _run_grouping(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    per_seed: list[tuple[Any, Any]],
) -> tuple[Any, int]:
    """Fake each seed to return its own champion; return (result, group count)."""
    champs = list(per_seed)

    def fake_run(self: Any) -> DiscoveryResult:
        champ = champs.pop(0)
        return DiscoveryResult(
            generations=[], top_strategies=[champ], total_candidates_evaluated=400,
            converged=False, convergence_generation=None,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    caplog.set_level(logging.INFO, logger="vibe_quant.discovery.pooling")
    res = pooling._run_multi_seed(
        len(per_seed), _mock_cfg(prior_trials_fn=lambda s, e: 30, top_k=2),
        lambda c: {}, "/tmp/vq_nb_grouping.json",
    )  # fmt: skip
    return res, _groups_logged(caplog)


def test_distant_identical_behaviour_merges_into_one_group(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Distance alone would keep them apart; the fingerprint makes them one group."""
    a, b = _chrom(), _distant_chrom()
    assert chromosome_distance(a, b) >= _MIN_DIST  # structurally distant precondition
    fit_a = _fit(mtm_daily_returns=_series(5.0))
    fit_b = _fit(mtm_daily_returns=_series(5.0))
    assert _behaviour_fingerprint(fit_a) == _behaviour_fingerprint(fit_b)

    res, groups = _run_grouping(monkeypatch, caplog, [(a, fit_a), (b, fit_b)])
    assert groups == 1
    assert len(res.top_strategies) == 1  # one top-K slot, not two


def test_distant_different_behaviour_stays_two_groups(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Distant genomes with different metrics are still two groups (unchanged)."""
    a, b = _chrom(), _distant_chrom()
    assert chromosome_distance(a, b) >= _MIN_DIST
    fit_a = _fit(sharpe_ratio=2.0, mtm_daily_returns=_series(5.0))
    fit_b = _fit(sharpe_ratio=3.0, mtm_daily_returns=_series(5.0))
    assert _behaviour_fingerprint(fit_a) != _behaviour_fingerprint(fit_b)

    res, groups = _run_grouping(monkeypatch, caplog, [(a, fit_a), (b, fit_b)])
    assert groups == 2
    assert len(res.top_strategies) == 2


def test_nan_metrics_never_merge(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A NaN field makes the fingerprint comparison False, so distant NaNs stay apart."""
    a, b = _chrom(), _distant_chrom()
    fit_a = _fit(profit_factor=float("nan"), mtm_daily_returns=_series(5.0))
    fit_b = _fit(profit_factor=float("nan"), mtm_daily_returns=_series(5.0))
    assert _behaviour_fingerprint(fit_a) != _behaviour_fingerprint(fit_b)  # NaN != NaN

    _, groups = _run_grouping(monkeypatch, caplog, [(a, fit_a), (b, fit_b)])
    assert groups == 2


def test_close_genomes_merge_regardless_of_behaviour(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """The structural rule alone still groups: close genomes with different metrics."""
    a, b = _chrom(), _chrom()
    assert chromosome_distance(a, b) < _MIN_DIST  # structurally close precondition
    fit_a = _fit(sharpe_ratio=2.0, mtm_daily_returns=_series(5.0))
    fit_b = _fit(sharpe_ratio=3.0, mtm_daily_returns=_series(5.0))
    assert _behaviour_fingerprint(fit_a) != _behaviour_fingerprint(fit_b)

    res, groups = _run_grouping(monkeypatch, caplog, [(a, fit_a), (b, fit_b)])
    assert groups == 1
    assert len(res.top_strategies) == 1
