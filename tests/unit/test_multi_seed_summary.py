"""Multi-seed SUMMARY must not abort on a non-finite seed Sharpe (bd vibe-quant-k5pt1).

``_run_multi_seed``'s MULTI-SEED SUMMARY used to feed the raw per-seed best
Sharpes straight into ``statistics.mean/median/min/max/stdev``. A seed whose
champion carries ``sharpe_ratio=nan`` makes ``statistics.stdev([nan, ...])``
raise ``AttributeError: 'float' object has no attribute 'numerator'``, aborting
the whole run BEFORE grouping/notes/champion selection. The summary now computes
its stats over FINITE values only and reports the count of non-finite seeds, so
such runs complete and still log useful numbers (all-non-finite -> ``n/a``).

These tests drive the REAL ``_run_multi_seed`` (only ``DiscoveryPipeline.run``
is faked to return chosen per-seed champions) and read the summary from the log.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tests.unit.test_dsr_cumulative_n import _chrom, _fit, _mock_cfg, _series
from vibe_quant.discovery import __main__ as cli
from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult

if TYPE_CHECKING:
    from pytest import LogCaptureFixture, MonkeyPatch


def _run_summary(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    per_seed: list[list[tuple[Any, Any]]],
) -> tuple[DiscoveryResult, str]:
    """Fake each seed to return its own champion list; return (result, log text)."""
    seeds = list(per_seed)

    def fake_run(self: Any) -> DiscoveryResult:
        return DiscoveryResult(
            generations=[], top_strategies=list(seeds.pop(0)),
            total_candidates_evaluated=400, converged=False,
            convergence_generation=None,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    caplog.set_level(logging.INFO, logger="vibe_quant.discovery.__main__")
    res = cli._run_multi_seed(
        len(per_seed), _mock_cfg(prior_trials_fn=lambda s, e: 30, top_k=2),
        lambda c: {}, "/tmp/vq_ec_nansharpe_summary.json",
    )  # fmt: skip
    return res, caplog.text


def _chrom_fit(sharpe: float, seed: int) -> tuple[Any, Any]:
    return (_chrom(), _fit(sharpe_ratio=sharpe, mtm_daily_returns=_series(1.0, seed=seed)))


def test_nan_seed_sharpe_does_not_abort_and_stats_use_finite_only(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """One NaN seed, one finite: the run completes and the stats ignore the NaN."""
    res, text = _run_summary(
        monkeypatch,
        caplog,
        [[_chrom_fit(float("nan"), seed=1)], [_chrom_fit(2.0, seed=2)]],
    )
    assert isinstance(res, DiscoveryResult)  # completed, not aborted
    assert "(2 groups)" in text or "groups from" in text  # reached grouping/merged log
    assert "Best Sharpe: mean=2.00 median=2.00 min=2.00 max=2.00" in text
    assert "Non-finite seed Sharpes: 1/2" in text


def test_all_nan_seed_sharpes_log_na(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Every seed non-finite: still completes and logs ``n/a`` for the Sharpe block."""
    res, text = _run_summary(
        monkeypatch,
        caplog,
        [[_chrom_fit(float("nan"), seed=1)], [_chrom_fit(float("nan"), seed=2)]],
    )
    assert isinstance(res, DiscoveryResult)
    assert "Best Sharpe: n/a" in text
    assert "Non-finite seed Sharpes: 2/2" in text


def test_best_sharpe_ignores_non_finite_in_any_position(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """max() over NaN is order-dependent; best Sharpe must be finite-only either way."""
    res, text = _run_summary(
        monkeypatch,
        caplog,
        [
            [_chrom_fit(2.0, seed=1), _chrom_fit(float("nan"), seed=1)],
            [_chrom_fit(float("nan"), seed=2), _chrom_fit(3.0, seed=2)],
        ],
    )
    assert isinstance(res, DiscoveryResult)
    # finite bests are 2.0 and 3.0 regardless of where the NaN sits -> mean 2.50
    assert "Best Sharpe: mean=2.50 median=2.50 min=2.00 max=3.00" in text
    # every seed HAS a finite best (a non-finite one was ignored), so no warning
    assert "Non-finite seed Sharpes" not in text


def test_inf_champion_ignored_inside_a_seed(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """A +inf top strategy must not become the seed's best when a finite one exists.

    ``max([2.0, inf])`` is ``inf``, so the per-seed filter must drop non-finite
    values -- an ``isnan``-only guard would keep the inf and inflate the stats.
    """
    res, text = _run_summary(
        monkeypatch,
        caplog,
        [
            [_chrom_fit(2.0, seed=1), _chrom_fit(float("inf"), seed=1)],
            [_chrom_fit(3.0, seed=2), _chrom_fit(float("inf"), seed=2)],
        ],
    )
    assert isinstance(res, DiscoveryResult)
    assert "Best Sharpe: mean=2.50 median=2.50 min=2.00 max=3.00" in text
    assert "Non-finite seed Sharpes" not in text


def test_inf_seed_sharpe_is_non_finite_and_stats_use_finite_only(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """An all-+-inf seed is reported non-finite and excluded from the stats."""
    res, text = _run_summary(
        monkeypatch,
        caplog,
        [
            [_chrom_fit(float("-inf"), seed=1)],
            [_chrom_fit(2.0, seed=2)],
            [_chrom_fit(float("-inf"), seed=3)],
        ],
    )
    assert isinstance(res, DiscoveryResult)
    assert "Best Sharpe: mean=2.00 median=2.00 min=2.00 max=2.00" in text
    assert "Non-finite seed Sharpes: 2/3" in text


def test_all_inf_seed_sharpes_log_na(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Every seed non-finite (mixed +inf/-inf): completes and logs ``n/a``."""
    res, text = _run_summary(
        monkeypatch,
        caplog,
        [[_chrom_fit(float("inf"), seed=1)], [_chrom_fit(float("-inf"), seed=2)]],
    )
    assert isinstance(res, DiscoveryResult)
    assert "Best Sharpe: n/a" in text
    assert "Non-finite seed Sharpes: 2/2" in text
