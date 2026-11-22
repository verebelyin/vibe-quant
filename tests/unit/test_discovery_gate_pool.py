"""``--gate-pool all``: every train-passer is screened, the survivors capped by DSR rank.

vibe-quant-clmmc.2. The default ``top_k`` keeps only the 5 best-by-fitness diverse
genomes for the gates; ``all`` screens every genome that passes the train gate, then
keeps at most ``top_k`` survivors ranked by the DSR statistic (diversity first) and
records WHY each other survivor was left out. Default runs stay byte-identical
(``test_discovery_notes_golden.py``).
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.discovery import record
from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.gate_chain import TrialBudget
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import (
    DiscoveryConfig,
    DiscoveryPipeline,
    _cap_champions,
)

if TYPE_CHECKING:
    from pathlib import Path


def _chrom(i: int, indicator: str = "RSI", **kw: Any) -> StrategyChromosome:
    """Two genes, SL 2 / TP 4: no complexity or SL/TP penalty; ``i`` = entry threshold - 20."""
    base: dict[str, Any] = {
        "entry_genes": [StrategyGene(indicator, {"period": 14.0}, ConditionType.LT, 20.0 + i, None)],
        "exit_genes": [StrategyGene(indicator, {"period": 14.0}, ConditionType.GT, 70.0, None)],
        "stop_loss_pct": 2.0, "take_profit_pct": 4.0, "direction": Direction.LONG,
    }  # fmt: skip
    base.update(kw)
    return StrategyChromosome(**base)


class _FixedOps:
    """GA operators that hand the pipeline a fixed population (one generation only)."""

    def __init__(self, chroms: list[StrategyChromosome]) -> None:
        self.chroms = chroms

    def initialize(self, size: int, seeds: object = None) -> list[StrategyChromosome]:
        assert size == len(self.chroms)
        return [c.clone() for c in self.chroms]

    def random(self) -> StrategyChromosome:  # pragma: no cover - one generation
        raise AssertionError("no evolution in a one-generation run")

    mutate = crossover = random  # type: ignore[assignment]

    def is_valid(self, genome: StrategyChromosome) -> bool:  # pragma: no cover
        return True


class _Ranked:
    """Mock backtest: genome ``i`` scores below genome ``i-1``; DSR (legacy input) passes only for ``dsr_pass``.

    Score falls with ``i`` through the return; a kurtosis of 200 makes the legacy DSR
    insignificant at the same Sharpe (p 0.12 vs 0.027 at N 10, T 366).
    """

    def __init__(
        self, dsr_pass: set[int], zero_score: set[int] | None = None, flat: bool = False
    ) -> None:
        self.dsr_pass = dsr_pass
        self.zero_score = zero_score or set()
        self.flat = flat

    def __call__(self, chrom: StrategyChromosome) -> dict[str, float | int]:
        i = int(chrom.entry_genes[0].threshold) - 20
        kurt = 3.0 if i in self.dsr_pass else 200.0
        if i in self.zero_score:  # train passer whose overtrade penalty eats its raw score
            return {"sharpe_ratio": -1.0, "max_drawdown": 1.0, "profit_factor": 0.0,
                    "total_trades": 5000, "total_return": 1e-9, "skewness": 0.0, "kurtosis": kurt}  # fmt: skip
        return {
            "sharpe_ratio": 3.0, "max_drawdown": 0.1, "profit_factor": 1.5, "total_trades": 200,
            "total_return": 0.0 if self.flat else 1.9 - 0.1 * i, "skewness": 0.0, "kurtosis": kurt,
        }  # fmt: skip


def _config(gate_pool: str, **kw: Any) -> DiscoveryConfig:
    base: dict[str, Any] = {
        "population_size": 10, "max_generations": 1, "elite_count": 1, "top_k": 5,
        "min_trades": 50, "max_workers": None, "timeframe": "4h",
        "start_date": "2024-01-01", "end_date": "2025-01-01", "require_bootstrap_ci": False,
        "require_buy_hold": False, "require_dsr": True, "min_diversity_distance": 0.0,
        "gate_pool": gate_pool,
    }  # fmt: skip
    base.update(kw)
    return DiscoveryConfig(**base)


def _run(
    gate_pool: str, fn: _Ranked, **kw: Any
) -> tuple[Any, DiscoveryPipeline, list[StrategyChromosome]]:
    chroms = [_chrom(i) for i in range(10)]
    random.seed(1)
    pipe = DiscoveryPipeline(_config(gate_pool, **kw), fn, genome_ops=_FixedOps(chroms))  # type: ignore[arg-type]
    return pipe.run(), pipe, chroms


def test_sixth_genome_becomes_champion() -> None:
    """A2: the only DSR passer ranks 6th by fitness: ``all`` finds it, ``top_k`` never tests it."""
    all_res, all_pipe, chroms = _run("all", _Ranked({5}))
    ranked = sorted(all_pipe._all_scored, key=lambda t: -t[1].adjusted_score)
    assert ranked[5][0].uid == chroms[5].uid  # precondition: 6th by fitness

    assert [c.uid for c, _ in all_res.top_strategies] == [chroms[5].uid]
    assert len(all_res.guardrail_rejections) == 9

    top_res, _, top_chroms = _run("top_k", _Ranked({5}))
    assert top_res.top_strategies == []
    assert [r["uid"] for r in top_res.guardrail_rejections] == [c.uid for c in top_chroms[:5]]
    assert {r["stage"] for r in top_res.guardrail_rejections} == {"guardrails"}


def test_zero_score_train_passer_is_screened() -> None:
    """A4: adjusted_score 0 (penalties) but a train passer: in the ``all`` pool, not in ``_all_scored``."""
    res, pipe, chroms = _run("all", _Ranked({5}, zero_score={9}))

    assert chroms[9].uid not in {c.uid for c, _ in pipe._all_scored}
    zero = next(fr for k, fr in pipe.fitness_cache.items() if fr.total_trades == 5000)
    assert zero.adjusted_score == 0.0
    rejected = {r["uid"]: r for r in res.guardrail_rejections}
    assert rejected[chroms[9].uid]["stage"] == "guardrails"  # it reached the gates
    assert [c.uid for c, _ in res.top_strategies] == [chroms[5].uid]


def test_empty_pool_is_a_clean_no_champion_outcome() -> None:
    """A5: no train passer: 0 champions, no rejections, the no-positive-score reason, total 0."""
    res, _, _ = _run("all", _Ranked(set(), flat=True))
    assert res.top_strategies == [] and res.guardrail_rejections == []

    rows, counts = record.rejection_ledger(res.guardrail_rejections, "all")
    notes = record.no_champion_notes(
        {}, best_raw_score=0.0, guardrail_rejections=rows, rejection_counts=counts
    )
    assert notes["outcome"] == "no_viable_strategies"
    assert "(no candidate scored above zero)" in str(notes["reason"])
    assert notes["rejection_counts"] == {"total": 0, "by_stage": {}, "by_kind": {}}


# --- the champion cap -----------------------------------------------------------------------


def _fit(score: float = 0.5, ret: float = 0.3) -> FitnessResult:
    return FitnessResult(
        sharpe_ratio=2.0, max_drawdown=0.1, profit_factor=1.5, total_trades=200,
        total_return=ret, complexity_penalty=0.0, overtrade_penalty=0.0, sl_tp_penalty=0.0,
        raw_score=score, adjusted_score=score, passed_filters=True, filter_results={},
    )  # fmt: skip


_DIVERSE = ["RSI", "CCI", "WILLR", "MFI", "ROC", "ADX", "STOCH"]


def _diverse(n: int) -> list[StrategyChromosome]:
    """``n`` genomes pairwise >= 0.15 apart (different indicators, alternating direction)."""
    from vibe_quant.discovery.distance import chromosome_distance

    out = [
        _chrom(k, _DIVERSE[k], direction=Direction.LONG if k % 2 == 0 else Direction.SHORT)
        for k in range(n)
    ]
    assert all(chromosome_distance(a, b) >= 0.15 for a in out for b in out if a is not b)
    return out


def test_cap_by_dsr_stat() -> None:
    """A8: 7 diverse survivors: the 5 highest DSR statistics are kept, ranks 6 and 7 capped."""
    chroms = _diverse(7)
    psr = [3.1, 2.9, 2.7, 2.5, 2.3, 2.1, 1.9]
    # adjusted score runs the other way: the cap must rank by the DSR statistic
    survivors = [(c, _fit(0.1 + 0.01 * k, ret=0.1 + 0.01 * k)) for k, c in enumerate(chroms)]
    dsr = {c.uid: {"psr_stat": p} for c, p in zip(chroms, psr, strict=True)}

    kept, capped = _cap_champions(list(reversed(survivors)), dsr, top_k=5, min_distance=0.15)

    assert [c.uid for c, _ in kept] == [c.uid for c in chroms[:5]]
    assert [(c.uid, reason) for (c, _), reason in capped] == [
        (
            chroms[5].uid,
            "Passed every gate; not kept: champions capped at top_k=5 (rank 6 by DSR statistic)",
        ),
        (
            chroms[6].uid,
            "Passed every gate; not kept: champions capped at top_k=5 (rank 7 by DSR statistic)",
        ),
    ]


def test_cap_reasons() -> None:
    """A9: a near clone and a behavioural twin of the best are left out, each with its reason."""
    from vibe_quant.discovery.distance import chromosome_distance

    x, z, w = _diverse(3)
    y = x.clone()
    y.uid = "near-x"
    y.stop_loss_pct = 2.6
    assert 0.0 < chromosome_distance(x, y) < 0.15
    fit_x = _fit(0.5, ret=0.3)
    survivors = [(w, fit_x), (z, _fit(0.4, ret=0.2)), (y, _fit(0.45, ret=0.25)), (x, fit_x)]
    dsr = {x.uid: {"psr_stat": 3.0}, y.uid: {"psr_stat": 2.9}, z.uid: {"psr_stat": 2.8},
           w.uid: {"psr_stat": 2.7}}  # fmt: skip

    kept, capped = _cap_champions(survivors, dsr, top_k=5, min_distance=0.15)

    assert [c.uid for c, _ in kept] == [x.uid, z.uid]
    assert [(c.uid, reason) for (c, _), reason in capped] == [
        (y.uid, f"Passed every gate; not kept: within distance 0.15 of champion {x.uid}"),
        (w.uid, f"Passed every gate; not kept: same behaviour fingerprint as champion {x.uid}"),
    ]


def test_cap_without_dsr_stat_ranks_by_score_then_dsl_key() -> None:
    """A10: DSR off / legacy records: adjusted score desc, ties by DSL key asc, input order free."""
    chroms = _diverse(3)
    keys = {c.uid: DiscoveryPipeline._dsl_key(c) for c in chroms}
    survivors = [(c, _fit(0.5, ret=0.3 + 0.01 * k)) for k, c in enumerate(chroms)]
    dsr = {c.uid: {"status": "skipped", "reason": "dsr off"} for c in chroms}
    by_key = sorted(chroms, key=lambda c: keys[c.uid])

    for order in (survivors, list(reversed(survivors))):
        kept, capped = _cap_champions(order, dsr, top_k=2, min_distance=0.15)
        assert [c.uid for c, _ in kept] == [c.uid for c in by_key[:2]]
        assert [(c.uid, r) for (c, _), r in capped] == [
            (
                by_key[2].uid,
                "Passed every gate; not kept: champions capped at top_k=2 (rank 3 by adjusted score)",
            )
        ]


# --- DSR status for every screened genome ---------------------------------------------------


@pytest.mark.parametrize("gate_pool", ["all", "top_k"])
def test_dsr_status_for_every_screened_genome(gate_pool: str) -> None:
    """A11: ``all`` records why DSR never ran (worst-symbol reject, DSR off); ``top_k`` adds nothing."""
    import dataclasses

    worst = dataclasses.replace(_fit(), symbol_stats={"BTCUSDT": (0.2, 80), "ETHUSDT": (0.0, 80)})
    bad, good = _chrom(1), _chrom(2)

    pipe = DiscoveryPipeline(_config(gate_pool), _Ranked(set()))  # type: ignore[arg-type]
    pipe.gates.screen([(bad, worst)], TrialBudget(10, {}))
    off = DiscoveryPipeline(_config(gate_pool, require_dsr=False), _Ranked(set()))  # type: ignore[arg-type]
    off.gates.screen([(good, _fit())], TrialBudget(10, {}))

    if gate_pool == "all":
        assert pipe.gates.dsr_records[bad.uid] == {
            "status": "skipped",
            "reason": "worst_symbol_train",
        }
        assert off.gates.dsr_records[good.uid] == {"status": "skipped", "reason": "dsr off"}
    else:
        assert bad.uid not in pipe.gates.dsr_records
        assert off.gates.dsr_records == {}


# --- the GA is untouched by the pool flag ---------------------------------------------------


def _ga_run(gate_pool: str) -> tuple[Any, DiscoveryPipeline]:
    from vibe_quant.discovery.mock_backtest import mock_backtest

    cfg = DiscoveryConfig(
        population_size=8, max_generations=3, convergence_generations=3, top_k=2,
        min_trades=10, max_workers=None, timeframe="4h", start_date="2024-01-01",
        end_date="2025-01-01", require_buy_hold=False, gate_pool=gate_pool,  # type: ignore[arg-type]
    )  # fmt: skip
    random.seed(11)
    pipe = DiscoveryPipeline(cfg, mock_backtest)
    return pipe.run(), pipe


def test_ga_identical_across_pool_flag(saved_pools: None) -> None:
    """A14: same seed: evaluations, DSR trial counts, generation bests and GA uids identical."""
    top, top_pipe = _ga_run("top_k")
    pool, pool_pipe = _ga_run("all")

    assert pool.total_candidates_evaluated == top.total_candidates_evaluated
    assert pool.dsr_trials == top.dsr_trials
    assert [(g.best_fitness, g.mean_fitness, g.best_chromosome.uid) for g in pool.generations] == [
        (g.best_fitness, g.mean_fitness, g.best_chromosome.uid) for g in top.generations
    ]
    assert [c.uid for c, _ in pool_pipe._all_scored] == [c.uid for c, _ in top_pipe._all_scored]
    assert list(pool_pipe.fitness_cache) == list(top_pipe.fitness_cache)


@pytest.fixture
def saved_pools() -> Any:
    """``_apply_indicator_pool_filter`` mutates module globals: restore them after the run."""
    from vibe_quant.discovery.genome import MA_POOL
    from vibe_quant.discovery.operators import _INDICATOR_NAMES, _MA_NAMES, INDICATOR_POOL

    saved = dict(INDICATOR_POOL), list(_INDICATOR_NAMES), dict(MA_POOL), list(_MA_NAMES)
    yield
    INDICATOR_POOL.clear()
    INDICATOR_POOL.update(saved[0])
    _INDICATOR_NAMES[:] = saved[1]
    MA_POOL.clear()
    MA_POOL.update(saved[2])
    _MA_NAMES[:] = saved[3]


def test_seeded_all_run_notes_identical(tmp_path: Path) -> None:
    """A17: the same seeded ``--gate-pool all`` CLI run twice writes identical notes."""
    from tests.unit.test_discovery_notes_golden import MOCK_ARGV, notes_bytes
    from vibe_quant.discovery import run

    argv = [*MOCK_ARGV, "--no-bootstrap-ci", "--gate-pool", "all"]
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = notes_bytes(tmp_path / "a", argv, run.MockBacktests())
    second = notes_bytes(tmp_path / "b", argv, run.MockBacktests())

    assert first == second
    notes = json.loads(first)
    assert notes["gate_pool"] == "all"
    counts = notes["rejection_counts"]
    assert counts["total"] == sum(counts["by_stage"].values())


def test_pipeline_caps_survivors_with_gate_pool_cap_rows() -> None:
    """The champion cap runs in the pipeline: 7 survivors -> 5 champions + 2 cap rows (rev-ga1-1 F1)."""
    res, _, chroms = _run("all", _Ranked({0, 1, 2, 3, 4, 5, 6}))
    assert [c.uid for c, _ in res.top_strategies] == [c.uid for c in chroms[:5]]
    cap = [r for r in res.guardrail_rejections if r["stage"] == "gate_pool_cap"]
    assert [r["uid"] for r in cap] == [chroms[5].uid, chroms[6].uid]
    assert cap[0]["reasons"] == [
        "Passed every gate; not kept: champions capped at top_k=5 (rank 6 by adjusted score)"
    ]
