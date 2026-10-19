"""An explicit discovery indicator pool restricts the PriceVsMA genes too.

Before vibe-quant-nn7i7 the pool only pruned the scalar-threshold pool, so a run
launched with ``--indicator-pool ROC,RSI`` still drew KAMA/VIDYA/FRAMA
price-vs-MA genes. The pool now also restricts ``genome.MA_POOL``: a MA name in
the pool enables that MA's genes, an explicit pool without MA names yields none,
and ``indicator_pool=None`` keeps the default behaviour (pinned bit for bit by
``single_direction_digests.json``).
"""

from __future__ import annotations

import logging
import random

import pytest

from tests.unit.test_discovery_pipeline import _make_config, _mock_backtest
from vibe_quant.discovery import genome, operators
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    PriceVsMAConditionGene,
    StrategyChromosome,
    StrategyGene,
    crossover,
    initialize_population,
    mutate,
    validate_indicator_pool,
)
from vibe_quant.discovery.pipeline import DiscoveryPipeline

_ALL_MA_NAMES = ["FRAMA", "KAMA", "VIDYA"]


def _filter_pool(pool: list[str] | None) -> None:
    pipe = DiscoveryPipeline(_make_config(indicator_pool=pool), _mock_backtest)
    pipe._apply_indicator_pool_filter()


def _ma_genes_of(population: list[StrategyChromosome]) -> list[PriceVsMAConditionGene]:
    return [
        gene
        for chrom in population
        for gene in (
            chrom.ma_entry_genes
            + chrom.ma_exit_genes
            + chrom.short_ma_entry_genes
            + chrom.short_ma_exit_genes
        )
    ]


def _ma_types(population: list[StrategyChromosome]) -> set[str]:
    return {gene.indicator_type for gene in _ma_genes_of(population)}


def _evolve(population: list[StrategyChromosome], rounds: int = 20) -> list[StrategyChromosome]:
    """Twenty crossover+mutate rounds at full mutation rate (add/remove included)."""
    random.seed(20261019)
    pop = population
    for _ in range(rounds):
        children: list[StrategyChromosome] = []
        for i in range(0, len(pop) - 1, 2):
            child_a, child_b = crossover(pop[i], pop[i + 1])
            children.append(mutate(child_a, 1.0))
            children.append(mutate(child_b, 1.0))
        pop = children
    return pop


def _seed_with_kama() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
        direction=Direction.LONG,
        ma_entry_genes=[
            PriceVsMAConditionGene("KAMA", {"period": 10.0}, ConditionType.GT)
        ],
    )


_MA_LISTS = ["ma_entry_genes", "ma_exit_genes", "short_ma_entry_genes", "short_ma_exit_genes"]


def _both_seed_with_kama_in(lists: list[str]) -> StrategyChromosome:
    """A BOTH seed with valid scalar genes on each side and KAMA in ``lists``."""
    chrom = StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
        direction=Direction.BOTH,
        short_entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        short_exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
    )
    for name in lists:
        setattr(chrom, name, [PriceVsMAConditionGene("KAMA", {"period": 10.0}, ConditionType.GT)])
    return chrom


class TestPoolWithoutMaNames:
    def test_no_ma_genes_in_the_seeded_population_or_after_twenty_rounds(self) -> None:
        random.seed(11)
        _filter_pool(["ROC", "RSI"])

        assert operators._MA_NAMES == []
        assert genome.MA_POOL == {}

        population = initialize_population(12)
        assert _ma_genes_of(population) == []

        assert _ma_genes_of(_evolve(population)) == []

    def test_a_seed_ma_gene_outside_the_pool_is_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        random.seed(14)
        _filter_pool(["RSI", "VIDYA"])

        with caplog.at_level(logging.INFO, logger="vibe_quant.discovery.operators"):
            population = initialize_population(4, seed_chromosomes=[_seed_with_kama()])

        assert "Warm-start: seeded 1/4 slots" in caplog.text
        assert population[0].ma_entry_genes == []
        assert _ma_types(population) <= {"VIDYA"}


class TestSeedMaListsFiltered:
    @pytest.mark.parametrize("ma_list", _MA_LISTS)
    def test_forbidden_ma_gene_dropped_from_each_list_and_seed_admitted(
        self, ma_list: str
    ) -> None:
        random.seed(15)
        _filter_pool(["RSI", "VIDYA"])

        population = initialize_population(
            4, seed_chromosomes=[_both_seed_with_kama_in([ma_list])]
        )

        seed = population[0]
        assert seed.direction is Direction.BOTH
        assert getattr(seed, ma_list) == []
        assert seed.entry_genes[0].indicator_type == "RSI"  # the seed itself was admitted

    def test_seed_with_kama_in_all_four_lists_is_admitted_with_all_dropped(self) -> None:
        random.seed(16)
        _filter_pool(["RSI", "VIDYA"])

        population = initialize_population(
            4, seed_chromosomes=[_both_seed_with_kama_in(_MA_LISTS)]
        )

        seed = population[0]
        assert seed.direction is Direction.BOTH
        assert seed.entry_genes[0].indicator_type == "RSI"
        for name in _MA_LISTS:
            assert getattr(seed, name) == []


class TestPoolWithVidya:
    def test_only_vidya_ma_genes_in_the_seeded_population_and_after_twenty_rounds(self) -> None:
        random.seed(12)
        _filter_pool(["RSI", "VIDYA"])

        assert operators._MA_NAMES == ["VIDYA"]

        population = initialize_population(24)
        assert _ma_types(population) == {"VIDYA"}

        assert _ma_types(_evolve(population)) == {"VIDYA"}


class TestPoolNone:
    def test_default_pool_keeps_every_ma(self) -> None:
        random.seed(13)
        _filter_pool(None)

        assert sorted(genome.MA_POOL) == _ALL_MA_NAMES
        assert sorted(operators._MA_NAMES) == _ALL_MA_NAMES

        population = initialize_population(24)
        types = _ma_types(population)
        assert types, "the default pool must still draw MA genes"
        assert types <= set(_ALL_MA_NAMES)


class TestPoolValidation:
    def test_ma_names_are_valid_pool_names(self) -> None:
        validate_indicator_pool(["RSI", "VIDYA"])
        validate_indicator_pool(["VIDYA", "KAMA", "FRAMA", "ROC"])

    def test_unknown_name_still_fails(self) -> None:
        with pytest.raises(ValueError, match="not available for GA discovery"):
            validate_indicator_pool(["RSI", "NOPE"])

    def test_ma_names_alone_cannot_build_the_genome(self) -> None:
        with pytest.raises(ValueError, match="scalar-threshold"):
            validate_indicator_pool(["VIDYA"])


def test_run_notes_record_an_explicit_pool_and_default_runs_add_no_key() -> None:
    """A harness checks the pool the run recorded (scripts/research/power config_violations)."""
    from vibe_quant.discovery.__main__ import _run_provenance_notes
    from vibe_quant.discovery.pipeline import DiscoveryResult

    result = DiscoveryResult(
        generations=[], top_strategies=[], total_candidates_evaluated=10, converged=True,
        convergence_generation=1,
    )  # fmt: skip
    kwargs = dict(
        use_mock=True, eval_windows_count=1, eval_windows=None, symbol_agg="portfolio",
        split_ratio=0.0, direction=None, cross_window_months=[], cross_window_min_sharpe=0.5,
        wfa_oos_step_days=0, wfa_min_consistency=0.75, num_seeds=1, seed=42,
        bootstrap_min_sharpe=0.0, holdout_min_sharpe=0.0, symbols=["BTCUSDT"],
    )  # fmt: skip
    explicit = _run_provenance_notes(result, indicator_pool=["ROC", "RSI"], **kwargs)  # type: ignore[arg-type]
    assert explicit["indicator_pool"] == ["ROC", "RSI"]
    assert "indicator_pool" not in _run_provenance_notes(result, **kwargs)  # type: ignore[arg-type]
