"""Tests for discovery dashboard tab module."""

from __future__ import annotations

import pytest

from vibe_quant.dashboard.pages.discovery import (
    DISCOVERY_SYMBOLS,
    build_fitness_chart_data,
    build_results_table,
    chromosome_to_yaml,
    render_discovery_tab,
)
from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.operators import StrategyChromosome as OpsChromosome
from vibe_quant.discovery.operators import StrategyGene as OpsGene
from vibe_quant.discovery.pipeline import (
    DiscoveryConfig,
    DiscoveryResult,
    GenerationResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ops_chromosome(
    n_entry: int = 2,
    n_exit: int = 1,
    sl: float = 3.0,
    tp: float = 6.0,
) -> OpsChromosome:
    """Create a test operators.StrategyChromosome."""
    from vibe_quant.discovery.operators import ConditionType, Direction

    entry = [
        OpsGene(
            indicator_type="RSI",
            parameters={"period": 14.0},
            condition=ConditionType.LT,
            threshold=30.0,
        )
        for _ in range(n_entry)
    ]
    exit_ = [
        OpsGene(
            indicator_type="RSI",
            parameters={"period": 14.0},
            condition=ConditionType.GT,
            threshold=70.0,
        )
        for _ in range(n_exit)
    ]
    return OpsChromosome(
        entry_genes=entry,
        exit_genes=exit_,
        stop_loss_pct=sl,
        take_profit_pct=tp,
        direction=Direction.LONG,
    )


def _make_fitness_result(
    sharpe: float = 1.5,
    max_dd: float = 0.15,
    pf: float = 2.0,
    trades: int = 100,
    score: float = 0.7,
) -> FitnessResult:
    """Create a test FitnessResult."""
    return FitnessResult(
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        profit_factor=pf,
        total_trades=trades,
        complexity_penalty=0.02,
        raw_score=score + 0.02,
        adjusted_score=score,
        passed_filters=True,
        filter_results={},
    )


def _make_generation_result(gen: int, best: float, mean: float, worst: float) -> GenerationResult:
    """Create a test GenerationResult."""
    chrom = _make_ops_chromosome()
    return GenerationResult(
        generation=gen,
        best_fitness=best,
        mean_fitness=mean,
        worst_fitness=worst,
        best_chromosome=chrom,
        population_size=50,
        num_passed_filters=40,
    )


# ---------------------------------------------------------------------------
# Module import tests
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Verify module can be imported and key symbols exist."""

    def test_import_render_function(self) -> None:
        """render_discovery_tab is callable."""
        assert callable(render_discovery_tab)

    def test_import_render_alias(self) -> None:
        """render alias exists."""
        from vibe_quant.dashboard.pages.discovery import render

        assert callable(render)
        assert render.__name__ == render_discovery_tab.__name__

    def test_import_helpers(self) -> None:
        """Helper functions are importable."""
        assert callable(build_fitness_chart_data)
        assert callable(build_results_table)
        assert callable(chromosome_to_yaml)

    def test_discovery_symbols_constant(self) -> None:
        """DISCOVERY_SYMBOLS has expected entries."""
        assert "BTCUSDT" in DISCOVERY_SYMBOLS
        assert "ETHUSDT" in DISCOVERY_SYMBOLS
        assert "SOLUSDT" in DISCOVERY_SYMBOLS

    def test_app_includes_discovery_page(self) -> None:
        """app.py references discovery page file."""
        from vibe_quant.dashboard import app

        # st.navigation API uses file paths, not imports
        assert hasattr(app, "_PAGES_DIR")
        assert (app._PAGES_DIR / "discovery.py").exists()


# ---------------------------------------------------------------------------
# Configuration helper tests
# ---------------------------------------------------------------------------


class TestDiscoveryConfig:
    """Test DiscoveryConfig construction from dashboard params."""

    def test_default_config(self) -> None:
        """Default config is valid."""
        cfg = DiscoveryConfig()
        assert cfg.population_size == 50
        assert cfg.max_generations == 100
        assert cfg.mutation_rate == 0.1
        assert cfg.elite_count == 2

    def test_custom_config(self) -> None:
        """Custom config values are stored."""
        cfg = DiscoveryConfig(
            population_size=100,
            max_generations=200,
            mutation_rate=0.2,
            elite_count=5,
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        assert cfg.population_size == 100
        assert cfg.symbols == ["BTCUSDT", "ETHUSDT"]
        assert cfg.timeframe == "5m"

    def test_invalid_elite_count_raises(self) -> None:
        """elite_count >= population_size raises ValueError."""
        with pytest.raises(ValueError, match="elite_count"):
            DiscoveryConfig(population_size=10, elite_count=10)

    def test_invalid_mutation_rate_raises(self) -> None:
        """mutation_rate outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="mutation_rate"):
            DiscoveryConfig(mutation_rate=1.5)

    def test_slider_range_bounds(self) -> None:
        """Config accepts values at slider boundary limits."""
        cfg_min = DiscoveryConfig(population_size=10, max_generations=10, elite_count=1)
        assert cfg_min.population_size == 10

        cfg_max = DiscoveryConfig(population_size=200, max_generations=500, elite_count=9)
        assert cfg_max.population_size == 200


# ---------------------------------------------------------------------------
# Fitness chart data tests
# ---------------------------------------------------------------------------


class TestFitnessChartData:
    """Test build_fitness_chart_data formatting."""

    def test_empty_generations(self) -> None:
        """Empty input returns empty lists."""
        data = build_fitness_chart_data([])
        assert data["generation"] == []
        assert data["best"] == []
        assert data["mean"] == []
        assert data["worst"] == []

    def test_single_generation(self) -> None:
        """Single generation data is correct."""
        gens = [_make_generation_result(0, best=0.8, mean=0.5, worst=0.2)]
        data = build_fitness_chart_data(gens)

        assert data["generation"] == [0.0]
        assert data["best"] == [0.8]
        assert data["mean"] == [0.5]
        assert data["worst"] == [0.2]

    def test_multiple_generations(self) -> None:
        """Multiple generations produce parallel lists."""
        gens = [
            _make_generation_result(0, 0.5, 0.3, 0.1),
            _make_generation_result(1, 0.7, 0.4, 0.15),
            _make_generation_result(2, 0.8, 0.5, 0.2),
        ]
        data = build_fitness_chart_data(gens)

        assert len(data["generation"]) == 3
        assert data["generation"] == [0.0, 1.0, 2.0]
        assert data["best"] == [0.5, 0.7, 0.8]
        # best should be monotonically non-decreasing in a healthy run
        assert data["best"][-1] >= data["best"][0]

    def test_data_keys(self) -> None:
        """Output dict has exactly the expected keys."""
        data = build_fitness_chart_data([])
        assert set(data.keys()) == {"generation", "best", "mean", "worst"}

    def test_all_lists_same_length(self) -> None:
        """All output lists have the same length."""
        gens = [_make_generation_result(i, 0.5 + i * 0.1, 0.3, 0.1) for i in range(5)]
        data = build_fitness_chart_data(gens)
        lengths = {len(v) for v in data.values()}
        assert len(lengths) == 1
        assert lengths.pop() == 5


# ---------------------------------------------------------------------------
# Results table tests
# ---------------------------------------------------------------------------


class TestResultsTable:
    """Test build_results_table formatting."""

    def test_empty_strategies(self) -> None:
        """Empty input returns empty list."""
        rows = build_results_table([])
        assert rows == []

    def test_single_strategy(self) -> None:
        """Single strategy produces one row with correct fields."""
        chrom = _make_ops_chromosome(n_entry=2, n_exit=1)
        fr = _make_fitness_result(sharpe=1.5, max_dd=0.15, pf=2.0, trades=100, score=0.7)
        rows = build_results_table([(chrom, fr)])

        assert len(rows) == 1
        row = rows[0]
        assert row["Rank"] == 1
        assert row["Sharpe"] == 1.5
        assert row["MaxDD"] == 0.15
        assert row["PF"] == 2.0
        assert row["Total Trades"] == 100
        assert row["Genes"] == 3  # 2 entry + 1 exit
        assert row["Score"] == 0.7

    def test_multiple_strategies_ranked(self) -> None:
        """Multiple strategies have sequential ranks."""
        pairs = [
            (_make_ops_chromosome(), _make_fitness_result(score=0.9)),
            (_make_ops_chromosome(), _make_fitness_result(score=0.7)),
            (_make_ops_chromosome(), _make_fitness_result(score=0.5)),
        ]
        rows = build_results_table(pairs)

        assert len(rows) == 3
        assert rows[0]["Rank"] == 1
        assert rows[1]["Rank"] == 2
        assert rows[2]["Rank"] == 3

    def test_gene_count_calculation(self) -> None:
        """Gene count = entry + exit genes."""
        chrom = _make_ops_chromosome(n_entry=4, n_exit=2)
        fr = _make_fitness_result()
        rows = build_results_table([(chrom, fr)])

        assert rows[0]["Genes"] == 6

    def test_row_keys(self) -> None:
        """Each row has all expected keys."""
        chrom = _make_ops_chromosome()
        fr = _make_fitness_result()
        rows = build_results_table([(chrom, fr)])

        expected_keys = {"Rank", "Sharpe", "MaxDD", "PF", "Total Trades", "Genes", "Score"}
        assert set(rows[0].keys()) == expected_keys


# ---------------------------------------------------------------------------
# Chromosome YAML export tests
# ---------------------------------------------------------------------------


class TestChromosomeYaml:
    """Test chromosome_to_yaml conversion."""

    def test_produces_valid_yaml(self) -> None:
        """Output is valid YAML string."""
        import yaml as _yaml

        chrom = _make_ops_chromosome()
        yaml_str = chromosome_to_yaml(chrom)

        parsed = _yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)

    def test_contains_indicators(self) -> None:
        """YAML contains indicators section."""
        import yaml as _yaml

        chrom = _make_ops_chromosome(n_entry=2, n_exit=1)
        yaml_str = chromosome_to_yaml(chrom)
        parsed = _yaml.safe_load(yaml_str)

        assert "indicators" in parsed
        assert len(parsed["indicators"]) > 0

    def test_contains_entry_conditions(self) -> None:
        """YAML contains entry_conditions."""
        import yaml as _yaml

        chrom = _make_ops_chromosome()
        yaml_str = chromosome_to_yaml(chrom)
        parsed = _yaml.safe_load(yaml_str)

        assert "entry_conditions" in parsed

    def test_contains_stop_loss(self) -> None:
        """YAML contains stop_loss config."""
        import yaml as _yaml

        chrom = _make_ops_chromosome()
        yaml_str = chromosome_to_yaml(chrom)
        parsed = _yaml.safe_load(yaml_str)

        assert "stop_loss" in parsed
        assert parsed["stop_loss"]["type"] == "fixed_pct"

    def test_name_prefix(self) -> None:
        """Strategy name starts with genome_ (canonical naming from genome.chromosome_to_dsl)."""
        import yaml as _yaml

        chrom = _make_ops_chromosome()
        yaml_str = chromosome_to_yaml(chrom)
        parsed = _yaml.safe_load(yaml_str)

        assert parsed["name"].startswith("genome_")


# ---------------------------------------------------------------------------
# DiscoveryResult integration tests
# ---------------------------------------------------------------------------


class TestDiscoveryResultFormatting:
    """Test formatting DiscoveryResult for display."""

    def _make_result(self, n_gens: int = 5, converged: bool = False) -> DiscoveryResult:
        gens = [
            _make_generation_result(i, 0.3 + i * 0.1, 0.2 + i * 0.05, 0.1)
            for i in range(n_gens)
        ]
        top = [
            (_make_ops_chromosome(), _make_fitness_result(score=0.8)),
            (_make_ops_chromosome(), _make_fitness_result(score=0.6)),
        ]
        return DiscoveryResult(
            generations=gens,
            top_strategies=top,
            total_candidates_evaluated=n_gens * 50,
            converged=converged,
            convergence_generation=n_gens - 1 if converged else None,
        )

    def test_convergence_info(self) -> None:
        """Converged result has convergence_generation set."""
        result = self._make_result(converged=True)
        assert result.converged is True
        assert result.convergence_generation is not None

    def test_non_convergence_info(self) -> None:
        """Non-converged result has convergence_generation=None."""
        result = self._make_result(converged=False)
        assert result.converged is False
        assert result.convergence_generation is None

    def test_chart_data_from_result(self) -> None:
        """Can build chart data from DiscoveryResult.generations."""
        result = self._make_result(n_gens=10)
        data = build_fitness_chart_data(result.generations)
        assert len(data["generation"]) == 10

    def test_table_from_result(self) -> None:
        """Can build table from DiscoveryResult.top_strategies."""
        result = self._make_result()
        rows = build_results_table(result.top_strategies)
        assert len(rows) == 2
        assert rows[0]["Rank"] == 1
