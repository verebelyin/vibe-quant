"""A discovery run over a port's own parameters (``genome_ops`` = ParamOps).

R34: the per-generation / top-K logs read ``genome_summary``. R36: the DSR raw trial
count is the number of DISTINCT rendered DSL bodies. The legacy path stays
bit-identical (test_discovery_legacy_rng_golden.py, test_discovery_default_parity.py).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_gate_chain_contract import _fit
from tests.unit.test_param_genome_consumers import _genome, _long_port, _short_port
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline, GenerationResult

if TYPE_CHECKING:
    from pathlib import Path


def _config(**overrides: Any) -> DiscoveryConfig:
    base: dict[str, Any] = {
        "population_size": 4,
        "max_generations": 2,
        "min_trades": 50,
        "symbols": ["BTCUSDT"],
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",
        "max_workers": None,
        "require_dsr": False,
        "eval_windows": 1,
    }
    base.update(overrides)
    return DiscoveryConfig(**base)


def _unused_backtest(_genome: Any) -> dict[str, float | int]:
    raise AssertionError("logging must not run a backtest")


@pytest.mark.parametrize(
    ("make_port", "other", "best_summary", "directions"),
    [
        (_long_port, {"fast": 12, "slow": 55}, "port=ma_long fast=10 slow=40", "{'long': 2}"),
        (_short_port, {"pinned": 3}, None, "{'short': 2}"),
    ],
)
def test_positive_fitness_generation_logs_param_genome(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    make_port: Any,
    other: dict[str, Any],
    best_summary: str | None,
    directions: str,
) -> None:
    """R34: a generation with positive fitness logs a ParamGenome through its summary."""
    port = make_port(tmp_path)
    best = _genome(port)
    second = _genome(port, **other)
    results = [_fit(adjusted_score=0.9), _fit(adjusted_score=0.4)]
    pipe = DiscoveryPipeline(_config(), _unused_backtest)
    generation = GenerationResult(
        generation=0,
        best_fitness=0.9,
        mean_fitness=0.65,
        worst_fitness=0.4,
        best_chromosome=best.clone(),  # type: ignore[arg-type]
        population_size=2,
        num_passed_filters=2,
    )
    with caplog.at_level(logging.INFO, logger="vibe_quant.discovery.pipeline"):
        pipe._log_generation(
            gen=0,
            population=[best, second],  # type: ignore[list-item]
            fitness_results=results,
            generation_results=[generation],
            best_idx=0,
            gen_elapsed=1.0,
            total_elapsed=1.0,
            evals_this_gen=2,
        )
    text = caplog.text
    assert f"Best: uid={best.uid} port={port.name} " in text
    if best_summary is not None:
        assert best_summary in text
    assert f"directions={directions}" in text
    assert "indicators={}" in text


# ---------------------------------------------------------------------------
# A whole run over a port's params
# ---------------------------------------------------------------------------


def _scored_backtest(genome: Any) -> dict[str, float | int]:
    """Deterministic metrics from the genome's values (a port genome has no genes)."""
    slow = float(genome.values["slow"])
    fast = float(genome.values["fast"])
    return {
        "sharpe_ratio": 1.0 + (slow - fast) / 100.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": 200,
        "total_return": 0.3,
    }


def _port_run_config(**overrides: Any) -> DiscoveryConfig:
    return _config(
        require_bootstrap_ci=False,
        train_test_split=0.0,
        min_diversity_distance=0.0,
        entropy_threshold=0.0,
        **overrides,
    )


@pytest.mark.parametrize("crowding", [False, True])
def test_param_run_evolves_and_returns_param_champions(
    tmp_path: Path, crowding: bool, caplog: pytest.LogCaptureFixture
) -> None:
    from vibe_quant.discovery.param_genome import ParamGenome, ParamOps

    port = _long_port(tmp_path)
    pipe = DiscoveryPipeline(
        _port_run_config(population_size=6, max_generations=3, use_crowding=crowding),
        _scored_backtest,
        genome_ops=ParamOps(port),
    )
    with caplog.at_level(logging.INFO, logger="vibe_quant.discovery.pipeline"):
        result = pipe.run()
    assert result.top_strategies, "the best params must survive the gates"
    champion, fit = result.top_strategies[0]
    assert isinstance(champion, ParamGenome)
    assert champion.port is port
    # the fake sharpe grows with slow - fast: the best genome must beat the default (10, 40)
    best_gap = champion.values["slow"] - champion.values["fast"]  # type: ignore[operator]
    assert best_gap >= 30
    assert fit.error is None
    assert "Top " in caplog.text
    assert "port=ma_long fast=" in caplog.text


def test_param_run_top_strategies_log_the_summary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pipe = DiscoveryPipeline(_config(), _unused_backtest)
    genome = _genome(_short_port(tmp_path), upper=75.0)
    with caplog.at_level(logging.INFO, logger="vibe_quant.discovery.pipeline"):
        pipe._log_top_strategies([(genome, _fit(adjusted_score=0.8))])  # type: ignore[list-item]
    assert (
        f"#1 uid={genome.uid} score=0.8000 sharpe=0.50 pf=1.50 dd=10.0% return=30.0% trades=200 | "
        "port=rsi_short period=14 upper=75.0 lower=30.0 pinned=3"
    ) in caplog.text


def test_param_immigrants_replace_non_elite_members_with_port_genomes(tmp_path: Path) -> None:
    from vibe_quant.discovery.param_genome import ParamGenome, ParamOps

    port = _long_port(tmp_path)
    pipe = DiscoveryPipeline(
        _config(entropy_threshold=0.5, immigrant_fraction=0.5),
        _scored_backtest,
        genome_ops=ParamOps(port),
    )
    clones = [_genome(port) for _ in range(6)]  # monoculture: entropy 0
    known: list[Any] = [_fit(adjusted_score=0.5)] * 6
    new_pop, new_known = pipe._maybe_inject_immigrants(clones, known)  # type: ignore[arg-type]
    old_pop: list[Any] = clones
    replaced = [i for i, (old, new) in enumerate(zip(old_pop, new_pop, strict=True)) if old is not new]
    assert len(replaced) == 3
    assert 0 not in replaced  # the elite slot is protected
    assert all(isinstance(new_pop[i], ParamGenome) for i in replaced)
    assert [new_known[i] for i in replaced] == [None, None, None]


def test_param_population_entropy_is_one_for_distinct_values_and_zero_for_clones(
    tmp_path: Path,
) -> None:
    from vibe_quant.discovery.diversity import population_entropy

    port = _long_port(tmp_path)
    clones = [_genome(port) for _ in range(4)]
    spread = [_genome(port, fast=5 + i, slow=30 + 5 * i) for i in range(4)]
    assert population_entropy(clones) == 0.0  # type: ignore[arg-type]
    assert population_entropy(spread) == pytest.approx(1.0)  # type: ignore[arg-type]


def test_port_run_refuses_direction_and_indicator_pool(tmp_path: Path) -> None:
    from vibe_quant.discovery.param_genome import ParamOps

    ops = ParamOps(_long_port(tmp_path))
    with pytest.raises(ValueError, match="excludes config.direction and config.indicator_pool"):
        DiscoveryPipeline(_config(direction="long"), _scored_backtest, genome_ops=ops)
    with pytest.raises(ValueError, match="excludes config.direction and config.indicator_pool"):
        DiscoveryPipeline(_config(indicator_pool=["RSI"]), _scored_backtest, genome_ops=ops)


# ---------------------------------------------------------------------------
# R36: the DSR raw trial count is the number of DISTINCT rendered DSL bodies
# ---------------------------------------------------------------------------


def _flat_backtest(_genome: Any) -> dict[str, float | int]:
    return {"sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
            "total_trades": 200, "total_return": 0.3}


def test_dsr_raw_count_is_distinct_rendered_bodies(tmp_path: Path) -> None:
    """Clones and value aliases (75.0 / 75 / 75+1e-11) share one body; a changed value is a new one."""
    from vibe_quant.discovery.param_genome import ParamGenome

    port = _short_port(tmp_path)
    pipe = DiscoveryPipeline(_config(), _flat_backtest)
    base = _genome(port, upper=75.0)
    aliases = [
        ParamGenome(port=port, values={**base.values, "upper": 75}),
        ParamGenome(port=port, values={**base.values, "upper": 75.00000000001}),
        ParamGenome(port=port, values=dict(base.values)),
    ]
    assert len({base.uid, *(a.uid for a in aliases)}) == 4
    changed = _genome(port, upper=80.0)
    other = _genome(port, period=15)
    population = [base, *aliases, changed, other, changed.clone()]
    results = pipe._evaluate_new(population)  # type: ignore[arg-type]
    assert pipe._total_evaluated == 3
    assert len(pipe.fitness_cache) == 3
    assert results[0] is results[1] is results[2] is results[3]
    assert results[4] is results[6]
    assert len({id(r) for r in results}) == 3


def test_dsr_trial_record_counts_distinct_bodies_with_injected_prior_ledger(
    tmp_path: Path,
) -> None:
    """The gate record keeps its fields; ``raw`` is the distinct count; prior trials come from the ledger."""
    from vibe_quant.discovery.param_genome import ParamOps

    ledger_calls: list[tuple[str, str]] = []

    def scratch_ledger(start: str, end: str) -> int | None:
        ledger_calls.append((start, end))
        return 7

    port = _long_port(tmp_path)
    pipe = DiscoveryPipeline(
        _port_run_config(
            population_size=6, max_generations=3, require_dsr=True, prior_trials_fn=scratch_ledger
        ),
        _scored_backtest,
        genome_ops=ParamOps(port),
    )
    result = pipe.run()
    assert result.dsr_trials is not None
    assert result.dsr_trials["raw"] == result.total_candidates_evaluated == len(pipe.fitness_cache)
    assert result.dsr_trials["prior_n"] == 7
    assert ledger_calls == [("2024-01-01", "2024-06-01")]
    assert set(result.dsr_trials) >= {"raw", "prior_n", "run_n_eff", "gate_n", "dsr_rule"}
    assert result.dsr_trials["gate_n"] == result.dsr_trials["run_n_eff"] + 7  # type: ignore[operator]


# ---------------------------------------------------------------------------
# The bar preflight follows the timeframes the rendered port loads
# ---------------------------------------------------------------------------


def test_port_preflight_checks_bars_and_resolves_ticks_for_every_loaded_timeframe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 1h port loading a 15m indicator checks 1h AND 15m bars; ticks follow the finest (15m)."""
    from dataclasses import replace

    from tests.unit.test_fill_tick_preflight import _Fn, _Resolver
    from vibe_quant.discovery.param_genome import ParamOps
    from vibe_quant.screening import nt_runner

    port = _long_port(tmp_path)
    tree = dict(port.dsl_tree)
    tree["additional_timeframes"] = ["15m"]
    indicators = {name: dict(spec) for name, spec in tree["indicators"].items()}  # type: ignore[attr-defined]
    indicators["slow_ma"]["timeframe"] = "15m"
    tree["indicators"] = indicators
    port = replace(port, dsl_tree=tree)
    bar_types: list[str] = []
    monkeypatch.setattr(nt_runner, "require_bars_in_window", lambda c, b, s, e: bar_types.append(b))
    resolver = _Resolver()
    monkeypatch.setattr("vibe_quant.data.fill_ticks.resolve_fill_ticks", resolver)
    fn = _Fn("2024-02-01", "2024-03-01", timeframe="1h")
    pipe = DiscoveryPipeline(_config(), fn, genome_ops=ParamOps(port))

    pipe._preflight_bars()

    assert set(bar_types) == {
        "BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL",
        "BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL",
    }
    assert [(call[0], call[1]) for call in resolver.calls] == [(["BTCUSDT"], "15m")]
    assert fn.fill_ticks is not None and fn.fill_ticks.timeframe == "15m"
