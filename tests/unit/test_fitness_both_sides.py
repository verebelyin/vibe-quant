"""inydc T3: complexity and guardrails count the short-side genes (both sides).

``operators.scalar_gene_count`` (T1) sums entry+exit+short_entry+short_exit (MA
genes excluded, as on main). T3 makes the three duplicated
``len(entry_genes) + len(exit_genes)`` gene counts use it, so a direction=both
genome is penalised and guardrailed for BOTH sides:

* ``fitness._evaluate_single`` (the Occam / complexity penalty, R20),
* ``pipeline._validate_top_strategies`` (the ``max_complexity=10`` guardrail),
* ``scripts/research/power/gatechain.evaluate_candidate`` (same guardrail).

R22 is a property of the T1 DSL: a both genome's short side is part of its
``dsl_body_key``, so two genomes with the same long side and different short
sides are two distinct strategies for the DSR N (pipeline `_dsl_key` dedupe).
Long/short counts are unchanged: with empty short lists ``scalar_gene_count``
equals the old ``len(entry_genes) + len(exit_genes)``.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))

import gatechain  # noqa: E402

from tests.unit.test_power_gatechain import (  # noqa: E402
    N_DAYS,
    _flat_closes,
    _flat_window,
    _pass_holdout,
    _series,
    _w3,
    _window_factory,
)
from vibe_quant.discovery.fitness import (  # noqa: E402
    COMPLEXITY_PENALTY_CAP,
    FitnessResult,
    evaluate_population,
)
from vibe_quant.discovery.genome import chromosome_to_dsl  # noqa: E402
from vibe_quant.discovery.guardrails import check_complexity  # noqa: E402
from vibe_quant.discovery.operators import (  # noqa: E402
    MAX_SCALAR_GENES,
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
    _random_chromosome,
    scalar_gene_count,
)
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline  # noqa: E402

DIGEST_SEED = 7
COUNT = 200


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _rsi(condition: ConditionType, threshold: float, period: float = 14.0) -> StrategyGene:
    return StrategyGene("RSI", {"period": period}, condition, threshold)


def _long_11() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_rsi(ConditionType.LT, 30.0)],
        exit_genes=[_rsi(ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.LONG,
    )


def _both(n_entry: int, n_exit: int, n_short_entry: int, n_short_exit: int) -> StrategyChromosome:
    """A both chromosome with given long/short gene counts (RSI14, thresholds valid)."""
    return StrategyChromosome(
        entry_genes=[_rsi(ConditionType.LT, 30.0 + i) for i in range(n_entry)],
        exit_genes=[_rsi(ConditionType.GT, 70.0 + i) for i in range(n_exit)],
        short_entry_genes=[_rsi(ConditionType.GT, 70.0 + i) for i in range(n_short_entry)],
        short_exit_genes=[_rsi(ConditionType.LT, 30.0 + i) for i in range(n_short_exit)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.BOTH,
    )


def _bt(_chrom: StrategyChromosome) -> dict[str, float | int]:
    return {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.2,
        "profit_factor": 1.5,
        "total_trades": 100,
        "total_return": 0.05,
    }


def _fit(**kw: Any) -> FitnessResult:
    base: dict[str, Any] = {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": 100,
        "total_return": 0.3,
        "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0,
        "sl_tp_penalty": 0.0,
        "raw_score": 0.5,
        "adjusted_score": 0.5,
        "passed_filters": True,
        "filter_results": {},
        "trade_returns": tuple([0.01] * 100),
    }
    base.update(kw)
    return FitnessResult(**base)


def _pipe() -> DiscoveryPipeline:
    cfg = DiscoveryConfig(
        symbols=["BTC/USDT"],
        start_date="2024-01-01",
        end_date="2024-06-01",
        min_trades=1,
        require_dsr=False,
        require_bootstrap_ci=False,
        max_workers=1,
    )
    return DiscoveryPipeline(cfg, backtest_fn=_bt)


# ---------------------------------------------------------------------------
# R20: the complexity penalty counts BOTH sides (fitness.py site)
# ---------------------------------------------------------------------------


def test_both_complexity_counts_short_genes() -> None:
    # long 1+1 = 2 genes: exactly the free pair -> 0.0 (unchanged behaviour).
    long_res = evaluate_population([_long_11()], _bt, min_trades=1)
    assert long_res[0].complexity_penalty == 0.0

    # both 1+1 long + 1+1 short = 4 genes -> (4 - 2) * 0.02 = 0.04.
    both_res = evaluate_population([_both(1, 1, 1, 1)], _bt, min_trades=1)
    assert both_res[0].complexity_penalty == pytest.approx(0.04)
    assert both_res[0].adjusted_score == pytest.approx(both_res[0].raw_score - 0.04, abs=1e-6)

    # both 3+2 long + 3+2 short = 10 genes -> 0.16 clamped to the 0.1 cap.
    big_res = evaluate_population([_both(3, 2, 3, 2)], _bt, min_trades=1)
    assert big_res[0].complexity_penalty == pytest.approx(0.1)
    assert big_res[0].complexity_penalty == pytest.approx(COMPLEXITY_PENALTY_CAP)


def test_complexity_guardrail_equality_edge_and_existing_reason() -> None:
    # 3+2 long + 3+2 short = 10 genes: exactly max_complexity -> passes.
    edge = _both(3, 2, 3, 2)
    assert scalar_gene_count(edge) == 10
    assert check_complexity(scalar_gene_count(edge), MAX_SCALAR_GENES) == (True, None)

    # 3+3 long + 3+2 short = 11 genes: rejected, with the untouched reason text.
    over = _both(3, 3, 3, 2)
    assert scalar_gene_count(over) == 11
    assert check_complexity(scalar_gene_count(over), MAX_SCALAR_GENES) == (
        False,
        "Too complex: 11 genes > 10 max",
    )


# ---------------------------------------------------------------------------
# R20: the pipeline guardrail site counts BOTH sides too
# ---------------------------------------------------------------------------


def test_pipeline_guardrail_counts_short_genes() -> None:
    pipe = _pipe()
    fit = _fit()

    # 10 genes: at the max -> kept (the short side is counted, not double-rejected).
    at_edge = _both(3, 2, 3, 2)
    assert [c for c, _ in pipe._validate_top_strategies([(at_edge, fit)], 1)] == [at_edge]

    # 11 genes: rejected by the complexity guardrail with the existing reason.
    over = _both(3, 3, 3, 2)
    assert pipe._validate_top_strategies([(over, fit)], 1) == []
    [rejection] = pipe._guardrail_rejections
    assert rejection["stage"] == "guardrails"
    assert "Too complex: 11 genes > 10 max" in rejection["reasons"]


# ---------------------------------------------------------------------------
# R20: the gate-chain site counts BOTH sides too
# ---------------------------------------------------------------------------


def test_gatechain_complexity_marginal_counts_short_genes() -> None:
    settings = _w3()
    windows = _window_factory({}, _flat_window())
    closes = _flat_closes(N_DAYS)
    fit = _fit(mtm_daily_returns=_series(2.0, N_DAYS, 21))

    at_edge = _both(3, 2, 3, 2)  # 10 genes -> complexity passes
    out_edge = gatechain.evaluate_candidate(
        at_edge, fit, _pass_holdout(), windows, closes, settings
    )
    assert out_edge["marginals"]["complexity"] is True

    over = _both(3, 3, 3, 2)  # 11 genes -> complexity rejects
    out_over = gatechain.evaluate_candidate(over, fit, _pass_holdout(), windows, closes, settings)
    assert out_over["marginals"]["complexity"] is False


# ---------------------------------------------------------------------------
# R22: the short side is part of the strategy identity (DSR N dedupe)
# ---------------------------------------------------------------------------


def test_r22_short_side_changes_dsl_body_key() -> None:
    same_long_a = _both(1, 1, 1, 1)
    same_long_b = _both(1, 1, 1, 1)
    same_long_b.short_entry_genes[0] = _rsi(ConditionType.GT, 80.0)

    dsl_a = chromosome_to_dsl(same_long_a)
    dsl_b = chromosome_to_dsl(same_long_b)
    # The long side is untouched by the different short side...
    assert dsl_a["entry_conditions"]["long"] == dsl_b["entry_conditions"]["long"] == ["rsi_entry_0 < 30"]
    assert dsl_a["exit_conditions"]["long"] == dsl_b["exit_conditions"]["long"] == ["rsi_exit_0 > 70"]
    # ...and the literal short conditions differ.
    assert dsl_a["entry_conditions"]["short"] == ["rsi_short_entry_0 > 70"]
    assert dsl_b["entry_conditions"]["short"] == ["rsi_short_entry_0 > 80"]

    key_a = DiscoveryPipeline._dsl_key(same_long_a)
    key_b = DiscoveryPipeline._dsl_key(same_long_b)
    assert not key_a.startswith("uid:") and not key_b.startswith("uid:")
    assert key_a != key_b

    # True duplicates still collapse to one cache slot / one DSR trial.
    assert DiscoveryPipeline._dsl_key(_both(1, 1, 1, 1)) == key_a


# ---------------------------------------------------------------------------
# Long/short counts are unchanged (empty short lists -> legacy expression)
# ---------------------------------------------------------------------------


def test_scalar_gene_count_matches_legacy_expression_over_population() -> None:
    random.seed(DIGEST_SEED)
    for _ in range(COUNT):
        chrom = _random_chromosome(random.choice([Direction.LONG, Direction.SHORT]))
        assert chrom.short_entry_genes == [] and chrom.short_exit_genes == []
        legacy = len(chrom.entry_genes) + len(chrom.exit_genes)
        assert scalar_gene_count(chrom) == legacy
