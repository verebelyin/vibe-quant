"""inydc T4: real-engine proof that direction=both trades its own short side (R23-R26b).

Rows R23/R24/R26/R26b of ``data/swarm/20261018-grind-dir/design-inydc.md``, run on the
REAL NautilusTrader screening engine (``vibe_quant.screening.NTScreeningRunner`` — the
same entry point ``scripts/research/power/layer2.worker_screen`` uses) over BTCUSDT 4h,
2024-03-01..2024-06-01. That window fires the R23 short rule 15 times and the R24 board
long rule 18 times; a window with fewer than 3 short trades does not prove the short side
trades and must be changed (and the change stated).

Before the fix ``genome.chromosome_to_dsl`` copied the long entry/exit conditions into the
short side and the compiler's ``if long ... elif short`` (long-first, untouched) meant a
BOTH strategy never took a short trade. Board evidence (main, BTCUSDT 4h
2023-06-01..2026-06-01, RSI(14) crosses_below 30 / crosses_above 60, SL4 TP8): long
sharpe -0.3729222589307 142 trades == both -0.3729222589307 142 trades; short
0.06788740870648444 112 trades.

Measured on this base (verbatim, for the journal):
  R23 SHORT rule alone: sharpe -0.6372539506147427, 15 trades, total_return
      -0.023196310019727943, profit_factor 0.7898324782023829, max_drawdown
      0.06914824652329685. BOTH(dead long + the same short rule) equals all five exactly,
      with 0 long / 15 short trades.
  R24 board LONG rule alone: sharpe 0.20772222893030862, 18 trades, total_return
      0.007795585667368038, profit_factor 1.0409366547018863, max_drawdown
      0.1257044208146125. BOTH(identical rule on both sides) equals all five exactly, with
      18 long / 0 short trades.
  R26b the OLD stored both DSL (identical conditions on both sides, as runs 920-954
      stored) still compiles and replays to exactly the R24 LONG numbers: 18 long,
      0 short.

Engine facts and side counting are read through the shipped entry point: ``runner({})``
returns a ``BacktestMetrics`` and the trade side is read off the engine's positions with
``validation.extraction.all_positions`` / ``position_direction``, captured by a spy on
``NTScreeningRunner._extract_metrics`` (the existing convention,
tests/unit/test_screening_fill_mode.py) so the run itself is untouched.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_discovery_pipeline import _make_config, _mock_backtest
from vibe_quant.discovery.__main__ import _champion_entry
from vibe_quant.discovery.genome import chromosome_to_dsl, serializable_to_chromosome
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import DiscoveryPipeline
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.validation.extraction import all_positions, position_direction

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vibe_quant.screening.types import BacktestMetrics

# The five metrics R23/R24/R26b compare with exact ``==`` (both runs are deterministic and
# share one engine, so equality is bit-exact).
_METRICS = ("sharpe_ratio", "total_trades", "total_return", "profit_factor", "max_drawdown")

_CATALOG = Path(__file__).resolve().parents[2] / "data" / "catalog"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "4h"
_START = "2024-03-01"
_END = "2024-06-01"

_BAR_DIR = _CATALOG / "data" / "bar"
_4H_BAR_TYPE = f"{_SYMBOL}-PERP.BINANCE-4-HOUR-LAST-EXTERNAL"
_1M_BAR_TYPE = f"{_SYMBOL}-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"


def _skip_without_catalog() -> None:
    """Repo skip pattern (tests/unit/test_audit_dsl_natr.py): skip when data/catalog is absent.

    4h bars are the strategy's own data; the 1m LAST series is the screening fill ticks a
    coarser strategy needs, so a catalog with only one of the two cannot run this file.
    """
    if not (_BAR_DIR / _4H_BAR_TYPE).is_dir() or not (_BAR_DIR / _1M_BAR_TYPE).is_dir():
        pytest.skip("BTC 4h/1m catalog data not available")


@contextmanager
def _counting_sides(counts: dict[str, int]) -> Iterator[None]:
    """Count the positions of ONE run by side, inside that run.

    The spy sits on ``NTScreeningRunner._extract_metrics`` (the existing convention,
    tests/unit/test_screening_fill_mode.py) and the previous value is restored on exit, so
    back-to-back runs each count their own trades only.
    """
    original = NTScreeningRunner._extract_metrics

    def _spy(
        self: Any, params: Any, bt_result: Any, engine: Any, *args: Any, **kwargs: Any
    ) -> Any:
        for position in all_positions(engine):
            counts[position_direction(position)] += 1
        return original(self, params, bt_result, engine, *args, **kwargs)

    NTScreeningRunner._extract_metrics = _spy  # type: ignore[method-assign]
    try:
        yield
    finally:
        NTScreeningRunner._extract_metrics = original  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _rsi(condition: ConditionType, threshold: float) -> StrategyGene:
    """RSI(14) gene — one period for every side, so warm-up is identical everywhere."""
    return StrategyGene("RSI", {"period": 14.0}, condition, threshold)


def _short_rule_chromosome() -> StrategyChromosome:
    """R23: the short rule on its own (RSI14 crosses_above 70 in, crosses_below 40 out)."""
    return StrategyChromosome(
        entry_genes=[_rsi(ConditionType.CROSSES_ABOVE, 70.0)],
        exit_genes=[_rsi(ConditionType.CROSSES_BELOW, 40.0)],
        stop_loss_pct=4.0,
        take_profit_pct=8.0,
        direction=Direction.SHORT,
    )


def _dead_long_short_rule_both_chromosome() -> StrategyChromosome:
    """R23: BOTH whose long side can never fire and whose short side is the rule above.

    Long entry is ``RSI14 < 0`` (RSI is never negative), long exit uses the same
    indicator+period as the short side (R23 ruling: warm-up identical), and the short
    SL/TP are the base ones (``sl_short=sl``, ``tp_short=tp``).
    """
    return StrategyChromosome(
        entry_genes=[_rsi(ConditionType.LT, 0.0)],
        exit_genes=[_rsi(ConditionType.CROSSES_ABOVE, 60.0)],
        stop_loss_pct=4.0,
        take_profit_pct=8.0,
        direction=Direction.BOTH,
        short_entry_genes=[_rsi(ConditionType.CROSSES_ABOVE, 70.0)],
        short_exit_genes=[_rsi(ConditionType.CROSSES_BELOW, 40.0)],
        stop_loss_short_pct=4.0,
        take_profit_short_pct=8.0,
    )


def _board_long_chromosome() -> StrategyChromosome:
    """R24/R26b: the board rule as LONG (RSI14 crosses_below 30 / crosses_above 60, SL4 TP8)."""
    return StrategyChromosome(
        entry_genes=[_rsi(ConditionType.CROSSES_BELOW, 30.0)],
        exit_genes=[_rsi(ConditionType.CROSSES_ABOVE, 60.0)],
        stop_loss_pct=4.0,
        take_profit_pct=8.0,
        direction=Direction.LONG,
    )


def _board_both_chromosome() -> StrategyChromosome:
    """R24: BOTH with the identical non-trivial board rule on both sides (tie -> long)."""
    chrom = _board_long_chromosome()
    chrom.direction = Direction.BOTH
    chrom.short_entry_genes = [_rsi(ConditionType.CROSSES_BELOW, 30.0)]
    chrom.short_exit_genes = [_rsi(ConditionType.CROSSES_ABOVE, 60.0)]
    return chrom


def _old_stored_both_dsl() -> dict[str, object]:
    """R26b: the shape runs 920-954 stored — one condition copied to both sides.

    Pre-fix ``chromosome_to_dsl`` wrote the long conditions into the short side too, so the
    stored DSL references the SAME indicator names on both sides and carries no per-side
    SL/TP. It is replayed verbatim (never re-derived).
    """
    return {
        "name": "genome_old_stored_both_rsi",
        "timeframe": _TIMEFRAME,
        "indicators": {
            "rsi_entry_0": {"type": "RSI", "period": 14},
            "rsi_exit_0": {"type": "RSI", "period": 14},
        },
        "entry_conditions": {
            "long": ["rsi_entry_0 crosses_below 30"],
            "short": ["rsi_entry_0 crosses_below 30"],
        },
        "exit_conditions": {
            "long": ["rsi_exit_0 crosses_above 60"],
            "short": ["rsi_exit_0 crosses_above 60"],
        },
        "stop_loss": {"type": "fixed_pct", "percent": 4.0},
        "take_profit": {"type": "fixed_pct", "percent": 8.0},
    }


# ---------------------------------------------------------------------------
# the real engine seam
# ---------------------------------------------------------------------------


def _screen_dsl(dsl: dict[str, Any]) -> tuple[BacktestMetrics, dict[str, int]]:
    """Screen ``dsl`` on BTCUSDT 4h and count its trades by side.

    The side split is read from the engine's own positions (``all_positions`` +
    ``position_direction``) inside the run, so metrics and counts come from ONE run.
    """
    counts = {"LONG": 0, "SHORT": 0}
    runner = NTScreeningRunner(
        dsl, [_SYMBOL], _START, _END, catalog_path=str(_CATALOG), funding_archive_path=None
    )
    with _counting_sides(counts):
        metrics = runner({})
    # sum(counts) == the reported total: the split is that run's trade list
    assert sum(counts.values()) == metrics.total_trades
    return metrics, counts


def _screen_chromosome(chrom: StrategyChromosome) -> tuple[BacktestMetrics, dict[str, int]]:
    dsl = chromosome_to_dsl(chrom)
    dsl["timeframe"] = _TIMEFRAME
    return _screen_dsl(dsl)


def _assert_metrics_equal(left: BacktestMetrics, right: BacktestMetrics) -> None:
    for field in _METRICS:
        assert getattr(left, field) == getattr(right, field), field


def _dsl_without_name(dsl: dict[str, object]) -> dict[str, object]:
    """Every DSL key but ``name``.

    ``chromosome_to_serializable`` stores no uid, so a restored chromosome draws a fresh
    one and the rebuilt DSL's ``name`` is ``genome_<new_uid>`` (chief ruling): the replay
    comparison is exact on every other key, name aside.
    """
    return {key: value for key, value in dsl.items() if key != "name"}


# ---------------------------------------------------------------------------
# R23: BOTH trades its own short side
# ---------------------------------------------------------------------------


def test_r23_both_with_a_dead_long_side_trades_the_short_rule() -> None:
    """R23: BOTH(dead long, short=rule, sl_short=sl) == SHORT(rule) on all five, long 0."""
    _skip_without_catalog()
    both_metrics, both_counts = _screen_chromosome(_dead_long_short_rule_both_chromosome())
    short_metrics, short_counts = _screen_chromosome(_short_rule_chromosome())

    # The window must fire the short rule both ways (less than 3 means: change the window).
    assert short_counts["LONG"] == 0
    assert short_counts["SHORT"] >= 3
    assert both_counts["LONG"] == 0
    assert both_counts["SHORT"] == short_metrics.total_trades
    _assert_metrics_equal(both_metrics, short_metrics)


# ---------------------------------------------------------------------------
# R24: the same-bar tie stays long-first
# ---------------------------------------------------------------------------


def test_r24_identical_sides_tie_resolves_long() -> None:
    """R24: BOTH(identical non-trivial rule both sides) == LONG(rule); no short trade."""
    _skip_without_catalog()
    both_metrics, both_counts = _screen_chromosome(_board_both_chromosome())
    long_metrics, long_counts = _screen_chromosome(_board_long_chromosome())

    assert long_counts["LONG"] >= 3 and long_counts["SHORT"] == 0
    assert both_counts["SHORT"] == 0
    assert both_counts["LONG"] == long_metrics.total_trades
    _assert_metrics_equal(both_metrics, long_metrics)


# ---------------------------------------------------------------------------
# R26: within-run replay of every exported entry
# ---------------------------------------------------------------------------


def test_r26_exported_dsl_replays_from_the_stored_chromosome() -> None:
    """R26: a mock-backtest BOTH run's entries: dsl == chromosome_to_dsl(round trip)."""
    cfg = _make_config(
        direction="both", population_size=10, max_generations=4, top_k=3, elite_count=2
    )
    state = random.getstate()  # the GA draws from the module RNG; restore it, never leak it
    random.seed(20261018)
    try:
        result = DiscoveryPipeline(cfg, _mock_backtest).run()
    finally:
        random.setstate(state)

    assert result.top_strategies, "mock BOTH run found no champion"
    for chrom, fitness in result.top_strategies:
        assert chrom.direction is Direction.BOTH
        assert chrom.short_entry_genes and chrom.short_exit_genes
        entry = _champion_entry(chrom, fitness, timeframe=cfg.timeframe, dsr=None, buy_hold=None)
        rebuilt = chromosome_to_dsl(serializable_to_chromosome(entry["chromosome"]))
        rebuilt["timeframe"] = cfg.timeframe
        exported = entry["dsl"]
        assert isinstance(exported, dict) and isinstance(rebuilt, dict)
        # name is genome_<uid> and uid is not serialized (chief ruling), so it is the ONLY
        # key allowed to differ; every other key must match exactly.
        assert exported["name"] == f"genome_{chrom.uid}"
        assert _dsl_without_name(exported) == _dsl_without_name(rebuilt)
        differing = {key for key in set(exported) | set(rebuilt) if exported.get(key) != rebuilt.get(key)}
        assert differing == {"name"}, differing
        # teeth: the BOTH genome's own short side survives the replay, not the long one
        assert exported["entry_conditions"]["short"] == rebuilt["entry_conditions"]["short"]  # type: ignore[index]
        assert exported["exit_conditions"]["short"] == rebuilt["exit_conditions"]["short"]  # type: ignore[index]
        assert exported["entry_conditions"]["short"] != exported["entry_conditions"]["long"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# R26b: an OLD stored both champion replays long-only
# ---------------------------------------------------------------------------


def test_r26b_old_stored_both_dsl_still_replays_long_only() -> None:
    """R26b: the stored both DSL (identical sides) still compiles and trades like LONG."""
    _skip_without_catalog()
    stored = _old_stored_both_dsl()
    # Still compiles: the stored DSL is replayed as stored, never re-derived.
    validated = validate_strategy_dict(stored)
    assert validated.entry_conditions.long and validated.entry_conditions.short

    stored_metrics, stored_counts = _screen_dsl(dict(stored))
    long_metrics, _ = _screen_chromosome(_board_long_chromosome())

    assert stored_counts["LONG"] >= 3
    assert stored_counts["SHORT"] == 0
    assert stored_counts["LONG"] == long_metrics.total_trades
    _assert_metrics_equal(stored_metrics, long_metrics)
