"""Shared loader for persisted warm-start seed chromosomes.

Both the discovery API launch router and the discovery CLI subprocess must
agree on whether a given seed run can supply warm-start chromosomes; keeping
the loader here (rather than private to ``__main__``) makes that a single
source of truth.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

from vibe_quant.db.result_notes import parse_notes
from vibe_quant.discovery.genome import serializable_to_chromosome
from vibe_quant.discovery.operators import Direction

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.operators import StrategyChromosome

logger = logging.getLogger(__name__)


def _is_legacy_both(chrom: StrategyChromosome) -> bool:
    """A pre-inydc direction=both seed, which carries no short-side genes (R7)."""
    return (
        chrom.direction == Direction.BOTH
        and not chrom.short_entry_genes
        and not chrom.short_exit_genes
    )


def _legacy_both_to_long(chrom: StrategyChromosome) -> StrategyChromosome:
    """Load a legacy direction=both seed as long — what it actually traded (R7).

    The compiler checks long before short, so a pre-inydc both chromosome was
    long-only. The single stop/take-profit it really used is the long override
    when set, else the generic value (R7b); the per-direction fields are cleared.
    """
    chrom.direction = Direction.LONG
    if chrom.stop_loss_long_pct is not None:
        chrom.stop_loss_pct = chrom.stop_loss_long_pct
    if chrom.take_profit_long_pct is not None:
        chrom.take_profit_pct = chrom.take_profit_long_pct
    chrom.stop_loss_long_pct = None
    chrom.stop_loss_short_pct = None
    chrom.take_profit_long_pct = None
    chrom.take_profit_short_pct = None
    return chrom


def load_seed_chromosomes(
    state: StateManager,
    run_id: int,
) -> list[StrategyChromosome] | None:
    """Load top chromosomes from a prior discovery run for warm-starting.

    Reads the ``chromosome`` field from stored ``top_strategies`` if available.
    Returns ``None`` when the run has no result row, no (or non-string) notes,
    or no ``top_strategies`` entry carrying a usable ``chromosome`` dict. The
    caller decides whether an explicit warm-start request that cannot be
    honored is an operator error (the launch API refuses it) or a runtime
    backstop (the CLI subprocess raises).
    """
    result = state.get_backtest_result(run_id)
    if result is None:
        return None
    notes = result.get("notes", "")
    if not notes or not isinstance(notes, str):
        return None
    try:
        data = cast("dict[str, object]", parse_notes(notes))
        strategies = cast("list[dict[str, object]]", data.get("top_strategies", []))
        chromosomes: list[StrategyChromosome] = []
        converted = 0
        for entry in strategies:
            chrom_data = entry.get("chromosome")
            if chrom_data and isinstance(chrom_data, dict):
                chrom = serializable_to_chromosome(chrom_data)
                if _is_legacy_both(chrom):
                    chrom = _legacy_both_to_long(chrom)
                    converted += 1
                chromosomes.append(chrom)
        if converted:
            # One summary warning per seed load, not per seed (R7d).
            logger.warning(
                "Warm-start: loaded %d legacy direction=both seed chromosome(s) as long",
                converted,
            )
        return chromosomes if chromosomes else None
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        logger.warning("Failed to load seed chromosomes from run %d", run_id, exc_info=True)
        return None
