"""Shared loader for persisted warm-start seed chromosomes.

Both the discovery API launch router and the discovery CLI subprocess must
agree on whether a given seed run can supply warm-start chromosomes; keeping
the loader here (rather than private to ``__main__``) makes that a single
source of truth.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from vibe_quant.discovery.genome import serializable_to_chromosome

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.operators import StrategyChromosome

logger = logging.getLogger(__name__)


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
        data = json.loads(notes)
        strategies = data.get("top_strategies", [])
        chromosomes: list[StrategyChromosome] = []
        for entry in strategies:
            chrom_data = entry.get("chromosome")
            if chrom_data and isinstance(chrom_data, dict):
                chromosomes.append(serializable_to_chromosome(chrom_data))
        return chromosomes if chromosomes else None
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        logger.warning("Failed to load seed chromosomes from run %d", run_id, exc_info=True)
        return None
