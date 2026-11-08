"""Shared loader for persisted warm-start seed chromosomes.

Both the discovery API launch router and the discovery CLI subprocess must
agree on whether a given seed run can supply warm-start chromosomes; keeping
the loader here (rather than private to ``__main__``) makes that a single
source of truth.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, overload

from vibe_quant.db.result_notes import parse_notes
from vibe_quant.discovery.genome import (
    SeedIncompatibleError,
    genome_kind,
    serializable_to_genome,
)
from vibe_quant.discovery.operators import Direction

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.genome import Genome
    from vibe_quant.discovery.operators import StrategyChromosome
    from vibe_quant.discovery.param_genome import ParamGenome
    from vibe_quant.discovery.port import Port

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


def _require_compatible(chrom_data: dict[str, object], port: Port | None) -> None:
    """Refuse a stored seed that cannot join this run (R32).

    Legacy seeds go to legacy runs (``port is None``); param seeds go to a run of
    the SAME port file revision. Anything else raises ``SeedIncompatibleError``.
    """
    stored_kind = chrom_data.get("kind", "legacy")
    if port is None:
        if stored_kind != "legacy":
            msg = "seed run holds param genomes (a --port run); pass --port to warm-start from it"
            raise SeedIncompatibleError(msg)
        return
    if stored_kind == "legacy":
        msg = f"seed run holds legacy chromosomes; a run of port '{port.name}' needs param genomes"
        raise SeedIncompatibleError(msg)
    meta = chrom_data.get("port")
    stored_name = meta.get("name") if isinstance(meta, dict) else None
    stored_sha = meta.get("sha256") if isinstance(meta, dict) else None
    if stored_name != port.name:
        msg = f"seed genome belongs to port {stored_name!r}, not {port.name!r}"
        raise SeedIncompatibleError(msg)
    if stored_sha != port.sha256:
        msg = f"seed genome was produced by a different revision of port '{port.name}'"
        raise SeedIncompatibleError(msg)


@overload
def load_seed_chromosomes(
    state: StateManager, run_id: int, port: None = None
) -> list[StrategyChromosome] | None: ...


@overload
def load_seed_chromosomes(
    state: StateManager, run_id: int, port: Port
) -> list[ParamGenome] | None: ...


def load_seed_chromosomes(
    state: StateManager,
    run_id: int,
    port: Port | None = None,
) -> list[StrategyChromosome] | list[ParamGenome] | None:
    """Load top chromosomes from a prior discovery run for warm-starting.

    ``port`` is the run's port (None = legacy run). A seed run of the other kind,
    another port, or another revision of the port file raises
    ``SeedIncompatibleError`` (not swallowed like an unreadable run).

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
        data = parse_notes(notes)
        if not isinstance(data, dict):
            raise TypeError("notes are not an object")
        strategies = data.get("top_strategies", [])
        if not isinstance(strategies, list) or not all(isinstance(e, dict) for e in strategies):
            raise TypeError("top_strategies is not a list of objects")
        chromosomes: list[Genome] = []
        converted = 0
        for entry in strategies:
            chrom_data = entry.get("chromosome")
            if chrom_data and isinstance(chrom_data, dict):
                _require_compatible(chrom_data, port)
                chrom = serializable_to_genome(chrom_data, port)
                # Repair is a legacy fix (R7/R31): a param genome is loaded as stored.
                if genome_kind(chrom) == "legacy" and _is_legacy_both(chrom):  # type: ignore[arg-type]
                    chrom = _legacy_both_to_long(chrom)  # type: ignore[arg-type]
                    converted += 1
                chromosomes.append(chrom)
        if converted:
            # One summary warning per seed load, not per seed (R7d).
            logger.warning(
                "Warm-start: loaded %d legacy direction=both seed chromosome(s) as long",
                converted,
            )
        return chromosomes if chromosomes else None  # type: ignore[return-value]
    except SeedIncompatibleError:
        raise
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        logger.warning("Failed to load seed chromosomes from run %d", run_id, exc_info=True)
        return None
