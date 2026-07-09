"""Validation-vs-screening consistency check (vibe-quant-o11tp).

A screening champion that collapses under realistic fills (latency +
FillModel) is an overfit promotion — e.g. Batch 41's genome_6509e60a4ea9
went from screening Sharpe 5.40 to validation Sharpe -2.78. Nothing warned
the operator. After each validation run this module compares the result
against the strategy's screening reference and records flags.

Reference lookup order:
1. Latest completed standalone screening run for the strategy_id
   (``sweep_results`` row).
2. The discovery run's persisted champion metrics, matched by generated
   strategy name (``genome_<uid>``) in the discovery notes payload —
   discovery-exported strategies usually have no standalone screening run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager

logger = logging.getLogger(__name__)

# val Sharpe below this fraction of screening Sharpe counts as a collapse
COLLAPSE_RATIO = 0.5
# relative trade-count divergence beyond this raises a distinct warning
TRADE_DIVERGENCE = 0.10


@dataclass
class ScreeningReference:
    """Screening-tier metrics a validation run is compared against."""

    sharpe: float
    trades: int
    source: str  # e.g. "screening_run:842" or "discovery_run:848"


@dataclass
class ConsistencyReport:
    """Outcome of the validation-vs-screening comparison."""

    reference: ScreeningReference
    val_sharpe: float
    val_trades: int
    flags: list[str] = field(default_factory=list)

    @property
    def is_flagged(self) -> bool:
        return bool(self.flags)

    def to_dict(self) -> dict[str, object]:
        return {
            "flags": self.flags,
            "screen_sharpe": self.reference.sharpe,
            "screen_trades": self.reference.trades,
            "screen_source": self.reference.source,
            "val_sharpe": self.val_sharpe,
            "val_trades": self.val_trades,
        }


def assess_consistency(
    reference: ScreeningReference,
    val_sharpe: float,
    val_trades: int,
) -> ConsistencyReport:
    """Compare validation metrics against the screening reference."""
    report = ConsistencyReport(
        reference=reference, val_sharpe=val_sharpe, val_trades=val_trades
    )

    if reference.sharpe > 0 and val_sharpe < 0:
        report.flags.append(
            f"validation-collapse: Sharpe sign flip "
            f"({reference.sharpe:.2f} screening → {val_sharpe:.2f} validation)"
        )
    elif reference.sharpe > 0 and val_sharpe < COLLAPSE_RATIO * reference.sharpe:
        report.flags.append(
            f"validation-collapse: Sharpe {val_sharpe:.2f} is below "
            f"{COLLAPSE_RATIO:.0%} of screening {reference.sharpe:.2f}"
        )

    if reference.trades > 0:
        divergence = abs(val_trades - reference.trades) / reference.trades
        if divergence > TRADE_DIVERGENCE:
            report.flags.append(
                f"trade-count-divergence: {reference.trades} screening → "
                f"{val_trades} validation ({divergence:.0%} > {TRADE_DIVERGENCE:.0%})"
            )

    return report


def find_screening_reference(
    state: StateManager, strategy_id: int, strategy_name: str
) -> ScreeningReference | None:
    """Locate screening-tier metrics for a strategy, if any exist."""
    row = state.conn.execute(
        """
        SELECT br.id, sr.sharpe_ratio, sr.total_trades
        FROM backtest_runs br
        JOIN sweep_results sr ON sr.run_id = br.id
        WHERE br.strategy_id = ? AND br.run_mode = 'screening'
              AND br.status = 'completed'
        ORDER BY br.id DESC LIMIT 1
        """,
        (strategy_id,),
    ).fetchone()
    if row is not None and row[1] is not None:
        return ScreeningReference(
            sharpe=float(row[1]),
            trades=int(row[2] or 0),
            source=f"screening_run:{row[0]}",
        )

    return _reference_from_discovery_notes(state, strategy_name)


def _reference_from_discovery_notes(
    state: StateManager, strategy_name: str
) -> ScreeningReference | None:
    """Match a discovery-exported strategy back to its champion metrics."""
    rows = state.conn.execute(
        """
        SELECT br.id, res.notes
        FROM backtest_runs br
        JOIN backtest_results res ON res.run_id = br.id
        WHERE br.run_mode = 'discovery' AND br.status = 'completed'
              AND res.notes LIKE ?
        ORDER BY br.id DESC LIMIT 5
        """,
        (f'%{strategy_name.removeprefix("genome_")}%',),
    ).fetchall()
    for run_id, notes in rows:
        try:
            payload = json.loads(notes)
        except (TypeError, json.JSONDecodeError):
            continue
        strategies = payload.get("top_strategies")
        if not isinstance(strategies, list):
            continue
        for entry in strategies:
            if not isinstance(entry, dict):
                continue
            dsl = entry.get("dsl")
            name = dsl.get("name") if isinstance(dsl, dict) else None
            if name != strategy_name:
                continue
            sharpe = entry.get("sharpe")
            trades = entry.get("trades")
            if isinstance(sharpe, (int, float)) and isinstance(trades, (int, float)):
                return ScreeningReference(
                    sharpe=float(sharpe),
                    trades=int(trades),
                    source=f"discovery_run:{run_id}",
                )
    return None
