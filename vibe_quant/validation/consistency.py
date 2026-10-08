"""Validation-vs-screening consistency check (vibe-quant-o11tp).

A screening champion that collapses under realistic fills (latency +
FillModel) is an overfit promotion — e.g. Batch 41's genome_6509e60a4ea9
went from screening Sharpe 5.40 to validation Sharpe -2.78. Nothing warned
the operator. After each validation run this module compares the result
against the strategy's screening reference and records flags.

Reference lookup order:
1. Latest completed standalone screening run for the strategy_id: the
   ``sweep_results`` row whose parameters match the validated parameters,
   else the best finite-Sharpe row with trades > 0. (An arbitrary
   ``LIMIT 1`` row used to be picked -- e.g. sharpe=-inf / 0 trades -- so the
   collapse flag could never fire; bd vibe-quant-e70tl.15.)
2. The discovery run's persisted champion metrics, matched by generated
   strategy name (``genome_<uid>``) in the discovery notes payload —
   discovery-exported strategies usually have no standalone screening run.
   The continuous full-range headline (``full_range_*``) is preferred over
   the multi-window GA aggregate.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date
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
    # Window the reference metrics cover (ISO dates); None if unknown.
    start_date: str | None = None
    end_date: str | None = None
    # symbol_agg="worst" champions: trades are not portfolio-comparable unless
    # the persisted trades_sum was used -- then the comparison is skipped.
    skip_trades: bool = False
    notes: list[str] = field(default_factory=list)


def _worst_mode_trades(
    payload: dict[str, object],
    symbols: object,
    block: dict[str, object],
    sum_key: str,
    trades: int,
) -> tuple[int, bool, list[str]]:
    """(trades, skip_trades, notes) for a discovery reference.

    In ``symbol_agg=worst`` multi-symbol runs the persisted ``trades`` is the
    MIN symbol while validation is a portfolio run, so compare ``trades_sum``;
    legacy notes without it skip the trade comparison.
    """
    if payload.get("symbol_agg") != "worst":
        return trades, False, []
    try:
        n_symbols = len(json.loads(symbols) if isinstance(symbols, str) else symbols)  # type: ignore[arg-type]
    except (TypeError, json.JSONDecodeError):
        n_symbols = 2
    if n_symbols < 2:
        return trades, False, []
    notes = ["reference Sharpe = worst symbol (discovery symbol_agg=worst)"]
    total = block.get(sum_key)
    if isinstance(total, (int, float)):
        return int(total), False, notes
    notes.append(
        "trade comparison skipped: reference trades are the MIN per symbol "
        "(no trades_sum persisted) while validation is a portfolio run"
    )
    return trades, True, notes


def _window_days(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    try:
        days = (date.fromisoformat(end[:10]) - date.fromisoformat(start[:10])).days
    except ValueError:
        return None
    return days if days > 0 else None


@dataclass
class ConsistencyReport:
    """Outcome of the validation-vs-screening comparison."""

    reference: ScreeningReference
    val_sharpe: float
    val_trades: int
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

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
            "notes": self.notes,
        }


def assess_consistency(
    reference: ScreeningReference,
    val_sharpe: float,
    val_trades: int,
    val_window: tuple[str, str] | None = None,
) -> ConsistencyReport:
    """Compare validation metrics against the screening reference.

    Trade counts are only comparable over the same window. When both windows
    are known and differ (e.g. a discovery champion's full-range count vs a
    holdout-only validation), trades are compared as per-day rates.
    """
    report = ConsistencyReport(
        reference=reference, val_sharpe=val_sharpe, val_trades=val_trades,
        notes=list(reference.notes),
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

    ref_days = _window_days(reference.start_date, reference.end_date)
    val_days = _window_days(*val_window) if val_window else None
    if reference.skip_trades:
        pass
    elif reference.trades > 0 and ref_days and val_days and ref_days != val_days:
        ref_rate = reference.trades / ref_days
        val_rate = val_trades / val_days
        divergence = abs(val_rate - ref_rate) / ref_rate
        if divergence > TRADE_DIVERGENCE:
            report.flags.append(
                f"trade-rate-divergence: {ref_rate:.3f}/day screening "
                f"({reference.trades} over {ref_days}d) → {val_rate:.3f}/day validation "
                f"({val_trades} over {val_days}d) ({divergence:.0%} > {TRADE_DIVERGENCE:.0%})"
            )
    elif reference.trades > 0:
        divergence = abs(val_trades - reference.trades) / reference.trades
        if divergence > TRADE_DIVERGENCE:
            report.flags.append(
                f"trade-count-divergence: {reference.trades} screening → "
                f"{val_trades} validation ({divergence:.0%} > {TRADE_DIVERGENCE:.0%})"
            )

    return report


# Run-level launch knobs stored next to strategy params in run parameters
_NON_STRATEGY_PARAM_KEYS = frozenset(
    {"sweep", "overfitting_filters", "initial_balance", "leverage", "detail_timeframe"}
)


def _normalize_params(params: object) -> dict[str, float | str] | None:
    """Comparable form: dot/underscore-insensitive keys, numeric values as float."""
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return None
    if not isinstance(params, dict):
        return None
    normalized: dict[str, float | str] = {}
    for key, value in params.items():
        if key in _NON_STRATEGY_PARAM_KEYS:
            continue
        norm_key = str(key).replace(".", "_")
        if isinstance(value, bool):
            normalized[norm_key] = str(value)
        elif isinstance(value, (int, float)):
            normalized[norm_key] = float(value)
        else:
            normalized[norm_key] = str(value)
    return normalized


def find_screening_reference(
    state: StateManager,
    strategy_id: int,
    strategy_name: str,
    validated_params: dict[str, object] | None = None,
    val_window: tuple[str, str] | None = None,
) -> ScreeningReference | None:
    """Locate screening-tier metrics for a strategy, if any exist.

    Args:
        validated_params: Strategy parameters the validation run used; the
            screening row with the same parameters is the like-for-like
            reference.
        val_window: The validation run's (start, end) dates. A validation on
            a discovery run's holdout window is compared against the
            champion's holdout metrics (same window).
    """
    run_row = state.conn.execute(
        """
        SELECT br.id, br.start_date, br.end_date FROM backtest_runs br
        WHERE br.strategy_id = ? AND br.run_mode = 'screening'
              AND br.status = 'completed'
              AND EXISTS (SELECT 1 FROM sweep_results sr WHERE sr.run_id = br.id)
        ORDER BY br.id DESC LIMIT 1
        """,
        (strategy_id,),
    ).fetchone()
    if run_row is not None:
        rows = state.conn.execute(
            "SELECT parameters, sharpe_ratio, total_trades FROM sweep_results "
            "WHERE run_id = ? ORDER BY id",
            (run_row[0],),
        ).fetchall()
        usable = [
            (params, float(sharpe), int(trades or 0))
            for params, sharpe, trades in rows
            if sharpe is not None and math.isfinite(float(sharpe)) and int(trades or 0) > 0
        ]
        wanted = _normalize_params(validated_params or {})
        for params, sharpe, trades in usable:
            if wanted is not None and _normalize_params(params) == wanted:
                return ScreeningReference(
                    sharpe=sharpe,
                    trades=trades,
                    source=f"screening_run:{run_row[0]}",
                    start_date=run_row[1],
                    end_date=run_row[2],
                )
        if usable:
            _, sharpe, trades = max(usable, key=lambda r: r[1])
            return ScreeningReference(
                sharpe=sharpe,
                trades=trades,
                source=f"screening_run:{run_row[0]}:best",
                start_date=run_row[1],
                end_date=run_row[2],
            )

    return _reference_from_discovery_notes(
        state, strategy_name, val_window, _strategy_dsl(state, strategy_id)
    )


def _dsl_body(dsl: dict[str, object]) -> str:
    """Canonical JSON of a DSL minus ``name`` (same rule as discovery promote)."""
    body = {k: v for k, v in dsl.items() if k != "name"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)


def _strategy_dsl(state: StateManager, strategy_id: int) -> dict[str, object] | None:
    row = state.conn.execute(
        "SELECT dsl_config FROM strategies WHERE id = ?", (strategy_id,)
    ).fetchone()
    if row is None:
        return None
    try:
        dsl = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    except (TypeError, json.JSONDecodeError):
        return None
    return dsl if isinstance(dsl, dict) else None


def _reference_from_discovery_notes(
    state: StateManager,
    strategy_name: str,
    val_window: tuple[str, str] | None = None,
    strategy_dsl: dict[str, object] | None = None,
) -> ScreeningReference | None:
    """Match a discovery-exported strategy back to its champion metrics.

    Seeded runs reproduce identical genome names and promote renames a clashing
    strategy (``genome_x_2``), so when the validated strategy's DSL is known the
    champion is matched purely by DSL body (name ignored). With no DSL, falls
    back to name equality. Runs are scanned newest-first with no cutoff; the
    first match wins.
    """
    wanted_body = _dsl_body(strategy_dsl) if strategy_dsl else None
    sql = """
        SELECT br.id, res.notes, br.start_date, br.end_date, br.symbols
        FROM backtest_runs br
        JOIN backtest_results res ON res.run_id = br.id
        WHERE br.run_mode = 'discovery' AND br.status = 'completed'
    """
    params: tuple[str, ...] = ()
    if wanted_body is None:
        sql += " AND res.notes LIKE ?"
        params = (f'%{strategy_name.removeprefix("genome_")}%',)
    # DSL-known path scans every completed run: no sound SQL prefilter exists
    # for a renamed strategy (hundreds of runs; JSON parse stops at first hit).
    rows = state.conn.execute(sql + " ORDER BY br.id DESC", params)
    for run_id, notes, run_start, run_end, run_symbols in rows:
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
            if not isinstance(dsl, dict):
                continue
            if wanted_body is not None:
                if _dsl_body(dsl) != wanted_body:
                    continue
            elif dsl.get("name") != strategy_name:
                continue
            # Validated on the holdout window → compare like for like.
            holdout = entry.get("holdout")
            holdout_dates = payload.get("holdout_dates")
            if (
                val_window is not None
                and isinstance(holdout, dict)
                and isinstance(holdout_dates, list)
                and len(holdout_dates) == 2
                and [d[:10] for d in val_window] == [str(d)[:10] for d in holdout_dates]
                and isinstance(holdout.get("sharpe"), (int, float))
                and isinstance(holdout.get("trades"), (int, float))
            ):
                h_trades, h_skip, h_notes = _worst_mode_trades(
                    payload, run_symbols, holdout, "trades_sum", int(holdout["trades"])
                )
                return ScreeningReference(
                    sharpe=float(holdout["sharpe"]),
                    trades=h_trades,
                    source=f"discovery_run:{run_id}:holdout",
                    start_date=str(holdout_dates[0]),
                    end_date=str(holdout_dates[1]),
                    skip_trades=h_skip,
                    notes=h_notes,
                )
            # Continuous full-range headline when persisted (bd rewru); the
            # GA aggregate is a mean over sub-windows.
            sharpe = entry.get("full_range_sharpe", entry.get("sharpe"))
            trades = entry.get("full_range_trades", entry.get("trades"))
            if isinstance(sharpe, (int, float)) and isinstance(trades, (int, float)):
                f_trades, f_skip, f_notes = _worst_mode_trades(
                    payload, run_symbols, entry, "full_range_trades_sum", int(trades)
                )
                return ScreeningReference(
                    sharpe=float(sharpe),
                    trades=f_trades,
                    source=f"discovery_run:{run_id}",
                    start_date=run_start,
                    end_date=run_end,
                    skip_trades=f_skip,
                    notes=f_notes,
                )
    return None
