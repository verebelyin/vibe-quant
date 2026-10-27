"""Validation-vs-screening consistency check (vibe-quant-o11tp).

A screening champion that collapses under realistic fills (latency +
FillModel) is an overfit promotion — e.g. Batch 41's genome_6509e60a4ea9
went from screening Sharpe 5.40 to validation Sharpe -2.78. Nothing warned
the operator. After each validation run this module compares the result
against the strategy's screening reference and records flags.

Reference lookup order:
1. Latest completed standalone screening run for the strategy_id that executed
   the SAME effective DSL (``parameters.dsl_override`` when it carried one, else
   the stored DSL; compared as canonical bodies): the
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
from typing import TYPE_CHECKING, cast

from vibe_quant.db.result_notes import parse_notes
from vibe_quant.dsl.identity import dsl_body_key
from vibe_quant.sizing_settings import SIZING_FIELDS, SizingConfigError, recorded_sizing

if TYPE_CHECKING:
    from collections.abc import Mapping

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


def refused_reference_report(
    refused_run_ids: list[int], val_sharpe: float, val_trades: int
) -> dict[str, object]:
    """Notes payload for a validation that had screening runs but none it may compare.

    Every candidate was refused (another effective DSL, or unreadable parameters), so no
    screening metric appears here -- only the flag, which reads as a failure to anything
    that treats flags as one.
    """
    ids = ", ".join(str(i) for i in refused_run_ids)
    return {
        "flags": [
            f"reference-refused: {len(refused_run_ids)} screening run(s) of this strategy ran "
            f"a different effective DSL or unreadable parameters (ids {ids}); "
            "metrics not compared"
        ],
        "screen_sharpe": None,
        "screen_trades": None,
        "screen_source": None,
        "val_sharpe": val_sharpe,
        "val_trades": val_trades,
        "notes": [],
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


# Run-level launch knobs stored next to strategy params in run parameters.
# The sizing settings are run-level too: they never appear in a sweep row, so leaving them
# in a validation run's params would stop the exact-row match from ever succeeding (S28).
# ``dsl_override`` is the DSL the run executed, not a parameter of it.
_NON_STRATEGY_PARAM_KEYS = frozenset(
    {
        "sweep",
        "overfitting_filters",
        "initial_balance",
        "leverage",
        "detail_timeframe",
        "dsl_override",
        *SIZING_FIELDS,
    }
)


def _discovery_recorded_sizing(payload: Mapping[str, object]) -> dict[str, float] | None:
    """The sizing a discovery run recorded, or None when it is unreadable (S29c/S29d).

    A run whose notes carry no ``sizing`` key at all was launched at the defaults
    (``{}``). A run whose ``sizing`` key is PRESENT but not an object (``null``, a list,
    a string, a number) or holds an invalid value is unreadable: it must be skipped rather
    than compared at the defaults, consistent with discovery's own promote/replay rule (S23b).
    """
    if "sizing" not in payload:
        return {}
    raw = payload["sizing"]
    if not isinstance(raw, dict):
        return None
    try:
        return recorded_sizing(raw).to_params()
    except SizingConfigError:
        return None


def _resolved_param_key(key: str, config_fields: tuple[str, ...] | None) -> str:
    """The compiled config field ``key`` names, else its dot-folded spelling.

    Screening's ONE resolver (:func:`~vibe_quant.screening.nt_runner.resolve_strategy_params`)
    owns dot notation and the alias table, so the consistency check must ask it rather than
    keep a copy: ``take_profit.risk_reward_ratio`` and ``take_profit_risk_reward`` are the
    same parameter (vibe-quant-xh4xn).
    """
    if config_fields:
        from vibe_quant.screening.nt_runner import (
            UnknownStrategyParamError,
            resolve_strategy_params,
        )

        try:
            resolved = resolve_strategy_params({key: None}, config_fields)
        except UnknownStrategyParamError:
            logger.debug(
                "Parameter %r names no compiled config field; comparing it by spelling", key
            )
        else:
            return next(iter(resolved))
    return key.replace(".", "_")


def _normalize_params(
    params: object,
    config_fields: tuple[str, ...] | None = None,
    ambiguities: list[str] | None = None,
) -> dict[str, float | str] | None:
    """Comparable form: keys resolved to config fields, numeric values as float.

    Two keys naming the same field make the row ambiguous: None is returned instead of
    silently keeping one of them, and the reason (naming both keys) is appended to
    ``ambiguities`` so the consistency report can say why.
    """
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return None
    if not isinstance(params, dict):
        return None
    normalized: dict[str, float | str] = {}
    source_key: dict[str, str] = {}
    for key, value in params.items():
        if key in _NON_STRATEGY_PARAM_KEYS:
            continue
        norm_key = _resolved_param_key(str(key), config_fields)
        if norm_key in normalized:
            reason = (
                f"ambiguous parameters: {source_key[norm_key]!r} and {str(key)!r} "
                f"both name {norm_key!r}"
            )
            logger.warning("Ambiguous screening parameters: %s; skipping the exact-row match", reason)
            if ambiguities is not None and reason not in ambiguities:
                ambiguities.append(reason)
            return None
        if isinstance(value, bool):
            normalized[norm_key] = str(value)
        elif isinstance(value, (int, float)):
            normalized[norm_key] = float(value)
        else:
            normalized[norm_key] = str(value)
        source_key[norm_key] = str(key)
    return normalized


def find_screening_reference(
    state: StateManager,
    strategy_id: int,
    strategy_name: str,
    validated_params: dict[str, object] | None = None,
    val_window: tuple[str, str] | None = None,
    val_symbols: list[str] | None = None,
    val_sizing: Mapping[str, float] | None = None,
) -> ScreeningReference | None:
    """The reference of :func:`lookup_screening_reference`, without its refusals."""
    return lookup_screening_reference(
        state, strategy_id, strategy_name, validated_params, val_window, val_symbols, val_sizing
    )[0]


def lookup_screening_reference(
    state: StateManager,
    strategy_id: int,
    strategy_name: str,
    validated_params: dict[str, object] | None = None,
    val_window: tuple[str, str] | None = None,
    val_symbols: list[str] | None = None,
    val_sizing: Mapping[str, float] | None = None,
) -> tuple[ScreeningReference | None, list[int]]:
    """Locate screening-tier metrics for a strategy, if any exist.

    Returns ``(reference, refused_run_ids)``: the ids of candidate screening runs refused
    (another effective DSL, or unreadable parameters) so the caller can record that the
    comparison was not made instead of passing silently.

    Args:
        validated_params: Strategy parameters the validation run used; the
            screening row with the same parameters is the like-for-like
            reference. Its ``dsl_override``, when present, is the DSL this run
            executed instead of the strategy's stored one.
        val_window: The validation run's (start, end) dates. A validation on
            a discovery run's holdout window is compared against the
            champion's holdout metrics (same window).
        val_symbols: The validation run's symbols. Together with
            ``val_window`` it restricts the reference to a screening run over
            the same window and symbol set — a Sharpe from another window or
            portfolio is not a like-for-like reference.
        val_sizing: The validation run's sizing params (``to_params()`` of its
            settings). A reference must have recorded the same sizing: the same
            strategy at a different exposure cap is a different policy. ``None``
            means no sizing filter at all.

    A screening run that executed a different effective DSL (it launched with
    ``parameters.dsl_override``) is refused, with the reason logged: its rows are another
    strategy's, so neither an exact row nor its best-Sharpe row may be the reference
    (vibe-quant-e2af5). With no run of the same DSL the lookup returns None rather than
    comparing across strategies.
    """
    override = _dsl_override(validated_params)
    stored_dsl = _strategy_dsl(state, strategy_id)
    executed_dsl = override if override is not None else stored_dsl
    wanted_identity = _dsl_identity(executed_dsl, strategy_name)
    fields = _config_fields(executed_dsl, strategy_name)
    ambiguities: list[str] = []
    wanted = _normalize_params(validated_params or {}, fields, ambiguities)

    rejected: list[str] = []
    refused_ids: list[int] = []
    candidates, unreadable = _screening_candidates(
        state, strategy_id, val_window, val_symbols, val_sizing
    )
    for run_id in unreadable:
        reason = f"screening run {run_id} has unreadable parameters, so its effective DSL is unknown"
        logger.warning("Refusing screening reference: %s", reason)
        rejected.append(reason)
        refused_ids.append(run_id)
    for run_id, start, end, run_parameters in candidates:
        usable = [
            (params, float(sharpe), int(trades or 0))
            for params, sharpe, trades in state.list_sweep_reference_rows(run_id)
            if sharpe is not None and math.isfinite(float(sharpe)) and int(trades or 0) > 0
        ]
        if not usable:
            continue
        run_override = _dsl_override(run_parameters)
        run_identity = _dsl_identity(
            run_override if run_override is not None else stored_dsl, strategy_name
        )
        if run_identity != wanted_identity:
            reason = (
                f"screening run {run_id} executed a different effective DSL than the "
                "validation run"
            )
            logger.warning("Refusing screening reference: %s", reason)
            rejected.append(reason)
            refused_ids.append(run_id)
            continue
        for params, sharpe, trades in usable:
            if wanted is not None and _normalize_params(params, fields, ambiguities) == wanted:
                return (
                    ScreeningReference(
                        sharpe=sharpe,
                        trades=trades,
                        source=f"screening_run:{run_id}",
                        start_date=start,
                        end_date=end,
                    ),
                    refused_ids,
                )
        _, sharpe, trades = max(usable, key=lambda r: r[1])
        return (
            ScreeningReference(
                sharpe=sharpe,
                trades=trades,
                source=f"screening_run:{run_id}:best",
                start_date=start,
                end_date=end,
                notes=[
                    f"{reason}; no exact parameter row matched, reference is the best-Sharpe row"
                    for reason in ambiguities
                ],
            ),
            refused_ids,
        )

    if rejected:
        logger.warning(
            "Strategy %d has no screening reference on its own effective DSL: %s",
            strategy_id,
            "; ".join(rejected),
        )
    return (
        _reference_from_discovery_notes(state, strategy_name, val_window, executed_dsl, val_sizing),
        refused_ids,
    )


def _strategy_dsl(state: StateManager, strategy_id: int) -> dict[str, object] | None:
    raw = state.get_strategy_dsl_config_raw(strategy_id)
    if raw is None:
        return None
    try:
        dsl = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return None
    return dsl if isinstance(dsl, dict) else None


def _run_parameters(parameters: object) -> dict[str, object] | None:
    """A run row's ``parameters`` column as an object, or None when it is unreadable."""
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError:
            logger.debug("Run parameters column is not JSON; its effective DSL is unknown")
            return None
    return cast("dict[str, object]", parameters) if isinstance(parameters, dict) else None


def _dsl_override(parameters: Mapping[str, object] | None) -> dict[str, object] | None:
    """The DSL a run launched with instead of the stored one, or None.

    ``screening/__main__.py`` runs a nonempty ``parameters.dsl_override`` RAW, INSTEAD of
    the strategy's stored DSL, so that run's sweep rows describe another strategy body.
    """
    override = parameters.get("dsl_override") if parameters else None
    if isinstance(override, dict) and override:
        return cast("dict[str, object]", override)
    return None


def _dsl_identity(dsl: dict[str, object] | None, strategy_name: str) -> str | None:
    """Canonical identity of a DSL a run executes, or None when there is no DSL.

    Both sides are translated (editor DslConfig -> canonical StrategyDSL) and validated,
    so the comparison is on the canonical body (``dsl_body_key``: sorted keys, ``name``
    dropped) rather than raw text -- key order and defaulted fields must not fake a
    difference. A DSL that does not compile falls back to its raw body key, tagged so the
    two forms can never compare equal.
    """
    if dsl is None:
        return None
    from vibe_quant.dsl.schema import StrategyDSL
    from vibe_quant.dsl.translator import translate_dsl_config

    try:
        canonical = StrategyDSL.model_validate(
            translate_dsl_config(dsl, strategy_name=strategy_name)
        ).model_dump()
    except (ValueError, TypeError, KeyError, AttributeError):
        logger.debug("Strategy DSL does not compile; comparing its raw body")
        return "raw:" + dsl_body_key(dsl)
    return "canonical:" + dsl_body_key(canonical)


def _screening_candidates(
    state: StateManager,
    strategy_id: int,
    val_window: tuple[str, str] | None,
    val_symbols: list[str] | None,
    val_sizing: Mapping[str, float] | None,
) -> tuple[list[tuple[int, str, str, dict[str, object]]], list[int]]:
    """The strategy's completed screening runs, newest first, plus unreadable ones.

    ``(candidates, unreadable_ids)``. A candidate is ``(id, start_date, end_date,
    parameters)`` for every run that passes the validation's window / symbol / sizing
    filters -- the same ones ``StateManager.get_latest_screening_run_with_sweeps``
    applies, which only ever returns the newest run. The reference now has to look
    further back: the newest run may have executed a different effective DSL and be
    refused.

    The ``parameters`` column is read RAW and parsed BEFORE the sizing filter: a run whose
    column is unreadable has an unknown effective DSL, so it is reported in
    ``unreadable_ids`` (never silently dropped by the sizing filter) and the lookup goes on
    to older runs. ``list_backtest_runs`` -- which ``json.loads`` it unguarded -- cannot
    be used here.
    """
    from vibe_quant.db.state_manager import _recorded_sizing_params

    wanted_sizing = None if val_sizing is None else dict(val_sizing)
    candidates: list[tuple[int, str, str, dict[str, object]]] = []
    unreadable: list[int] = []
    runs = state.list_screening_run_references(strategy_id)
    for run_id, start, end, run_symbols, raw_parameters in runs:
        if val_window is not None and (start, end) != tuple(val_window):
            continue
        if val_symbols is not None:
            try:
                symbols = json.loads(run_symbols or "[]")
            except (TypeError, json.JSONDecodeError):
                logger.debug(
                    "Screening run %d has unreadable symbols; skipping it", run_id
                )
                continue
            if sorted(symbols) != sorted(val_symbols):
                continue
        run_parameters = _run_parameters(raw_parameters)
        if run_parameters is None:
            unreadable.append(run_id)
            continue
        if wanted_sizing is not None and (
            _recorded_sizing_params(run_parameters) != wanted_sizing
        ):
            continue
        candidates.append((run_id, start, end, run_parameters))
    return candidates, unreadable


def _config_fields(
    dsl: dict[str, object] | None, strategy_name: str
) -> tuple[str, ...] | None:
    """The executed DSL's compiled config fields, or None when it does not compile.

    ``dsl`` is the DSL the run executed -- its ``dsl_override`` when it carried one, else
    the strategy's stored DSL. Either is translated first (editor DslConfig -> canonical
    StrategyDSL, named ``strategy_name``) exactly as the screening and validation runners do.

    Best-effort like the rest of this advisory check: without fields the comparison keeps
    its dot-folding alone rather than losing the reference.
    """
    from vibe_quant.dsl.compiler import CompilerError
    from vibe_quant.dsl.parser import DSLParseError
    from vibe_quant.dsl.translator import translate_dsl_config
    from vibe_quant.screening.nt_runner import config_fields_for

    if dsl is None:
        return None
    try:
        return config_fields_for(translate_dsl_config(dsl, strategy_name=strategy_name))
    except (DSLParseError, CompilerError, ValueError, TypeError, KeyError, AttributeError):
        logger.debug(
            "Strategy %r does not compile; comparing parameters by dot-folded spelling",
            strategy_name,
        )
        return None


def _reference_from_discovery_notes(
    state: StateManager,
    strategy_name: str,
    val_window: tuple[str, str] | None = None,
    strategy_dsl: dict[str, object] | None = None,
    val_sizing: Mapping[str, float] | None = None,
) -> ScreeningReference | None:
    """Match a discovery-exported strategy back to its champion metrics.

    Seeded runs reproduce identical genome names and promote renames a clashing
    strategy (``genome_x_2``), so when the validated strategy's DSL is known the
    champion is matched purely by DSL body (name ignored). With no DSL, falls
    back to name equality. Runs are scanned newest-first with no cutoff; the
    first match wins. ``val_sizing`` skips a run whose ``notes.sizing`` is a
    different (or unreadable) sizing, like the screening-run lookup does.
    """
    wanted_body = dsl_body_key(strategy_dsl) if strategy_dsl else None
    wanted_sizing = None if val_sizing is None else dict(val_sizing)
    notes_like = (
        None if wanted_body is not None else f'%{strategy_name.removeprefix("genome_")}%'
    )
    # DSL-known path scans every completed run: no sound SQL prefilter exists
    # for a renamed strategy (hundreds of runs; JSON parse stops at first hit).
    rows = state.list_completed_discovery_notes(notes_like)
    for run_id, notes, run_start, run_end, run_symbols in rows:
        try:
            payload = cast("dict[str, object]", parse_notes(notes))
        except (TypeError, json.JSONDecodeError):
            continue
        strategies = payload.get("top_strategies")
        if not isinstance(strategies, list):
            continue
        if wanted_sizing is not None and (
            _discovery_recorded_sizing(payload) != wanted_sizing
        ):
            continue
        for entry in strategies:
            if not isinstance(entry, dict):
                continue
            dsl = entry.get("dsl")
            if not isinstance(dsl, dict):
                continue
            if wanted_body is not None:
                if dsl_body_key(dsl) != wanted_body:
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
