"""Shadow forward tracker: enrol a frozen strategy, then score it on days it never saw.

Design arch-c7-2 §2. :func:`enrol` freezes a strategy's inputs (DSL, launch parameters,
venue, latency, detail, entry order, instrument spec, funding fallback rate) and starts its
clock at the next UTC midnight F. :func:`update_all` re-runs every live row on a private
catalog snapshot through the frozen-run seam (:func:`~vibe_quant.validation.variant.run_frozen`)
from ``run_start`` (F - warm-up) to the data coverage day ``C + 1``, checks the run
reproduces every stored day bit for bit with unchanged inputs, then decides the new days
``committed_through + 1 .. C - 1`` one by one (:func:`~vibe_quant.shadow.estat.first_verdict`).
Each row's days, stats, state and events are written in one transaction.

No orders: nothing here imports the paper node or an execution client, and a pass only
records ``state='passed'`` plus one ``paper_proposal`` note (:data:`PROPOSALS_ENABLED`); the
proposal waits for overfit-auditor, risk-officer and the user before any paper session.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from vibe_quant.bankroll.report import dsl_hasher
from vibe_quant.data import catalog as catalog_mod
from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH
from vibe_quant.db.connection import DEFAULT_DB_PATH
from vibe_quant.db.state_manager import (
    ShadowDuplicate,
    ShadowUpdateRunning,
    StateManager,
    shadow_update_lock,
)
from vibe_quant.dsl.compiler import compiler_version_hash
from vibe_quant.dsl.parser import DSLParseError
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.shadow import estat
from vibe_quant.shadow.coverage import coverage_day, remove_stale_snapshots, snapshot
from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache
from vibe_quant.validation.runner import ValidationRunnerError
from vibe_quant.validation.variant import resolve_frozen, run_frozen

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vibe_quant.validation.results import ValidationResult

# Bumped with every declared comparability break (docs/claude/discovery.md). A row whose
# stored days reproduce under a new epoch keeps tracking; one that drifts is retired.
ENGINE_EPOCH: Final = 1
# On since 2026-11-23: journal RESULT shadow tracker null check PASS + the user's OK.
PROPOSALS_ENABLED: Final = True
PROPOSAL_SUFFIX: Final = "; proposed for paper (overfit-auditor, risk-officer and your OK decide)"
FAMILY: Final = "F1"
DEFAULT_BUDGET_S: Final = 1800.0
# Step 0 leaves younger lockless snapshots alone (another tracker on another state DB
# sharing this catalog may be between mkdtemp and its owner lock).
SNAPSHOT_AGE_FLOOR_S: Final = DEFAULT_BUDGET_S
PREREG_MIN_CHARS: Final = 40
WARMUP_DEFAULT: Final = 180
WARMUP_MIN: Final = 30
WARMUP_MAX: Final = 730
BACKDATE_MAX: Final = 1500
NO_SLOT_NOTE: Final = "no evidence slot left; a new family needs your OK"
NO_SLOT_PASS_NOTE: Final = "evidence ≥ 100 without a slot"
BUDGET_NOTE: Final = "deferred: update budget"
# Frozen-identity keys that are not run inputs: the window moves every update.
_WINDOW_KEYS: Final = frozenset({"start_date", "end_date"})
_LIVE_STATES: Final = ("tracking", "stalled")
_EPOCH_DAY: Final = date(1970, 1, 1)
_DAY_NS: Final = 86_400 * 10**9

logger = logging.getLogger(__name__)

__all__ = [
    "ENGINE_EPOCH",
    "PROPOSALS_ENABLED",
    "PROPOSAL_SUFFIX",
    "EnrolRefused",
    "EnrolRequest",
    "ShadowUpdateRunning",
    "UpdateSummary",
    "enrol",
    "update_all",
]


class EnrolRefused(Exception):
    """An enrolment request that cannot be tracked honestly; nothing was written."""


@dataclass(frozen=True)
class EnrolRequest:
    """What to freeze. ``backdate_days`` (1..1500) makes a test row: no slot, never a proposal."""

    name: str
    source_kind: str
    dsl_json: str
    symbols: Sequence[str]
    parameters: Mapping[str, object]
    pre_registration: str
    max_dd_limit: float
    warmup_days: int = WARMUP_DEFAULT
    source_ref: Mapping[str, object] = field(default_factory=dict)
    strategy_id: int | None = None
    backdate_days: int | None = None


@dataclass
class UpdateSummary:
    """One update pass: engine runs and the per-row outcome text."""

    trigger: str
    runs: int = 0
    snapshots_removed: int = 0
    rows: list[tuple[int, str, str]] = field(default_factory=list)  # (id, state, note)


@dataclass
class _Change:
    """One row's update: written in a single transaction."""

    note: str
    days: list[tuple[int, float]] = field(default_factory=list)
    fields: dict[str, object] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    to_state: str | None = None
    reason: str | None = None
    bust_day: date | None = None  # a forward bust: decide the days before it first (R14)


def _day_index(d: date) -> int:
    return (d - _EPOCH_DAY).days


def _midnight_ns(d: date) -> int:
    return _day_index(d) * _DAY_NS


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_request(req: EnrolRequest) -> None:
    """The request's own fields; each refusal names its field."""
    chars = len(req.pre_registration.strip())
    if chars < PREREG_MIN_CHARS:
        raise EnrolRefused(
            f"pre_registration must have at least {PREREG_MIN_CHARS} characters (got {chars})"
        )
    if not (isinstance(req.max_dd_limit, (int, float)) and 0 < req.max_dd_limit <= 1):
        raise EnrolRefused(f"max_dd_limit must be in (0, 1], got {req.max_dd_limit!r}")
    if not (_is_int(req.warmup_days) and WARMUP_MIN <= req.warmup_days <= WARMUP_MAX):
        raise EnrolRefused(
            f"warmup_days must be {WARMUP_MIN}..{WARMUP_MAX}, got {req.warmup_days!r}"
        )
    bd = req.backdate_days
    if bd is not None and not (_is_int(bd) and 1 <= bd <= BACKDATE_MAX):
        raise EnrolRefused(f"backdate_days must be 1..{BACKDATE_MAX}, got {bd!r}")
    if not req.symbols:
        raise EnrolRefused("symbols must not be empty")


def _shadow_dsl_hash(name: str, dsl: object) -> str | None:
    """``dsl_hasher`` with this tracker's wording when the DSL cannot be read.

    ``dsl_hasher`` logs "bankroll: ..." on a bad DSL; reached from here that warning names
    the wrong subsystem, so readability is checked first and the bankroll hasher only sees
    a DSL it accepts (enrolments are rare, the second parse costs nothing).
    """
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.dsl.translator import translate_dsl_config

    if not isinstance(dsl, dict):
        return None
    try:
        validate_strategy_dict(translate_dsl_config(dsl, strategy_name=name))
    except Exception as exc:  # noqa: BLE001 -- any invalid DSL is "unreadable"
        logger.warning("shadow: DSL of %r is unreadable (%s)", name, type(exc).__name__)
        return None
    return dsl_hasher(name)(dsl)


def _check_duplicate(state: StateManager, req: EnrolRequest) -> None:
    """Refuse a non-test request a live row already tracks, before any engine work.

    The hash is the one the run identity records (``dsl_hasher``); an unreadable DSL
    skips to ``resolve_frozen``, which refuses it. The insert repeats the check inside
    its transaction, so two concurrent enrolments cannot both pass.
    """
    if req.backdate_days is not None:
        return
    try:
        dsl = json.loads(req.dsl_json)
    except ValueError:
        return
    sha = _shadow_dsl_hash(req.name, dsl)
    refusal = state.shadow_duplicate(sha, req.symbols, req.parameters) if sha else None
    if refusal is not None:
        raise EnrolRefused(refusal)


def _insert_row(state: StateManager, row: Mapping[str, object]) -> int:
    """Insert the row; a twin enrolled since the pre-check is refused like it."""
    try:
        return state.insert_shadow_strategy(row)
    except ShadowDuplicate as exc:
        raise EnrolRefused(str(exc)) from exc


def _run_config(
    symbols: Sequence[str], parameters: Mapping[str, object], start: date, end: date
) -> dict[str, object]:
    """A ``backtest_runs``-shaped config for the window ``[start, end)``."""
    return {
        "symbols": list(symbols),
        "parameters": dict(parameters),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }


def _instrument_identity(symbols: Sequence[str]) -> dict[str, str]:
    """Canonical JSON of each symbol's instrument spec (fees, margins, increments)."""
    from nautilus_trader.model.instruments import CryptoPerpetual

    return {
        sym: json.dumps(
            CryptoPerpetual.to_dict(catalog_mod.create_instrument(sym)),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for sym in symbols
    }


def _frozen_extras(symbols: Sequence[str], archive: Path) -> dict[str, object]:
    """Frozen inputs the run identity does not carry."""
    return {
        "instrument": _instrument_identity(symbols),
        "funding_fallback_rate": FundingCalculator(archive).default_rate,
    }


def _history_refusal(
    symbols: Sequence[str],
    *,
    catalog: Path,
    archive: Path,
    run_start: date,
    first: date,
    enrolled: datetime,
) -> str | None:
    """Why the warm-up ``[run_start, F)`` is not fully covered, else None.

    Funding is checked through ``enrolled``: a settlement later on the enrol day has not
    happened yet, and the update's coverage day checks that day like any other.
    """
    until_ns = (enrolled - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1) * 1000
    cm = catalog_mod.CatalogManager(catalog)
    clear_rate_cache()
    funding = FundingCalculator(archive)
    for sym in symbols:
        span = cm.get_bar_date_range(sym, "1m")
        if span is None:
            return f"no 1m data for {sym} in the catalog"
        first_open = span[0]
        first_full = first_open.date()
        if first_open != datetime.combine(first_full, datetime.min.time(), tzinfo=UTC):
            first_full += timedelta(days=1)
        if first_full > run_start:
            return f"1m data for {sym} starts {first_full}, after warm-up start {run_start}"
        missing = funding.uncovered_settlements(
            sym, _midnight_ns(run_start), min(_midnight_ns(first) - 1, until_ns)
        )
        if missing:
            when = datetime.fromtimestamp(missing[0] // 10**9, tz=UTC).isoformat()
            return (
                f"funding for {sym} misses {len(missing)} settlement(s) in the warm-up, "
                f"first {when}"
            )
    return None


def enrol(
    state: StateManager,
    req: EnrolRequest,
    *,
    now: datetime,
    catalog: Path | None = None,
    archive: Path | None = None,
) -> int:
    """Freeze ``req`` and start its forward clock at the next UTC midnight.

    Args:
        state: The state DB.
        req: What to track.
        now: The enrolment instant (timezone-aware).
        catalog: The bar catalog (default ``data/catalog``, read at call time).
        archive: The raw-data archive (default ``data/archive/raw_data.db``).

    Returns:
        The new row id.

    Raises:
        ValueError: ``now`` is timezone-naive.
        EnrolRefused: a bad field, a DSL/run error, or an uncovered warm-up; 0 rows written.
    """
    if now.utcoffset() is None:
        raise ValueError(f"now must be timezone-aware, got {now.isoformat()}")
    _check_request(req)
    _check_duplicate(state, req)
    catalog = Path(catalog or catalog_mod.DEFAULT_CATALOG_PATH)
    archive = Path(archive or DEFAULT_ARCHIVE_PATH)
    if not archive.exists():
        raise EnrolRefused(f"archive {archive} does not exist")
    enrolled = now.astimezone(UTC) - timedelta(days=req.backdate_days or 0)
    first = enrolled.date() + timedelta(days=1)
    run_start = first - timedelta(days=req.warmup_days)
    try:
        identity = resolve_frozen(
            strategy_name=req.name,
            dsl_json=req.dsl_json,
            run_config=_run_config(req.symbols, req.parameters, run_start, first),
            catalog_path=catalog,
            archive_path=archive,
        )
    except ValidationRunnerError as exc:
        raise EnrolRefused(str(exc)) from exc
    refusal = _history_refusal(
        req.symbols,
        catalog=catalog,
        archive=archive,
        run_start=run_start,
        first=first,
        enrolled=enrolled,
    )
    if refusal is not None:
        raise EnrolRefused(refusal)
    is_test = req.backdate_days is not None
    compiler_hash = compiler_version_hash()
    shadow_id = _insert_row(
        state,
        {
            "name": req.name,
            "source_kind": req.source_kind,
            "source_ref": dict(req.source_ref),
            "strategy_id": req.strategy_id,
            "dsl_json": req.dsl_json,
            "dsl_sha256": identity["dsl_sha256"],
            "compiler_hash": compiler_hash,
            "compiler_hash_last": compiler_hash,
            "symbols": list(req.symbols),
            "timeframe": identity["timeframe"],
            "parameters": dict(req.parameters),
            "frozen_identity": {**identity, **_frozen_extras(req.symbols, archive)},
            "enrolled_at": enrolled.isoformat(timespec="microseconds"),
            "first_day": first.isoformat(),
            "run_start": run_start.isoformat(),
            "is_test": int(is_test),
            "family": None if is_test else FAMILY,
            "pre_registration": req.pre_registration,
            "max_dd_limit": float(req.max_dd_limit),
            "engine_epoch": ENGINE_EPOCH,
            "engine_epoch_last": ENGINE_EPOCH,
        },
    )
    if not is_test and _row(state, shadow_id)["slot"] is None:
        state.commit_shadow_update(
            shadow_id,
            new_days=[],
            fields={"status_note": NO_SLOT_NOTE},
            events=[{"kind": "note", "message": NO_SLOT_NOTE}],
        )
    return shadow_id


def _row(state: StateManager, shadow_id: int) -> dict[str, object]:
    return next(r for r in state.list_shadow_strategies() if r["id"] == shadow_id)


# --- update ------------------------------------------------------------------------------


def _timeframes(row: Mapping[str, object], detail: object) -> set[str] | str:
    """Every bar timeframe the frozen DSL reads plus the detail (fill) timeframe, or the
    first error of a stored DSL that no longer validates."""
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.dsl.translator import translate_dsl_config

    try:
        dsl = validate_strategy_dict(
            translate_dsl_config(json.loads(str(row["dsl_json"])), strategy_name=str(row["name"]))
        )
    except (DSLParseError, ValueError) as exc:  # JSONDecodeError is a ValueError
        details = exc.details if isinstance(exc, DSLParseError) else []
        return details[0] if details else str(exc).splitlines()[0]
    return dsl.get_all_timeframes() | ({detail} if isinstance(detail, str) else set())


def _failed_stream(
    symbols: Sequence[str], failed_streams: Mapping[str, set[str]], *, mark: bool
) -> str | None:
    """The first required stream that failed in this data update, as stall text."""
    required = {"klines", "funding", "mark"} if mark else {"klines", "funding"}
    for sym in symbols:
        hit = sorted(set(failed_streams.get(sym, ())) & required)
        if hit:
            return f"data update failed for {sym} ({', '.join(hit)})"
    return None


def _bust_day(result: ValidationResult) -> tuple[bool, date | None]:
    """(bust?, its UTC day): NT force-stop clock first, else the liquidation bust instant."""
    probe = result.mtm_probe
    bust_ts = probe.bust_ts if probe is not None else None
    if not result.engine_force_stopped and bust_ts is None:
        return False, None
    stop = result.notes.get("engine_force_stop")
    stopped_at = stop.get("stopped_at") if isinstance(stop, dict) else None
    if isinstance(stopped_at, str):
        return True, datetime.fromisoformat(stopped_at).astimezone(UTC).date()
    if bust_ts is not None:
        return True, _EPOCH_DAY + timedelta(days=bust_ts // _DAY_NS)
    return True, None


def _changed_inputs(
    frozen: Mapping[str, object], identity: Mapping[str, object], extras: Mapping[str, object]
) -> list[str]:
    current = {**identity, **extras}
    keys = (frozen.keys() | current.keys()) - _WINDOW_KEYS
    return sorted(k for k in keys if frozen.get(k) != current.get(k))


def _check_run(
    row: Mapping[str, object],
    result: ValidationResult,
    identity: Mapping[str, object],
    *,
    run_start: date,
    first: date,
    c_day: date,
    archive: Path,
) -> _Change | None:
    """The run's own failures, first hit wins (design §2 step 6, before the prefix check)."""
    symbols = list(row["symbols"])  # type: ignore[call-overload]
    frozen = row["frozen_identity"]
    assert isinstance(frozen, dict)
    changed = _changed_inputs(frozen, identity, _frozen_extras(symbols, archive))
    if changed:
        return _Change("", to_state="retired", reason=f"inputs changed: {', '.join(changed)}")
    bust, day = _bust_day(result)
    if bust:
        if day is None:
            return _Change("", to_state="stalled", reason="engine force-stopped at an unknown time")
        if day >= first:
            return _Change(
                "",
                to_state="futile",
                reason=f"account bust on {day}",
                fields={"verdict_day": day.isoformat()},
                bust_day=day,
            )
        return _Change("", to_state="retired", reason=f"warm-up bust on {day} not seen at enrol")
    daily = result.mtm_probe.daily if result.mtm_probe is not None else None
    if daily is None:
        return _Change("", to_state="stalled", reason="no daily series")
    expected = (c_day + timedelta(days=1) - run_start).days
    got_first = _EPOCH_DAY + timedelta(days=daily.first_day)
    if got_first != run_start or len(daily) != expected:
        return _Change(
            "",
            to_state="stalled",
            reason=(
                f"engine returned {len(daily)} days from {got_first}, "
                f"expected {expected} from {run_start}"
            ),
        )
    if result.funding_fallback_settlements > 0:
        n = result.funding_fallback_settlements
        return _Change(
            "", to_state="stalled", reason=f"funding fallback charged on {n} settlement(s)"
        )
    return None


def _drift(
    stored: Sequence[tuple[int, float]], values: Sequence[float], run_start_idx: int
) -> str | None:
    """The first stored day the re-run does not reproduce (IEEE ``==``: -0.0 == 0.0)."""
    for day, ret in stored:
        now = values[day - run_start_idx]
        if now != ret:
            when = _EPOCH_DAY + timedelta(days=day)
            return f"day {when}: stored {ret!r} now {now!r}"
    return None


def _decide(
    row: Mapping[str, object],
    stored: Sequence[tuple[int, float]],
    values: Sequence[float],
    *,
    run_start_idx: int,
    first: date,
    last: date,
) -> _Change:
    """Append days ``committed + 1 .. last`` and stop at the first verdict day."""
    first_idx = _day_index(first)
    new = [
        (d, values[d - run_start_idx]) for d in range(first_idx + len(stored), _day_index(last) + 1)
    ]
    x = np.array([r for _, r in stored] + [r for _, r in new], dtype=float)
    slot = row["slot"]
    can_pass = slot is not None or bool(row["is_test"])
    verdict = estat.first_verdict(
        x,
        start_index=len(stored),
        max_dd_limit=float(row["max_dd_limit"]),  # type: ignore[arg-type]
        can_pass=can_pass,
    )
    end = verdict[0] if verdict is not None else len(x) - 1
    keep = new[: end + 1 - len(stored)]
    stats = estat.forward_stats(x[: end + 1])
    through = first + timedelta(days=end)
    change = _Change(
        f"advanced {len(keep)} day(s) to {through}",
        days=keep,
        fields={"committed_through": through.isoformat(), "stats": asdict(stats)},
    )
    if verdict is not None:
        change.to_state, change.reason = verdict[1], verdict[2]
        change.fields["verdict_day"] = through.isoformat()
        if verdict[1] == "passed" and PROPOSALS_ENABLED and not row["is_test"] and slot is not None:
            change.reason = f"{verdict[2]}{PROPOSAL_SUFFIX}"
            change.events.append(
                {"kind": "note", "message": f"paper_proposal: evidence pass on {through}"}
            )
        return change
    if not can_pass and stats.e_value >= estat.E_PASS:
        change.note = NO_SLOT_PASS_NOTE
        prior = row["stats"]
        prior_e = prior.get("e_value", 1.0) if isinstance(prior, dict) else 1.0
        if prior_e < estat.E_PASS:
            change.events.append({"kind": "note", "message": NO_SLOT_PASS_NOTE})
    return change


def _evaluate(
    row: Mapping[str, object],
    state: StateManager,
    *,
    today: date,
    failed_streams: Mapping[str, set[str]],
    catalog: Path,
    archive: Path,
    compiler_hash: str,
    summary: UpdateSummary,
) -> _Change:
    """Design §2 steps 2-7 for one live row."""
    symbols = list(row["symbols"])  # type: ignore[call-overload]
    frozen = row["frozen_identity"]
    assert isinstance(frozen, dict)
    params = row["parameters"]
    assert isinstance(params, dict)
    mark = LiquidationSettings.from_params(params).on
    failed = _failed_stream(symbols, failed_streams, mark=mark)
    if failed is not None:
        return _Change("", to_state="stalled", reason=failed)
    first = date.fromisoformat(str(row["first_day"]))
    run_start = date.fromisoformat(str(row["run_start"]))
    committed = (
        date.fromisoformat(str(row["committed_through"]))
        if row["committed_through"]
        else first - timedelta(days=1)
    )
    timeframes = _timeframes(row, frozen.get("detail_timeframe"))
    if isinstance(timeframes, str):
        return _Change("", to_state="retired", reason=f"invalid DSL: {timeframes}")
    c_day, gap = coverage_day(
        symbols,
        timeframes,
        catalog=catalog,
        archive=archive,
        mark=mark,
        run_start=run_start,
    )
    c_day = min(c_day, today - timedelta(days=1))  # a day is never complete before it ends
    last = c_day - timedelta(days=1)  # day C carries the run-end close-out
    if last <= committed:
        if gap is not None:
            return _Change(f"no new data since {committed}", to_state="stalled", reason=gap)
        return _Change(f"no new data since {committed}")
    bust: _Change | None = None
    while True:
        summary.runs += 1
        try:
            with snapshot(catalog, symbols) as snap:
                result, identity = run_frozen(
                    strategy_name=str(row["name"]),
                    dsl_json=str(row["dsl_json"]),
                    run_config=_run_config(symbols, params, run_start, c_day + timedelta(days=1)),
                    catalog_path=snap,
                    archive_path=archive,
                    detail_override=frozen.get("detail_timeframe"),
                    latency_override=frozen.get("latency_preset"),
                )
        except Exception as exc:  # a snapshot or engine failure stalls this row only
            return _Change(
                "", to_state="stalled", reason=f"engine error: {type(exc).__name__}: {exc}"
            )
        failure = _check_run(
            row, result, identity, run_start=run_start, first=first, c_day=c_day, archive=archive
        )
        if failure is None:
            break
        # A forward bust on D: a daily cadence decided every day through D - 2 before the
        # bust showed (R14). Re-run to end = D so those days get the normal rules first.
        bd = failure.bust_day
        if bd is None or bd - timedelta(days=2) <= committed or bd > c_day:
            return failure
        bust, c_day = failure, bd - timedelta(days=1)
    last = c_day - timedelta(days=1)
    assert result.mtm_probe is not None and result.mtm_probe.daily is not None
    values = result.mtm_probe.daily.values
    stored = state.get_shadow_days(int(row["id"]))  # type: ignore[call-overload]
    epoch_last = int(row["engine_epoch_last"])  # type: ignore[call-overload]
    drift = _drift(stored, values, _day_index(run_start))
    if drift is not None:
        if epoch_last != ENGINE_EPOCH:
            drift += f"; declared engine change epoch {epoch_last}→{ENGINE_EPOCH}"
        return _Change("", to_state="retired", reason=drift)
    change = _decide(
        row, stored, values, run_start_idx=_day_index(run_start), first=first, last=last
    )
    if epoch_last != ENGINE_EPOCH:
        change.fields["engine_epoch_last"] = ENGINE_EPOCH
        change.events.insert(
            0,
            {
                "kind": "note",
                "message": f"engine epoch {epoch_last}→{ENGINE_EPOCH}: stored days reproduce",
            },
        )
    if row["compiler_hash_last"] != compiler_hash:
        change.fields["compiler_hash_last"] = compiler_hash
        change.events.insert(
            0,
            {
                "kind": "note",
                "message": f"compiler hash {row['compiler_hash_last']}→{compiler_hash}: "
                "stored days reproduce",
            },
        )
    if change.to_state is None and bust is not None:
        change.to_state, change.reason = bust.to_state, bust.reason
        change.fields.update(bust.fields)
    if change.to_state is None:
        if gap is not None:
            change.to_state, change.reason = "stalled", gap
        elif row["state"] == "stalled":
            change.to_state, change.reason = "tracking", f"data complete through {last}"
    return change


def _commit(state: StateManager, row: Mapping[str, object], change: _Change, now: str) -> str:
    """Write one row's change (days, fields, events) in one transaction; returns its state."""
    fields = dict(change.fields)
    events = list(change.events)
    old = str(row["state"])
    new = change.to_state or old
    if change.to_state is not None:
        fields["state_reason"] = change.reason
        if new != old:
            fields["state"] = new
            fields["state_changed_at"] = now
            prefix = "[TEST] " if row["is_test"] else ""
            events.append(
                {
                    "kind": "state_change",
                    "from_state": old,
                    "to_state": new,
                    "message": f"{prefix}{change.reason}",
                }
            )
    fields["status_note"] = change.note or change.reason
    fields["last_update_at"] = now
    state.commit_shadow_update(
        int(row["id"]),  # type: ignore[call-overload]
        new_days=change.days,
        fields=fields,
        events=events,
    )
    return new


def _mark_stale_proposals(state: StateManager, rows: Sequence[Mapping[str, object]]) -> None:
    """Passed rows measured on an older engine epoch: flag once (verdict and days unchanged)."""
    for row in rows:
        epoch_last = int(row["engine_epoch_last"])  # type: ignore[call-overload]
        if (
            row["state"] == "passed"
            and not row["is_test"]
            and not row["proposal_stale"]
            and epoch_last < ENGINE_EPOCH
        ):
            msg = (
                f"evidence measured on engine epoch {epoch_last}; "
                f"paper decision must re-check under {ENGINE_EPOCH}"
            )
            state.commit_shadow_update(
                int(row["id"]),  # type: ignore[call-overload]
                new_days=[],
                fields={"proposal_stale": 1},
                events=[
                    {
                        "kind": "state_change",
                        "from_state": "passed",
                        "to_state": "passed",
                        "message": msg,
                    }
                ],
            )


def update_all(
    state: StateManager,
    *,
    today: date,
    trigger: str,
    failed_streams: Mapping[str, set[str]],
    budget_s: float = DEFAULT_BUDGET_S,
    catalog: Path | None = None,
    archive: Path | None = None,
) -> UpdateSummary:
    """Advance every ``tracking``/``stalled`` row as far as complete data allows.

    Args:
        state: The state DB (its file is the one-updater lock's key).
        today: The current UTC date; no day at or after it is ever committed.
        trigger: Who asked ("data-update", "cli", "api"), for the summary.
        failed_streams: Symbol -> streams (``klines``, ``funding``, ``mark``) that failed in
            the data update that triggered this one; such rows stall without a run.
        budget_s: Wall-clock budget; rows not started within it are deferred.
        catalog: The bar catalog (default ``data/catalog``).
        archive: The raw-data archive (default ``data/archive/raw_data.db``).

    Raises:
        ShadowUpdateRunning: another update of this state DB is running.
        FileNotFoundError: the archive does not exist.
    """
    catalog = Path(catalog or catalog_mod.DEFAULT_CATALOG_PATH)
    archive = Path(archive or DEFAULT_ARCHIVE_PATH)
    if not archive.exists():  # RawDataArchive would create an empty one
        raise FileNotFoundError(f"archive {archive} does not exist")
    summary = UpdateSummary(trigger=trigger)
    with shadow_update_lock(state.db_path or DEFAULT_DB_PATH):
        start = time.monotonic()
        summary.snapshots_removed = remove_stale_snapshots(catalog, min_age_s=SNAPSHOT_AGE_FLOOR_S)
        all_rows = state.list_shadow_strategies()
        _mark_stale_proposals(state, all_rows)
        live = sorted(
            (r for r in all_rows if r["state"] in _LIVE_STATES),
            key=lambda r: (str(r["last_update_at"] or ""), int(r["id"])),  # type: ignore[call-overload]
        )
        compiler_hash = compiler_version_hash()
        for row in live:
            sid = int(row["id"])  # type: ignore[call-overload]
            if time.monotonic() - start >= budget_s:
                state.commit_shadow_update(
                    sid, new_days=[], fields={"status_note": BUDGET_NOTE}, events=[]
                )
                summary.rows.append((sid, str(row["state"]), BUDGET_NOTE))
                continue
            try:
                change = _evaluate(
                    row,
                    state,
                    today=today,
                    failed_streams=failed_streams,
                    catalog=catalog,
                    archive=archive,
                    compiler_hash=compiler_hash,
                    summary=summary,
                )
            except (ShadowUpdateRunning, sqlite3.Error):
                raise  # the state DB itself: fail the whole update closed
            except Exception as exc:  # one row's failure stalls that row, never the others
                reason = f"update error: {type(exc).__name__}: {exc}"
                change = _Change("", to_state="stalled", reason=reason)
            now = datetime.now(UTC).isoformat(timespec="microseconds")
            new_state = _commit(state, row, change, now)
            summary.rows.append((sid, new_state, change.note or str(change.reason)))
    return summary
