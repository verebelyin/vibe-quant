"""The carry DSR trial ledger: one JSON line per trial outcome (design x0axf.14 §2).

The deflated Sharpe of ``evaluate`` charges ``N = N_PRIOR_TRIALS + |distinct spec
hashes in the ledger u {own}|``. This module owns the ledger lines, their
registration rules and that reducer:

* every line carries ``outcome`` in ``completed | ruined | liquidated`` and
  ``failed_at`` (decision/bust ns or ``null``). A failure line nulls
  ``weekly_sharpe``/``weeks``/``total_return``/``max_drawdown_daily``. A line
  without ``outcome`` (written before this field) reads as ``completed``;
* :func:`failure_outcome` decides whether a failed run is a trial at all: only a
  :class:`~vibe_quant.portfolio.spec.CarryEquityTooSmall` raised AFTER the first
  decision is (data, preflight, ledger and execution errors are not);
* :func:`record_completion` / :func:`record_failure` append under an exclusive
  ``flock`` after reading and validating every existing line; a failed write is
  truncated back to the old size, so the ledger never holds a partial record;
* :func:`trial_counts` charges each hash once (failures included) and takes its
  variance input from the hash's most recent FINITE completed Sharpe, so a later
  failure or null row never erases an observation.

Any unreadable, undecodable, malformed or truncated ledger -- including a
``NaN``/``Infinity`` constant, a ``weekly_sharpe`` that is not ``null`` or a
finite float, or Sharpes whose variance overflows a float -- is a
:class:`CarryLedgerError` naming the path (and line): the CLI exits 2. An append
is refused when the ledger with the new record would be such a ledger.
``fcntl`` is POSIX-only (macOS/Linux are the only targets).
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe_quant.portfolio.metrics import _WEEKS_PER_YEAR
from vibe_quant.portfolio.spec import CarryEquityTooSmall, CarryLiquidated

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from vibe_quant.portfolio.spec import CarrySpec

# The three journal trials of `docs/discovery-journal.md` that already spent
# DSR budget on this idea (R38): RESULT H-11 (189 coins) and its 66-coin
# secondary (2026-10-11), and PRE-REGISTRATION H-11F4 (4 coins, 2026-10-12).
# H-11F (29 coins) was withdrawn before any evaluation -> not a trial.
N_PRIOR_TRIALS = 3
COMPLETED = "completed"
RUINED = "ruined"
LIQUIDATED = "liquidated"
OUTCOMES = (COMPLETED, RUINED, LIQUIDATED)
METRIC_KEYS = ("weekly_sharpe", "weeks", "total_return", "max_drawdown_daily")

# Failures recorded as ``liquidated``; checked before the CarryEquityTooSmall base.
_LIQUIDATED: tuple[type[CarryEquityTooSmall], ...] = (CarryLiquidated,)

REPO_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)
_NS_PER_S = 10**9
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_CHUNK = 1 << 20
_MAX_FLOAT_INT = int(sys.float_info.max)  # a larger JSON int has no finite float value


class CarryLedgerError(ValueError):
    """The trial ledger cannot be read, parsed or appended to (fail closed, exit 2)."""


def jsonable(value: Any) -> Any:
    """Non-finite floats become ``null``, so ``allow_nan=False`` never trips (R52)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def dumps(payload: Any, **kw: Any) -> str:
    """Strict JSON of ``payload``: NaN/+-inf already mapped to ``null`` (R52)."""
    return json.dumps(jsonable(payload), allow_nan=False, **kw)


def git_sha() -> str:
    """The checkout's HEAD, or ``"unknown"`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        logger.warning("git rev-parse failed (%s); trial git_sha recorded as unknown", type(error).__name__)
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _line(
    engine: str,
    spec: CarrySpec,
    sha: str,
    metrics: Mapping[str, Any],
    outcome: str,
    failed_at: int | None,
) -> str:
    now_ns = time.time_ns()
    created_at = (
        _EPOCH + timedelta(seconds=now_ns // _NS_PER_S, microseconds=now_ns % _NS_PER_S // 1000)
    ).isoformat()
    return dumps(
        {
            "spec_hash": spec.spec_hash(),
            "spec": spec.to_json(),
            "engine": engine,
            "git_sha": sha,
            "created_at": created_at,
            "ts": now_ns,
            **metrics,
            "outcome": outcome,
            "failed_at": failed_at,
        },
        sort_keys=True,
    )


def completion_line(
    engine: str, spec: CarrySpec, summary: Mapping[str, Any], sha: str
) -> str:
    """A completed run's line: headline metrics, ``outcome "completed"``, ``failed_at null``."""
    return _line(engine, spec, sha, {key: summary[key] for key in METRIC_KEYS}, COMPLETED, None)


def failure_line(engine: str, spec: CarrySpec, outcome: str, failed_at: int, sha: str) -> str:
    """A ruined/liquidated run's line: every metric ``null``, ``failed_at`` = the failure ns."""
    if outcome not in (RUINED, LIQUIDATED):
        raise ValueError(f"failure outcome must be ruined or liquidated, got {outcome!r}")
    return _line(engine, spec, sha, dict.fromkeys(METRIC_KEYS), outcome, failed_at)


def failure_outcome(error: BaseException, decisions: Sequence[int]) -> str | None:
    """The trial outcome of a failed run, or ``None`` when the failure is not a trial.

    Only a :class:`CarryEquityTooSmall` whose ``decision_ns`` is strictly after
    the first decision is a trial (``liquidated`` for a liquidation subclass,
    checked first, else ``ruined``). A null time or a stop at/before the first
    decision is an input failure; any other error class is not a trial.
    """
    if not isinstance(error, CarryEquityTooSmall) or not decisions:
        return None
    at = error.decision_ns
    if at is None or at <= decisions[0]:
        return None
    if isinstance(error, _LIQUIDATED):
        return LIQUIDATED
    return RUINED


def outcome_of(record: Mapping[str, Any]) -> str:
    """A record's outcome; a legacy line without the field is ``completed``."""
    return str(record.get("outcome", COMPLETED))


def _finite_sharpe(record: Mapping[str, Any]) -> float | None:
    value = record.get("weekly_sharpe")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _no_constant(name: str) -> None:
    """``json.loads`` hook: ``NaN``/``Infinity`` are not JSON, so not a ledger value."""
    raise ValueError(f"non-JSON constant {name}")


def _sharpe_ok(value: object) -> bool:
    """``weekly_sharpe`` is ``null``/absent or a number with a finite float value."""
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        return abs(value) <= _MAX_FLOAT_INT
    return math.isfinite(value)


def _parse_record(text: str, path: Path, number: int) -> dict[str, Any]:
    try:
        record = json.loads(text, parse_constant=_no_constant)
    except ValueError as error:
        raise CarryLedgerError(f"trial ledger {path} line {number}: malformed ({error})") from error
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("spec_hash"), str)
        or not record["spec_hash"]
        or outcome_of(record) not in OUTCOMES
        or not _sharpe_ok(record.get("weekly_sharpe"))
    ):
        raise CarryLedgerError(
            f"trial ledger {path} line {number}: malformed (not a trial record)"
        )
    return record


def _variance(records: Sequence[Mapping[str, Any]]) -> float | None:
    """Per-period variance of each hash's most recent finite completed Sharpe (file order).

    ``None`` with fewer than two such values; ``OverflowError`` when not a float.
    """
    latest: dict[str, float] = {}
    for record in records:
        sharpe = _finite_sharpe(record)
        if outcome_of(record) == COMPLETED and sharpe is not None:
            latest[str(record["spec_hash"])] = sharpe
    sharpes = [value / math.sqrt(_WEEKS_PER_YEAR) for value in latest.values()]
    if len(sharpes) < 2:
        return None
    return statistics.variance(sharpes)  # exact sum; OverflowError when out of float range


def _check_variance(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    """The DSR must be able to read the ledger: its Sharpe variance is a finite float."""
    try:
        _variance(records)
    except OverflowError as error:
        raise CarryLedgerError(
            f"trial ledger {path}: weekly_sharpe variance not representable ({error})"
        ) from error


def _parse(data: bytes, path: Path) -> list[dict[str, Any]]:
    """Every record of the ledger bytes; a partial or bad line raises naming its number."""
    lines = data.split(b"\n")
    if lines[-1]:
        raise CarryLedgerError(f"trial ledger {path} line {len(lines)}: truncated (no newline)")
    records: list[dict[str, Any]] = []
    for number, raw in enumerate(lines[:-1], start=1):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CarryLedgerError(
                f"trial ledger {path} line {number}: malformed ({error})"
            ) from error
        if text.strip():
            records.append(_parse_record(text, path, number))
    _check_variance(records, path)
    return records


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while chunk := os.pread(fd, _CHUNK, offset):
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def read_trials(path: Path) -> list[dict[str, Any]]:
    """One coherent snapshot of the ledger under a shared lock; absent -> ``[]``."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return []
    except OSError as error:
        raise CarryLedgerError(f"trial ledger {path} unreadable: {error}") from error
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        data = _read_all(fd)
    except OSError as error:
        raise CarryLedgerError(f"trial ledger {path} unreadable: {error}") from error
    finally:
        os.close(fd)
    return _parse(data, path)


def _append_bytes(fd: int, payload: bytes) -> None:
    """One ``os.write`` of the whole record; a short write is an error."""
    written = os.write(fd, payload)
    if written != len(payload):
        raise OSError(f"short write: {written} of {len(payload)} bytes")


def _register(
    path: Path, line: str, should_append: Callable[[list[dict[str, Any]]], bool]
) -> bool:
    """Lock, read + validate every line, decide, append one whole record. True = appended.

    The ledger WITH the new record must still be valid (finite DSR variance), so a
    record the DSR could not read is refused before the append.
    """
    new = _parse_record(line, Path("<new record>"), 1)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    except OSError as error:
        raise CarryLedgerError(f"trial ledger {path} unreadable: {error}") from error
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            data = _read_all(fd)
        except OSError as error:
            raise CarryLedgerError(f"trial ledger {path} unreadable: {error}") from error
        records = _parse(data, path)
        if not should_append(records):
            return False
        _check_variance([*records, new], path)
        try:
            _append_bytes(fd, (line + "\n").encode("utf-8"))
        except OSError as error:
            os.ftruncate(fd, len(data))
            raise CarryLedgerError(f"trial ledger {path} append failed: {error}") from error
        return True
    finally:
        os.close(fd)  # releases the lock


def record_completion(path: Path, line: str, *, history: bool) -> bool:
    """Register a completed run; returns whether a line was appended.

    ``history=True`` (``run``) always appends: one history row per completed run.
    ``history=False`` (``evaluate`` ensuring its own trial) appends only when the
    hash has no completed row, or every completed row has a null Sharpe and this
    one is finite; otherwise the ledger bytes stay unchanged.
    """
    new = json.loads(line)
    spec_hash = new["spec_hash"]
    finite = _finite_sharpe(new) is not None

    def should_append(records: list[dict[str, Any]]) -> bool:
        if history:
            return True
        completed = [
            r for r in records if r["spec_hash"] == spec_hash and outcome_of(r) == COMPLETED
        ]
        if not completed:
            return True
        return finite and all(_finite_sharpe(r) is None for r in completed)

    return _register(path, line, should_append)


def record_failure(path: Path, line: str) -> bool:
    """Register a failure line unless one with the same (hash, outcome) exists."""
    new = json.loads(line)
    key = (new["spec_hash"], outcome_of(new))

    def should_append(records: list[dict[str, Any]]) -> bool:
        return all((r["spec_hash"], outcome_of(r)) != key for r in records)

    return _register(path, line, should_append)


def trial_counts(
    records: Sequence[Mapping[str, Any]], spec_hash: str
) -> tuple[int, float | None]:
    """``(N, trials_sharpe_variance)`` from ledger records and the evaluated spec (R38/R41).

    ``N = N_PRIOR_TRIALS + |recorded hashes u {spec_hash}|``, each hash once,
    failures included. The variance (PER-PERIOD weekly Sharpe units) takes per
    hash its most recent finite completed Sharpe in file order; a failure-only or
    null-only hash adds to ``N`` only. ``None`` with fewer than two such values.
    """
    hashes = {str(record["spec_hash"]) for record in records}
    return N_PRIOR_TRIALS + len(hashes | {spec_hash}), _variance(records)
