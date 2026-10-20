"""Bounded per-process memo of indicator TAIL values for discovery runs.

Generated strategies re-run every ``compute_fn`` indicator over the rolling
:class:`~vibe_quant.dsl.pta_buffer.PtaBuffer` on every bar. GA children repeat
their parents' indicator/param sets, so on byte-identical bars the same pure
call runs again and again: a discovery run spends a large share of its CPU on
recomputation (measured 1.1-1.9x wall-clock in the POC). This module stores
what the generated code actually reads -- the last value of each returned
Series -- keyed on ``(function object, params reprs)`` plus the buffer's
blake2b-128 content fingerprint, and returns it on a hit.

Scope and safety
----------------
* Discovery only. :func:`session` is opened from exactly one place,
  :meth:`vibe_quant.discovery.backtest_fn.NTBacktestFn._run_single`; screening
  sweeps, validation and paper/live never open it, so there the generated code
  calls two no-op functions (``memo_lookup`` returns ``MEMO_MISS`` without even
  hashing, ``memo_store`` returns immediately).
* Bit-identical by construction: a hit returns the scalar the same pure
  function produced from byte-identical input on an identically-keyed call.
  The fingerprint covers the buffered bar count and every OHLCV value.
* Single-threaded by design (like ``prefix_memo``): one module-global store per
  process, unlocked. Discovery uses one process per worker and each worker
  evaluates one backtest at a time.

Kill switch and cap
-------------------
``VIBE_QUANT_IND_MEMO=0`` disables the memo (read at import and by
:func:`refresh_enabled`; any other value -- including ``""`` and ``"false"`` --
keeps it on, the ``VIBE_QUANT_PTA_MEMO`` precedent). ``session()`` is then a
no-op and :func:`snapshot` reports ``state == "disabled"``.

``VIBE_QUANT_IND_MEMO_MAX_MB`` is the per-process ESTIMATED cap, in whole MB,
default 64. It is parsed at every :func:`session` entry (so paper and
validation never parse it); a value that is not an integer >= 1 raises
``ValueError`` there. The estimate is ``150 + 105 * len(dict)`` bytes per
stored result (``150`` for a single output or ``None``), calibrated against
tracemalloc for the nested ``gen[bucket][fingerprint]`` layout: each bucket
tuple is stored once per generation and the per-bar fingerprint bytes object is
shared by every indicator of that bar. An insert that would take the young
generation past half the cap first rotates it into the old generation (dropping the previous old one, counted in ``evictions`` as entries),
so neither generation passes half the cap and ``est_bytes`` stays within the cap.

Adaptive switch-off
-------------------
Generation 1 of a discovery run has no reuse at all (measured: 19 runs, 0 hits
each), so the pipeline marks those sessions ``judge=False`` and they are never
judged. Every judged session with at least one call appends ``(hits, calls)``
to a window of the last ``WINDOW`` (100) sessions; a full window averaging
below ``MIN_RATE`` (10%) turns the memo OFF for ``backoff`` sessions
(``OFF_INITIAL`` 100, doubling to ``OFF_MAX`` 1600) and then re-probes. A full
window at or above the floor resets the backoff. The store is kept while OFF.

Monitoring
----------
:func:`snapshot` is this process's 15-key cumulative state; :func:`run_summary`
aggregates it with the worker snapshots the parent received via
:func:`record_worker` (last snapshot per ``proc_id`` wins) into the 16 keys the
pipeline logs per generation/run and persists as ``notes["ind_memo"]``:
``enabled``, ``max_mb``, ``processes``, ``hits``, ``misses``, ``hit_rate``,
``bypassed``, ``evictions``, ``switch_offs``, ``off_processes``,
``parent_sessions``, ``entries_max``, ``est_mb_max``, ``rss_growth_mb_max``,
``peak_rss_mb_max_worker``, ``peak_rss_mb_parent``. ``max_mb`` is the largest
worker cap (the parent's own when no worker reported); ``evictions > 0`` means the
cap is too small for the run and ``VIBE_QUANT_IND_MEMO_MAX_MB`` should be
raised.

The perf harness ``scripts/agents/perf_phase.py`` includes this memo by
default, so its numbers are memo-on; ``VIBE_QUANT_IND_MEMO=0`` gives the
pre-memo baseline.
"""

from __future__ import annotations

import os
import resource
import sys
from collections import deque
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, cast
from uuid import uuid4

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from vibe_quant.dsl.pta_buffer import PtaBuffer

    _ComputeFn = Callable[[pd.DataFrame, dict[str, object]], object]

MEMO_MISS: Final[object] = object()

EST_ENTRY: Final[int] = 150
EST_DICT_TAIL: Final[int] = 105
DEFAULT_MAX_MB: Final[int] = 64

_ENABLED_ENV: Final[str] = "VIBE_QUANT_IND_MEMO"
_MAX_MB_ENV: Final[str] = "VIBE_QUANT_IND_MEMO_MAX_MB"

WINDOW = 100
MIN_RATE = 0.10
OFF_INITIAL = 100
OFF_MAX = 1600

_Bucket = tuple[object, tuple[tuple[str, str], ...]]
_Store = dict[_Bucket, dict[bytes, object]]


class _Tail:
    """Stand-in for a one-output Series: the generated code reads ``iloc[-1]``."""

    __slots__ = ("_v",)

    def __init__(self, value: object) -> None:
        self._v = value

    def __len__(self) -> int:
        return 1

    @property
    def iloc(self) -> _Tail:
        return self

    def __getitem__(self, index: int) -> object:
        if index in (-1, 0):
            return self._v
        raise IndexError(index)


class _EmptyTail:
    """Stand-in for an empty Series: only ``len(...) == 0`` is ever asked."""

    __slots__ = ()

    def __len__(self) -> int:
        return 0


_EMPTY: Final = _EmptyTail()
_NOT_STORED: Final[object] = object()


_enabled = os.environ.get(_ENABLED_ENV, "1") != "0"
_state = "on" if _enabled else "disabled"
_off_left = 0
# None = at OFF_INITIAL: read lazily so tests can patch the constants.
_backoff: int | None = None
_depth = 0
_window: deque[tuple[int, int]] = deque()
_PROCID = uuid4().hex

_young: _Store = {}
_old: _Store = {}
_young_bytes = 0
_old_bytes = 0

_hits = 0
_misses = 0
_bypassed = 0
_evictions = 0
_sessions = 0
_switch_offs = 0

_max_bytes = DEFAULT_MAX_MB * 1_000_000
_rss_at_first_session: int | None = None
_remote: dict[str, dict[str, object]] = {}


def is_active() -> bool:
    """Whether lookups/stores currently use the memo (inside an active session)."""
    return _enabled and _state == "on" and _depth > 0


def refresh_enabled() -> None:
    """Re-read ``VIBE_QUANT_IND_MEMO`` (tests flip it at runtime)."""
    global _enabled, _state
    _enabled = os.environ.get(_ENABLED_ENV, "1") != "0"
    if not _enabled:
        _state = "disabled"
    elif _state == "disabled":
        _state = "on"


def _max_bytes_from_env() -> int:
    raw = os.environ.get(_MAX_MB_ENV, "")
    if raw == "":
        return DEFAULT_MAX_MB * 1_000_000
    msg = f"{_MAX_MB_ENV} must be an integer >= 1, got {raw!r}"
    try:
        mb = int(raw)
    except ValueError:
        raise ValueError(msg) from None
    if mb < 1:
        raise ValueError(msg)
    return mb * 1_000_000


def _peak_rss_bytes() -> int:
    """Peak RSS in bytes (``ru_maxrss`` is bytes on darwin, KiB elsewhere)."""
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw * (1 if sys.platform == "darwin" else 1024)


def _bucket(fn: _ComputeFn, params: dict[str, object]) -> _Bucket:
    return (fn, tuple(sorted((k, repr(v)) for k, v in params.items())))


def _est_entry(value: object) -> int:
    if isinstance(value, dict):
        return EST_ENTRY + EST_DICT_TAIL * len(value)
    return EST_ENTRY


def _wrap(res: object) -> object:
    """The value the generated code reads, or ``_NOT_STORED`` for anything else."""
    if res is None:
        return None
    if isinstance(res, pd.Series):
        return _EMPTY if len(res) == 0 else _Tail(res.iloc[-1])
    if isinstance(res, dict):
        if not all(value is None or isinstance(value, pd.Series) for value in res.values()):
            return _NOT_STORED
        return {
            key: (None if value is None else (_EMPTY if len(value) == 0 else _Tail(value.iloc[-1])))
            for key, value in res.items()
        }
    return _NOT_STORED


def _entries() -> int:
    return sum(len(tails) for tails in _young.values()) + sum(len(tails) for tails in _old.values())


def _rotate() -> None:
    global _young, _old, _young_bytes, _old_bytes, _evictions
    _evictions += sum(len(tails) for tails in _old.values())
    _old, _old_bytes = _young, _young_bytes
    _young, _young_bytes = {}, 0


def _insert_young(bucket: _Bucket, fp: bytes, value: object) -> None:
    global _young_bytes
    size = _est_entry(value)
    if _young_bytes + size > _max_bytes // 2:  # projected, so no generation passes half the cap
        _rotate()
    tails = _young.get(bucket)
    if tails is None:
        _young[bucket] = {fp: value}
    else:
        tails[fp] = value
    _young_bytes += size


def memo_lookup(fn: _ComputeFn, buf: PtaBuffer, params: dict[str, object]) -> object:
    """The stored tail for ``(fn, params)`` at ``buf``'s content, else ``MEMO_MISS``."""
    global _hits, _misses, _bypassed, _old_bytes
    if not is_active():
        return MEMO_MISS
    if buf.cap == 0:
        _bypassed += 1
        return MEMO_MISS
    bucket = _bucket(fn, params)
    fp = buf.fingerprint()
    tails = _young.get(bucket)
    if tails is not None:
        value = tails.get(fp, MEMO_MISS)
        if value is not MEMO_MISS:
            _hits += 1
            return value
    tails = _old.get(bucket)
    if tails is not None:
        value = tails.pop(fp, MEMO_MISS)
        if value is not MEMO_MISS:
            _hits += 1
            _old_bytes -= _est_entry(value)
            if not tails:
                del _old[bucket]
            _insert_young(bucket, fp, value)  # promote: the live generation
            return value
    _misses += 1
    return MEMO_MISS


def memo_store(fn: _ComputeFn, buf: PtaBuffer, params: dict[str, object], res: object) -> None:
    """Store the tail(s) of ``res``; a non-storable shape only bumps ``bypassed``."""
    global _bypassed
    if not is_active() or buf.cap == 0:
        return
    value = _wrap(res)
    if value is _NOT_STORED:
        _bypassed += 1
        return
    _insert_young(_bucket(fn, params), buf.fingerprint(), value)


def _judge_window(judge: bool, hits: int, misses: int) -> None:
    global _state, _off_left, _backoff, _switch_offs
    if _state != "on" or not judge:
        return
    calls = hits + misses
    if calls == 0:
        return
    _window.append((hits, calls))
    while len(_window) > WINDOW:
        _window.popleft()
    if len(_window) < WINDOW:
        return
    total = sum(c for _, c in _window)
    if sum(h for h, _ in _window) / total < MIN_RATE:
        backoff = OFF_INITIAL if _backoff is None else _backoff
        _state = "off"
        _off_left = backoff
        _backoff = min(2 * backoff, OFF_MAX)
        _switch_offs += 1
        _window.clear()
    else:
        _backoff = OFF_INITIAL


@contextmanager
def session(*, judge: bool = True) -> Iterator[None]:
    """Open the memo for the body; the ONLY way the memo turns on."""
    global _depth, _sessions, _max_bytes, _rss_at_first_session, _off_left, _state
    if not _enabled or _depth > 0:
        yield
        return
    _depth = 1
    try:
        _sessions += 1
        if _rss_at_first_session is None:
            _rss_at_first_session = _peak_rss_bytes()
        _max_bytes = _max_bytes_from_env()
        if _state == "off":
            try:
                yield
            finally:  # a failed evaluation still spends its OFF session
                _off_left -= 1
                if _off_left <= 0:
                    _state = "on"
                    _window.clear()
            return
        hits = _hits
        misses = _misses
        yield
        _judge_window(judge, _hits - hits, _misses - misses)
    finally:
        _depth = 0


def clear() -> None:
    """Reset the store, counters, adaptive state and remote snapshots."""
    global _young, _old, _young_bytes, _old_bytes, _hits, _misses, _bypassed
    global _evictions, _sessions, _switch_offs, _state, _off_left, _backoff
    global _rss_at_first_session, _max_bytes
    _young, _old = {}, {}
    _young_bytes = _old_bytes = 0
    _hits = _misses = _bypassed = _evictions = _sessions = _switch_offs = 0
    _window.clear()
    _state = "on" if _enabled else "disabled"
    _off_left = 0
    _backoff = None
    _rss_at_first_session = None
    _max_bytes = DEFAULT_MAX_MB * 1_000_000
    _remote.clear()


def snapshot() -> dict[str, object]:
    """This process's cumulative memo state (15 keys; see the module docstring)."""
    return {
        "proc_id": _PROCID,
        "pid": os.getpid(),
        "enabled": _enabled,
        "state": _state,
        "max_bytes": _max_bytes,
        "entries": _entries(),
        "est_bytes": _young_bytes + _old_bytes,
        "hits": _hits,
        "misses": _misses,
        "bypassed": _bypassed,
        "evictions": _evictions,
        "sessions": _sessions,
        "switch_offs": _switch_offs,
        "peak_rss_bytes": _peak_rss_bytes(),
        "rss_at_first_session_bytes": _rss_at_first_session,
    }


def record_worker(snap: Mapping[str, object]) -> None:
    """Parent side: keep the LAST snapshot a process sent (keyed by ``proc_id``)."""
    _remote[str(snap["proc_id"])] = dict(snap)


def _int(snap: Mapping[str, object], key: str) -> int:
    return cast("int", snap[key])


def _mb(value: int) -> float:
    return round(value / 1_000_000, 1)


def run_summary() -> dict[str, object]:
    """Aggregate the worker snapshots + this process into the 16 monitoring keys."""
    own = snapshot()
    own_id = str(own["proc_id"])
    procs = dict(_remote)
    procs[own_id] = own  # a remote copy of our own proc_id never wins

    hits = sum(_int(p, "hits") for p in procs.values())
    misses = sum(_int(p, "misses") for p in procs.values())
    calls = hits + misses
    growths = [
        _int(p, "peak_rss_bytes") - _int(p, "rss_at_first_session_bytes")
        for p in procs.values()
        if p["rss_at_first_session_bytes"] is not None
    ]
    worker_peaks = [_int(p, "peak_rss_bytes") for proc, p in procs.items() if proc != own_id]
    # The workers hold the memos; the parent's cap is the default until its first session.
    worker_caps = [_int(p, "max_bytes") for proc, p in procs.items() if proc != own_id]
    cap = max(worker_caps) if worker_caps else _int(own, "max_bytes")
    return {
        "enabled": own["enabled"],
        "max_mb": cap // 1_000_000,
        "processes": len(procs),
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / calls, 4) if calls else None,
        "bypassed": sum(_int(p, "bypassed") for p in procs.values()),
        "evictions": sum(_int(p, "evictions") for p in procs.values()),
        "switch_offs": sum(_int(p, "switch_offs") for p in procs.values()),
        "off_processes": sum(1 for p in procs.values() if p["state"] == "off"),
        "parent_sessions": _int(own, "sessions"),
        "entries_max": max(_int(p, "entries") for p in procs.values()),
        "est_mb_max": _mb(max(_int(p, "est_bytes") for p in procs.values())),
        "rss_growth_mb_max": _mb(max(growths)) if growths else None,
        "peak_rss_mb_max_worker": _mb(max(worker_peaks)) if worker_peaks else None,
        "peak_rss_mb_parent": _mb(_int(own, "peak_rss_bytes")),
    }
