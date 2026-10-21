"""Per-process counts of the reasons the fast tier fell back to the NT engine.

``FastBacktestFn`` with ``nt_fallback=True`` (discovery ``--evaluator fast``) runs a
genome the replay cannot reproduce exactly on the NT engine instead of raising. Before
this module those fallbacks were silent: nobody could tell how many genomes fell back
or why (run 981: 43 confirm backtests at 65-132 s; a fallback costs 65-130 s on 1m).

:func:`record` is called once per fallback (one symbol x one window) with the
``FastTierUnsupported`` message; every digit run is normalised to ``N`` so a per-bar
id or a loaded-timeframe list collapses into one reason. The store is per process, like
:mod:`vibe_quant.dsl.indicator_memo`: workers (a forked one starts empty under a fresh
``proc_id``) send :func:`snapshot` back to the parent, which keeps the LAST snapshot per ``proc_id`` (:func:`record_worker`), and
:func:`summary` sums those with this process's own counts -- the own ``proc_id``'s
remote copy never wins. Observability only: no metric, fill, fitness or gate changes.

A run's ``notes["evaluator"]["nt_fallbacks"]`` is :func:`summary` expanded
(``{"total": int, "by_reason": dict[str, int]}``, keys sorted).
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, cast
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Mapping

_PROCID = uuid4().hex

_counts: dict[str, int] = {}
_remote: dict[str, dict[str, object]] = {}


def _reset_in_forked_child() -> None:
    """A forked worker starts a fresh store under its own ``proc_id``.

    ``fork`` copies this module: without the reset every worker would report the
    parent's ``proc_id`` (merged into one identity, then overwritten by the parent's own
    counts in :func:`summary`) and resend the parent's history as its own.
    """
    global _PROCID
    _PROCID = uuid4().hex
    _counts.clear()
    _remote.clear()


if hasattr(os, "register_at_fork"):  # POSIX; spawn re-imports the module instead
    os.register_at_fork(after_in_child=_reset_in_forked_child)


def _key(reason: str) -> str:
    """The reason with every digit run collapsed to ``N``."""
    return re.sub(r"\d+", "N", reason)


def record(reason: str) -> None:
    """Count one fallback for ``reason`` in this process."""
    key = _key(reason)
    _counts[key] = _counts.get(key, 0) + 1


def snapshot() -> dict[str, object]:
    """This process's cumulative counts (``proc_id`` + ``counts``)."""
    return {"proc_id": _PROCID, "counts": dict(_counts)}


def record_worker(snap: Mapping[str, object]) -> None:
    """Parent side: keep the LAST snapshot a process sent (keyed by ``proc_id``)."""
    _remote[str(snap["proc_id"])] = dict(snap)


def summary() -> dict[str, object]:
    """``{"total", "by_reason"}`` over this process + every recorded worker snapshot."""
    own = snapshot()
    own_id = str(own["proc_id"])
    procs = dict(_remote)
    procs[own_id] = own  # a remote copy of our own proc_id never wins
    by_reason: dict[str, int] = {}
    for snap in procs.values():
        for key, count in cast("Mapping[str, int]", snap["counts"]).items():
            by_reason[key] = by_reason.get(key, 0) + int(count)
    return {"total": sum(by_reason.values()), "by_reason": dict(sorted(by_reason.items()))}


def clear() -> None:
    """Reset the counts and the recorded worker snapshots (tests, new runs)."""
    _counts.clear()
    _remote.clear()
