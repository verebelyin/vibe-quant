"""Fast-tier fallback accounting (vibe-quant-z8vqv.7): counts by normalised reason.

The fast tier falls back to NT on shapes it cannot replay. These tests pin the pure
per-process store: digit-normalised keys, cumulative snapshots, one snapshot per
worker process (last wins) and the own-process-wins aggregation rule.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.fast_tier import fallbacks

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    fallbacks.clear()
    yield
    fallbacks.clear()


def _snap(proc_id: str, counts: dict[str, int]) -> dict[str, Any]:
    return {"proc_id": proc_id, "counts": dict(counts)}


def test_record_normalises_digits_and_counts_cumulatively() -> None:
    fallbacks.record("open tick gaps through the take-profit at bar 1234")
    fallbacks.record("open tick gaps through the take-profit at bar 9")
    fallbacks.record("time filters")

    snap = fallbacks.snapshot()
    assert snap["counts"] == {"open tick gaps through the take-profit at bar N": 2, "time filters": 1}
    assert fallbacks.summary() == {
        "total": 3,
        "by_reason": {"open tick gaps through the take-profit at bar N": 2, "time filters": 1},
    }


def test_snapshot_is_this_process_and_returns_a_copy() -> None:
    first = fallbacks.snapshot()
    fallbacks.record("time filters")
    assert first["counts"] == {}
    snap = fallbacks.snapshot()
    snap["counts"]["forged"] = 7
    assert fallbacks.snapshot()["counts"] == {"time filters": 1}


def test_record_worker_last_snapshot_per_proc_wins() -> None:
    fallbacks.record_worker(_snap("pid-a", {"time filters": 1}))
    fallbacks.record_worker(_snap("pid-b", {"non-default fee schedule": 2}))
    fallbacks.record_worker(_snap("pid-a", {"time filters": 3}))

    assert fallbacks.summary() == {
        "total": 5,
        "by_reason": {"non-default fee schedule": 2, "time filters": 3},
    }


def test_summary_own_process_beats_a_remote_copy_of_its_proc_id() -> None:
    fallbacks.record("time filters")
    own = fallbacks.snapshot()
    fallbacks.record_worker(_snap(str(own["proc_id"]), {"time filters": 9}))

    assert fallbacks.summary() == {"total": 1, "by_reason": {"time filters": 1}}


def test_summary_keys_are_sorted() -> None:
    for reason in ("time filters", "non-default fee schedule", "one symbol per account only"):
        fallbacks.record(reason)
    assert list(fallbacks.summary()["by_reason"]) == [
        "non-default fee schedule",
        "one symbol per account only",
        "time filters",
    ]


def test_clear_resets_counts_and_remote_snapshots() -> None:
    fallbacks.record("time filters")
    fallbacks.record_worker(_snap("pid-a", {"time filters": 2}))
    fallbacks.clear()

    assert fallbacks.snapshot()["counts"] == {}
    assert fallbacks.summary() == {"total": 0, "by_reason": {}}


def _child_view(conn: Any) -> None:
    conn.send((fallbacks.snapshot(), fallbacks.summary()))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs the fork start method")
def test_forked_child_starts_empty_under_a_fresh_proc_id() -> None:
    fallbacks.record("time filters")
    fallbacks.record_worker(_snap("pid-a", {"time filters": 4}))
    parent_id = fallbacks.snapshot()["proc_id"]

    ctx = multiprocessing.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_child_view, args=(send,))
    child.start()
    snap, summary = recv.recv()
    child.join()

    assert snap["proc_id"] != parent_id
    assert snap["counts"] == {}
    assert summary == {"total": 0, "by_reason": {}}
    # the parent's own store is untouched by the child's reset
    assert fallbacks.snapshot()["proc_id"] == parent_id
    assert fallbacks.summary() == {"total": 5, "by_reason": {"time filters": 5}}
