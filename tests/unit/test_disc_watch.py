"""Tests for scripts/agents/disc_watch.py — the discovery run watcher.

Everything runs against a tmp_path logs dir and a tmp sqlite `backtest_runs` table;
the poll loop's ``sleep`` is injected so a pass advances on command and nothing ever
waits. A fake sleep that runs out of mutations raises rather than letting the loop
spin, and one that is called at all when none were queued fails the same way.
"""

from __future__ import annotations

import importlib.util
import io
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "disc_watch.py"
_spec = importlib.util.spec_from_file_location("disc_watch", _SCRIPT)
assert _spec is not None and _spec.loader is not None
disc_watch = importlib.util.module_from_spec(_spec)
sys.modules["disc_watch"] = disc_watch
_spec.loader.exec_module(disc_watch)

RUN = "938"
LOG_NAME = f"discovery_{RUN}_20261017_193121.log"


def _make_db(path: Path, statuses: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE backtest_runs (id TEXT PRIMARY KEY, status TEXT, run_mode TEXT)")
    conn.executemany(
        "INSERT INTO backtest_runs VALUES (?, ?, 'discovery')", list(statuses.items())
    )
    conn.commit()
    conn.close()
    return path


def _db(tmp_path: Path, statuses: dict[str, str]) -> Path:
    return _make_db(tmp_path / "state.db", statuses)


def _set_status(db: Path, run_id: str, status: str) -> None:
    conn = sqlite3.connect(db)
    conn.execute("UPDATE backtest_runs SET status = ? WHERE id = ?", (status, run_id))
    conn.commit()
    conn.close()


def _gen(
    gen: int,
    total: int,
    best: float,
    *,
    gen_time: float = 138.9,
    eta: float = 278.0,
    zero: int = 7,
    zero_total: int = 8,
) -> str:
    return (
        f"2026-10-17 19:31:21,095 [INFO] vibe_quant.discovery.pipeline: === GEN {gen}/{total} === "
        f"best={best:.4f} mean=0.0253 median=0.0000 std=0.0669 | zero_score={zero}/{zero_total} | "
        f"gen_time={gen_time:.1f}s total=139s ETA={eta:.0f}s (initial)"
    )


def _complete(gens: int = 2, evaluated: int = 14, total: int = 253) -> str:
    return (
        "2026-10-17 19:33:15,086 [INFO] vibe_quant.discovery.pipeline: "
        f"=== DISCOVERY COMPLETE: {total}s total (126.4s/gen avg) | {gens} gens | "
        f"{evaluated} evaluated | converged=True ==="
    )


def _log(logs: Path, lines: Sequence[str], name: str = LOG_NAME) -> Path:
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / name
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _append(path: Path, *lines: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(f"{line}\n")


def _run(
    run_ids: Sequence[str],
    logs: Path,
    db: Path,
    *,
    all_gens: bool = False,
    mutations: Sequence[Callable[[], None]] = (),
) -> tuple[int, str]:
    out = io.StringIO()
    queue = list(mutations)

    def sleep(_seconds: float) -> None:
        if not queue:
            raise AssertionError("watch() slept with no mutation left to apply")
        queue.pop(0)()

    code = disc_watch.watch(
        run_ids, logs_dir=logs, db_path=db, all_gens=all_gens, out=out, sleep=sleep
    )
    assert not queue, f"{len(queue)} unused mutations"
    return code, out.getvalue()


def test_exit_zero_after_completed_run_drains_late_lines(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    log = _log(logs, [_gen(1, 3, 0.2022)])
    db = _db(tmp_path, {RUN: "running"})

    def finish() -> None:
        _set_status(db, RUN, "completed")
        _append(log, _complete())

    code, out = _run([RUN], logs, db, mutations=[finish])

    assert code == 0
    assert out.splitlines()[0] == "run 938: gen 1/3 best=0.2022 zero=7/8 (139s, ETA 278s)"
    assert "run 938: COMPLETE 2 gens, 14 evaluated, 253s" in out
    assert out.splitlines()[-1] == "run 938: STATUS completed"
    assert out.count("run 938: STATUS completed") == 1


def test_state_db_is_opened_read_only(tmp_path: Path) -> None:
    db = _db(tmp_path, {RUN: "completed"})
    conn = disc_watch.open_db(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO backtest_runs VALUES ('x', 'running', 'discovery')")
    finally:
        conn.close()


def test_no_kill_without_kill_after(tmp_path: Path) -> None:
    db = _db(tmp_path, {RUN: "running"})
    logs = tmp_path / "logs"
    _log(logs, [_complete()])
    code, out = _run([RUN], logs, db, mutations=[lambda: _set_status(db, RUN, "completed")])
    assert code == 0 and "KILL" not in out
