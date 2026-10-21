"""Tests for scripts/agents/disc_watch.py — the discovery run watcher.

Everything runs against a tmp_path logs dir and a tmp sqlite `backtest_runs` table;
the poll loop's ``sleep`` is injected so a pass advances on command and nothing ever
waits. A fake sleep that runs out of mutations raises rather than letting the loop
spin, and one that is called at all when none were queued fails the same way.
"""

from __future__ import annotations

import importlib.util
import io
import os
import shutil
import sqlite3
import subprocess
import sys
import time
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


_GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=agent@example.com", "-c", "user.name=agent", *args],
        cwd=cwd,
        env={**os.environ, **_GIT_ENV},
        check=True,
        capture_output=True,
        text=True,
    )


def _worktree_repo(tmp_path: Path) -> tuple[Path, Path]:
    """A throwaway main checkout plus a linked worktree, both with scripts/agents/."""
    main = tmp_path / "main"
    (main / "scripts" / "agents").mkdir(parents=True)
    _git(main, "init", "--initial-branch=main")
    _git(main, "commit", "--allow-empty", "-m", "init")
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "--detach", str(wt))
    (wt / "scripts" / "agents").mkdir(parents=True)
    return main, wt


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


def test_gen_line_printed_only_when_best_improves(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(logs, [_gen(1, 3, 0.2022), _gen(2, 3, 0.1500), _gen(3, 3, 0.2500)])
    db = _db(tmp_path, {RUN: "completed"})

    code, out = _run([RUN], logs, db)

    assert code == 0
    assert "run 938: gen 1/3 best=0.2022 zero=7/8 (139s, ETA 278s)" in out
    assert "gen 2/3" not in out
    assert "run 938: gen 3/3 best=0.2500 zero=7/8 (139s, ETA 278s)" in out
    assert "run 938: STATUS completed" in out


def test_all_gens_prints_every_gen(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(logs, [_gen(1, 3, 0.2022), _gen(2, 3, 0.1500), _gen(3, 3, 0.2500)])
    db = _db(tmp_path, {RUN: "completed"})

    code, out = _run([RUN], logs, db, all_gens=True)

    assert code == 0
    assert [line for line in out.splitlines() if ": gen " in line] == [
        "run 938: gen 1/3 best=0.2022 zero=7/8 (139s, ETA 278s)",
        "run 938: gen 2/3 best=0.1500 zero=7/8 (139s, ETA 278s)",
        "run 938: gen 3/3 best=0.2500 zero=7/8 (139s, ETA 278s)",
    ]


def test_complete_line(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(logs, [_complete()])
    db = _db(tmp_path, {RUN: "completed"})

    code, out = _run([RUN], logs, db)

    assert code == 0
    assert "run 938: COMPLETE 2 gens, 14 evaluated, 253s" in out


def test_champion_summaries_and_errors(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(
        logs,
        [
            _gen(1, 3, 0.2022),
            "2026-10-17 19:33:15,087 [WARNING] vibe_quant.discovery.pipeline: No champion passed "
            "every gate — run persists 0 champions (1 rejections)",
            "2026-10-17 19:33:15,087 [INFO] vibe_quant.discovery.pipeline: "
            "2/3 top strategies passed guardrails",
            "2026-10-17 19:34:00,000 [ERROR] vibe_quant.discovery.pipeline: boom",
            "2026-10-17 19:34:00,001 [INFO] vibe_quant.api.server: champion seen elsewhere",
        ],
    )
    db = _db(tmp_path, {RUN: "completed"})

    code, out = _run([RUN], logs, db)

    assert code == 0
    assert (
        "run 938: No champion passed every gate — run persists 0 champions (1 rejections)" in out
    )
    assert "top strategies passed guardrails" not in out
    assert "run 938: boom" in out
    assert "champion seen elsewhere" not in out


def test_traceback_prints_frames_and_the_exception_line(tmp_path: Path) -> None:
    """The real run-937 tail: the decisive `ValueError:` line must not be cut off."""
    logs = tmp_path / "logs"
    failed_message = (
        "warm-start run 928 has no persisted champions to seed from (guardrails may have "
        "rejected all candidates); launch without seed_run_id instead"
    )
    _log(
        logs,
        [
            "Traceback (most recent call last):",
            '  File "/repo/vibe_quant/discovery/__main__.py", line 1672, in main',
            "    raise ValueError(",
            "    ...<3 lines>...",
            "    )",
            f"ValueError: {failed_message}",
            f"Discovery failed: {failed_message}",
        ],
    )
    db = _db(tmp_path, {RUN: "failed"})

    code, out = _run([RUN], logs, db)

    assert code == 1
    assert out.splitlines() == [
        "run 938: Traceback (most recent call last):",
        'run 938: File "/repo/vibe_quant/discovery/__main__.py", line 1672, in main',
        "run 938: raise ValueError(",
        "run 938: ...<3 lines>...",
        "run 938: )",
        f"run 938: ValueError: {failed_message}",
        "run 938: STATUS failed",
    ]


def test_traceback_caps_the_block_at_25_lines(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(
        logs,
        ["Traceback (most recent call last):", *(f"  frame {i}" for i in range(1, 31))],
    )
    db = _db(tmp_path, {RUN: "completed"})

    code, out = _run([RUN], logs, db)

    assert code == 0
    assert [line for line in out.splitlines() if "frame " in line] == [
        f"run 938: frame {i}" for i in range(1, 26)
    ]


def test_traceback_of_a_chained_exception_prints_each_exception_line(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(
        logs,
        [
            "Traceback (most recent call last):",
            "  File a.py, line 1, in <module>",
            "    raise ValueError('first')",
            "ValueError: first",
            "During handling of the above exception, another exception occurred:",
            "Traceback (most recent call last):",
            "  File b.py, line 2, in <module>",
            "    raise KeyError('second')",
            "KeyError: second",
        ],
    )
    db = _db(tmp_path, {RUN: "completed"})

    code, out = _run([RUN], logs, db)

    assert code == 0
    assert "run 938: ValueError: first" in out
    assert "run 938: KeyError: second" in out


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


def test_exit_one_when_run_failed(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(logs, [_gen(1, 1, 0.1000), _complete(gens=1, evaluated=3, total=10)])
    db = _db(tmp_path, {RUN: "failed"})

    code, out = _run([RUN], logs, db)

    assert code == 1
    assert "run 938: STATUS failed" in out


def test_unknown_run_id_exits_two(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    db = _db(tmp_path, {RUN: "completed"})

    code, out = _run(["999"], logs, db)

    assert code == 2
    assert out == "run 999: UNKNOWN run id\n"


def test_log_appearing_late_is_waited_for_once(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    db = _db(tmp_path, {RUN: "running"})

    def start_log() -> None:
        _log(logs, [_gen(1, 2, 0.2022)])

    def finish() -> None:
        _set_status(db, RUN, "completed")
        _append(logs / LOG_NAME, _complete(gens=2, evaluated=8, total=200))

    code, out = _run([RUN], logs, db, mutations=[start_log, finish])

    assert code == 0
    assert out.count("run 938: waiting for log") == 1
    assert "run 938: gen 1/2 best=0.2022 zero=7/8 (139s, ETA 278s)" in out
    assert "run 938: COMPLETE 2 gens, 8 evaluated, 200s" in out
    assert "run 938: STATUS completed" in out


def test_newest_log_for_a_run_id_wins(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _log(logs, [_gen(1, 1, 0.1111)], name=f"discovery_{RUN}_20261017_190000.log")
    _log(logs, [_gen(1, 1, 0.2222)], name=f"discovery_{RUN}_20261017_193121.log")
    db = _db(tmp_path, {RUN: "completed"})

    code, out = _run([RUN], logs, db)

    assert code == 0
    assert "run 938: gen 1/1 best=0.2222" in out
    assert "0.1111" not in out


def test_state_db_is_opened_read_only(tmp_path: Path) -> None:
    db = _db(tmp_path, {RUN: "completed"})
    conn = disc_watch.open_db(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO backtest_runs VALUES ('x', 'running', 'discovery')")
    finally:
        conn.close()


def test_docstring_shows_monitor_usage() -> None:
    assert disc_watch.__doc__ is not None
    assert 'Monitor(command="scripts/agents/disc_watch.py 938 940"' in disc_watch.__doc__


def test_main_checkout_root_resolves_the_worktrees_common_dir(tmp_path: Path) -> None:
    main, wt = _worktree_repo(tmp_path)

    root = disc_watch.main_checkout_root(wt / "scripts" / "agents")

    assert root == main.resolve()


def test_main_checkout_root_falls_back_to_the_script_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    script_dir = tmp_path / "copy" / "scripts" / "agents"
    script_dir.mkdir(parents=True)

    assert disc_watch.main_checkout_root(script_dir) == script_dir.parents[1]


def test_defaults_come_from_the_main_checkout_not_the_cwd(tmp_path: Path) -> None:
    """A decoy cwd tree must lose: the worktree copy resolves logs/DB from the main checkout."""
    main, wt = _worktree_repo(tmp_path)
    script = wt / "scripts" / "agents" / "disc_watch.py"
    shutil.copy(_SCRIPT, script)
    _log(main / "logs", [_gen(1, 1, 0.2022), _complete(gens=1, evaluated=4, total=60)])
    _make_db(main / "data" / "state" / "vibe_quant.db", {RUN: "completed"})

    elsewhere = tmp_path / "elsewhere"
    _log(elsewhere / "logs", [_gen(1, 9, 0.9999)])
    _make_db(elsewhere / "data" / "state" / "vibe_quant.db", {RUN: "failed"})

    proc = subprocess.run(
        [sys.executable, str(script), RUN],
        cwd=elsewhere,
        env={key: value for key, value in os.environ.items() if key != "VIBE_QUANT_DB"},
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "run 938: COMPLETE 1 gens, 4 evaluated, 60s" in proc.stdout
    assert "run 938: STATUS completed" in proc.stdout
    assert "0.9999" not in proc.stdout
    assert "STATUS failed" not in proc.stdout


def test_explicit_logs_and_db_flags_win_over_the_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "777"
    logs = tmp_path / "logs"
    _log(
        logs,
        [_gen(1, 1, 0.2022), _complete(gens=1, evaluated=2, total=30)],
        name=f"discovery_{run_id}_20261017_193121.log",
    )
    db = _db(tmp_path, {run_id: "completed"})
    monkeypatch.chdir(tmp_path)

    code = disc_watch.main([run_id, "--logs", str(logs), "--db", str(db)])

    assert code == 0


def test_env_db_is_read_when_no_db_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No --db: VIBE_QUANT_DB names the state DB, not the main-checkout default (ovf0e).

    The main-checkout default is pointed at an absent dir, so `elif env_db` is the only branch
    that can reach the run's status: the mutant `elif False` falls through to a DB that does not
    exist, and main() then exits 2 (cannot open the state DB). A 0 therefore proves main() read
    the env DB and saw the run "completed" (failed/cancelled would be 1, unknown would be 2).
    """
    run_id = "777"
    logs = tmp_path / "logs"
    _log(
        logs,
        [_gen(1, 1, 0.2022), _complete(gens=1, evaluated=2, total=30)],
        name=f"discovery_{run_id}_20261017_193121.log",
    )
    db = _db(tmp_path, {run_id: "completed"})
    monkeypatch.setenv("VIBE_QUANT_DB", str(db))
    monkeypatch.setattr(
        disc_watch, "main_checkout_root", lambda *_args, **_kwargs: tmp_path / "main"
    )

    assert disc_watch.main([run_id, "--logs", str(logs)]) == 0


def test_db_flag_wins_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--db beats VIBE_QUANT_DB: the env DB says the run failed, the flag DB says completed."""
    run_id = "777"
    logs = tmp_path / "logs"
    _log(
        logs,
        [_gen(1, 1, 0.2022), _complete(gens=1, evaluated=2, total=30)],
        name=f"discovery_{run_id}_20261017_193121.log",
    )
    env_db = _db(tmp_path / "env", {run_id: "failed"})
    flag_db = _db(tmp_path / "flag", {run_id: "completed"})
    monkeypatch.setenv("VIBE_QUANT_DB", str(env_db))

    # exit 0 comes from the flag DB's "completed", not the env DB's "failed" (which would be 1)
    assert disc_watch.main([run_id, "--logs", str(logs), "--db", str(flag_db)]) == 0


def test_kill_after_cancels_a_running_run_once_and_waits_for_its_end(tmp_path: Path) -> None:
    db = _db(tmp_path, {RUN: "running"})
    logs = tmp_path / "logs"
    _log(logs, [])
    now = [0.0]
    killed: list[str] = []

    def kill(run_id: str) -> str:
        killed.append(run_id)
        return "HTTP 204"

    def tick(seconds: float, status: str | None = None) -> Callable[[], None]:
        def step() -> None:
            now[0] += seconds
            if status:
                _set_status(db, RUN, status)

        return step

    out = io.StringIO()
    queue = [tick(200), tick(200), tick(5), tick(5, "cancelled")]

    def sleep(_s: float) -> None:
        queue.pop(0)()

    code = disc_watch.watch(
        [RUN], logs_dir=logs, db_path=db, out=out, sleep=sleep,
        kill_after=360, kill=kill, clock=lambda: now[0],
    )  # fmt: skip
    assert killed == [RUN]  # exactly once, though the run stayed "running" two more passes
    assert f"run {RUN}: KILL after 400s (HTTP 204)" in out.getvalue()
    assert f"run {RUN}: STATUS cancelled" in out.getvalue()
    assert code == 1  # cancelled counts as a bad end, like any cancel


def test_no_kill_without_kill_after(tmp_path: Path) -> None:
    db = _db(tmp_path, {RUN: "running"})
    logs = tmp_path / "logs"
    _log(logs, [_complete()])
    code, out = _run([RUN], logs, db, mutations=[lambda: _set_status(db, RUN, "completed")])
    assert code == 0 and "KILL" not in out


# --- chunked reads and the kill deadline (bead vibe-quant-ab16c) -------------------

# The run-976 flood shape: ~205 bytes, no HEADER_RE match, so _handle_line ignores it.
_SIZING_LINE = (
    "\x1b[1m2026-03-12T21:43:59.999000000Z\x1b[0m \x1b[1;33m[WARN] "
    "BACKTESTER-001.Genome5d677cf003d6Strategy: Sizing: notional 4.{digit} below minimum "
    "5.00000000 USDT -- entry skipped\x1b[0m\n"
)
MIB = 1024 * 1024


def _sizing_block(total: int) -> bytes:
    """``total`` bytes of run-976-shaped Sizing lines (built in one shot, never per line)."""
    sample = _SIZING_LINE.format(digit=0)
    count = total // len(sample) + 1
    return "".join(_SIZING_LINE.format(digit=i % 10) for i in range(count)).encode()[:total]


def _error_line(msg: str) -> bytes:
    return f"2026-10-17 19:34:00,000 [ERROR] vibe_quant.discovery.pipeline: {msg}\n".encode()


def _bulk_log(
    logs: Path, run_id: str, *, total: int = 20 * MIB, errors: Sequence[str] = ()
) -> Path:
    """A big Sizing flood for ``run_id`` with ERROR lines spliced at start/middle/end."""
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"discovery_{run_id}_20261017_193121.log"
    body = _sizing_block(total)
    cuts = [0, len(body) // 2, len(body)]
    for cut, msg in sorted(zip(cuts, errors, strict=False), reverse=True):
        block = _error_line(msg)
        body = block + body if cut == 0 else body[:cut] + b"\n" + block + body[cut:]
    path.write_bytes(body)
    return path


def test_kill_fires_on_first_pass_before_reading_a_huge_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R19: the deadline is checked for every run before any log byte is handled."""
    logs = tmp_path / "logs"
    _bulk_log(logs, "938")
    _bulk_log(logs, "939")
    db = _db(tmp_path, {"938": "running", "939": "running"})
    order: list[str] = []

    def kill(run_id: str) -> str:
        order.append(f"kill {run_id}")
        _set_status(db, run_id, "cancelled")
        return "HTTP 202"

    original = disc_watch._handle_line

    def spy(run, line, *, out, all_gens):  # noqa: ANN001, ANN202
        order.append(f"handle {run.run_id}")
        original(run, line, out=out, all_gens=all_gens)

    monkeypatch.setattr(disc_watch, "_handle_line", spy)
    out = io.StringIO()

    def sleep(_seconds: float) -> None:
        raise AssertionError("watch() slept while a backlog existed")

    code = disc_watch.watch(
        ["938", "939"], logs_dir=logs, db_path=db, out=out, sleep=sleep,
        kill_after=0, kill=kill, clock=lambda: 0.0,
    )  # fmt: skip

    assert code == 1
    assert out.getvalue().splitlines()[:2] == [
        "run 938: KILL after 0s (HTTP 202)",
        "run 939: KILL after 0s (HTTP 202)",
    ]
    assert order[:2] == ["kill 938", "kill 939"]
    assert order[2:] and all(entry.startswith("handle ") for entry in order[2:])


def test_one_pass_reads_at_most_one_chunk(tmp_path: Path) -> None:
    """R19b: one pass over a 20 MB log consumes at most READ_CHUNK bytes."""
    logs = tmp_path / "logs"
    path = _bulk_log(logs, RUN)
    run = disc_watch.Run(run_id=RUN, log=path)

    lines = disc_watch._read_new_lines(run, terminal=False)

    assert run.offset == disc_watch.READ_CHUNK
    assert path.stat().st_size > disc_watch.READ_CHUNK
    assert lines


def test_chunked_catch_up_handles_every_line_without_sleeping(tmp_path: Path) -> None:
    """R19c: ERROR lines at the start, middle and end all survive the chunked catch-up."""
    logs = tmp_path / "logs"
    _bulk_log(logs, RUN, errors=["boom-first", "boom-middle", "boom-last"])
    db = _db(tmp_path, {RUN: "completed"})

    code, out = _run([RUN], logs, db)  # its sleep raises: a backlog must never reach it

    assert code == 0
    assert out.count(": boom-") == 3
    for marker in ("boom-first", "boom-middle", "boom-last"):
        assert f"run 938: {marker}" in out


def test_multibyte_char_across_chunk_boundary_is_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R20: a 2-byte character cut by the chunk boundary comes out whole, no U+FFFD."""
    logs = tmp_path / "logs"
    logs.mkdir()
    path = logs / LOG_NAME
    header = "2026-10-17 19:34:00,000 [ERROR] vibe_quant.discovery.pipeline: caf"
    payload = f"{header}é boom\n".encode()
    path.write_bytes(payload)
    monkeypatch.setattr(disc_watch, "READ_CHUNK", payload.index("é".encode()) + 1)
    run = disc_watch.Run(run_id=RUN, log=path)

    assert disc_watch._read_new_lines(run, terminal=False) == []
    lines = disc_watch._read_new_lines(run, terminal=False)

    assert lines == [f"{header}é boom"]
    assert "\ufffd" not in "".join(lines)


def test_terminal_flush_waits_for_the_last_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R20b: a final line without a newline is flushed once, whole, only at EOF."""
    logs = tmp_path / "logs"
    logs.mkdir()
    path = logs / LOG_NAME
    path.write_bytes(b"first\nsecond\npartial-no-newline")
    monkeypatch.setattr(disc_watch, "READ_CHUNK", 10)
    run = disc_watch.Run(run_id=RUN, log=path)

    first = disc_watch._read_new_lines(run, terminal=True)
    assert run.offset == 10 and first == ["first"]

    emitted = list(first)
    while run.offset < path.stat().st_size or run.pending:
        size = path.stat().st_size
        for line in disc_watch._read_new_lines(run, terminal=True):
            if run.offset < size:
                assert line in {"first", "second"}, f"truncated line {line!r} emitted early"
            emitted.append(line)

    assert emitted == ["first", "second", "partial-no-newline"]


@pytest.mark.timing
def test_read_8mib_of_sizing_lines_is_linear(tmp_path: Path) -> None:
    """8 MiB of run-976-shaped lines in one call: linear, well under the 5.83 s quadratic."""
    logs = tmp_path / "logs"
    logs.mkdir()
    path = logs / LOG_NAME
    data = _sizing_block(8 * MIB)
    path.write_bytes(data)
    run = disc_watch.Run(run_id=RUN, log=path)

    started = time.perf_counter()
    lines = disc_watch._read_new_lines(run, terminal=False)
    elapsed = time.perf_counter() - started

    assert len(lines) == data.count(b"\n")
    assert elapsed < 0.5, f"8 MiB of Sizing lines took {elapsed:.2f}s"
# --- end chunked reads ------------------------------------------------------------
