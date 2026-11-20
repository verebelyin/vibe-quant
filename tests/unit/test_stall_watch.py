"""scripts/agents/stall_watch.py — automatic stall supervisor for swarm workers."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import ModuleType


_WATCH_PATH = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "stall_watch.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stall_watch", _WATCH_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stall_watch"] = mod
    spec.loader.exec_module(mod)
    return mod


sw = _load()


def _run_watch(job: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_WATCH_PATH), "watch", "--job", str(job), *extra],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        env={**os.environ, "SWARM_BOARD": str(job / "board")},
    )


@pytest.fixture
def sleeper() -> Generator[Callable[[], subprocess.Popen[bytes]]]:
    procs: list[subprocess.Popen[bytes]] = []

    def spawn() -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        procs.append(proc)
        return proc

    yield spawn
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def _staged_worker(job: Path, pid: int, tmp_path: Path) -> Path:
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    log = tmp_path / "w1.log"
    log.write_text("thinking...\n", encoding="utf-8")
    sw.register_start(job, "w1", pid, log, cwd)
    stale = time.time() - 30 * 60
    os.utime(log, (stale, stale))
    return log


def test_once_pass_prints_stall_and_posts_to_bus(
    tmp_path: Path, sleeper: Callable[[], subprocess.Popen[bytes]]
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    proc = sleeper()
    log = _staged_worker(job, proc.pid, tmp_path)
    out = _run_watch(job, "--once")
    assert re.search(rf"STALL w1 pid={proc.pid} log_idle=\d+m edit_idle=(?:\d+m|unknown)", out.stdout)
    assert "KILLED" not in out.stdout
    assert proc.poll() is None
    msgs = [json.loads(line) for line in (job / "bus" / "messages.jsonl").read_text().splitlines()]
    assert any(
        m.get("to") == "board" and m.get("topic") == "blockers" and m.get("kind") == "blocker"
        and "w1" in str(m.get("body")) for m in msgs
    )
    assert any(m.get("to") == "chief" and m.get("kind") == "blocker" for m in msgs)
    assert log.exists()


# Runs INSIDE a child with its own session (start_new_session=True): a regression that
# killpg's the "launcher" group can then only hit this throwaway session, never pytest's.
_HARNESS = r"""
import importlib.util, os, subprocess, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location("sw", sys.argv[1])
sw = importlib.util.module_from_spec(spec)
sys.modules["sw"] = sw
spec.loader.exec_module(sw)
job, tmp, mode = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
a = subprocess.Popen(["sleep", "30"])  # same group as this launcher
b = subprocess.Popen(["sleep", "30"])
try:
    cwd = tmp / "cwd"; cwd.mkdir(exist_ok=True)
    log = tmp / "a.log"; log.write_text("x\n")
    sw.register_start(job, "a", a.pid, log, cwd)
    stale = time.time() - 1800
    os.utime(log, (stale, stale))
    if mode == "inproc":
        sw.watch(job, once=True, kill=True)
    else:
        subprocess.run([sys.executable, sys.argv[1], "watch", "--job", str(job), "--once", "--kill"],
                       start_new_session=True, check=True, capture_output=True, timeout=60)
    a.wait(timeout=10)
    assert b.poll() is None, "sibling in the launcher group was killed"
    print("HARNESS-OK")
finally:
    for p in (a, b):
        if p.poll() is None:
            p.kill()
        p.wait()
"""


def _isolated_kill(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    job = tmp_path / "job"
    job.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-c", _HARNESS, str(_WATCH_PATH), str(job), str(tmp_path), mode],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True, env={**os.environ, "SWARM_BOARD": str(job / "board")},
    )
    try:
        out, err = proc.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)  # unreaped, so the pgid is still ours
        out, err = proc.communicate()
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def test_kill_same_group_worker_spares_supervisor_group(tmp_path: Path) -> None:
    res = _isolated_kill(tmp_path, "inproc")
    assert res.returncode == 0 and "HARNESS-OK" in res.stdout, (res.returncode, res.stdout, res.stderr)


def test_signal_refuses_pid_one_and_below(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(sw.os, "kill", lambda *a: calls.append(a))
    monkeypatch.setattr(sw.os, "killpg", lambda *a: calls.append(a))
    for pid in (1, 0, -5):
        with pytest.raises(OSError):
            sw._signal_worker(pid)
    assert calls == []


def test_kill_rechecks_pstart_before_signalling(
    tmp_path: Path, sleeper: Callable[[], subprocess.Popen[bytes]], capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    proc = sleeper()  # own session: a regression can only hit this throwaway process
    worker = sw.Worker("w1", proc.pid, tmp_path / "w.log", tmp_path, time.time(), pstart="Thu Jan  1 00:00:00 1970")
    sw._kill_worker(job, worker, "body")
    out = capsys.readouterr().out
    assert "KILL-FAILED w1" in out and "KILLED w1" not in out
    assert proc.poll() is None
    assert not (job / "workers.jsonl").exists()


def test_unverifiable_start_time_never_kills_or_counts_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    calls: list[object] = []
    monkeypatch.setattr(sw.os, "kill", lambda pid, sig: calls.append((pid, sig)) if sig else None)
    monkeypatch.setattr(sw.os, "killpg", lambda *a: calls.append(a))
    monkeypatch.setattr(sw, "proc_start", lambda pid: "")
    monkeypatch.setattr(sw, "is_alive", lambda pid: True)
    worker = sw.Worker("w1", 424242, tmp_path / "w.log", tmp_path, time.time(), pstart="")
    sw._kill_worker(job, worker, "body")
    out = capsys.readouterr().out
    assert "KILL-FAILED w1 start time unverifiable" in out and "KILLED" not in out
    assert "WARN" in out
    assert calls == [] and not (job / "workers.jsonl").exists()
    monkeypatch.setattr(sw, "proc_start", lambda pid: "Mon Jan  1 00:00:00 2026")  # recorded "" only
    sw._kill_worker(job, worker, "body")
    assert "KILL-FAILED w1 start time unverifiable" in capsys.readouterr().out
    assert calls == []
    monkeypatch.setattr(sw, "proc_start", lambda pid: "")
    sw.register_start(job, "w2", 424243, "l", "c")  # recorded pstart is "" too
    assert sw.live_workers(job) == []
