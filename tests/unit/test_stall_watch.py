"""scripts/agents/stall_watch.py — automatic stall supervisor for swarm workers."""

from __future__ import annotations

import contextlib
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


def _events(job: Path) -> list[dict[str, object]]:
    path = job / "workers.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


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


def test_registry_round_trip(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    pid = os.getpid()
    start = sw.register_start(job, "w1", pid, tmp_path / "w1.log", tmp_path / "src")
    end = sw.register_end(job, "w1", pid, 0)
    events = _events(job)
    assert [e["event"] for e in events] == ["start", "end"]
    assert events[0]["agent"] == "w1" and events[0]["pid"] == pid
    assert events[0]["log"] == str(tmp_path / "w1.log")
    assert events[0]["cwd"] == str(tmp_path / "src")
    assert events[1]["rc"] == 0
    assert events == [start, end]
    for ev in events:
        assert isinstance(ev["ts"], float) and ev["ts"] <= time.time()
    assert sw.live_workers(job) == []

    subprocess.run(
        [
            sys.executable, str(_WATCH_PATH), "register-start",
            "--job", str(job), "--agent", "w9", "--pid", str(pid),
            "--log", "a.log", "--cwd", "src",
        ],
        capture_output=True, text=True, timeout=60, check=True,
    )
    subprocess.run(
        [
            sys.executable, str(_WATCH_PATH), "register-end",
            "--job", str(job), "--agent", "w9", "--pid", str(pid), "--rc", "3",
        ],
        capture_output=True, text=True, timeout=60, check=True,
    )
    cli_events = _events(job)[2:]
    assert [e["event"] for e in cli_events] == ["start", "end"]
    assert cli_events[0]["agent"] == "w9" and cli_events[0]["log"] == "a.log"
    assert cli_events[1]["rc"] == 3
    assert sw.live_workers(job) == []


def test_liveness_current_pid_alive_dead_pid_not(tmp_path: Path) -> None:
    dead = subprocess.Popen(["sleep", "30"])
    dead.kill()
    dead.wait()
    assert sw.is_alive(os.getpid()) is True
    assert sw.is_alive(dead.pid) is False

    job = tmp_path / "job"
    job.mkdir()
    sw.register_start(job, "dead", dead.pid, "d.log", "c")
    sw.register_start(job, "alive", os.getpid(), "a.log", "c")
    live = sw.live_workers(job)
    assert [(w.agent, w.pid) for w in live] == [("alive", os.getpid())]


def test_stall_decision_table() -> None:
    now = 1_000_000.0
    th = sw.Thresholds(log_idle_min=10.0, edit_idle_min=20.0)
    cases = [
        # (name, log_mtime, log_size_history, edit_mtime, expected)
        ("log idle", now - 30 * 60, (10.0, 10.0), now - 60, True),
        ("log idle at threshold", now - 10 * 60, (10.0,), now - 60, True),
        ("log idle just below threshold", now - 9.9 * 60, (7.0, 7.0), now - 5 * 60, False),
        ("thinking without editing", now - 60, (100.0, 300.0), now - 25 * 60, True),
        ("thinking without editing at threshold", now - 60, (100.0, 101.0), now - 20 * 60, True),
        ("touched log without growth + no edits", now - 60, (300.0, 300.0), now - 25 * 60, False),
        ("edit idle below threshold with growth", now - 60, (1.0, 2.0), now - 19.9 * 60, False),
        ("healthy: recent log and recent edits", now - 60, (100.0, 200.0), now - 30, False),
        ("healthy: single sample, recent edits", now - 30, (50.0,), now - 30, False),
        ("healthy: quiet log but edited recently", now - 5 * 60, (500.0,), now - 2 * 60, False),
        ("edit idle unknown: growing log never edit-stalls", now - 60, (100.0, 300.0), None, False),
        ("edit idle unknown: log idle still stalls", now - 30 * 60, (10.0,), None, True),
        ("growth before the recent window is not growth", now - 60, (100.0, 300.0, 300.0, 300.0), now - 25 * 60, False),
        ("growth within the recent window", now - 60, (300.0, 300.0, 350.0, 350.0), now - 25 * 60, True),
    ]
    for name, log_mtime, sizes, edit_mtime, expected in cases:
        got = sw.is_stalled(now, log_mtime, sizes, edit_mtime, th)
        assert got is expected, f"{name}: is_stalled={got}, expected {expected}"


def test_recovery_then_re_stall_fires_second_alert() -> None:
    th = sw.Thresholds(log_idle_min=10.0, edit_idle_min=20.0)
    now = 10_000.0
    alerted = False
    fires: list[bool] = []

    def step(at: float, log_mtime: float, sizes: tuple[float, ...], edit_mtime: float) -> None:
        nonlocal alerted
        stalled = sw.is_stalled(at, log_mtime, sizes, edit_mtime, th)
        alerted, fire = sw.alert_once(alerted, stalled)
        fires.append(fire)

    step(now, now - 30 * 60, (10.0, 10.0), now - 30 * 60)  # silent 30m -> alert
    step(now + 60, now - 30 * 60, (10.0, 10.0), now - 30 * 60)  # still stalled -> no spam
    step(now + 120, now + 120, (10.0, 20.0), now + 120)  # recovered (log + edit)
    step(now + 120 + 30 * 60, now + 120, (20.0, 20.0), now + 120)  # silent again -> alert
    assert fires == [True, False, False, True]


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


def test_once_pass_with_kill_terminates_worker(
    tmp_path: Path, sleeper: Callable[[], subprocess.Popen[bytes]]
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    proc = sleeper()
    _staged_worker(job, proc.pid, tmp_path)
    out = _run_watch(job, "--once", "--kill")
    assert re.search(rf"STALL w1 pid={proc.pid} log_idle=\d+m edit_idle=(?:\d+m|unknown)", out.stdout)
    assert "KILLED w1" in out.stdout
    proc.wait(timeout=10)
    ends = [e for e in _events(job) if e["event"] == "end"]
    assert len(ends) == 1
    assert ends[0]["rc"] == -15
    assert "killed" in str(ends[0].get("note", "")).lower()


def test_watch_exits_when_no_live_workers(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    dead = subprocess.Popen(["sleep", "30"])
    dead.kill()
    dead.wait()
    sw.register_start(job, "w1", dead.pid, "w1.log", "c")
    sw.register_end(job, "w1", dead.pid, 0)
    proc = _run_watch(job)  # no --once: must exit on its own
    assert proc.stdout == ""


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return proc.stdout


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


def test_kill_non_leader_worker_spares_siblings_and_launcher(tmp_path: Path) -> None:
    # supervisor in ITS OWN group; workers are non-leaders of the launcher's group
    res = _isolated_kill(tmp_path, "sub")
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


def test_proc_start_uses_stable_locale_and_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    real = subprocess.run

    def spy(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return real(*args, **kwargs)  # type: ignore[call-overload,no-any-return]

    monkeypatch.setattr(sw.subprocess, "run", spy)
    sw.proc_start(os.getpid())
    env = seen["env"]
    assert isinstance(env, dict) and env["LC_ALL"] == "C" and env["TZ"] == "UTC"


def test_pstart_mismatch_on_live_pid_warns(
    tmp_path: Path, sleeper: Callable[[], subprocess.Popen[bytes]], capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    proc = sleeper()
    _staged_worker(job, proc.pid, tmp_path)
    events = _events(job)
    events[0]["pstart"] = "Thu Jan  1 00:00:00 1970"
    (job / "workers.jsonl").write_text(json.dumps(events[0]) + "\n", encoding="utf-8")
    assert sw.live_workers(job) == []
    assert f"WARN w1 pid={proc.pid}" in capsys.readouterr().out


def test_zombie_is_not_alive() -> None:
    z = subprocess.Popen(["sleep", "30"], start_new_session=True)
    z.kill()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and sw.is_alive(z.pid):
            time.sleep(0.05)
        assert sw.is_alive(z.pid) is False  # killed but unreaped = zombie
    finally:
        z.wait()


def test_pstart_mismatch_is_not_live_and_not_killed(
    tmp_path: Path, sleeper: Callable[[], subprocess.Popen[bytes]]
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    proc = sleeper()
    _staged_worker(job, proc.pid, tmp_path)
    events = _events(job)
    events[0]["pstart"] = "Thu Jan  1 00:00:00 1970"  # some other process now owns this pid
    (job / "workers.jsonl").write_text(json.dumps(events[0]) + "\n", encoding="utf-8")
    assert sw.live_workers(job) == []
    out = _run_watch(job, "--once", "--kill")
    assert "STALL" not in out.stdout
    assert "KILLED" not in out.stdout
    assert proc.poll() is None


def test_edit_activity_counts_recent_commit_in_clean_repo(tmp_path: Path) -> None:
    now = time.time()
    started = now - 30 * 60
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    f = repo / "a.txt"
    f.write_text("x", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "work")
    commit_ts = float(_git(repo, "log", "-1", "--format=%ct").strip())
    old = started - 3600
    os.utime(f, (old, old))  # the commit is the only recent evidence
    worker = sw.Worker("w1", os.getpid(), tmp_path / "w.log", repo, started)
    edit = sw.edit_idle_since(worker)
    assert edit is not None and edit >= commit_ts - 5
    th = sw.Thresholds(log_idle_min=10.0, edit_idle_min=20.0)
    assert sw.is_stalled(now, now - 60, (100.0, 300.0), edit, th) is False


def test_edit_activity_non_git_recent_file_not_stalled(tmp_path: Path) -> None:
    now = time.time()
    started = now - 30 * 60
    cwd = tmp_path / "plain"
    cwd.mkdir()
    f = cwd / "notes.txt"
    f.write_text("thinking", encoding="utf-8")
    mtime = now - 60
    os.utime(f, (mtime, mtime))
    worker = sw.Worker("w1", os.getpid(), tmp_path / "w.log", cwd, started)
    edit = sw.edit_idle_since(worker)
    assert edit is not None and edit >= mtime - 1
    th = sw.Thresholds(log_idle_min=10.0, edit_idle_min=20.0)
    assert sw.is_stalled(now, now - 60, (100.0, 300.0), edit, th) is False


def test_edit_activity_unknown_never_edit_stalls(tmp_path: Path) -> None:
    now = time.time()
    started = now - 30 * 60
    cwd = tmp_path / "empty"
    cwd.mkdir()
    worker = sw.Worker("w1", os.getpid(), tmp_path / "w.log", cwd, started)
    assert sw.edit_idle_since(worker) is None
    th = sw.Thresholds(log_idle_min=10.0, edit_idle_min=20.0)
    assert sw.is_stalled(now, now - 60, (100.0, 300.0), None, th) is False
    assert sw.is_stalled(now, now - 30 * 60, (100.0,), None, th) is True


def test_watch_loop_alerts_once_per_episode(
    tmp_path: Path, sleeper: Callable[[], subprocess.Popen[bytes]]
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    proc = sleeper()
    _staged_worker(job, proc.pid, tmp_path)
    out = _run_watch(job, "--interval-s", "0", "--max-passes", "5")
    assert out.stdout.count("STALL") == 1
    assert "KILLED" not in out.stdout
    assert proc.poll() is None


def test_kill_failed_when_no_signal_delivered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    dead = subprocess.Popen(["sleep", "30"])
    dead.kill()
    dead.wait()
    worker = sw.Worker("w1", dead.pid, tmp_path / "w.log", tmp_path, time.time())
    sw._kill_worker(job, worker, "test body")
    out = capsys.readouterr().out
    assert "KILL-FAILED w1" in out
    assert "KILLED w1" not in out
    assert not (job / "workers.jsonl").exists()


def test_changed_files_resolves_porcelain_paths_from_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    sub = repo / "sub"
    sub.mkdir(parents=True)
    _git(repo, "init", "-q")
    f = sub / "x.txt"
    f.write_text("x", encoding="utf-8")
    _git(repo, "add", "sub/x.txt")
    assert sw.changed_files(sub) == [f.resolve()]


def test_edit_activity_ignores_ignored_dirs_in_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = time.time() - 30 * 60
    repo = tmp_path / "repo"
    (repo / "junk").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / ".gitignore").write_text("junk/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    old = started - 3600
    monkeypatch.setenv("GIT_COMMITTER_DATE", f"@{int(old)} +0000")
    _git(repo, "commit", "-q", "-m", "init")
    os.utime(repo / ".gitignore", (old, old))
    (repo / "junk" / "cache.bin").write_text("other actor", encoding="utf-8")  # recent, gitignored
    # the commit predates start, so only the ignored write could (wrongly) count
    worker = sw.Worker("w1", os.getpid(), tmp_path / "w.log", repo, started)
    assert sw.edit_idle_since(worker) is None


def test_changed_files_untracked_nested_and_unicode(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "newdir").mkdir(parents=True)
    _git(repo, "init", "-q")
    inner = repo / "newdir" / "\u00e9 x.txt"
    inner.write_text("x", encoding="utf-8")
    assert sw.changed_files(repo) == [inner.resolve()]


def test_changed_files_rename_uses_new_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "old.txt").write_text("content that is long enough for rename detection\n" * 3, encoding="utf-8")
    _git(repo, "add", "old.txt")
    _git(repo, "commit", "-q", "-m", "c")
    _git(repo, "mv", "old.txt", "new.txt")
    assert sw.changed_files(repo) == [(repo / "new.txt").resolve()]


def test_non_git_walk_skips_cache_dirs(tmp_path: Path) -> None:
    started = time.time() - 30 * 60
    cwd = tmp_path / "plain"
    for d in ("__pycache__", ".mypy_cache", ".pnpm-store", ".beads"):
        (cwd / d).mkdir(parents=True)
        (cwd / d / "f").write_text("noise", encoding="utf-8")
    worker = sw.Worker("w1", os.getpid(), tmp_path / "w.log", cwd, started)
    assert sw.edit_idle_since(worker) is None
