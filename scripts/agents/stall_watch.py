#!/usr/bin/env python3
"""Automatic stall supervisor for swarm workers.

Workers can register via the CLI (the orchestrator wires ``cmd-task.sh``) into ``<job>/workers.jsonl``:

    stall_watch.py register-start --job <job_dir> --agent <name> --pid <pid> --log <path> --cwd <path>
    stall_watch.py register-end   --job <job_dir> --agent <name> --pid <pid> --rc <int> [--note TEXT]

Then the supervisor watches the job until every live worker has ended:

    stall_watch.py watch --job <job_dir> [--log-idle-min 10] [--edit-idle-min 20] [--interval-s 30] [--kill] [--once]

Live workers are start events without a matching end whose pid is still alive and
whose process start time (recorded as ``pstart``) still matches — a reused pid is
treated as dead and never killed.
A worker STALLs when its log has been silent for >= --log-idle-min minutes, or
(the log has grown within the last few samples AND its files have not been edited
for >= --edit-idle-min minutes = "thinking without editing"; when edit activity is
UNKNOWN — no evidence of any edit at all — that stall never fires). Each stall
episode is reported once:

    STALL <agent> pid=<pid> log_idle=<m>m edit_idle=<m>m|unknown

and posted (best effort) to the job bus on #blockers plus a DM to chief; with
--kill the worker gets SIGTERM — killpg only when it leads its own group
(pgid == pid) that is not the supervisor's — plus register_end(rc=-15) when the signal was delivered, else
a KILL-FAILED line.
Exits when no live workers remain (or after one pass with --once).
Stdlib only; runs with any python3.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REGISTRY_NAME = "workers.jsonl"
KILL_RC = -15
LOG_GROWTH_WINDOW = 3
WALK_SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "data", "__pycache__", ".pnpm-store", ".beads"})
WALK_CAP = 5000


@dataclass(frozen=True)
class Thresholds:
    log_idle_min: float = 10.0
    edit_idle_min: float = 20.0


@dataclass(frozen=True)
class Worker:
    agent: str
    pid: int
    log: Path
    cwd: Path
    started: float
    pstart: str = ""


def _append_event(job_dir: Path, event: dict[str, object]) -> dict[str, object]:
    path = job_dir / REGISTRY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return event


def _ps_env() -> dict[str, str]:
    """``ps`` output varies with locale/timezone; pin both so recorded and fresh values compare."""
    return {**os.environ, "LC_ALL": "C", "TZ": "UTC"}


def proc_start(pid: int) -> str:
    """Process start time as stable ``ps`` text ('' when unavailable)."""
    try:
        proc = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=10, check=False, env=_ps_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def register_start(job_dir: Path, agent: str, pid: int, log: str | Path, cwd: str | Path) -> dict[str, object]:
    return _append_event(
        job_dir,
        {
            "event": "start", "agent": agent, "pid": pid, "log": str(log), "cwd": str(cwd),
            "pstart": proc_start(pid), "ts": time.time(),
        },
    )


def register_end(job_dir: Path, agent: str, pid: int, rc: int, note: str | None = None) -> dict[str, object]:
    event: dict[str, object] = {"event": "end", "agent": agent, "pid": pid, "rc": rc, "ts": time.time()}
    if note:
        event["note"] = note
    return _append_event(job_dir, event)


def read_registry(job_dir: Path) -> list[dict[str, object]]:
    path = job_dir / REGISTRY_NAME
    if not path.exists():
        return []
    out: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def is_alive(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return not _is_zombie(pid)


def _is_zombie(pid: int) -> bool:
    try:
        proc = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True, text=True, timeout=10, check=False, env=_ps_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.strip().startswith("Z")


_WARNED: set[tuple[str, int]] = set()


def _as_int(value: object) -> int:
    return int(str(value))


def live_workers(job_dir: Path) -> list[Worker]:
    open_workers: list[Worker] = []
    for event in read_registry(job_dir):
        if event.get("event") == "start":
            open_workers.append(
                Worker(
                    agent=str(event.get("agent", "")),
                    pid=_as_int(event.get("pid", 0)),
                    log=Path(str(event.get("log", ""))),
                    cwd=Path(str(event.get("cwd", ""))),
                    started=float(str(event.get("ts", 0.0))),
                    pstart=str(event.get("pstart", "")),
                )
            )
        elif event.get("event") == "end":
            key = (str(event.get("agent", "")), _as_int(event.get("pid", 0)))
            for i, w in enumerate(open_workers):
                if (w.agent, w.pid) == key:
                    del open_workers[i]
                    break
    live: list[Worker] = []
    for w in open_workers:
        if not is_alive(w.pid):
            continue
        fresh = proc_start(w.pid)
        if fresh and fresh == w.pstart:
            live.append(w)
        elif (w.agent, w.pid) not in _WARNED:
            _WARNED.add((w.agent, w.pid))
            why = "start time differs from registration" if fresh and w.pstart else "start time unverifiable"
            print(f"WARN {w.agent} pid={w.pid} alive but {why}; not supervised", flush=True)
    return live


def is_stalled(
    now: float,
    log_mtime: float,
    log_size_history: Sequence[float],
    edit_mtime: float | None,
    thresholds: Thresholds,
) -> bool:
    """Stall decision, pure in its inputs.

    Stalled when the log has been silent for >= log_idle_min ("dead quiet") or
    when nothing has been edited for >= edit_idle_min while the log has grown
    within the recent samples ("thinking without editing"). ``log_size_history``
    is the chronological list of observed log sizes; growth counts only inside
    the recent window, not cumulatively since the first sample. ``edit_mtime``
    is the newest edit evidence, or None when edit activity is UNKNOWN — an
    unknown edit-idle never triggers the edit stall.
    """
    log_idle_min = thresholds.log_idle_min * 60
    edit_idle_min = thresholds.edit_idle_min * 60
    if now - log_mtime >= log_idle_min:
        return True
    if edit_mtime is None:
        return False
    window = log_size_history[-LOG_GROWTH_WINDOW:]
    log_grew = len(window) >= 2 and window[-1] > window[0]
    return now - edit_mtime >= edit_idle_min and log_grew


def alert_once(alerted: bool, stalled: bool) -> tuple[bool, bool]:
    """Episode tracking: (new_alerted, fire_alert). One alert per stall episode; re-alert after recovery."""
    fire = stalled and not alerted
    return stalled, fire


def _repo_root(cwd: Path) -> Path | None:
    """git toplevel for ``cwd``, or None outside a git repo."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    top = proc.stdout.strip()
    return Path(top).resolve() if top else None


def changed_files(cwd: Path) -> list[Path]:
    """Uncommitted changed files under ``cwd`` (empty outside a git repo).

    gitignore-aware (``git status``), every untracked file listed, NUL-separated.
    git paths are repo-root-relative: they resolve against the toplevel from
    ``rev-parse --show-toplevel``, not against cwd.
    """
    root = _repo_root(cwd)
    if root is None:
        return []
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "status", "--porcelain", "-z", "--untracked-files=all"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    base = cwd.resolve()
    out: list[Path] = []
    fields = proc.stdout.split("\0")
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if len(entry) < 4:
            continue
        if entry[0] in "RC" or entry[1] in "RC":
            i += 1  # rename/copy: the next field is the origin path
        path = (root / entry[3:]).resolve()
        if path.is_relative_to(base):
            out.append(path)
    return out


def _last_commit_ts(cwd: Path) -> float | None:
    """Epoch seconds of the newest commit from ``cwd`` (None outside git or without commits)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "log", "-1", "--format=%ct"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _is_cache_dir(name: str) -> bool:
    return name.startswith(".") and name.endswith("_cache")


def _newest_modified_since(cwd: Path, since: float) -> float | None:
    """Newest mtime under ``cwd`` newer than ``since`` (bounded walk, non-git dirs only)."""
    newest: float | None = None
    budget = WALK_CAP
    stack = [cwd]
    while stack and budget > 0:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    if budget <= 0:
                        break
                    budget -= 1
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in WALK_SKIP_DIRS and not _is_cache_dir(entry.name):
                            stack.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    try:
                        mtime = entry.stat().st_mtime
                    except OSError:
                        continue
                    if mtime > since and (newest is None or mtime > newest):
                        newest = mtime
        except OSError:
            continue
    return newest


def edit_idle_since(worker: Worker) -> float | None:
    """Newest evidence of the worker's edits, or None when edit activity is UNKNOWN.

    Evidence in a git repo: mtimes of uncommitted changed files (gitignore-aware,
    so other actors' ignored/state files do not count) and the last commit when
    newer than start. Outside git: a bounded walk for files modified after
    start. A committed-clean tree is not "idle since start"; with no evidence at
    all the caller must treat edit-idle as UNKNOWN.
    """
    times: list[float] = []
    in_git = _repo_root(worker.cwd) is not None
    for path in changed_files(worker.cwd):
        try:
            times.append(path.stat().st_mtime)
        except OSError:
            continue
    commit_ts = _last_commit_ts(worker.cwd)
    if commit_ts is not None and commit_ts > worker.started:
        times.append(commit_ts)
    if not in_git:
        walked = _newest_modified_since(worker.cwd, worker.started)
        if walked is not None:
            times.append(walked)
    return max(times) if times else None


def _post_alerts(job_dir: Path, body: str) -> None:
    """Best effort: #blockers board post + DM to chief on the job bus."""
    bus_py = Path(__file__).resolve().parent / "bus.py"
    base = [sys.executable, str(bus_py), "--bus", str(job_dir / "bus"), "--as", "stall-watch"]
    commands = [
        [*base, "board", "post", "--topic", "blockers", "--kind", "blocker", "--body", body],
        [*base, "post", "--to", "chief", "--kind", "blocker", "--body", body],
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, capture_output=True, timeout=60, check=False)
        except (OSError, subprocess.SubprocessError):
            continue


def _signal_worker(pid: int) -> None:
    """SIGTERM a worker without ever signalling a group it merely shares.

    killpg only when the worker leads its own group (pgid == pid) and that is not
    the supervisor's. Otherwise (shared launcher group, or our own) a targeted
    ``os.kill``. pid <= 1 (``os.kill`` would broadcast to a group, or hit init)
    and our own pid are refused outright.
    """
    if pid <= 1 or pid == os.getpid():
        raise OSError(f"refusing to signal pid {pid}")
    pg = os.getpgid(pid)
    if pg == pid and pg != os.getpgid(0):
        os.killpg(pg, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)


def _kill_worker(job_dir: Path, worker: Worker, body: str) -> None:
    # re-verify identity right before signalling (the pid may have exited or been reused since the pass began)
    if not is_alive(worker.pid):
        print(f"KILL-FAILED {worker.agent} pid gone or reused", flush=True)
        return
    fresh = proc_start(worker.pid)
    if not fresh or not worker.pstart:  # "" == "" would pass the identity check unverified
        print(f"WARN {worker.agent} pid={worker.pid} start time unverifiable; not killing", flush=True)
        print(f"KILL-FAILED {worker.agent} start time unverifiable", flush=True)
        return
    if fresh != worker.pstart:
        print(f"KILL-FAILED {worker.agent} pid gone or reused", flush=True)
        return
    try:
        _signal_worker(worker.pid)
    except OSError as exc:
        print(f"KILL-FAILED {worker.agent} {exc.strerror or exc}", flush=True)
        return
    register_end(job_dir, worker.agent, worker.pid, KILL_RC, note=f"killed by stall_watch: {body}")
    print(f"KILLED {worker.agent}", flush=True)


def watch(
    job_dir: Path,
    *,
    log_idle_min: float = 10.0,
    edit_idle_min: float = 20.0,
    interval_s: float = 30.0,
    kill: bool = False,
    once: bool = False,
    max_passes: int | None = None,
) -> int:
    thresholds = Thresholds(log_idle_min=log_idle_min, edit_idle_min=edit_idle_min)
    alerted: dict[tuple[str, int], bool] = {}
    sizes: dict[tuple[str, int], list[float]] = {}
    passes = 0
    while True:
        passes += 1
        workers = live_workers(job_dir)
        if not workers:
            return 0
        live_keys = {(w.agent, w.pid) for w in workers}
        for key in list(alerted):
            if key not in live_keys:
                del alerted[key]
                sizes.pop(key, None)
        for worker in workers:
            key = (worker.agent, worker.pid)
            now = time.time()
            log_mtime = worker.log.stat().st_mtime if worker.log.exists() else worker.started
            log_size = float(worker.log.stat().st_size) if worker.log.exists() else 0.0
            sizes.setdefault(key, []).append(log_size)
            edit_mtime = edit_idle_since(worker)
            stalled = is_stalled(now, log_mtime, sizes[key], edit_mtime, thresholds)
            alerted[key], fire = alert_once(alerted.get(key, False), stalled)
            if not fire:
                continue
            log_idle_m = (now - log_mtime) / 60
            edit_idle = "unknown" if edit_mtime is None else f"{(now - edit_mtime) / 60:.0f}m"
            print(
                f"STALL {worker.agent} pid={worker.pid} log_idle={log_idle_m:.0f}m edit_idle={edit_idle}",
                flush=True,
            )
            body = (
                f"{worker.agent} stalled: log idle {log_idle_m:.0f}m, "
                f"edit idle {edit_idle} (pid {worker.pid})"
            )
            _post_alerts(job_dir, body)
            if kill:
                _kill_worker(job_dir, worker, body)
        if once:
            return 0
        if max_passes is not None and passes >= max_passes:
            return 0
        time.sleep(interval_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="stall_watch.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    rs = sub.add_parser("register-start", help="record a worker start in <job>/workers.jsonl")
    rs.add_argument("--job", type=Path, required=True)
    rs.add_argument("--agent", required=True)
    rs.add_argument("--pid", type=int, required=True)
    rs.add_argument("--log", required=True)
    rs.add_argument("--cwd", required=True)
    re_end = sub.add_parser("register-end", help="record a worker end in <job>/workers.jsonl")
    re_end.add_argument("--job", type=Path, required=True)
    re_end.add_argument("--agent", required=True)
    re_end.add_argument("--pid", type=int, required=True)
    re_end.add_argument("--rc", type=int, required=True)
    re_end.add_argument("--note")
    w = sub.add_parser("watch", help="supervise live workers until they all end")
    w.add_argument("--job", type=Path, required=True)
    w.add_argument("--log-idle-min", type=float, default=10.0)
    w.add_argument("--edit-idle-min", type=float, default=20.0)
    w.add_argument("--interval-s", type=float, default=30.0)
    w.add_argument("--kill", action="store_true", help="SIGTERM a stalled worker (killpg only for a worker leading its own group)")
    w.add_argument("--once", action="store_true", help="a single pass, then exit")
    w.add_argument("--max-passes", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.cmd == "register-start":
        print(json.dumps(register_start(args.job, args.agent, args.pid, args.log, args.cwd)))
    elif args.cmd == "register-end":
        print(json.dumps(register_end(args.job, args.agent, args.pid, args.rc, note=args.note)))
    elif args.cmd == "watch":
        return watch(
            args.job,
            log_idle_min=args.log_idle_min,
            edit_idle_min=args.edit_idle_min,
            interval_s=args.interval_s,
            kill=args.kill,
            once=args.once,
            max_passes=args.max_passes,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
