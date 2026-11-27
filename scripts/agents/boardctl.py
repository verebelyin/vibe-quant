#!/usr/bin/env python3
"""The swarm's local Vercel Board: pinned install, start/stop, and the `vb` launcher.

    boardctl.py up [--dir D] [--port N] [--no-inspect]   # start `vb serve` (detached) or report it up
    boardctl.py down [--dir D]                           # stop the server whose pid is in D/serve.pid
    boardctl.py vb <args...>                             # the real `vb`, behind a secret scan
    boardctl.py url [--port N]                           # print the default board URL (starts nothing)

The npm package `@vercel/board` is pinned in ``scripts/agents/board/package-lock.json`` and
installed OUTSIDE every checkout, in ``$VQ_BOARD_HOME`` or
``~/.cache/vibe-quant/board/<first 12 hex of sha256(package-lock.json)>``, so all worktrees
share one install and a version bump is a new directory. The real board lives in
``<main checkout>/.board`` on port 4100; ``--dir``/``--port`` override both (tests always do).

Every value a `vb` write sends is scanned first: the text of post|reply|edit|schedule|tasks
(argument, ``--file``, or stdin ``-``), each ``k=v`` value of `vb label` and of ``--label``,
``--idempotency-key``, ``--avatar``, and ``--attach`` files (text up to 1 MB; a larger or
binary file is refused unscanned). One that looks like a secret is refused with exit 2 before
anything is sent. ``up`` and ``down`` hold one flock on ``<dir>/boardctl.lock`` for their whole
check -> start -> ready -> pid-file transition. Never ``npx vb``: `vb` on the public npm
registry is an unrelated package.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

BOARD_PKG_DIR = Path(__file__).resolve().parent / "board"
DEFAULT_PORT = 4100
MIN_NODE = (22, 18)
_TEXT_COMMANDS = frozenset(("post", "reply", "edit", "schedule", "tasks"))
_ATTACH_MAX = 1 << 20  # bytes of attachment text the scan reads; larger is refused
_LOCK_NAME = "boardctl.lock"
# Global `vb` flags that take a value (01-cli.md § Configuration): skipped when finding the command.
_VALUE_FLAGS = frozenset(("--url", "--scope", "--team", "--name", "--as", "--avatar"))
_UP_WAIT_S = 20.0
_DOWN_WAIT_S = 5.0
_WHOAMI_TIMEOUT_S = 3.0


@functools.lru_cache(maxsize=8)
def _git_common_dir(here: str) -> str:
    """git's common dir for the directory ``here`` (as bus.py asks it)."""
    return subprocess.run(
        ["git", "-C", here, "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main_checkout() -> Path:
    """The main checkout (not a worktree), via git's common dir: every worktree finds one board."""
    return Path(_git_common_dir(str(Path(__file__).resolve().parent))).parent


def default_url(port: int = DEFAULT_PORT) -> str:
    return f"http://127.0.0.1:{port}/api/board"


def board_home() -> Path:
    """Where the pinned package is installed: $VQ_BOARD_HOME, else a dir keyed by the lock file."""
    raw = os.environ.get("VQ_BOARD_HOME")
    if raw:
        return Path(raw)
    digest = hashlib.sha256((BOARD_PKG_DIR / "package-lock.json").read_bytes()).hexdigest()[:12]
    return Path.home() / ".cache" / "vibe-quant" / "board" / digest


def _vb_bin(home: Path) -> Path:
    return home / "node_modules" / ".bin" / "vb"


def ensure_installed() -> Path:
    """Install the pinned package on first use (flock-guarded) and return the real `vb` path.

    ``.ok`` is written last, so an install that died halfway is redone (``npm ci`` starts clean).
    Raises RuntimeError with the reason when npm is missing or the install fails."""
    home = board_home()
    ok = home / ".ok"
    if ok.exists():
        return _vb_bin(home)
    home.mkdir(parents=True, exist_ok=True)
    with open(home / ".install.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ok.exists():  # another process finished it while we waited
            return _vb_bin(home)
        npm = shutil.which("npm")
        if npm is None:
            raise RuntimeError("npm not found on PATH: install Node >= 22.18 (it ships npm)")
        for name in ("package.json", "package-lock.json"):
            shutil.copyfile(BOARD_PKG_DIR / name, home / name)
        proc = subprocess.run(
            [npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
            cwd=home,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not _vb_bin(home).exists():
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
            raise RuntimeError(f"npm ci in {home} failed ({proc.returncode}): " + " | ".join(tail))
        ok.write_text("ok\n")
    return _vb_bin(home)


# ------------------------------------------------------------------ secret scan (copy of bus.py's)

_SECRET_ASSIGN_RE = re.compile(
    r"\b[A-Z0-9_]*(?:TOKEN|SECRET|API_KEY|APIKEY|PASSWORD|PRIVATE_KEY)[A-Z0-9_]*\s*[=:]\s*['\"]?[^\s'\"]{12,}"
)
_ENV_LINE_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}=\S", re.MULTILINE)


def looks_like_secret(text: str) -> bool:
    """A credential assignment, or an env dump (>= 5 KEY=value lines).

    2026-10-17: a double-quoted --body ran a backticked `env` and posted
    CLAUDE_CODE_MESSAGING_TOKEN to the bus (and through a worker's model provider)."""
    return bool(_SECRET_ASSIGN_RE.search(text)) or len(_ENV_LINE_RE.findall(text)) >= 5


def _command_index(argv: list[str]) -> int | None:
    """Index of the subcommand: the first token that is neither a flag nor a global flag's value."""
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in _VALUE_FLAGS:
            i += 2
        elif arg.startswith("-"):
            i += 1
        else:
            return i
    return None


def _wants_stdin(rest: list[str]) -> bool:
    """`-` as the text, or `--file -`."""
    return "-" in rest or "--file=-" in rest


def _flag_values(argv: list[str], flag: str) -> list[str]:
    """Every value of ``flag`` in ``argv``, as ``flag v`` or ``flag=v``."""
    out = []
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            out.append(argv[i + 1])
        elif arg.startswith(flag + "="):
            out.append(arg.split("=", 1)[1])
    return out


def _attachment_problem(path: str) -> str | None:
    """Why the attachment at ``path`` must not be sent, or None (a missing file: `vb` says so)."""
    try:
        with open(path, "rb") as f:
            data = f.read(_ATTACH_MAX + 1)
    except OSError:
        return None
    if len(data) > _ATTACH_MAX or b"\0" in data:
        return f"attachment {path} is binary or over 1 MB, so it cannot be scanned"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return f"attachment {path} is binary or over 1 MB, so it cannot be scanned"
    return "the attachment looks like a secret" if looks_like_secret(text) else None


def _secret_reason(argv: list[str], stdin: bytes | None) -> str | None:
    """Why this `vb` call must not be sent, or None. The reason names the argument kind, never the value."""
    at = _command_index(argv)
    cmd = argv[at] if at is not None else "call"
    checks: list[tuple[str, str]] = [
        ("label value", v.split("=", 1)[-1]) for v in _flag_values(argv, "--label")
    ]
    checks += [("idempotency key", v) for v in _flag_values(argv, "--idempotency-key")]
    checks += [("avatar", v) for v in _flag_values(argv, "--avatar")]
    rest = argv[at + 1 :] if at is not None else []
    if cmd == "label":
        checks += [
            ("label value", a.split("=", 1)[1]) for a in rest if "=" in a and not a.startswith("-")
        ]
    if cmd in _TEXT_COMMANDS:
        checks += [("text", a) for a in rest if a != "-"]
        for path in _flag_values(rest, "--file"):
            if path != "-":
                with contextlib.suppress(OSError):  # the real `vb` reports a missing file
                    checks.append(
                        ("text", Path(path).read_text(encoding="utf-8", errors="replace"))
                    )
        if stdin is not None:
            checks.append(("text", stdin.decode("utf-8", errors="replace")))
    for kind, value in checks:
        if looks_like_secret(value):
            return f"vb {cmd} refused: the {kind} looks like a secret (KEY=value credential or env dump); nothing was sent"
    for path in _flag_values(argv, "--attach"):
        problem = _attachment_problem(path)
        if problem is not None:
            return f"vb {cmd} refused: {problem}; nothing was sent"
    return None


def _client_env(env: dict[str, str] | None) -> dict[str, str]:
    out = dict(os.environ if env is None else env)
    out.setdefault("BOARD_URL", default_url())
    return out


def run_vb(
    argv: list[str], *, stdin: str | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the real `vb` for a Python caller, behind the same secret scan as the launcher.

    A refusal returns exit 2 with the reason on stderr and sends nothing."""
    reason = _secret_reason(argv, stdin.encode() if stdin is not None else None)
    if reason is not None:
        return subprocess.CompletedProcess(["vb", *argv], 2, "", reason + "\n")
    return subprocess.run(
        [str(ensure_installed()), *argv],
        input=stdin,
        env=_client_env(env),
        capture_output=True,
        text=True,
        check=False,
    )


KB_READS_LOG = "data/swarm/_board-reads.jsonl"


def _log_kb_read(argv: list[str], at: int | None) -> None:
    """Append one line per `vb read kb/...` to the main checkout's KB_READS_LOG (tests:
    $VQ_KB_READS_LOG; agent = $BOARD_PROFILE), so the gate's kb-search check and retros can see who searched. Best
    effort: a logging error never blocks the read."""
    if at is None or argv[at] != "read":
        return
    rest = [a for a in argv[at + 1 :] if not a.startswith("-")]
    if not rest or not rest[0].startswith("kb"):
        return
    row = {
        "ts": time.time(),
        "agent": os.environ.get("BOARD_PROFILE", ""),
        "args": argv[at + 1 :],
    }
    try:
        path = Path(os.environ.get("VQ_KB_READS_LOG") or main_checkout() / KB_READS_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except (OSError, subprocess.SubprocessError, ValueError):
        pass


def _cmd_vb(argv: list[str]) -> int:
    """The `scripts/agents/bin/vb` launcher: scan, then hand over to the real binary."""
    try:
        real = ensure_installed()
    except RuntimeError as exc:
        print(f"vb: {exc}", file=sys.stderr)
        return 1
    env = _client_env(None)
    at = _command_index(argv)
    _log_kb_read(argv, at)
    scanned = at is not None and argv[at] in _TEXT_COMMANDS and _wants_stdin(argv[at + 1 :])
    data = sys.stdin.buffer.read() if scanned else None
    reason = _secret_reason(argv, data)
    if reason is not None:
        print(reason, file=sys.stderr)
        return 2
    if data is None:
        os.execve(real, [str(real), *argv], env)
    return subprocess.run([str(real), *argv], input=data, env=env, check=False).returncode


# ------------------------------------------------------------------ server


def node_version() -> tuple[int, int, int] | None:
    """`node --version` as a tuple, or None when Node is missing or unreadable."""
    node = shutil.which("node")
    if node is None:
        return None
    out = subprocess.run([node, "--version"], capture_output=True, text=True, check=False).stdout
    m = re.match(r"v(\d+)\.(\d+)\.(\d+)", out.strip())
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def node_problem(version: tuple[int, int, int] | None) -> str | None:
    """Why this Node cannot run the board, or None when it can."""
    need = ".".join(map(str, MIN_NODE))
    if version is None:
        return f"Node not found on PATH: the board needs Node >= {need}"
    if version[:2] < MIN_NODE:
        return f"Node {'.'.join(map(str, version))} is too old: the board needs Node >= {need}"
    return None


def _board_up(real: Path, url: str) -> bool:
    """True when `vb whoami` reaches a board at ``url`` with this env's token.

    Bounded: a program that holds the port but never answers HTTP would hang `vb` ~8 s."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("BOARD_NAME", "BOARD_TOKEN", "BOARD_PROFILE")
    }
    env["BOARD_URL"] = url
    try:
        proc = subprocess.run(
            [str(real), "whoami"],
            env=env,
            capture_output=True,
            text=True,
            timeout=_WHOAMI_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def _port_free(port: int) -> bool:
    """Nothing accepts on 127.0.0.1:``port``. A connect probe, not a bind: a bind fails for
    30-60 s after `down` while the old server's connections sit in TIME_WAIT."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.connect(("127.0.0.1", port))
        except ConnectionRefusedError:
            return True
        except OSError:  # a timeout or another error: something holds the port
            return False
    return False


def _alive(pid: int) -> bool:
    with contextlib.suppress(ChildProcessError):
        if os.waitpid(pid, os.WNOHANG)[0] == pid:  # our own child that exited: reap it
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _resolve_dir(dir: Path | None) -> Path:
    return dir if dir is not None else main_checkout() / ".board"


@contextlib.contextmanager
def _dir_lock(data: Path) -> Iterator[None]:
    """One exclusive flock on ``<data>/boardctl.lock``: two `up`s (every session start runs one)
    or an `up` and a `down` never interleave their check -> start -> ready -> pid-file steps."""
    data.mkdir(parents=True, exist_ok=True)
    with open(data / _LOCK_NAME, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def up(dir: Path | None, port: int | None, inspect: bool = True) -> int:
    """Start `vb serve` detached on ``dir``/``port`` unless a board already answers there."""
    problem = node_problem(node_version())
    if problem is not None:
        print(f"boardctl up: {problem}", file=sys.stderr)
        return 1
    try:
        real = ensure_installed()
    except RuntimeError as exc:
        print(f"boardctl up: {exc}", file=sys.stderr)
        return 1
    data = _resolve_dir(dir)
    port = DEFAULT_PORT if port is None else port
    with _dir_lock(data):
        return _up_locked(real, data, port, inspect)


def _up_locked(real: Path, data: Path, port: int, inspect: bool) -> int:
    url = default_url(port)
    if not _port_free(port) and _board_up(real, url):
        print(f"board already up at {url}")
        return 0
    if not _port_free(port):
        print(
            f"boardctl up: port {port} is in use by another program (not a board this env can reach)",
            file=sys.stderr,
        )
        return 1
    cmd = [str(real), "serve", "--port", str(port), "--dir", str(data), "--no-open"]
    if not inspect:
        cmd.append("--no-inspect")
    with open(data / "serve.log", "ab") as log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (data / "serve.pid").write_text(f"{proc.pid}\n")
    deadline = time.monotonic() + _UP_WAIT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        if _board_up(real, url):
            print(f"board up at {url}")
            if inspect:
                print(f"inspector at http://127.0.0.1:{port}")
            return 0
        time.sleep(0.1)
    _stop(proc.pid)
    (data / "serve.pid").unlink(missing_ok=True)
    tail = (data / "serve.log").read_text(errors="replace").strip().splitlines()[-5:]
    print(f"boardctl up: the board did not come up at {url}: " + " | ".join(tail), file=sys.stderr)
    return 1


def _stop(pid: int) -> None:
    """SIGINT (the server's Ctrl-C path, which removes its token), then SIGTERM, then SIGKILL."""
    for sig, wait in ((signal.SIGINT, _DOWN_WAIT_S), (signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, sig)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if not _alive(pid):
                return
            time.sleep(0.05)


def _is_board_serve(pid: int) -> bool:
    """Guard against a recycled pid: the process must be a `vb serve`."""
    out = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False
    )
    return " serve " in f" {out.stdout.strip()} "


def down(dir: Path | None) -> int:
    """Stop the server whose pid is in ``<dir>/serve.pid``; exit 0 when nothing runs."""
    data = _resolve_dir(dir)
    if not data.is_dir():
        print("board not running (no serve.pid)")
        return 0
    with _dir_lock(data):  # waits for an `up` in flight, so its pid file is there to read
        return _down_locked(data / "serve.pid")


def _down_locked(pid_file: Path) -> int:
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        print("board not running (no serve.pid)")
        return 0
    if _alive(pid) and _is_board_serve(pid):
        _stop(pid)
        if _alive(pid):
            print(f"boardctl down: pid {pid} did not exit", file=sys.stderr)
            return 1
        print(f"board stopped (pid {pid})")
    else:
        print(f"board not running (stale pid {pid})")
    pid_file.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["vb"]:
        return _cmd_vb(argv[1:])
    ap = argparse.ArgumentParser(
        prog="boardctl.py", description=__doc__.split("\n\n")[0] if __doc__ else None
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_up = sub.add_parser("up", help="start the board (or report it up)")
    p_up.add_argument("--dir", type=Path, default=None)
    p_up.add_argument("--port", type=int, default=None)
    p_up.add_argument("--no-inspect", action="store_true")
    p_down = sub.add_parser("down", help="stop the board")
    p_down.add_argument("--dir", type=Path, default=None)
    sub.add_parser("vb", help="run the real vb behind the secret scan")
    p_url = sub.add_parser("url", help="print the default board URL")
    p_url.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)
    if args.cmd == "url":
        print(default_url(args.port))
        return 0
    if args.cmd == "up":
        return up(args.dir, args.port, inspect=not args.no_inspect)
    return down(args.dir)


if __name__ == "__main__":
    sys.exit(main())
