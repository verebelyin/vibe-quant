#!/usr/bin/env python3
"""The swarm's local Vercel Board: pinned install, start/stop, and the `vb` launcher.

    boardctl.py up [--dir D] [--port N] [--no-inspect]   # start `vb serve` (detached) or report it up
    boardctl.py down [--dir D]                           # stop the server whose pid is in D/serve.pid
    boardctl.py vb <args...>                             # the real `vb`, behind a secret scan

The npm package `@vercel/board` is pinned in ``scripts/agents/board/package-lock.json`` and
installed OUTSIDE every checkout, in ``$VQ_BOARD_HOME`` or
``~/.cache/vibe-quant/board/<first 12 hex of sha256(package-lock.json)>``, so all worktrees
share one install and a version bump is a new directory. The real board lives in
``<main checkout>/.board`` on port 4100; ``--dir``/``--port`` override both (tests always do).

`vb post|reply|edit|schedule` text (argument, ``--file``, or stdin ``-``) that looks like a
secret is refused with exit 2 before anything is sent. Never ``npx vb``: `vb` on the public
npm registry is an unrelated package.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import hashlib
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

BOARD_PKG_DIR = Path(__file__).resolve().parent / "board"
DEFAULT_PORT = 4100
MIN_NODE = (22, 18)
_TEXT_COMMANDS = frozenset(("post", "reply", "edit", "schedule"))
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
        capture_output=True, text=True, check=True,
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
            cwd=home, capture_output=True, text=True, check=False,
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


def _secret_reason(argv: list[str], stdin: bytes | None) -> str | None:
    """Why this `vb` call must not be sent, or None. Only post/reply/edit/schedule are scanned."""
    at = _command_index(argv)
    if at is None or argv[at] not in _TEXT_COMMANDS:
        return None
    rest = argv[at + 1 :]
    texts = [a for a in rest if a != "-"]
    for i, arg in enumerate(rest):
        path = None
        if arg == "--file" and i + 1 < len(rest):
            path = rest[i + 1]
        elif arg.startswith("--file="):
            path = arg.split("=", 1)[1]
        if path is not None and path != "-":
            with contextlib.suppress(OSError):  # the real `vb` reports a missing file
                texts.append(Path(path).read_text(encoding="utf-8", errors="replace"))
    if stdin is not None:
        texts.append(stdin.decode("utf-8", errors="replace"))
    if any(looks_like_secret(t) for t in texts):
        return (
            f"vb {argv[at]} refused: the text looks like a secret (KEY=value credential or env dump); "
            "nothing was sent"
        )
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
        [str(ensure_installed()), *argv], input=stdin, env=_client_env(env),
        capture_output=True, text=True, check=False,
    )


def _cmd_vb(argv: list[str]) -> int:
    """The `scripts/agents/bin/vb` launcher: scan, then hand over to the real binary."""
    try:
        real = ensure_installed()
    except RuntimeError as exc:
        print(f"vb: {exc}", file=sys.stderr)
        return 1
    env = _client_env(None)
    at = _command_index(argv)
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
    env = {k: v for k, v in os.environ.items() if k not in ("BOARD_NAME", "BOARD_TOKEN", "BOARD_PROFILE")}
    env["BOARD_URL"] = url
    try:
        proc = subprocess.run(
            [str(real), "whoami"], env=env, capture_output=True, text=True,
            timeout=_WHOAMI_TIMEOUT_S, check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


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
    url = default_url(port)
    if not _port_free(port) and _board_up(real, url):
        print(f"board already up at {url}")
        return 0
    if not _port_free(port):
        print(f"boardctl up: port {port} is in use by another program (not a board this env can reach)", file=sys.stderr)
        return 1
    data.mkdir(parents=True, exist_ok=True)
    cmd = [str(real), "serve", "--port", str(port), "--dir", str(data), "--no-open"]
    if not inspect:
        cmd.append("--no-inspect")
    with open(data / "serve.log", "ab") as log:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
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
    out = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False)
    return " serve " in f" {out.stdout.strip()} "


def down(dir: Path | None) -> int:
    """Stop the server whose pid is in ``<dir>/serve.pid``; exit 0 when nothing runs."""
    pid_file = _resolve_dir(dir) / "serve.pid"
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
    ap = argparse.ArgumentParser(prog="boardctl.py", description=__doc__.split("\n\n")[0] if __doc__ else None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_up = sub.add_parser("up", help="start the board (or report it up)")
    p_up.add_argument("--dir", type=Path, default=None)
    p_up.add_argument("--port", type=int, default=None)
    p_up.add_argument("--no-inspect", action="store_true")
    p_down = sub.add_parser("down", help="stop the board")
    p_down.add_argument("--dir", type=Path, default=None)
    sub.add_parser("vb", help="run the real vb behind the secret scan")
    args = ap.parse_args(argv)
    if args.cmd == "up":
        return up(args.dir, args.port, inspect=not args.no_inspect)
    return down(args.dir)


if __name__ == "__main__":
    sys.exit(main())
