"""scripts/agents/kill-stray-browsers.sh kills orphaned automation Chromium only."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "kill-stray-browsers.sh"


def _orphan(exe: Path) -> int:
    """Start ``exe 300`` detached from this process (its parent exits, so ppid becomes 1)."""
    out = subprocess.run(
        ["bash", "-c", f'"{exe}" 300 >/dev/null 2>&1 & echo $!'],
        capture_output=True, text=True, check=True,
    )
    pid = int(out.stdout.strip())
    deadline = time.monotonic() + 5
    while _ppid(pid) != "1":
        assert time.monotonic() < deadline, "the probe never got reparented"
        time.sleep(0.05)
    return pid


def _ppid(pid: int) -> str:
    return subprocess.run(
        ["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return _ppid(pid) != ""  # a zombie reaped by launchd disappears from ps


def test_kills_an_orphaned_playwright_chromium_and_spares_other_orphans(tmp_path: Path) -> None:
    stray_exe = tmp_path / "ms-playwright" / "chromium_headless_shell-1" / "headless_shell"
    other_exe = tmp_path / "Google Chrome.app" / "Google Chrome"
    for exe in (stray_exe, other_exe):
        exe.parent.mkdir(parents=True)
        exe.symlink_to("/bin/sleep")
    stray, other = _orphan(stray_exe), _orphan(other_exe)
    try:
        out = subprocess.run([str(_SCRIPT)], capture_output=True, text=True, check=True)
        deadline = time.monotonic() + 5
        while _alive(stray):
            assert time.monotonic() < deadline, "the orphaned automation browser survived"
            time.sleep(0.05)
        assert str(stray) in out.stdout
        assert _alive(other)  # the user's own Chrome (any non-automation path) is never touched
    finally:
        for pid in (stray, other):
            if _alive(pid):
                os.kill(pid, 9)
