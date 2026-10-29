"""scripts/agents/bin/timeout — GNU `timeout` for agents on macOS (it has none).

Doctor lane-t3 (Oct 14-28): `timeout: command not found` in 17 agent threads; agents copy
Linux habits. The shim is a real executable on the PATH the bus protocol exports.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_SHIM = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "bin" / "timeout"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(_SHIM), *args], capture_output=True, text=True, timeout=60)


def test_a_command_that_finishes_keeps_its_exit_code_and_output() -> None:
    proc = _run("5", sys.executable, "-c", "print('hi'); raise SystemExit(3)")
    assert proc.returncode == 3
    assert proc.stdout == "hi\n"


def test_a_command_past_its_duration_is_killed_with_124() -> None:
    start = time.monotonic()
    proc = _run("0.5", sys.executable, "-c", "import time; time.sleep(30)")
    assert proc.returncode == 124
    assert time.monotonic() - start < 10


def test_duration_suffixes_and_bad_usage() -> None:
    assert _run("1s", sys.executable, "-c", "pass").returncode == 0
    assert _run("0.01m", sys.executable, "-c", "pass").returncode == 0
    bad = _run("soon", "true")
    assert bad.returncode == 125
    assert "invalid duration" in bad.stderr
    assert _run("5").returncode == 125


def test_a_missing_command_exits_127() -> None:
    assert _run("5", "no-such-command-xyz").returncode == 127
