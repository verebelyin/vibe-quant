"""The Claude Code hooks in .claude/settings.json after the board cutover (bead m0cbl.6).

SessionStart starts the board, then prints the board brief, in ONE command: Claude Code runs
an event's hooks in parallel, so a separate brief hook could read before the board is up. A
board that cannot be read prints ONE line naming board-up.sh, never nothing; so does the
prompt hook, so "no pending ask" and "board unreadable" look different. Each hook command
runs exactly as written in settings.json (``bash -c``) against a scratch project dir whose
``scripts/agents`` links to this checkout's, except ``board-up.sh``: a stub that starts a
scratch board (or fails), so the real board is never started here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.unit.conftest import UNROUTABLE_BOARD_URL, board_env, free_port

if TYPE_CHECKING:
    from tests.unit.conftest import BoardServer

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "scripts" / "agents"
SETTINGS = json.loads((ROOT / ".claude" / "settings.json").read_text())
UNAVAILABLE = "SWARM BOARD unavailable: run scripts/agents/board-up.sh"


def _commands(event: str) -> list[str]:
    return [h["command"] for group in SETTINGS["hooks"][event] for h in group["hooks"]]


def _one(event: str, needle: str) -> str:
    [cmd] = [c for c in _commands(event) if needle in c]
    return cmd


def _project(tmp_path: Path, board_up: str) -> Path:
    """A CLAUDE_PROJECT_DIR whose scripts/agents is this checkout's, with ``board_up`` as board-up.sh."""
    agents = tmp_path / "proj" / "scripts" / "agents"
    agents.mkdir(parents=True)
    for entry in AGENTS.iterdir():
        if entry.name != "board-up.sh":
            (agents / entry.name).symlink_to(entry)
    stub = agents / "board-up.sh"
    stub.write_text("#!/bin/sh\n" + board_up + "\n")
    stub.chmod(0o755)
    return tmp_path / "proj"


def _run(cmd: str, tmp_path: Path, project: Path, **env: str) -> subprocess.CompletedProcess[str]:
    full = {
        **os.environ,
        "CLAUDE_PROJECT_DIR": str(project),
        "SWARM_BOARD": str(tmp_path / "global"),
        "SWARM_JOBS": str(tmp_path / "jobs"),
        "SWARM_STORE": "board",
        "SWARM_BOARD_NS": "t" + uuid.uuid4().hex[:12],
        **env,
    }
    (tmp_path / "jobs").mkdir(exist_ok=True)
    return subprocess.run(["bash", "-c", cmd], env=full, capture_output=True, text=True, timeout=120, check=False)


@pytest.mark.slow(reason="a cold `vb serve` start inside the hook + bus.py brief with git/bd context: ~2.5 s alone")
def test_session_start_on_a_cold_board_starts_it_then_prints_the_brief(tmp_path: Path) -> None:
    port = free_port()
    data = tmp_path / "data"
    env = board_env(tmp_path / "xdg", f"http://127.0.0.1:{port}/api/board")
    ctl = f"{sys.executable} {AGENTS / 'boardctl.py'}"
    project = _project(tmp_path, f"exec {ctl} up --dir {data} --port {port} --no-inspect")
    try:
        done = _run(
            _one("SessionStart", "board brief"), tmp_path, project,
            BOARD_URL=env["BOARD_URL"], XDG_CONFIG_HOME=env["XDG_CONFIG_HOME"],
        )  # fmt: skip
    finally:
        subprocess.run([*ctl.split(), "down", "--dir", str(data)], env=env, capture_output=True, timeout=60, check=False)
    assert done.returncode == 0 and UNAVAILABLE not in done.stdout, done.stdout + done.stderr
    assert done.stdout.startswith("SWARM BOARD — shared memory") and "vb read kb/" in done.stdout


def test_session_start_prints_one_line_when_board_up_fails(tmp_path: Path) -> None:
    """board-up.sh fails (port held by another program, no Node, ...) and nothing answers."""
    project = _project(tmp_path, "echo 'boardctl up: port 4100 is in use' >&2; exit 1")
    done = _run(_one("SessionStart", "board brief"), tmp_path, project, BOARD_URL=UNROUTABLE_BOARD_URL)
    assert (done.returncode, done.stdout, done.stderr) == (0, UNAVAILABLE + "\n", "")


def test_prompt_hook_prints_one_line_when_the_board_is_unreadable(tmp_path: Path) -> None:
    project = _project(tmp_path, "exit 1")
    done = _run(_one("UserPromptSubmit", "pending --global"), tmp_path, project, BOARD_URL=UNROUTABLE_BOARD_URL)
    assert (done.returncode, done.stdout, done.stderr) == (0, UNAVAILABLE + "\n", "")


def test_prompt_hook_is_silent_with_no_ask_and_names_a_pending_one(board_server: BoardServer, tmp_path: Path) -> None:
    project = _project(tmp_path, "exit 1")
    env = {
        "BOARD_URL": board_server.env["BOARD_URL"], "XDG_CONFIG_HOME": board_server.env["XDG_CONFIG_HOME"],
        "SWARM_BOARD_NS": "t" + uuid.uuid4().hex[:12],
    }  # fmt: skip
    hook = _one("UserPromptSubmit", "pending --global")
    assert _run(hook, tmp_path, project, **env).stdout == ""
    ask = _run(f"python3 {AGENTS / 'bus.py'} --as cc-x-1 post --to chief --kind question --body 'A or B'", tmp_path, project, **env)
    assert ask.returncode == 0, ask.stderr
    done = _run(hook, tmp_path, project, **env)
    assert done.returncode == 0 and done.stderr == ""
    [line] = done.stdout.splitlines()
    assert line.startswith("UNANSWERED WORKER ASK (answer: bus.py --as chief reply --ref <id>): [")
    assert "cc-x-1 -> chief (question): A or B (" in line


def test_both_board_hooks_bound_each_board_request() -> None:
    for cmd in (_one("SessionStart", "board brief"), _one("UserPromptSubmit", "pending --global")):
        assert "SWARM_BOARD_TIMEOUT=3 python3" in cmd


def test_prompt_hook_names_the_chief_reply_command() -> None:
    assert "UNANSWERED WORKER ASK (answer: bus.py --as chief reply --ref <id>): " in _one("UserPromptSubmit", "pending --global")
