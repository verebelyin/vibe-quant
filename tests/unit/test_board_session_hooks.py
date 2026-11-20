"""The Claude Code hooks in .claude/settings.json after the board cutover (bead m0cbl.6).

SessionStart starts the board, then prints the board brief: a board that cannot be read
prints ONE line naming board-up.sh, never nothing. Each hook command runs exactly as written
in settings.json (``bash -c``, ``CLAUDE_PROJECT_DIR`` = this checkout); the board-up hook is
never run here (it would start the real board).
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.unit.conftest import BoardServer

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = json.loads((ROOT / ".claude" / "settings.json").read_text())
UNAVAILABLE = "SWARM BOARD unavailable: run scripts/agents/board-up.sh"


def _commands(event: str) -> list[str]:
    return [h["command"] for group in SETTINGS["hooks"][event] for h in group["hooks"]]


def _one(event: str, needle: str) -> str:
    [cmd] = [c for c in _commands(event) if needle in c]
    return cmd


def _run(cmd: str, tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    full = {
        **os.environ,
        "CLAUDE_PROJECT_DIR": str(ROOT),
        "SWARM_BOARD": str(tmp_path / "global"),
        "SWARM_JOBS": str(tmp_path / "jobs"),
        **env,
    }
    (tmp_path / "jobs").mkdir(exist_ok=True)
    return subprocess.run(["bash", "-c", cmd], env=full, capture_output=True, text=True, timeout=60, check=False)


def test_session_start_starts_the_board_before_the_brief() -> None:
    cmds = _commands("SessionStart")
    up = next(i for i, c in enumerate(cmds) if "board-up.sh" in c)
    brief = next(i for i, c in enumerate(cmds) if "board brief" in c)
    assert up < brief
    assert cmds[up].endswith("|| true") and ">/dev/null 2>&1" in cmds[up]


def test_brief_hook_prints_one_line_when_the_board_is_down(tmp_path: Path) -> None:
    """BOARD_URL is the conftest's unroutable URL: nothing answers there."""
    done = _run(_one("SessionStart", "board brief"), tmp_path, SWARM_STORE="board")
    assert (done.returncode, done.stdout, done.stderr) == (0, UNAVAILABLE + "\n", "")


@pytest.mark.slow(reason="a real `vb serve` start (session board_server) + bus.py brief with git/bd context: ~2.3 s alone")
def test_brief_hook_prints_the_brief_when_the_board_is_up(board_server: BoardServer, tmp_path: Path) -> None:
    done = _run(
        _one("SessionStart", "board brief"), tmp_path,
        SWARM_STORE="board", SWARM_BOARD_NS="t" + uuid.uuid4().hex[:12],
        BOARD_URL=board_server.env["BOARD_URL"], XDG_CONFIG_HOME=board_server.env["XDG_CONFIG_HOME"],
    )  # fmt: skip
    assert done.returncode == 0 and UNAVAILABLE not in done.stdout
    assert done.stdout.startswith("SWARM BOARD — shared memory") and "vb read kb/" in done.stdout


def test_prompt_hook_names_the_chief_reply_command() -> None:
    assert "UNANSWERED WORKER ASK (answer: bus.py --as chief reply --ref <id>): " in _one("UserPromptSubmit", "pending --global")
