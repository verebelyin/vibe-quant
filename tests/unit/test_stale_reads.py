"""Tests for scripts/agents/stale_reads.py — commands an agent ran against the WRONG tree.

A T3 child starts its shell in the MAIN checkout, so a command such as
`sed -n 1,80p scripts/agents/handoff_lint.py` with no `cd <worktree>` reads main's
STALE code (job 20261029-swarm-retro: cc-facts-1 did exactly that while its branch
had changed that file). The rules are literal command strings: relative repo paths
are stale unless the command cd's into the worktree; /tmp copies, docs/data reads
and absolute worktree paths are fine.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "stale_reads.py"
_spec = importlib.util.spec_from_file_location("stale_reads", _SCRIPT)
assert _spec is not None and _spec.loader is not None
stale_reads = importlib.util.module_from_spec(_spec)
sys.modules["stale_reads"] = stale_reads
_spec.loader.exec_module(stale_reads)

WT = Path("/Users/verebelyin/projects/vq-retro-facts")
MAIN = Path("/Users/verebelyin/projects/vibe-quant")
_AGENT = "cc-facts-1"
_THREAD = "thread:delegated-task:command%3Amcp%3A38fca8fb-bbef-4970-b4d6-f929b738ab98%3Adelegate-task%3A"


_DB_SEQ = 0


def _db(tmp_path: Path, rows: list[tuple[str, str, int, str | None]]) -> Path:
    """A tiny stand-in for the T3 state DB: (thread_id, type, ordinal, command) rows."""
    global _DB_SEQ
    _DB_SEQ += 1
    path = tmp_path / f"statev2-{_DB_SEQ}.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        "create table orchestration_v2_projection_turn_items "
        "(thread_id text, type text, ordinal integer, payload_json text)"
    )
    for thread_id, kind, ordinal, command in rows:
        payload = {"input": command} if command is not None else {"status": "completed"}
        con.execute(
            "insert into orchestration_v2_projection_turn_items values (?, ?, ?, ?)",
            (thread_id, kind, ordinal, json.dumps(payload)),
        )
    con.commit()
    con.close()
    return path


def _commands(tmp_path: Path, commands: list[str], agent: str = _AGENT) -> Path:
    """One command_execution row per command, ordinals in list order, for `agent`."""
    return _db(
        tmp_path,
        [(_THREAD + agent, "command_execution", i, cmd) for i, cmd in enumerate(commands)],
    )


def _stale(commands: list[str], tmp_path: Path, agent: str = _AGENT) -> list[str]:
    db = _commands(tmp_path, commands, agent)
    return stale_reads.stale_commands(db, agent, WT, MAIN)


class TestStaleCommands:
    def test_a_relative_read_with_no_cd_is_stale(self, tmp_path: Path) -> None:
        """T3 starts the child in main, so a bare relative path reads main."""
        assert _stale(["sed -n 1,80p scripts/agents/handoff_lint.py"], tmp_path) == [
            "sed -n 1,80p scripts/agents/handoff_lint.py"
        ]

    def test_pytest_with_pythonpath_pwd_and_no_cd_is_stale(self, tmp_path: Path) -> None:
        cmd = "PYTHONPATH=$PWD .venv/bin/python -m pytest tests/unit/test_x.py"
        assert _stale([cmd], tmp_path) == [cmd]

    def test_absolute_main_code_path_is_stale(self, tmp_path: Path) -> None:
        for cmd in (
            f"sed -n 1p {MAIN}/vibe_quant/metrics.py",
            f"head -20 {MAIN}/tests/unit/test_x.py",
        ):
            assert _stale([cmd], tmp_path) == [cmd], cmd

class TestCli:
    def _run(self, tmp_path: Path, *argv: str) -> int:
        return stale_reads.main([*argv])

    def test_missing_db_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = self._run(
            tmp_path, "--agent", _AGENT, "--worktree", str(WT),
            "--db", str(tmp_path / "nope.sqlite"),
        )  # fmt: skip
        assert rc == 2
        assert "no T3 history db" in capsys.readouterr().err

    def test_the_script_entry_point_exits_1_on_a_stale_command(self, tmp_path: Path) -> None:
        """The CLI runs as a script (`stale_reads.py --agent ...`), not only via main()."""
        db = _commands(tmp_path, ["cat vibe_quant/a.py"])
        proc = subprocess.run(
            [
                sys.executable, str(_SCRIPT), "--agent", _AGENT,
                "--worktree", str(WT), "--db", str(db),
            ],  # fmt: skip
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 1, proc.stderr
        assert proc.stdout.splitlines() == ["cat vibe_quant/a.py", "STALE 1 of 1 commands"]


class TestUnchangedPathReadExemption:
    """A plain read of paths byte-identical in the worktree and main is not stale.

    Four makers in one job failed the gate on `sed`/`rg` of main's swarm tooling that their
    branches never changed, costing a chief ruling and a gate rerun each time.
    """

    def _stale_u(self, commands: list[str], tmp_path: Path, changed: set[str]) -> list[str]:
        db = _commands(tmp_path, commands)
        stale, _total = stale_reads.stale_report(
            db, _AGENT, WT, MAIN, unchanged=lambda p, names_only=False: p not in changed
        )
        return stale

    def test_reads_of_changed_paths_stay_stale(self, tmp_path: Path) -> None:
        changed = {"scripts/agents/handoff_lint.py", "vibe_quant/x.py"}
        for cmd in (
            f"cd {MAIN} && sed -n 1,80p scripts/agents/handoff_lint.py",
            f"cd {MAIN} && cat scripts/agents/bus.py vibe_quant/x.py",
            f"cd {MAIN} && sed -n 1p scripts/agents/bus.py && cat vibe_quant/x.py",
        ):
            assert self._stale_u([cmd], tmp_path, changed=changed) == [cmd], cmd
