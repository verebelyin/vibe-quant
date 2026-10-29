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
    def test_cc_facts_example_is_stale(self, tmp_path: Path) -> None:
        """The job's real failure: cd into MAIN, then read a repo-relative code path."""
        cmd = f"cd {MAIN} && set -o pipefail; sed -n 1,80p scripts/agents/handoff_lint.py"
        assert _stale([cmd], tmp_path) == [cmd]

    def test_a_relative_read_with_no_cd_is_stale(self, tmp_path: Path) -> None:
        """T3 starts the child in main, so a bare relative path reads main."""
        assert _stale(["sed -n 1,80p scripts/agents/handoff_lint.py"], tmp_path) == [
            "sed -n 1,80p scripts/agents/handoff_lint.py"
        ]

    def test_pytest_with_pythonpath_pwd_and_no_cd_is_stale(self, tmp_path: Path) -> None:
        cmd = "PYTHONPATH=$PWD .venv/bin/python -m pytest tests/unit/test_x.py"
        assert _stale([cmd], tmp_path) == [cmd]

    def test_mypy_and_ruff_with_pythonpath_pwd_are_stale_without_cd(self, tmp_path: Path) -> None:
        for cmd in (
            "PYTHONPATH=$PWD .venv/bin/mypy vibe_quant/",
            "PYTHONPATH=$PWD .venv/bin/ruff check frontend/src",
        ):
            assert _stale([cmd], tmp_path) == [cmd], cmd

    def test_absolute_main_code_path_is_stale(self, tmp_path: Path) -> None:
        for cmd in (
            f"sed -n 1p {MAIN}/vibe_quant/metrics.py",
            f"head -20 {MAIN}/tests/unit/test_x.py",
        ):
            assert _stale([cmd], tmp_path) == [cmd], cmd

    def test_the_brief_blessed_absolute_main_tooling_call_is_fine(self, tmp_path: Path) -> None:
        """A brief tells a child to run `python3 <main>/scripts/agents/bus.py`: shared
        tooling from main is intended, so an ABSOLUTE scripts/ path is never stale."""
        cmd = (
            f"set -o pipefail; python3 {MAIN}/scripts/agents/bus.py --bus "
            f"{MAIN}/data/swarm/job/bus --as cc-board-1 inbox"
        )
        assert _stale([cmd], tmp_path) == []

    def test_the_brief_blessed_board_search_is_fine(self, tmp_path: Path) -> None:
        """Every brief tells the child to run `bus board search --path '<scope>'` first; a
        --path VALUE is a search needle, not a file the command reads."""
        cmd = (
            f"set -o pipefail; python3 {MAIN}/scripts/agents/bus.py --bus {MAIN}/data/swarm/job/bus "
            f"--as cc-board-1 board search --global --path 'scripts/agents/bus.py'"
        )
        assert _stale([cmd], tmp_path) == []

    def test_running_main_swarm_tooling_is_fine(self, tmp_path: Path) -> None:
        """Chief ruling R16: the bus protocol orders `python3 scripts/agents/<x>.py` from main
        (and the gate command is a bare `scripts/agents/swarm-check.sh`), so RUNNING the shared
        swarm tooling is exempt even with a relative path and a cd into main."""
        for cmd in (
            f"cd {MAIN} && python3 scripts/agents/bus.py --bus data/swarm/job/bus inbox",
            f"cd {MAIN} && .venv/bin/python scripts/agents/handoff_lint.py handoff.md",
            "scripts/agents/swarm-check.sh /Users/verebelyin/projects/vq-retro-facts --scope a.py",
        ):
            assert _stale([cmd], tmp_path) == [], cmd

    def test_a_bus_message_body_quoting_paths_is_fine(self, tmp_path: Path) -> None:
        """A DM/board body is prose: paths inside it are not reads (the bus protocol owns it)."""
        cmd = (
            f"cd {MAIN} && python3 scripts/agents/bus.py --bus data/swarm/job/bus --as cc-x-1 "
            f"ask --to chief --body 'I read scripts/agents/bus.py and tests/unit/test_x.py'"
        )
        assert _stale([cmd], tmp_path) == []

    def test_reading_main_code_from_main_is_still_stale(self, tmp_path: Path) -> None:
        """Reads of scripts/, tests/ and vibe_quant/ from main stay stale (R16)."""
        for cmd in (
            f"cd {MAIN} && sed -n 1,80p scripts/agents/handoff_lint.py",
            f"cd {MAIN} && cat tests/unit/test_x.py",
            f"cd {MAIN} && rg -n mutex vibe_quant/",
            f"cd {MAIN} && head -5 scripts/agents/bus.py",
        ):
            assert _stale([cmd], tmp_path) == [cmd], cmd

    def test_test_and_lint_runs_from_main_are_still_stale(self, tmp_path: Path) -> None:
        for cmd in (
            f"cd {MAIN} && PYTHONPATH=$PWD .venv/bin/python -m pytest tests/unit/test_x.py",
            f"cd {MAIN} && PYTHONPATH=$PWD .venv/bin/mypy vibe_quant",
        ):
            assert _stale([cmd], tmp_path) == [cmd], cmd

    def test_a_board_search_scope_loop_is_fine(self, tmp_path: Path) -> None:
        """The real cc-caps-1 command: a scope list looped into the exempt `bus board search`
        is data, not a read of main."""
        cmd = (
            "cd /Users/verebelyin/projects/vibe-quant && for p in scripts/agents/mutate.py "
            "tests/unit/test_mutate.py scripts/agents/swarm_check.py tests/unit/test_swarm_check.py "
            "docs/orchestration/prompts/bus-protocol.md docs/orchestration/prompts/cmd-task.md; "
            'do echo "=== $p"; python3 scripts/agents/bus.py --bus data/swarm/20261029-swarm-retro/bus '
            '--as cc-caps-1 board search --global --path "$p" --topic gotchas findings decisions '
            "2>&1 | head -40; done"
        )
        assert _stale([cmd], tmp_path) == []

    def test_a_bus_shell_function_wrapping_the_tooling_is_fine(self, tmp_path: Path) -> None:
        """Briefs define `bus() { python3 <main>/scripts/agents/bus.py ... "$@"; }` first."""
        cmd = (
            f'set -o pipefail; bus() {{ python3 {MAIN}/scripts/agents/bus.py --bus {MAIN}/data/b '
            '--as cc-x-1 "$@"; }; bus inbox'
        )
        assert _stale([cmd], tmp_path) == []

    def test_a_scope_loop_that_reads_the_files_is_stale(self, tmp_path: Path) -> None:
        cmd = f"cd {MAIN} && for p in scripts/agents/bus.py docs/x.md; do cat $p; done"
        assert _stale([cmd], tmp_path) == [cmd]

    def test_pythonpath_pwd_before_exempt_tooling_is_fine(self, tmp_path: Path) -> None:
        cmd = f"cd {MAIN} && PYTHONPATH=$PWD python3 scripts/agents/handoff_lint.py handoff.md"
        assert _stale([cmd], tmp_path) == []

    def test_direct_pytest_mypy_ruff_runs_from_main_are_stale(self, tmp_path: Path) -> None:
        for cmd in (
            ".venv/bin/python -m pytest",
            ".venv/bin/mypy",
            ".venv/bin/ruff check",
            f"cd {MAIN} && uv run pytest -q",
        ):
            assert _stale([cmd], tmp_path) == [cmd], cmd

    def test_direct_runs_in_the_worktree_or_a_scratch_copy_are_fine(self, tmp_path: Path) -> None:
        for cmd in (
            f"cd {WT} && .venv/bin/ruff check",
            "cd /tmp/probe && .venv/bin/python -m pytest",
            f"cd {MAIN} && rg -n ruff pyproject.toml",
        ):
            assert _stale([cmd], tmp_path) == [], cmd

    def test_reading_main_scripts_by_absolute_path_is_stale(self, tmp_path: Path) -> None:
        for cmd in (
            f"cat {MAIN}/scripts/agents/handoff_lint.py",
            f"cat {MAIN}/scripts/agents/swarm-check.sh",
        ):
            assert _stale([cmd], tmp_path) == [cmd], cmd

    def test_frontend_src_is_a_repo_code_root(self, tmp_path: Path) -> None:
        cmd = f"cd {MAIN} && grep -rn utcNow frontend/src"
        assert _stale([cmd], tmp_path) == [cmd]

    def test_a_cd_into_the_worktree_is_fine(self, tmp_path: Path) -> None:
        assert _stale([f"cd {WT} && sed -n 1,80p scripts/agents/handoff_lint.py"], tmp_path) == []

    def test_an_absolute_worktree_path_is_fine(self, tmp_path: Path) -> None:
        assert _stale([f"sed -n 1p {WT}/scripts/x.py"], tmp_path) == []

    def test_worktree_pytest_run_is_fine(self, tmp_path: Path) -> None:
        cmd = f"cd {WT} && PYTHONPATH=$PWD .venv/bin/python -m pytest tests/unit/test_x.py -q"
        assert _stale([cmd], tmp_path) == []

    def test_main_docs_reads_are_fine(self, tmp_path: Path) -> None:
        assert _stale([f"cd {MAIN} && cat docs/orchestration/README.md"], tmp_path) == []

    def test_main_bus_and_data_reads_are_fine(self, tmp_path: Path) -> None:
        cmds = [
            f"cd {MAIN} && python3 scripts/agents/bus.py --bus data/swarm/job/bus inbox",
            f"cd {MAIN} && cat data/swarm/job/handoffs/x.md",
        ]
        # the bus CLI is shared tooling run from main (R16); data/swarm/ is bus state, not code
        assert _stale(cmds, tmp_path) == []

    def test_a_tmp_copy_is_fine(self, tmp_path: Path) -> None:
        assert _stale(["cd /tmp/probe-copy && sed -n 1p scripts/agents/bus.py"], tmp_path) == []

    def test_a_command_with_no_repo_code_or_runner_is_fine(self, tmp_path: Path) -> None:
        assert _stale([f"cd {MAIN} && git log --oneline -5"], tmp_path) == []

    def test_another_agents_commands_are_ignored(self, tmp_path: Path) -> None:
        db = _db(
            tmp_path,
            [
                (_THREAD + _AGENT, "command_execution", 0, "sed -n 1p scripts/agents/bus.py"),
                (_THREAD + "cc-other-1", "command_execution", 1, "cat vibe_quant/metrics.py"),
                (_THREAD + _AGENT, "command_execution", 2, "sed -n 2p tests/unit/test_x.py"),
                (_THREAD + _AGENT, "assistant_message", 3, "not a command"),
            ],
        )
        assert stale_reads.stale_commands(db, _AGENT, WT, MAIN) == [
            "sed -n 1p scripts/agents/bus.py",
            "sed -n 2p tests/unit/test_x.py",
        ]

    def test_stale_commands_keep_ordinal_order(self, tmp_path: Path) -> None:
        db = _db(
            tmp_path,
            [
                (_THREAD + _AGENT, "command_execution", 7, "cat vibe_quant/b.py"),
                (_THREAD + _AGENT, "command_execution", 2, "cat vibe_quant/a.py"),
                (_THREAD + _AGENT, "command_execution", 5, "cd /x && echo hi"),
            ],
        )
        assert stale_reads.stale_commands(db, _AGENT, WT, MAIN) == [
            "cat vibe_quant/a.py",
            "cat vibe_quant/b.py",
        ]

    def test_total_counts_every_command(self, tmp_path: Path) -> None:
        db = _commands(
            tmp_path,
            ["cat vibe_quant/a.py", "cd /x && echo hi", "cat vibe_quant/b.py"],
        )
        stale, total = stale_reads.stale_report(db, _AGENT, WT, MAIN)
        assert (len(stale), total) == (2, 3)

    def test_a_malformed_payload_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        db = _db(
            tmp_path,
            [
                (_THREAD + _AGENT, "command_execution", 0, "{not json"),
                (_THREAD + _AGENT, "command_execution", 1, "cat vibe_quant/a.py"),
            ],
        )
        # the helper json-dumps its payload, so write the raw row by hand instead
        con = sqlite3.connect(db)
        con.execute(
            "update orchestration_v2_projection_turn_items set payload_json = 'not json' "
            "where ordinal = 0"
        )
        con.commit()
        con.close()
        assert stale_reads.stale_commands(db, _AGENT, WT, MAIN) == ["cat vibe_quant/a.py"]


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

    def test_stale_prints_each_command_then_the_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cmd = "sed -n 1p scripts/agents/bus.py"
        db = _commands(tmp_path, [cmd, "cd /x && echo hi"])
        rc = self._run(tmp_path, "--agent", _AGENT, "--worktree", str(WT), "--db", str(db))
        out = capsys.readouterr().out
        assert rc == 1
        assert out.splitlines() == [cmd, "STALE 1 of 2 commands"]

    def test_clean_run_exits_0(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        db = _commands(tmp_path, [f"cd {WT} && sed -n 1p scripts/agents/bus.py"])
        rc = self._run(tmp_path, "--agent", _AGENT, "--worktree", str(WT), "--db", str(db))
        out = capsys.readouterr().out
        assert rc == 0
        assert out.splitlines() == ["STALE 0 of 1 commands"]

    def test_printed_command_is_cut_to_160_chars(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cmd = "sed -n 1p vibe_quant/" + "x" * 300 + ".py"
        db = _commands(tmp_path, [cmd])
        assert self._run(tmp_path, "--agent", _AGENT, "--worktree", str(WT), "--db", str(db)) == 1
        printed = capsys.readouterr().out.splitlines()[0]
        assert printed == cmd[:160] and len(printed) == 160

    def test_main_flag_overrides_the_derived_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A deleted worktree cannot name its main: --main supplies it."""
        cmd = f"cat {MAIN}/scripts/agents/bus.py"
        db = _commands(tmp_path, [cmd])
        gone = tmp_path / "deleted-worktree"
        base = ("--agent", _AGENT, "--worktree", str(gone), "--db", str(db))
        assert self._run(tmp_path, *base) == 0
        assert self._run(tmp_path, *base, "--main", str(MAIN)) == 1
        assert capsys.readouterr().out.splitlines()[-2:] == [cmd, "STALE 1 of 1 commands"]

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
