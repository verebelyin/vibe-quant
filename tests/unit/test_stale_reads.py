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

    def test_the_bus_protocol_export_line_is_fine(self, tmp_path: Path) -> None:
        """The bus block's first line exports PATH=<main>/scripts/agents/bin: a value, no read."""
        cmd = (
            f"cd {MAIN} && export SWARM_AGENT=cc-x-1 SWARM_BUS={MAIN}/data/b "
            f"PATH={MAIN}/scripts/agents/bin:$PATH; python3 {MAIN}/scripts/agents/bus.py "
            f"--bus {MAIN}/data/b --as cc-x-1 inbox"
        )
        assert _stale([cmd], tmp_path) == []

    def test_an_export_before_a_main_read_is_still_stale(self, tmp_path: Path) -> None:
        cmd = f"cd {MAIN} && export X=1; cat scripts/agents/bus.py"
        assert _stale([cmd], tmp_path) == [cmd]

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

    def test_a_bare_scratch_root_then_a_relative_cd_is_fine(self, tmp_path: Path) -> None:
        cmd = "cd /private/tmp && mkdir -p probe && cd probe && PYTHONPATH=$PWD pytest tests/unit/test_slow.py"
        assert _stale([cmd], tmp_path) == []

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

    def test_the_base_flag_measures_reads_from_the_branch_point(self, tmp_path: Path) -> None:
        """`--base` is the branch point: main moving after it must not flag a read."""
        fixture = _branch_fixture(tmp_path)
        main, wt, base = fixture
        cmd = f"cd {main} && sed -n '1,80p' scripts/agents/a.py"
        db = _commands(tmp_path, [cmd])
        argv = ("--agent", _AGENT, "--worktree", str(wt), "--db", str(db))
        assert stale_reads.main([*argv]) == 1  # no base: main's HEAD moved a.py
        assert stale_reads.main([*argv, "--base", base]) == 0

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

    def test_reads_of_unchanged_paths_are_fine(self, tmp_path: Path) -> None:
        for cmd in (
            f"cd {MAIN} && sed -n 1,80p scripts/agents/mutate.py",
            f"cd {MAIN} && rg -n handoff scripts/agents/swarm_check.py scripts/agents/swarm-check.sh",
            f"cat {MAIN}/scripts/agents/handoff_lint.py",
            f"cd {MAIN} && rg -n mutex vibe_quant/",
            f"cd {MAIN} && ls tests/unit/ | wc -l",
        ):
            assert self._stale_u([cmd], tmp_path, changed=set()) == [], cmd

    def test_a_quoted_pipe_in_a_pattern_stays_one_read(self, tmp_path: Path) -> None:
        """The two real false positives: the `|` inside the quoted pattern split the read."""
        for cmd in (
            f'cd {MAIN} && grep -n "def main\\|add_argument" scripts/agents/mutate.py',
            f'cd {MAIN} && rg -n "handoff-contract|GATE PASS" docs/orchestration/README.md '
            "scripts/agents/handoff_lint.py",
        ):
            assert self._stale_u([cmd], tmp_path, changed=set()) == [], cmd

    def test_a_quoted_pipe_does_not_hide_a_changed_file(self, tmp_path: Path) -> None:
        cmd = f'cd {MAIN} && grep -n "a|b" scripts/agents/mutate.py'
        assert self._stale_u([cmd], tmp_path, changed={"scripts/agents/mutate.py"}) == [cmd]

    def test_reads_of_changed_paths_stay_stale(self, tmp_path: Path) -> None:
        changed = {"scripts/agents/handoff_lint.py", "vibe_quant/x.py"}
        for cmd in (
            f"cd {MAIN} && sed -n 1,80p scripts/agents/handoff_lint.py",
            f"cd {MAIN} && cat scripts/agents/bus.py vibe_quant/x.py",
            f"cd {MAIN} && sed -n 1p scripts/agents/bus.py && cat vibe_quant/x.py",
        ):
            assert self._stale_u([cmd], tmp_path, changed=changed) == [cmd], cmd

    def test_test_runs_and_python_imports_stay_stale(self, tmp_path: Path) -> None:
        for cmd in (
            f"cd {MAIN} && PYTHONPATH=$PWD .venv/bin/python -m pytest tests/unit/test_stale_reads.py",
            f"cd {MAIN} && sed -n 1p scripts/agents/bus.py && .venv/bin/mypy vibe_quant",
            f"cd {MAIN} && PYTHONPATH=$PWD .venv/bin/python -c 'import vibe_quant.metrics'",
        ):
            assert self._stale_u([cmd], tmp_path, changed=set()) == [cmd], cmd

    def test_without_the_predicate_every_main_read_is_stale(self, tmp_path: Path) -> None:
        cmd = f"cd {MAIN} && sed -n 1,80p scripts/agents/mutate.py"
        assert _stale([cmd], tmp_path) == [cmd]

    def test_unchanged_paths_reads_git(self, tmp_path: Path) -> None:
        main = tmp_path / "main"
        (main / "scripts" / "agents").mkdir(parents=True)
        (main / "scripts" / "agents" / "a.py").write_text("x = 1\n")
        (main / "vibe_quant").mkdir()
        (main / "vibe_quant" / "m.py").write_text("y = 1\n")

        def git(*args: str, cwd: Path = main) -> None:
            subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

        git("init", "-q", "-b", "main")
        git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
        git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
        wt = tmp_path / "wt"
        git("worktree", "add", "-q", "-b", "task", str(wt))
        (wt / "vibe_quant" / "m.py").write_text("y = 2\n")  # uncommitted product edit
        same = stale_reads.unchanged_paths(wt, main)
        assert same("scripts/agents/a.py") is True
        assert same("scripts/agents/") is True
        assert same("vibe_quant/m.py") is False
        assert same("vibe_quant/") is False
        git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "edit", cwd=wt)
        assert stale_reads.unchanged_paths(wt, main)("vibe_quant/m.py") is False  # committed
        (wt / "scripts" / "agents" / "new.py").write_text("z = 1\n")  # untracked in worktree
        assert stale_reads.unchanged_paths(wt, main)("scripts/agents/") is False
        (wt / "scripts" / "agents" / "new.py").unlink()
        (main / "scripts" / "agents" / "a.py").write_text("x = 9\n")  # main has a dirty edit
        assert stale_reads.unchanged_paths(wt, main)("scripts/agents/a.py") is False
        assert stale_reads.unchanged_paths(wt, None)("scripts/agents/a.py") is False
        assert stale_reads.unchanged_paths(tmp_path / "nope", main)("vibe_quant/m.py") is False


def _git(*args: str, cwd: Path) -> str:
    """Run git in `cwd` as the fixture's author; return its trimmed stdout."""
    proc = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return proc.stdout.strip()


def _branch_fixture(tmp_path: Path, *, added: str | None = None) -> tuple[Path, Path, str]:
    """A base commit, a task worktree, and main moving afterwards.

    The base commits `scripts/agents/a.py` and `scripts/agents/b.py`; the worktree's branch
    then edits b.py (and optionally ADDS `added` under scripts/agents/), and finally MAIN
    commits a change to a.py — a file the branch never touched. Returns (main, worktree,
    the base sha), so an unchanged-path check measured from the branch point is exercised.
    """
    main = tmp_path / "main"
    agents = main / "scripts" / "agents"
    agents.mkdir(parents=True)
    (agents / "a.py").write_text("a = 1\n")
    (agents / "b.py").write_text("b = 1\n")
    _git("init", "-q", "-b", "main", cwd=main)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".", cwd=main)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base", cwd=main)
    base = _git("rev-parse", "HEAD", cwd=main)
    wt = tmp_path / "wt"
    _git("worktree", "add", "-q", "-b", "task", str(wt), cwd=main)
    (wt / "scripts" / "agents" / "b.py").write_text("b = 2\n")
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "edit b", cwd=wt)
    if added is not None:
        (wt / "scripts" / "agents" / added).write_text("c = 1\n")
        _git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".", cwd=wt)
        _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "add", cwd=wt)
    (agents / "a.py").write_text("a = 2\n")
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "main moves", cwd=main)
    return main, wt, base


def _stale_from_base(
    commands: list[str], tmp_path: Path, fixture: tuple[Path, Path, str]
) -> list[str]:
    """Stale commands, with the worktree measured against the BRANCH POINT (base)."""
    main, wt, base = fixture
    db = _commands(tmp_path, commands)
    stale, _total = stale_reads.stale_report(
        db, _AGENT, wt, main, unchanged=stale_reads.unchanged_paths(wt, main, base=base)
    )
    return stale


class TestMainMovedAfterTheBranchPoint:
    """Job 20261106-p1tools: main landed commits after the worktree was cut, so a read was
    flagged as stale even though the BRANCH never changed the path. The branch point is the
    reference: main moving afterwards is not this round's stale copy."""

    def test_a_read_of_a_file_main_moved_and_the_branch_left_alone_is_fine(
        self, tmp_path: Path
    ) -> None:
        fixture = _branch_fixture(tmp_path)
        cmd = f"cd {fixture[0]} && sed -n '1,80p' scripts/agents/a.py"
        assert _stale_from_base([cmd], tmp_path, fixture) == []

    def test_a_read_of_a_file_the_branch_changed_is_stale(self, tmp_path: Path) -> None:
        """The maker reading main's OLD copy of a file its own branch edits is the bug."""
        fixture = _branch_fixture(tmp_path)
        cmd = f"cd {fixture[0]} && sed -n 1,5p scripts/agents/b.py"
        assert _stale_from_base([cmd], tmp_path, fixture) == [cmd]

    def test_a_test_run_from_main_is_still_stale(self, tmp_path: Path) -> None:
        """A lint/test run acts on main's copy whatever the branch point says."""
        fixture = _branch_fixture(tmp_path)
        cmd = f"cd {fixture[0]} && .venv/bin/mypy scripts/agents/a.py"
        assert _stale_from_base([cmd], tmp_path, fixture) == [cmd]


class TestLsComparesOnlyNames:
    """`ls` reveals NAMES, not contents: a content edit under a directory does not matter,
    an added/deleted/renamed file does (relative to the branch point)."""

    def test_ls_of_a_directory_the_branch_only_edited_is_fine(self, tmp_path: Path) -> None:
        fixture = _branch_fixture(tmp_path)  # the branch edited b.py's contents
        cmd = f"cd {fixture[0]} && ls scripts/agents/"
        assert _stale_from_base([cmd], tmp_path, fixture) == []

    def test_ls_is_stale_when_the_branch_added_a_file_there(self, tmp_path: Path) -> None:
        fixture = _branch_fixture(tmp_path, added="c.py")
        cmd = f"cd {fixture[0]} && ls scripts/agents/"
        assert _stale_from_base([cmd], tmp_path, fixture) == [cmd]


    def test_find_by_name_compares_only_names(self, tmp_path: Path) -> None:
        """`find -name` lists names like `ls` (cc-tvp-t3-1 failed the gate on one)."""
        fixture = _branch_fixture(tmp_path)  # the branch edited b.py's contents
        cmd = f"cd {fixture[0]} && find scripts/agents -name b.py"
        assert _stale_from_base([cmd], tmp_path, fixture) == []
        added = _branch_fixture(tmp_path / "added", added="c.py")
        cmd = f"cd {added[0]} && find scripts/agents -name c.py"
        assert _stale_from_base([cmd], tmp_path / "added", added) == [cmd]

    def test_find_with_exec_is_not_a_plain_read(self, tmp_path: Path) -> None:
        fixture = _branch_fixture(tmp_path)
        cmd = f"cd {fixture[0]} && find scripts/agents -name b.py -exec cat {{}} +"
        assert _stale_from_base([cmd], tmp_path, fixture) == [cmd]


_HEREDOC_BODY = (
    "cd /Users/verebelyin/projects/vibe-quant && sed -n 1,80p scripts/agents/swarm_check.py"
)


class TestAHeredocWrittenToAFileIsText:
    """A `cat >file`/`tee file` heredoc body is a script being WRITTEN, not read; the same
    body fed to an interpreter is code that runs and stays a candidate."""

    def test_cat_and_tee_heredocs_written_to_a_file_are_not_reads(self, tmp_path: Path) -> None:
        for opener in (
            "cat > /tmp/bench.sh <<'EOF'",
            "cat <<'EOF' > /tmp/bench.sh",
            "cat >> /tmp/bench.sh <<'EOF'",
            "tee /tmp/bench.sh <<'EOF'",
            "cat > /tmp/bench.sh <<-'EOF'",
        ):
            cmd = f"{opener}\n{_HEREDOC_BODY}\nEOF\nchmod +x /tmp/bench.sh"
            assert _stale([cmd], tmp_path) == [], opener

    def test_an_unquoted_tag_is_recognised(self, tmp_path: Path) -> None:
        cmd = f"cat > /tmp/bench.sh <<EOF\n{_HEREDOC_BODY}\nEOF"
        assert _stale([cmd], tmp_path) == []

    def test_the_same_body_fed_to_an_interpreter_is_code(self, tmp_path: Path) -> None:
        for opener in ("bash <<'EOF'", "python3 - <<'EOF'", "sh -s <<EOF"):
            cmd = f"{opener}\n{_HEREDOC_BODY}\nEOF"
            assert _stale([cmd], tmp_path) == [cmd], opener

    def test_an_interpreter_heredoc_redirecting_its_output_is_still_code(
        self, tmp_path: Path
    ) -> None:
        """Only a WRITER command makes a heredoc data: `bash -s <<EOF > log` runs the body."""
        for opener in ("bash -s <<'EOF' > /tmp/out.log", "python3 - <<'EOF' 2> /tmp/err.log"):
            cmd = f"{opener}\n{_HEREDOC_BODY}\nEOF"
            assert _stale([cmd], tmp_path) == [cmd], opener


class TestHeredocOwnership:
    """The heredoc belongs to ONE simple command; the delimiter is the shell's exact token."""

    def test_an_unrelated_writer_on_the_opener_line_does_not_exempt_an_interpreter(
        self, tmp_path: Path
    ) -> None:
        """`cat /tmp/banner; bash <<EOF > log` runs the body; bash owns the heredoc."""
        for opener in (
            "cat /tmp/banner; bash <<'EOF' > /tmp/out.log",
            "tee /tmp/banner </dev/null; bash <<'EOF'",
            "bash <<'EOF' | tee /tmp/out.log",
            "echo 'cat > f' && bash <<'EOF'",
            'bash <<\'EOF\' "tee /tmp/x"',
        ):
            cmd = f"{opener}\n{_HEREDOC_BODY}\nEOF"
            assert _stale([cmd], tmp_path) == [cmd], opener

    def test_a_writer_after_an_unrelated_command_still_writes(self, tmp_path: Path) -> None:
        for opener in (
            "echo hi; cat > /tmp/bench.sh <<'EOF'",
            "cat /tmp/banner && tee /tmp/bench.sh <<'EOF'",
        ):
            cmd = f"{opener}\n{_HEREDOC_BODY}\nEOF"
            assert _stale([cmd], tmp_path) == [], opener

    def test_each_heredoc_of_a_line_follows_its_own_command(self, tmp_path: Path) -> None:
        """A writer's body is text, the interpreter's body that follows it is code."""
        cmd = f"cat > /tmp/a.sh <<'A'; bash <<'B'\nthe text\nA\n{_HEREDOC_BODY}\nB"
        assert _stale([cmd], tmp_path) == [cmd]
        quiet = f"cat > /tmp/a.sh <<'A'; bash <<'B'\n{_HEREDOC_BODY}\nA\necho done\nB"
        assert _stale([quiet], tmp_path) == []

    def test_a_space_indented_apparent_delimiter_does_not_end_the_body(
        self, tmp_path: Path
    ) -> None:
        """Only an exact `EOF` line ends it; `  EOF` and `EOF ` are body text."""
        for fake in ("  EOF", "EOF ", "\tEOF"):
            cmd = f"cat > /tmp/bench.sh <<'EOF'\n{fake}\n{_HEREDOC_BODY}\nEOF"
            assert _stale([cmd], tmp_path) == [], repr(fake)

    def test_dash_heredoc_strips_only_leading_tabs(self, tmp_path: Path) -> None:
        tabbed = f"cat > /tmp/bench.sh <<-'EOF'\n{_HEREDOC_BODY}\n\t\tEOF\nsed 1p /tmp/x"
        assert _stale([tabbed], tmp_path) == []
        spaced = f"cat > /tmp/bench.sh <<-'EOF'\n  EOF\n{_HEREDOC_BODY}\nEOF"
        assert _stale([spaced], tmp_path) == []

    def test_a_real_terminator_still_ends_the_body(self, tmp_path: Path) -> None:
        """Text after the exact delimiter is a command again."""
        cmd = f"cat > /tmp/bench.sh <<'EOF'\nhello\nEOF\n{_HEREDOC_BODY}"
        assert _stale([cmd], tmp_path) == [cmd]

    def test_a_delimiter_with_punctuation_is_recognised(self, tmp_path: Path) -> None:
        """'END-SCRIPT', "a.b", \\EOF and E"O"F are whole delimiter tokens."""
        for opener, tag in (
            ("cat > /tmp/bench.sh <<'END-SCRIPT'", "END-SCRIPT"),
            ('cat > /tmp/bench.sh <<"a.b"', "a.b"),
            ("cat > /tmp/bench.sh <<END-SCRIPT", "END-SCRIPT"),
            ("cat > /tmp/bench.sh <<\\EOF", "EOF"),
            ('cat > /tmp/bench.sh <<E"O"F', "EOF"),
            ("cat > /tmp/bench.sh <<- 'END-SCRIPT'", "END-SCRIPT"),
        ):
            cmd = f"{opener}\n{_HEREDOC_BODY}\n{tag}"
            assert _stale([cmd], tmp_path) == [], opener

    def test_a_punctuated_delimiter_keeps_an_interpreter_body(self, tmp_path: Path) -> None:
        cmd = f"bash <<'END-SCRIPT'\n{_HEREDOC_BODY}\nEND-SCRIPT"
        assert _stale([cmd], tmp_path) == [cmd]

    def test_cat_without_an_output_file_is_not_a_writer(self, tmp_path: Path) -> None:
        for opener in ("cat <<'EOF'", "cat >&2 <<'EOF'", "cat 2> /tmp/err <<'EOF'", "tee <<'EOF'"):
            cmd = f"{opener}\n{_HEREDOC_BODY}\nEOF"
            assert _stale([cmd], tmp_path) == [cmd], opener

    def test_a_here_string_is_not_a_heredoc(self, tmp_path: Path) -> None:
        cmd = f"cat > /tmp/x <<< 'EOF'\n{_HEREDOC_BODY}\nEOF"
        assert _stale([cmd], tmp_path) == [cmd]

    def test_a_quoted_heredoc_operator_is_not_a_heredoc(self, tmp_path: Path) -> None:
        cmd = f"cat > /tmp/x '<<EOF'\n{_HEREDOC_BODY}\nEOF"
        assert _stale([cmd], tmp_path) == [cmd]


class TestAScopeValueIsGateData:
    """`--scope` is the gate's glob list of paths the diff may touch, not a read of them."""

    def test_a_scope_value_is_not_a_file_read(self, tmp_path: Path) -> None:
        cmd = (
            "scripts/agents/swarm-check.sh /Users/verebelyin/projects/vq-bench-slots-1 "
            "--scope scripts/agents/swarm_check.py"
        )
        assert _stale([cmd], tmp_path) == []

    def test_a_quoted_scope_list_is_not_a_file_read(self, tmp_path: Path) -> None:
        cmd = (
            "scripts/agents/swarm-check.sh /tmp/x "
            "--scope='scripts/agents/swarm_check.py,tests/unit/test_x.py'"
        )
        assert _stale([cmd], tmp_path) == []


class TestToolingArguments:
    """Running main's tooling is exempt; its arguments naming repo code act on main's copy."""

    def test_mutate_on_repo_code_from_main_is_stale(self, tmp_path: Path) -> None:
        for cmd in (
            f"cd {MAIN} && python3 scripts/agents/mutate.py --file vibe_quant/x.py --old a --new b "
            "-- .venv/bin/python -m pytest tests/unit/test_x.py",
            f"cd {MAIN} && scripts/agents/mutate.py --auto-diff abc --max 3 -- pytest tests/unit/t.py",
        ):
            assert _stale([cmd], tmp_path) == [cmd], cmd

    def test_the_same_run_in_the_worktree_is_fine(self, tmp_path: Path) -> None:
        cmd = (
            f"cd {WT} && python3 scripts/agents/mutate.py --file vibe_quant/x.py "
            "-- .venv/bin/python -m pytest tests/unit/test_x.py"
        )
        assert _stale([cmd], tmp_path) == []

    def test_protocol_commands_without_code_arguments_stay_clean(self, tmp_path: Path) -> None:
        for cmd in (
            f"cd {MAIN} && python3 scripts/agents/handoff_lint.py {MAIN}/data/swarm/j/handoffs/x.md",
            f"cd {MAIN} && python3 scripts/agents/bus.py --bus data/b --as cc-x-1 board search "
            "--global --path 'vibe_quant/x.py' --topic gotchas",
            f"python3 {MAIN}/scripts/agents/bus.py --bus {MAIN}/data/b --as cc-x-1 post --to chief "
            "--kind done --body 'fixed vibe_quant/x.py and tests/unit/test_x.py'",
        ):
            assert _stale([cmd], tmp_path) == [], cmd


def test_every_command_the_bus_protocol_block_prescribes_is_fine(tmp_path: Path) -> None:
    """Render the real template: an edit to it must not turn every maker's gate red.

    The block's own export line false-FAILed every maker of job 20261029-gaps-speed.
    """
    import re

    template = (Path(__file__).resolve().parents[2] / "docs/orchestration/prompts/bus-protocol.md").read_text()
    snippets = [
        s.replace("{ID}", "cc-x-1").replace("{BUS}", f"{MAIN}/data/swarm/j/bus").replace("{JOB}", f"{MAIN}/data/swarm/j")
        for s in re.findall(r"`([^`]+)`", template)
    ]
    prefix = next(s for s in snippets if s.startswith("export SWARM_AGENT="))
    commands = [f"cd {MAIN} && {prefix}"] + [
        f"cd {MAIN} && {prefix}; {s}" for s in snippets if s.startswith(("bus ", "python3 "))
    ]
    assert len(commands) >= 6
    assert _stale(commands, tmp_path) == []
