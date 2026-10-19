"""Tests for scripts/agents/mutate.py — the one-command mutation check.

Every test drives the CLI in-process with `mutate.main(argv)` against a throwaway git repo
in tmp_path, with a real test command (`sys.executable -m pytest -q`) running in the
scratch copy, so the copy/apply/run/restore/verify/report path is exercised end to end —
including the assertion that the original repo's bytes and `git status` never move. The one
status driven by a fake runner is TIMEOUT, since hanging a real pytest would cost the suite
a sleep.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "mutate.py"
_spec = importlib.util.spec_from_file_location("mutate", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mutate = importlib.util.module_from_spec(_spec)
sys.modules["mutate"] = mutate
_spec.loader.exec_module(mutate)

_MOD = """\
def is_positive(x: int) -> bool:
    return x > 0
"""
_TEST = """\
from vibe_quant.mod import is_positive


def test_positive() -> None:
    assert is_positive(1)


def test_zero() -> None:
    assert not is_positive(0)
"""
# `x > 0` -> `x >= 0` breaks test_zero (KILLED); `x > 0` -> `x >= 1` is equivalent for the
# inputs the tests use (SURVIVED).
_KILL = ("return x > 0", "return x >= 0")
_SURVIVE = ("return x > 0", "return x >= 1")
# `return = x > 0` is a syntax error, so pytest cannot even import the module: no assertion
# runs, and the probe is INVALID rather than a kill (round-2 F1).
_BROKEN = ("return x > 0", "return = x > 0")

_CMD = [sys.executable, "-m", "pytest", "-q"]

# Hermetic, non-interactive git: no user/system config, identity from the environment.
_GIT_ENV_OVERRIDES = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "mutate",
    "GIT_AUTHOR_EMAIL": "mutate@test",
    "GIT_COMMITTER_NAME": "mutate",
    "GIT_COMMITTER_EMAIL": "mutate@test",
}


def _git_env() -> dict[str, str]:
    drop = {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(_GIT_ENV_OVERRIDES)
    return env


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=_git_env(),
    )
    return out.stdout


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, env=_git_env())
    (repo / "vibe_quant").mkdir()
    (repo / "vibe_quant" / "__init__.py").write_text("")
    (repo / "vibe_quant" / "mod.py").write_text(_MOD)
    unit = repo / "tests" / "unit"
    unit.mkdir(parents=True)
    (unit / "test_mod.py").write_text(_TEST)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _run(repo: Path, *extra: str, cmd: list[str] | None = None) -> int:
    """Run one mutant via the CLI and return its exit code."""
    rc: int = mutate.main(["--repo", str(repo), *extra, "--", *(cmd or _CMD)])
    return rc


def _lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().out.splitlines()


def test_killed_mutant(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo(tmp_path)
    old, new = _KILL
    rc = _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new)
    out = _lines(capsys)
    assert rc == 0
    # Default name is <file>:<line of the match> — the mutated line is line 2.
    # pytest exit 1 = tests ran and failed, so this stays a kill (exit 2+ would be INVALID).
    assert out[0].startswith("KILLED vibe_quant/mod.py:2 (exit 1, ")
    assert out[0].endswith("s)")
    assert out[-1] == "MUTANTS killed=1 survived=0 errors=0"


def test_surviving_mutant(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo(tmp_path)
    old, new = _SURVIVE
    rc = _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new)
    out = _lines(capsys)
    assert rc == 1
    assert out[0].startswith("SURVIVED vibe_quant/mod.py:2 (")
    assert out[0].endswith("s)")
    assert out[-1] == "MUTANTS killed=0 survived=1 errors=0"


def test_old_matching_twice_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo(tmp_path)
    (repo / "vibe_quant" / "dup.py").write_text(
        "def below(a: int, b: int) -> bool:\n"
        "    return a <= b\n"
        "\n"
        "\n"
        "def within(a: int, b: int) -> bool:\n"
        "    return a <= b\n"
    )
    rc = _run(
        repo,
        "--file",
        "vibe_quant/dup.py",
        "--old",
        "return a <= b",
        "--new",
        "return a < b",
        "--name",
        "dup",
    )
    out = _lines(capsys)
    assert rc == 2
    assert out[0] == "ERROR dup: old text found 2 times in vibe_quant/dup.py"
    assert out[-1] == "MUTANTS killed=0 survived=0 errors=1"


def test_old_missing_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo(tmp_path)
    rc = _run(
        repo,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        "return x < 0",
        "--new",
        "return x <= 0",
        "--name",
        "absent",
    )
    out = _lines(capsys)
    assert rc == 2
    assert out[0] == "ERROR absent: old text found 0 times in vibe_quant/mod.py"
    assert out[-1] == "MUTANTS killed=0 survived=0 errors=1"


def test_failing_baseline_aborts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo(tmp_path)
    failing = [sys.executable, "-c", "raise SystemExit(3)"]
    rc = _run(
        repo, "--file", "vibe_quant/mod.py", "--old", _KILL[0], "--new", _KILL[1], cmd=failing
    )
    out = _lines(capsys)
    assert rc == 2
    assert out == ["ERROR baseline: tests fail without mutation"]


def test_run_leaves_the_original_repo_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    before_bytes = (repo / "vibe_quant" / "mod.py").read_bytes()
    before_status = _git(repo, "status", "--porcelain")
    old, new = _KILL
    rc = _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new)
    assert rc == 0
    assert (repo / "vibe_quant" / "mod.py").read_bytes() == before_bytes
    assert _git(repo, "status", "--porcelain") == before_status


def _record_mkdtemp(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every scratch dir built through tempfile.mkdtemp, so a test can assert
    mutate.py removed it afterwards."""
    created: list[Path] = []
    real_mkdtemp = mutate.tempfile.mkdtemp

    def fake_mkdtemp(prefix: str = "", dir: str | None = None) -> str:
        path = Path(real_mkdtemp(prefix=prefix, dir=dir))
        created.append(path)
        return str(path)

    monkeypatch.setattr(mutate.tempfile, "mkdtemp", fake_mkdtemp)
    return created


def test_scratch_copy_is_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _record_mkdtemp(monkeypatch)
    repo = _make_repo(tmp_path)
    old, new = _KILL
    assert _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new) == 0
    capsys.readouterr()
    assert created, "mutate.py must build its copy with tempfile.mkdtemp"
    assert created[0].name.startswith("mutate-")
    assert not created[0].exists()


def test_scratch_copy_is_removed_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _record_mkdtemp(monkeypatch)
    repo = _make_repo(tmp_path)

    def boom(cmd: object, cwd: object, env: object, timeout: object) -> Any:
        raise RuntimeError("boom")

    old, new = _KILL
    with pytest.raises(RuntimeError, match="boom"):
        mutate.main(
            [
                "--repo",
                str(repo),
                "--file",
                "vibe_quant/mod.py",
                "--old",
                old,
                "--new",
                new,
                "--",
                *_CMD,
            ],
            run=boom,
        )
    assert created, "mutate.py must build its copy with tempfile.mkdtemp"
    assert not created[0].exists()


def test_uncommitted_edit_is_what_gets_mutated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    # Commit says `x > 0`; the working tree says `x >= 1`. The mutant's `old` exists only in
    # the uncommitted text, so a tool reading the committed blob would report "0 times".
    (repo / "vibe_quant" / "mod.py").write_text(_MOD.replace("return x > 0", "return x >= 1"))
    rc = _run(
        repo,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        "return x >= 1",
        "--new",
        "return x >= 0",
        "--name",
        "uncommitted",
    )
    out = _lines(capsys)
    assert rc == 0
    assert out[0].startswith("KILLED uncommitted (exit 1, ")
    assert out[-1] == "MUTANTS killed=1 survived=0 errors=0"


def test_spec_runs_every_mutant(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo(tmp_path)
    spec = tmp_path / "mutants.json"
    spec.write_text(
        json.dumps(
            [
                {
                    "name": "strict",
                    "file": "vibe_quant/mod.py",
                    "old": _KILL[0],
                    "new": _KILL[1],
                },
                {
                    "name": "equivalent",
                    "file": "vibe_quant/mod.py",
                    "old": _SURVIVE[0],
                    "new": _SURVIVE[1],
                },
            ]
        )
    )
    rc = mutate.main(["--repo", str(repo), "--spec", str(spec), "--", *_CMD])
    out = _lines(capsys)
    assert rc == 1
    assert out[0].startswith("KILLED strict (exit 1, ")
    assert out[1].startswith("  first failure: ")
    assert out[2].startswith("SURVIVED equivalent (")
    assert out[-1] == "MUTANTS killed=1 survived=1 errors=0"


def test_data_dirs_are_symlinked_into_the_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    # data/ is ignored, so it is not in `git ls-files` and must be linked in instead: the
    # probe only passes (SURVIVED, rc 1) when the copy really has the market data.
    (repo / ".gitignore").write_text("data/\n")
    (repo / "data" / "catalog").mkdir(parents=True)
    (repo / "data" / "catalog" / "market.csv").write_text("price\n1\n")
    probe = (
        "import os, sys; p = 'data/catalog'; "
        "ok = os.path.islink(p) and os.path.isfile(os.path.join(p, 'market.csv')); "
        "sys.exit(0 if ok else 1)"
    )
    old, new = _KILL
    rc = _run(
        repo,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        old,
        "--new",
        new,
        cmd=[sys.executable, "-c", probe],
    )
    out = _lines(capsys)
    assert rc == 1
    assert out[0].startswith("SURVIVED ")
    assert out[-1] == "MUTANTS killed=0 survived=1 errors=0"


def test_timeout_is_labelled_and_counts_as_killed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    calls: list[int] = []

    def fake_run(cmd: object, cwd: object, env: object, timeout: object) -> Any:
        calls.append(1)
        if len(calls) == 1:
            return mutate.RunResult(0, 0.01)
        return mutate.RunResult(124, 30.0, timed_out=True)

    old, new = _KILL
    rc = mutate.main(
        [
            "--repo",
            str(repo),
            "--file",
            "vibe_quant/mod.py",
            "--old",
            old,
            "--new",
            new,
            "--",
            *_CMD,
        ],
        run=fake_run,
    )
    out = _lines(capsys)
    assert rc == 0
    assert out == ["TIMEOUT vibe_quant/mod.py:2", "MUTANTS killed=1 survived=0 errors=0"]


def test_pytest_collection_error_is_invalid_not_killed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mutant that only breaks the import makes pytest exit 2 before any assertion runs:
    reporting KILLED there would credit a mutant that was never exercised (round-2 F1)."""
    repo = _make_repo(tmp_path)
    old, new = _BROKEN
    rc = _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new, "--name", "syntax")
    out = _lines(capsys)
    assert rc == 2
    assert out[0] == (
        "INVALID syntax: test command exit 2 (collection/usage error, no assertion ran)"
    )
    assert out[-1] == "MUTANTS killed=0 survived=0 errors=1"


def test_non_pytest_command_exit_2_is_still_a_kill(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pytest exit-code rule must not leak onto other commands: there, exit 2 is a kill."""
    repo = _make_repo(tmp_path)
    probe = (
        "import pathlib, sys; "
        "sys.exit(2 if 'x >= 0' in pathlib.Path('vibe_quant/mod.py').read_text() else 0)"
    )
    old, new = _KILL
    rc = _run(
        repo,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        old,
        "--new",
        new,
        "--name",
        "probe",
        cmd=[sys.executable, "-c", probe],
    )
    out = _lines(capsys)
    assert rc == 0
    assert out[0].startswith("KILLED probe (exit 2, ")
    assert out[-1] == "MUTANTS killed=1 survived=0 errors=0"


def test_relative_executable_resolves_from_the_main_checkout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A task worktree has no `.venv` of its own (it lives in the main checkout), so
    `.venv/bin/python` in the test command must resolve there — and be used as found: it is a
    symlink to the base interpreter, and resolving it drops the venv's pytest (round-2 F2)."""
    venv = Path(sys.executable).parent.parent  # `<venv>/bin/python` -> `<venv>`
    if not (venv / "bin" / "python").exists():
        pytest.skip("the suite is not running from a venv")

    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(main)], check=True, env=_git_env())
    (main / ".gitignore").write_text(".venv\n")  # `.venv` symlinks a dir: `.venv/` misses it
    (main / "vibe_quant").mkdir()
    (main / "vibe_quant" / "__init__.py").write_text("")
    (main / "vibe_quant" / "mod.py").write_text(_MOD)
    unit = main / "tests" / "unit"
    unit.mkdir(parents=True)
    (unit / "test_mod.py").write_text(_TEST)
    (main / ".venv").symlink_to(venv, target_is_directory=True)
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "init")

    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", "-b", "task", str(wt))
    assert not (wt / ".venv").exists()

    old, new = _KILL
    rc = _run(
        wt,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        old,
        "--new",
        new,
        cmd=[".venv/bin/python", "-m", "pytest", "-q"],
    )
    out = _lines(capsys)
    assert rc == 0, out
    assert out[0].startswith("KILLED vibe_quant/mod.py:2 (exit 1, ")
    assert out[-1] == "MUTANTS killed=1 survived=0 errors=0"


def test_rev_uses_the_committed_text_not_the_working_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reviewers mutate the pinned commit: with an uncommitted edit in the tree, `--rev HEAD`
    still sees the committed bytes and never the maker's working copy (round-2 F3)."""
    repo = _make_repo(tmp_path)
    # Commit says `x > 0`; the working tree says `x >= 1`.
    (repo / "vibe_quant" / "mod.py").write_text(_MOD.replace("return x > 0", "return x >= 1"))

    rc = _run(
        repo,
        "--rev",
        "HEAD",
        "--file",
        "vibe_quant/mod.py",
        "--old",
        "return x > 0",
        "--new",
        "return x >= 0",
        "--name",
        "pinned",
    )
    out = _lines(capsys)
    assert rc == 0, out
    assert out[0].startswith("KILLED pinned (exit 1, ")
    assert out[-1] == "MUTANTS killed=1 survived=0 errors=0"

    # The uncommitted text is not in the copy, so a mutant on it matches 0 times.
    rc = _run(
        repo,
        "--rev",
        "HEAD",
        "--file",
        "vibe_quant/mod.py",
        "--old",
        "return x >= 1",
        "--new",
        "return x >= 0",
        "--name",
        "uncommitted",
    )
    out = _lines(capsys)
    assert rc == 2
    assert out[0] == "ERROR uncommitted: old text found 0 times in vibe_quant/mod.py"


def test_unknown_rev_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo(tmp_path)
    old, new = _KILL
    rc = _run(repo, "--rev", "deadbeef", "--file", "vibe_quant/mod.py", "--old", old, "--new", new)
    out = _lines(capsys)
    assert rc == 2
    assert out[0].startswith("ERROR repo: git archive deadbeef failed")


def test_killed_mutant_shows_first_failing_test_and_assertion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    old, new = _KILL
    _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new)
    out = _lines(capsys)
    assert out[0].startswith("KILLED ")
    assert out[1].startswith("  first failure: tests/unit/test_mod.py::test_zero: ")
    assert "assert not True" in out[1]
    assert out[-1] == "MUTANTS killed=1 survived=0 errors=0"


def test_killed_by_runtime_error_shows_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    _run(repo, "--file", "vibe_quant/mod.py", "--old", "return x > 0", "--new", "return 1 // 0 > x")
    out = _lines(capsys)
    assert out[0].startswith("KILLED ")
    assert out[1].startswith("  first failure: tests/unit/test_mod.py::test_")
    assert "ZeroDivisionError" in out[1]


def test_quiet_kills_hides_failure_detail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    old, new = _KILL
    _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new, "--quiet-kills")
    out = _lines(capsys)
    assert len(out) == 2
    assert out[0].startswith("KILLED ")


def test_survivor_has_no_failure_detail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo(tmp_path)
    old, new = _SURVIVE
    _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new)
    assert len(_lines(capsys)) == 2


def test_first_failure_parses_parametrized_id_with_spaces(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    (repo / "tests" / "unit" / "test_ids.py").write_text(
        "import pytest\nfrom vibe_quant.mod import is_positive\n\n\n"
        "@pytest.mark.parametrize('v', [pytest.param(-1, id='with spaces')])\n"
        "def test_value(v: int) -> None:\n    assert not is_positive(v + 1)\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "ids")
    _run(
        repo,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        "return x > 0",
        "--new",
        "return x >= 0",
        cmd=[*_CMD, "tests/unit/test_ids.py", "tests/unit/test_mod.py::test_positive"],
    )
    out = _lines(capsys)
    assert out[0].startswith("KILLED ")
    assert out[1].startswith("  first failure: tests/unit/test_ids.py::test_value[with spaces]: ")


def test_first_failure_parses_parametrized_id_containing_separator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    (repo / "tests" / "unit" / "test_ids.py").write_text(
        "import pytest\nfrom vibe_quant.mod import is_positive\n\n\n"
        "@pytest.mark.parametrize('v', [pytest.param(-1, id='with - separator')])\n"
        "def test_value(v: int) -> None:\n    assert not is_positive(v + 1)\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "ids")
    _run(
        repo,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        "return x > 0",
        "--new",
        "return x >= 0",
        cmd=[*_CMD, "tests/unit/test_ids.py", "tests/unit/test_mod.py::test_positive"],
    )
    out = _lines(capsys)
    assert out[0].startswith("KILLED ")
    assert out[1].startswith(
        "  first failure: tests/unit/test_ids.py::test_value[with - separator]: "
    )


def test_first_failure_keeps_bracketed_node_id_whole() -> None:
    # The message separator is the first " - " OUTSIDE square brackets; a " - " inside a
    # parametrize id must not split the node id, and the id is printed whole either way.
    assert mutate.first_failure("") == ""
    assert (
        mutate.first_failure("FAILED tests/t.py::test_a") == "  first failure: tests/t.py::test_a: "
    )
    assert (
        mutate.first_failure("FAILED tests/t.py::test_a - assert 1 == 2\n")
        == "  first failure: tests/t.py::test_a: assert 1 == 2"
    )
    assert (
        mutate.first_failure("FAILED tests/t.py::test_v[with - sep] - assert 0 == 1\n")
        == "  first failure: tests/t.py::test_v[with - sep]: assert 0 == 1"
    )


def test_shell_style_test_commands_are_normalised() -> None:
    norm = mutate.normalize_command
    assert norm(["tests/unit/test_x.py", "-q"]) == (
        [".venv/bin/python", "-m", "pytest", "tests/unit/test_x.py", "-q"],
        {},
    )
    assert norm(["pytest", "tests/unit/test_x.py"]) == (
        [".venv/bin/python", "-m", "pytest", "tests/unit/test_x.py"],
        {},
    )
    assert norm(["PYTHONPATH=/w", "FOO=1", ".venv/bin/python", "-m", "pytest", "t.py"]) == (
        [".venv/bin/python", "-m", "pytest", "t.py"],
        {"FOO": "1"},
    )
    assert norm(["/x/python", "-m", "pytest", "t.py"]) == (
        ["/x/python", "-m", "pytest", "t.py"],
        {},
    )


def test_auto_diff_mutates_added_code_lines_only(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "vibe_quant" / "gate.py").write_text(
        "def ok(x, n, flag):\n"
        '    """Gate.\n\n'
        "    passes when not over and below the cap\n"
        '    """\n'
        '    msg = "not and or <="\n'
        '    logger.info("x <= %s", n)\n'
        "    if x <= n and flag:\n"
        "        return 3\n"
        "    return 3\n"
    )
    mutants = mutate.diff_mutants(repo, base, "vibe_quant/*.py", 8)
    got = {(m.name, m.new.strip()) for m in mutants}
    assert got == {
        ("vibe_quant/gate.py:8:cmp", "if x < n and flag:"),
        ("vibe_quant/gate.py:9:int", "return 4"),  # the 8-space line is unique in the file
    }
    # docstring prose (line 4), string literals (6) and logging (7) yield no mutant; the
    # 4-space `return 3` (10) also occurs inside line 9, so it is not unique and is skipped


def test_auto_diff_cli_runs_the_generated_mutants(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "vibe_quant" / "cap.py").write_text("def capped(x):\n    return x >= 10\n")
    (repo / "tests" / "unit" / "test_cap.py").write_text(
        "from vibe_quant.cap import capped\n\n\ndef test_edge():\n    assert capped(10)\n"
    )
    rc = mutate.main(
        ["--repo", str(repo), "--auto-diff", base, "--", sys.executable, "-m", "pytest",
         "tests/unit/test_cap.py", "-q"]
    )  # fmt: skip
    assert rc == 0  # the >= -> > mutant was generated and killed by the edge test


def test_auto_diff_with_nothing_to_mutate_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    assert mutate.main(["--repo", str(repo), "--auto-diff", base, "--", "true"]) == 2
    assert "no mutable added line" in capsys.readouterr().out
