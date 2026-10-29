"""Tests for scripts/agents/mutate.py — the one-command mutation check.

Every test drives the CLI in-process with `mutate.main(argv)` against a throwaway copy of a
template git repo, with a real `sys.executable -m pytest -q` running in the scratch copy, so
the copy/apply/run/restore/verify/report path is exercised end to end — including the
assertion that the original repo's bytes and `git status` never move. Tests that assert
mutate.py's own plumbing (the copy, the restore, the quiet/survivor line count) rather than a
pytest verdict run a one-line `python -c` probe instead: same code path, and the probe's
non-zero exit is a kill exactly as a non-pytest command should be. The one status driven by a
fake runner is TIMEOUT, since hanging a real pytest would cost the suite a sleep.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def _probe(old: str, new: str) -> list[str]:
    """A non-pytest test command: pass while the original text is in the file, fail on the
    mutant — i.e. it exits 1 exactly when the mutation was applied (a kill)."""
    return [
        sys.executable,
        "-c",
        "import pathlib, sys; p = pathlib.Path('vibe_quant/mod.py').read_text(); "
        f"sys.exit(0 if {old!r} in p and {new!r} not in p else 1)",
    ]


@pytest.fixture(scope="module", autouse=True)
def _fast_scratch_pytest() -> Iterator[None]:
    """The scratch copy is a bare repo of stdlib-only files: loading the venv's plugin set
    (randomly/xdist/...) only pays pytest start-up. Verified: same FAILED/short-summary text."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        yield


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


def _build_template(root: Path) -> Path:
    repo = root / "repo"
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


_TEMPLATE: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def _template_repo(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Build the committed repo once: a private copy is 8ms, `git init`+`add`+`commit` 131ms."""
    global _TEMPLATE
    _TEMPLATE = _build_template(tmp_path_factory.mktemp("mutate-template"))


def _make_repo(tmp_path: Path) -> Path:
    """A private copy of the template repo: mutate.py only reads the repo, never writes it."""
    assert _TEMPLATE is not None, "the module template fixture did not run"
    repo = tmp_path / "repo"
    shutil.copytree(_TEMPLATE, repo)
    return repo


def _run(repo: Path, *extra: str, cmd: list[str] | None = None) -> int:
    """Run one mutant via the CLI and return its exit code."""
    rc: int = mutate.main(["--repo", str(repo), *extra, "--", *(cmd or _CMD)])
    return rc


def _lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().out.splitlines()


def test_killed_mutant(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """KILLED names the mutated line, its kind and the pytest exit, shows the first failing
    assertion, and never touches the original repo's bytes or `git status`."""
    repo = _make_repo(tmp_path)
    before_bytes = (repo / "vibe_quant" / "mod.py").read_bytes()
    before_status = _git(repo, "status", "--porcelain")
    old, new = _KILL
    rc = _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new)
    out = _lines(capsys)
    assert rc == 0
    assert out[0] == "BASELINE ran=2 skipped=0 deselected=0"
    # Default name is <file>:<line of the match> — the mutated line is line 2.
    # pytest exit 1 = tests ran and failed, so this stays a kill (exit 2+ would be INVALID).
    assert out[1].startswith("KILLED vibe_quant/mod.py:2 (assert, exit 1, ")
    assert out[1].endswith("s)")
    assert out[2].startswith("  first failure: tests/unit/test_mod.py::test_zero: ")
    assert "assert not True" in out[2]
    assert out[-1] == "MUTANTS killed=1 survived=0 errors=0"
    assert (repo / "vibe_quant" / "mod.py").read_bytes() == before_bytes
    assert _git(repo, "status", "--porcelain") == before_status


def test_surviving_mutant(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """SURVIVED prints one line only: a survivor has no failure detail to show."""
    repo = _make_repo(tmp_path)
    old, new = _SURVIVE
    rc = _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new)
    out = _lines(capsys)
    assert rc == 1
    assert out[0] == "BASELINE ran=2 skipped=0 deselected=0"
    assert out[1].startswith("SURVIVED vibe_quant/mod.py:2 (")
    assert out[1].endswith("s)")
    assert out[-1] == "MUTANTS killed=0 survived=1 errors=0"
    assert len(out) == 3  # a plain run that skipped and deselected nothing has no WARN
    assert not any(line.startswith("WARN") for line in out)


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
    assert out[1] == "ERROR dup: old text found 2 times in vibe_quant/dup.py"
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
    assert out[1] == "ERROR absent: old text found 0 times in vibe_quant/mod.py"
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
    assert _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new,
                cmd=_probe(old, new)) == 0  # fmt: skip
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
        cmd=_probe("return x >= 1", "return x >= 0"),
    )
    out = _lines(capsys)
    assert rc == 0
    assert out[0].startswith("KILLED uncommitted (exit 1, ")  # a probe: no BASELINE line
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
    assert out[1].startswith("KILLED strict (assert, exit 1, ")
    assert out[2].startswith("  first failure: ")
    assert out[3].startswith("SURVIVED equivalent (")
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
    # The fake baseline returns no output, so every summary count is 0.
    assert out == [
        "BASELINE ran=0 skipped=0 deselected=0",
        "TIMEOUT vibe_quant/mod.py:2",
        "MUTANTS killed=1 survived=0 errors=0",
    ]


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
    assert out[1] == (
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
    assert out[0] == "BASELINE ran=2 skipped=0 deselected=0"
    assert out[1].startswith("KILLED vibe_quant/mod.py:2 (assert, exit 1, ")
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
        cmd=_probe("return x > 0", "return x >= 0"),
    )
    out = _lines(capsys)
    assert rc == 0, out
    assert out[0].startswith("KILLED pinned (exit 1, ")  # a probe: no BASELINE line
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
    assert out[1] == "ERROR uncommitted: old text found 0 times in vibe_quant/mod.py"


def test_unknown_rev_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo(tmp_path)
    old, new = _KILL
    rc = _run(repo, "--rev", "deadbeef", "--file", "vibe_quant/mod.py", "--old", old, "--new", new)
    out = _lines(capsys)
    assert rc == 2
    assert out[0].startswith("ERROR repo: git archive deadbeef failed")


def test_killed_by_runtime_error_shows_error_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crash inside production code IS a kill — only a mutant that broke the program TEXT
    is invalid — and the line names the exception that escaped."""
    repo = _make_repo(tmp_path)
    _run(repo, "--file", "vibe_quant/mod.py", "--old", "return x > 0", "--new", "return 1 // 0 > x")
    out = _lines(capsys)
    assert out[1].startswith("KILLED vibe_quant/mod.py:2 (crash ZeroDivisionError, exit 1, ")
    assert out[2].startswith("  first failure: tests/unit/test_mod.py::test_")
    assert "ZeroDivisionError" in out[2]


def test_quiet_kills_hides_failure_detail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    old, new = _KILL
    _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new, "--quiet-kills",
         cmd=_probe(old, new))  # fmt: skip
    out = _lines(capsys)
    assert len(out) == 2  # the probe is not pytest, so no BASELINE line either
    assert out[0].startswith("KILLED ")


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
    assert out[1].startswith("KILLED ")
    assert out[2].startswith("  first failure: tests/unit/test_ids.py::test_value[with spaces]: ")


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
    assert out[1].startswith("KILLED ")
    assert out[2].startswith(
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


def test_auto_diff_with_no_candidates_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A test-only diff has no line under the default `vibe_quant/*.py` glob: that is a
    finished check with nothing to mutate (exit 0), not a usage error (bead aq9zy item 2)."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "tests" / "unit" / "test_more.py").write_text(
        "def test_more() -> None:\n    assert 1 == 1\n"
    )
    marker = tmp_path / "baseline-ran"
    rc = mutate.main(
        ["--repo", str(repo), "--auto-diff", base, "--",
         sys.executable, "-c",
         f"import pathlib; pathlib.Path({str(marker)!r}).write_text('x')"]
    )  # fmt: skip
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "MUTANTS no candidates"
    assert not marker.exists()  # nothing to mutate, so no baseline run is paid for
    assert "vibe_quant/*.py" in captured.err and base[:7] in captured.err


def test_auto_diff_max_below_one_is_a_usage_error(tmp_path: Path) -> None:
    """`--max 0` would print "no candidates" on a real mutable diff: refuse it instead."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    with pytest.raises(SystemExit) as exc:
        mutate.main(
            [
                "--repo",
                str(repo),
                "--auto-diff",
                base,
                "--max",
                "0",
                "--",
                sys.executable,
                "-c",
                "pass",
            ]
        )
    assert exc.value.code == 2


def test_auto_diff_deletes_added_call_and_assignment_lines(tmp_path: Path) -> None:
    """A line that only adds a call or an assignment has no cmp/bool/not/int mutant; deleting
    it (as `pass`, so the block stays valid) is the behaviour change to probe."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "vibe_quant" / "svc.py").write_text(
        'def run(s):\n    s.start()\n    s.mode = "x"\n    total = s.count\n    return total\n'
    )
    mutants = mutate.diff_mutants(repo, base, "vibe_quant/*.py", 8)
    got = {(m.name, m.new.strip()) for m in mutants}
    # The string assignment (3) and the return (5) are not deletable; neither is `def run`.
    assert got == {
        ("vibe_quant/svc.py:2:del", "pass"),
        ("vibe_quant/svc.py:4:del", "pass"),
    }
    assert mutants[0].new == "    pass"  # same indent as the deleted line


def test_auto_diff_swaps_added_bool_literals(tmp_path: Path) -> None:
    """`enabled = True` gets a `flag` mutant on top of its `del` one; a True inside a string
    literal is not a flag."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "vibe_quant" / "flag.py").write_text(
        "def build(s):\n"
        "    enabled = True\n"
        '    x = "True"\n'
        "    f(strict=False)\n"
        "    return enabled\n"
    )
    mutants = mutate.diff_mutants(repo, base, "vibe_quant/*.py", 8)
    got = {(m.name, m.new.strip()) for m in mutants}
    assert got == {
        ("vibe_quant/flag.py:2:flag", "enabled = False"),
        ("vibe_quant/flag.py:2:del", "pass"),
        ("vibe_quant/flag.py:4:flag", "f(strict=True)"),
        ("vibe_quant/flag.py:4:del", "pass"),
    }


def test_auto_diff_never_deletes_inert_lines(tmp_path: Path) -> None:
    """Docstring/comment/blank/pass/import/annotation/ellipsis/multi-line-statement lines are
    not deletable: deleting them changes nothing or breaks the file."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "vibe_quant" / "inert.py").write_text(
        '"""Module docstring."""\n'
        "\n"
        "# a comment\n"
        "import os\n"
        "from collections import deque\n"
        "\n"
        "def inert(x: int) -> None:\n"
        '    """Doc."""\n'
        "    pass\n"
        "    ...\n"
        "    y: int\n"
        "    foo(\n"
        "        x,\n"
        "    )\n"
        "    do()\n"
        "    a = 1; b = 2\n"
    )
    mutants = mutate.diff_mutants(repo, base, "vibe_quant/*.py", 99)
    # only the plain single-line call is deletable: the multi-line call and `a = 1; b = 2`
    # (two statements on one line) are not
    assert [m.name for m in mutants if m.name.endswith(":del")] == ["vibe_quant/inert.py:15:del"]
    # the int on `a = 1; b = 2` still yields its operator mutant, so this is not "no mutants"
    assert ("vibe_quant/inert.py:16:int", "    a = 2; b = 2") in {(m.name, m.new) for m in mutants}


def test_auto_diff_never_deletes_compound_header_lines(tmp_path: Path) -> None:
    """A simple statement that shares its line with a compound header (`try:`, `else:`,
    `if x:`, `with ...:`) is not a complete single-line statement: replacing the whole line
    with `pass` would delete the header too. Only `s.go()` is deletable here."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "vibe_quant" / "compound.py").write_text(
        "def compound(s, x, items):\n"
        "    try: s.start()\n"
        "    except Exception:\n"
        "        pass\n"
        "    if x:\n"
        "        pass\n"
        "    else: y = 2\n"
        "    with s: s.start()\n"
        "    for i in items: s.step(i)\n"
        "    while x: s.step(x)\n"
        "    s.go()\n"
    )
    mutants = mutate.diff_mutants(repo, base, "vibe_quant/*.py", 99)
    assert [m.name for m in mutants if m.name.endswith(":del")] == ["vibe_quant/compound.py:11:del"]


def test_auto_diff_cli_kills_delete_mutant(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "vibe_quant" / "hook.py").write_text('def setup(reg):\n    reg.append("ok")\n')
    (repo / "tests" / "unit" / "test_hook.py").write_text(
        "from vibe_quant.hook import setup\n\n\n"
        "def test_registers() -> None:\n"
        "    reg = []\n"
        "    setup(reg)\n"
        '    assert reg == ["ok"]\n'
    )
    rc = mutate.main(
        ["--repo", str(repo), "--auto-diff", base, "--", sys.executable, "-m", "pytest",
         "tests/unit/test_hook.py", "-q"]
    )  # fmt: skip
    assert rc == 0  # the `del` mutant empties reg, so the assertion kills it


def test_auto_diff_cli_reports_surviving_delete_mutant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "vibe_quant" / "hook.py").write_text('def setup(reg):\n    reg.append("ok")\n')
    (repo / "tests" / "unit" / "test_hook.py").write_text(
        "from vibe_quant.hook import setup\n\n\ndef test_setup() -> None:\n    setup([])\n"
    )
    rc = mutate.main(
        ["--repo", str(repo), "--auto-diff", base, "--", sys.executable, "-m", "pytest",
         "tests/unit/test_hook.py", "-q"]
    )  # fmt: skip
    out = _lines(capsys)
    assert rc == 1
    assert out[1].startswith("SURVIVED vibe_quant/hook.py:2:del (")
    assert out[-1] == "MUTANTS killed=0 survived=1 errors=0"


# ------------------------------------------------------- --compare-rev: a lost kill

def _weakened_repo(tmp_path: Path) -> tuple[Path, str]:
    """(repo, base sha): the template's strong test, then a commit that DELETES test_zero.

    The same mutant (`x > 0` -> `x >= 0`) is killed on the base tree and survives on HEAD,
    which is exactly a lost kill - what `--compare-rev` must flag."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "tests" / "unit" / "test_mod.py").write_text(
        "from vibe_quant.mod import is_positive\n\n\ndef test_positive() -> None:\n"
        "    assert is_positive(1)\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "drop the zero case")
    return repo, base


def test_compare_rev_flags_a_kill_lost_on_head(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, base = _weakened_repo(tmp_path)
    before_status = _git(repo, "status", "--porcelain")
    old, new = _KILL
    rc = _run(
        repo,
        "--compare-rev",
        base,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        old,
        "--new",
        new,
        cmd=[sys.executable, "-m", "pytest", "-q"],
    )
    out = _lines(capsys)
    assert rc == 1
    # One BASELINE line per phase comes first, prefixed by the side it belongs to.
    assert out[0] == "base BASELINE ran=2 skipped=0 deselected=0"
    assert out[1] == "head BASELINE ran=1 skipped=0 deselected=0"
    first = out[2].split()
    assert first[0] == "base=KILLED" and first[1] == "head=SURVIVED", out
    assert "vibe_quant/mod.py:2" in out[2]
    assert out[-1] == "MUTANTS compared=1 lost=1 errors=0"
    assert _git(repo, "status", "--porcelain") == before_status


def test_compare_rev_passes_a_mutant_killed_on_both(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same repo state on both sides: nothing was lost, so the exit code is 0 and the
    per-mutant line still names both sides and the whole multi-word mutant name."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    old, new = _KILL
    rc = _run(
        repo,
        "--compare-rev",
        base,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        old,
        "--new",
        new,
        "--name",
        "zero is positive",
        cmd=[sys.executable, "-m", "pytest", "-q"],
    )
    out = _lines(capsys)
    assert rc == 0
    assert out[2] == "base=KILLED head=KILLED zero is positive", out
    assert out[-1] == "MUTANTS compared=1 lost=0 errors=0"


def test_compare_rev_compares_against_the_named_rev_not_the_working_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both copies are built from real trees - base from `git archive <rev>`, head from the
    working tree - so a test that exists only on the base side is what kills the mutant."""
    repo, base = _weakened_repo(tmp_path)
    assert "test_zero" in subprocess.run(
        ["git", "-C", str(repo), "show", f"{base}:tests/unit/test_mod.py"],
        capture_output=True, text=True, check=True, env=_git_env(),
    ).stdout  # fmt: skip
    old, new = _KILL
    rc = _run(
        repo,
        "--compare-rev",
        base,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        old,
        "--new",
        new,
        cmd=[sys.executable, "-m", "pytest", "-q"],
    )
    assert rc == 1
    assert _lines(capsys)[2].startswith("base=KILLED head=SURVIVED ")


def test_compare_rev_needs_only_a_test_command(tmp_path: Path) -> None:
    """The flag belongs to the same source group as --rev: both cannot be given (one copy is
    pinned by a rev, the other IS the working tree)."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    with pytest.raises(SystemExit) as exc:
        mutate.main(
            ["--repo", str(repo), "--rev", base, "--compare-rev", base,
             "--file", "vibe_quant/mod.py", "--old", _KILL[0], "--new", _KILL[1], "--", *_CMD]
        )  # fmt: skip
    assert exc.value.code == 2


def test_compare_rev_counts_a_mutant_that_cannot_apply_as_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A side where the mutant cannot be applied (the line moved) is ERROR, not a verdict:
    the run is an error (exit 2), and the per-mutant line still names the other side."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "vibe_quant" / "mod.py").write_text(
        "def is_positive(x: int) -> bool:\n    return 0 < x\n"
    )  # same behaviour, the mutant's `old` text is gone
    _git(repo, "commit", "-qam", "the line the mutant targets already moved")
    rc = _run(
        repo,
        "--compare-rev",
        base,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        _KILL[0],
        "--new",
        _KILL[1],
        cmd=[sys.executable, "-m", "pytest", "-q"],
    )
    out = _lines(capsys)
    assert rc == 2
    assert out[2].startswith("base=KILLED head=ERROR"), out
    assert any("head: ERROR" in line for line in out), out
    assert out[-1] == "MUTANTS compared=1 lost=0 errors=1"


@pytest.mark.parametrize(
    ("cmd", "load1", "expected"),
    [
        (["pytest", "-n", "6"], 5.0, ["pytest", "-n", "6"]),  # load = half the cores: untouched
        (["pytest", "-n", "6"], 12.0, ["pytest", "-n", "2"]),  # busier than the cores: floor 2
        (["pytest", "-n", "auto", "-q"], 7.0, ["pytest", "-n", "3", "-q"]),
        (["pytest", "-n8"], 7.0, ["pytest", "-n3"]),
        (["pytest", "--numprocesses=8"], 7.0, ["pytest", "--numprocesses=3"]),
        (["pytest", "-n", "2"], 12.0, ["pytest", "-n", "2"]),  # already under the cap
        (["pytest", "-q"], 12.0, ["pytest", "-q"]),  # no xdist flag
    ],
)
def test_cap_workers_lowers_xdist_workers_to_the_free_cores_on_a_busy_host(
    cmd: list[str], load1: float, expected: list[str]
) -> None:
    out, note = mutate.cap_workers(cmd, load1, 10)
    assert out == expected
    assert (note is None) == (out == cmd)


# ------------------------------------------- a mutant that broke the program TEXT, not behaviour


def test_mutant_introducing_a_name_error_is_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mutant that types a name its own text invents is never exercised: pytest fails with
    NameError and nothing was asserted, so it is no verdict (rev-paperstart-1)."""
    repo = _make_repo(tmp_path)
    rc = _run(
        repo,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        "return x > 0",
        "--new",
        "return undefined_name > 0",
    )
    out = _lines(capsys)
    assert rc == 2
    assert out[1] == (
        "INVALID vibe_quant/mod.py:2: NameError from the mutant text (no behaviour tested)"
    )
    assert out[-1] == "MUTANTS killed=0 survived=0 errors=1"


def test_mutant_importing_a_missing_module_is_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mutant that imports a module that does not exist breaks the import, not the
    behaviour: no assertion runs, so the probe is invalid and the run is an error."""
    repo = _make_repo(tmp_path)
    rc = _run(
        repo,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        "return x > 0",
        "--new",
        "from missing_mod import thing\n    return thing(x) > 0",
    )
    out = _lines(capsys)
    assert rc == 2
    assert out[1] == (
        "INVALID vibe_quant/mod.py:2: ModuleNotFoundError from the mutant text "
        "(no behaviour tested)"
    )
    assert out[-1] == "MUTANTS killed=0 survived=0 errors=1"


def test_runtime_name_error_the_mutant_did_not_introduce_is_still_a_kill(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`not_defined_anywhere` exists in the code before the mutation, so calling it is a real
    crash in production code — a kill, not an invalid probe."""
    repo = _make_repo(tmp_path)
    # The latent NameError sits in a branch the baseline tests never take; the mutant widens
    # the comparison so test_zero enters it. The name is in neither the old nor the new text
    # OF THE MUTATED LINE, so the crash is production code breaking, not the mutant's typo.
    (repo / "vibe_quant" / "mod.py").write_text(
        "def is_positive(x: int) -> bool:\n"
        "    if x < 0:\n"
        "        return not_defined_anywhere(x)\n"
        "    return x > 0\n"
    )
    _git(repo, "commit", "-qam", "a latent NameError")
    rc = _run(repo, "--file", "vibe_quant/mod.py", "--old", "if x < 0:", "--new", "if x <= 0:")
    out = _lines(capsys)
    assert rc == 0, out
    assert out[1].startswith("KILLED vibe_quant/mod.py:2 (crash NameError, exit 1, ")
    assert out[-1] == "MUTANTS killed=1 survived=0 errors=0"


def test_xdist_x_failure_exit_2_is_classified_as_a_kill(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`-x` under xdist exits 2 on a real assertion failure (serial `-x` exits 1). Exit 2 with
    a FAILED node and an `Interrupted: stopping after` line is classified like exit 1, not
    INVALID (chief ruling R1, measured)."""
    if importlib.util.find_spec("xdist") is None:
        pytest.skip("pytest-xdist is not installed")
    repo = _make_repo(tmp_path)
    old, new = _KILL
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        rc = _run(
            repo,
            "--file",
            "vibe_quant/mod.py",
            "--old",
            old,
            "--new",
            new,
            cmd=[sys.executable, "-m", "pytest", "-q", "-x", "-n", "2"],
        )
    out = _lines(capsys)
    assert rc == 0, out
    assert out[1].startswith("KILLED vibe_quant/mod.py:2 (assert, exit 2, ")
    assert out[-1] == "MUTANTS killed=1 survived=0 errors=0"


# ------------------------------------------------------- how many tests the baseline ran


def test_baseline_reports_a_k_filter_and_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `-k` filter that ran 2 of 159 tests makes a survivor meaningless, so the run says so
    (revlite-e4-busenv-1 report)."""
    repo = _make_repo(tmp_path)
    old, new = _SURVIVE
    rc = _run(
        repo,
        "--file",
        "vibe_quant/mod.py",
        "--old",
        old,
        "--new",
        new,
        cmd=[sys.executable, "-m", "pytest", "-q", "-k", "test_positive"],
    )
    out = _lines(capsys)
    assert rc == 1
    assert out[0] == "BASELINE ran=1 skipped=0 deselected=1"
    assert out[1] == (
        "WARN baseline: 1 tests deselected (-k/-m filter): "
        "a survivor may only mean the filter missed the killing test"
    )
    assert out[2].startswith("SURVIVED ")


def test_baseline_warns_when_tests_skip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `--rev` copy has no `.git`, so needs_git tests skip and a survivor proves less."""
    repo = _make_repo(tmp_path)
    (repo / "tests" / "unit" / "test_skip.py").write_text(
        "import pytest\n\n\n"
        "@pytest.mark.skip(reason='needs_git')\n"
        "def test_needs_git() -> None:\n    pass\n"
    )
    old, new = _SURVIVE
    _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new)
    out = _lines(capsys)
    assert out[0] == "BASELINE ran=2 skipped=1 deselected=0"
    assert out[1] == "WARN baseline: 1 tests skipped (e.g. needs_git in a --rev copy)"


def test_baseline_counts_an_all_skipped_suite() -> None:
    """Every selected test skipped (a `--rev` copy without `.git`): the summary has no
    `passed`, and a survivor there proves nothing, so the skip count and WARN must show."""
    assert mutate.baseline_report("s [100%]\n======= 1 skipped in 0.01s =======\n") == [
        "BASELINE ran=0 skipped=1 deselected=0",
        "WARN baseline: 1 tests skipped (e.g. needs_git in a --rev copy)",
    ]


def test_outer_pytest_addopts_does_not_change_the_mutant_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-review gate exports `PYTEST_ADDOPTS=-q -rfE`: at `-qq` pytest 9 drops its
    summary line (the counts then read 0/0/0), and any `-k`/`-m` there would filter the
    mutant run from the outside. The copy runs the test command as given."""
    monkeypatch.setenv("PYTEST_ADDOPTS", "-q -rfE -k nothing_matches")
    repo = _make_repo(tmp_path)
    old, new = _KILL
    rc = _run(repo, "--file", "vibe_quant/mod.py", "--old", old, "--new", new)
    out = _lines(capsys)
    assert rc == 0, out
    assert out[0] == "BASELINE ran=2 skipped=0 deselected=0"
    assert out[1].startswith("KILLED vibe_quant/mod.py:2 (assert, exit 1, ")
    assert out[-1] == "MUTANTS killed=1 survived=0 errors=0"


# --------------------------------------------------- --auto-diff: which lines get mutated


def test_auto_diff_samples_branchy_lines_across_the_file(tmp_path: Path) -> None:
    """A file with 12 candidates under `--max 3` is sampled across the whole file (first,
    middle, last), not its first three lines."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    lines = ["def branchy(xs):"] + [f"    a{i} = xs < {i}" for i in range(1, 13)]
    (repo / "vibe_quant" / "spread.py").write_text("\n".join(lines) + "\n")
    mutants = mutate.diff_mutants(repo, base, "vibe_quant/*.py", 3)
    assert [m.name for m in mutants] == [
        "vibe_quant/spread.py:2:cmp",
        "vibe_quant/spread.py:8:cmp",  # middle of the file
        "vibe_quant/spread.py:13:cmp",  # last line
    ]


def test_auto_diff_puts_branchy_lines_first(tmp_path: Path) -> None:
    """When every candidate fits, the branchy lines (comparison, boolean, if/while) come
    before the plain arithmetic ones, whichever order the file has them in."""
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "vibe_quant" / "order.py").write_text(
        "def ordered(x):\n"
        "    total1 = f(1)\n"
        "    total2 = f(2)\n"
        "    if x < 3:\n"
        "        pass\n"
        "    while x < 4:\n"
        "        break\n"
        "    keep = x and 5\n"
    )
    mutants = mutate.diff_mutants(repo, base, "vibe_quant/*.py", 9)
    assert [m.name for m in mutants] == [
        "vibe_quant/order.py:4:cmp",
        "vibe_quant/order.py:6:cmp",
        "vibe_quant/order.py:8:bool",
        "vibe_quant/order.py:2:int",
        "vibe_quant/order.py:3:int",
    ]


def test_auto_diff_emits_a_minmax_mutant(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "vibe_quant" / "pick.py").write_text("def pick(a, b):\n    return min(a, b)\n")
    mutants = mutate.diff_mutants(repo, base, "vibe_quant/*.py", 8)
    assert [(m.name, m.new) for m in mutants] == [
        ("vibe_quant/pick.py:2:minmax", "    return max(a, b)")
    ]


def test_line_mutants_minmax_swaps_min_and_max() -> None:
    text = "    return min(a, b)"
    got = mutate.line_mutants("m.py", 3, text, text + "\n", None)
    assert [(m.name, m.new) for m in got] == [("m.py:3:minmax", "    return max(a, b)")]


def test_line_mutants_method_swap_table() -> None:
    cases = {
        "    lo = df.idxmin()": "    lo = df.idxmax()",
        "    r = x.floor": "    r = x.ceil",
        "    ok = any(v)": "    ok = all(v)",
        "    ok = s.startswith(t)": "    ok = s.endswith(t)",
        "    out = df.head(n)": "    out = df.tail(n)",
    }
    for text, want in cases.items():
        got = mutate.line_mutants("m.py", 7, text, text + "\n", None)
        assert [(m.name, m.new) for m in got] == [("m.py:7:method", want)], text


def test_line_mutants_kwnone_nones_the_first_call_keyword() -> None:
    text = "    run(x, timeout=limit)"
    got = mutate.line_mutants("m.py", 2, text, text + "\n", None)
    assert [(m.name, m.new) for m in got] == [
        ("m.py:2:kwnone", "    run(x, timeout=None)")
    ]


def test_line_mutants_kwnone_skips_def_defaults_and_bool_values() -> None:
    # A parameter default is not a call keyword: `def f(a=1):` gets the int mutant only.
    text = "def f(a=1):"
    got = mutate.line_mutants("m.py", 1, text, text + "\n", None)
    assert [m.name for m in got] == ["m.py:1:int"]
    # A bool value is `flag`'s job, so kwnone must not steal `f(strict=False)`.
    text = "    f(strict=False)"
    got = mutate.line_mutants("m.py", 4, text, text + "\n", None)
    assert [m.name for m in got] == ["m.py:4:flag"]


def test_line_mutants_scale_multiplies_a_float_literal() -> None:
    for text, want in {
        "    return x * 0.5": "    return x * 5.0",
        "    rate = x * 0.05": "    rate = x * 0.5",
        "    return x * 1e-3": "    return x * 0.01",
        "    return x * 1.5e3": "    return x * 15000.0",
    }.items():
        got = mutate.line_mutants("m.py", 4, text, text + "\n", None)
        assert [(m.name, m.new) for m in got] == [("m.py:4:scale", want)], text


# --- kwnone: the whole value, brackets balanced (reviewer finding B3) ----------------------


def _kw_none(code: str) -> str | None:
    """`code` with the kwnone replacement applied, or None when kwnone yields no mutant."""
    hit: tuple[int, int, str] | None = mutate._kwnone(code)  # type: ignore[attr-defined]
    if hit is None:
        return None
    start, end, replacement = hit
    return code[:start] + replacement + code[end:]


def test_line_mutants_kwnone_keeps_the_second_keyword_whole() -> None:
    """B3: a nested call in the FIRST keyword's value must not shift the mutant to the
    second keyword (`timeout=get_timeout(), retries=limit` -> `retries=None`)."""
    text = "    run(timeout=get_timeout(), retries=limit)"
    got = mutate.line_mutants("m.py", 2, text, text + "\n", None)
    assert got == [
        mutate.Mutant("m.py:2:kwnone", "m.py", text, "    run(timeout=None, retries=limit)")
    ]


def test_kwnone_nones_a_whole_nested_value() -> None:
    """The value span runs to the matching bracket, so a nested call or a subscript is
    replaced as one value. Digits are the `int` operator's job and preempt kwnone in
    line_mutants, so the span logic itself is pinned on the helper, in the reviewer's shapes."""
    assert _kw_none("f(a=g(h(1), 2))") == "f(a=None)"
    assert _kw_none("x = d(k=v[0], m=2)") == "x = d(k=None, m=2)"


def test_line_mutants_kwnone_nones_a_bracketed_value() -> None:
    text = "    f(a=g(h(x), y))"
    got = mutate.line_mutants("m.py", 5, text, text + "\n", None)
    assert [(m.name, m.new) for m in got] == [("m.py:5:kwnone", "    f(a=None)")]


def test_line_mutants_kwnone_skips_a_fragment_value() -> None:
    """A multi-line call's middle line: mutating a partial value would produce broken code."""
    text = "    timeout=get_timeout("
    assert [m.name for m in mutate.line_mutants("m.py", 3, text, text + "\n", None)] == []


def test_kwnone_skips_a_value_whose_bracket_does_not_close() -> None:
    assert _kw_none("f(a=g(h(x)") is None


# --- the hand-mutant cap: 3 per maker round, 5 per reviewer -------------------------------


class _CountingRunner:
    """A RunFn that passes every command without running it: the cap tests need no pytest."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Any, cwd: Path, env: Any, timeout: float) -> Any:
        self.calls.append([str(c) for c in cmd])
        return mutate.RunResult(0, 0.01)


def _cap_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent: str = "cc-x-1",
    cap: str | None = None,
) -> Path:
    """A job dir with a bus, exported as SWARM_AGENT/SWARM_BUS: returns the ledger path."""
    bus = tmp_path / "job" / "bus"
    bus.mkdir(parents=True)
    monkeypatch.setenv("SWARM_AGENT", agent)
    monkeypatch.setenv("SWARM_BUS", str(bus))
    if cap is not None:
        monkeypatch.setenv("SWARM_MUTANT_CAP", cap)
    return bus.parent / "mutants.jsonl"


def _many_lines(repo: Path, count: int = 10) -> None:
    """A module of `count` one-line assignments, so a spec can ask for that many mutants."""
    (repo / "vibe_quant" / "many.py").write_text("".join(f"V{i} = 1\n" for i in range(count)))


def _spec_file(tmp_path: Path, count: int, name: str = "mutants.json") -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            [
                {
                    "name": f"m{i}",
                    "file": "vibe_quant/many.py",
                    "old": f"V{i} = 1",
                    "new": f"V{i} = 2",
                }
                for i in range(count)
            ]
        )
    )
    return path


def _one_mutant(repo: Path, index: int) -> list[str]:
    return [
        "--repo", str(repo), "--file", "vibe_quant/many.py",
        "--old", f"V{index} = 1", "--new", f"V{index} = 3", "--", *_CMD,
    ]  # fmt: skip


def test_mutant_cap_refuses_a_maker_past_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = _cap_job(tmp_path, monkeypatch)
    repo = _make_repo(tmp_path)
    _many_lines(repo, 4)
    runner = _CountingRunner()

    assert mutate.main(
        ["--repo", str(repo), "--spec", str(_spec_file(tmp_path, 2)), "--", *_CMD], run=runner
    ) == 1  # every mutant survives the stub: exit 1, but the run happened and was counted
    assert mutate.main([*_one_mutant(repo, 0)], run=runner) == 1
    capsys.readouterr()
    calls_before = len(runner.calls)

    rc = mutate.main([*_one_mutant(repo, 1)], run=runner)
    out = capsys.readouterr().out
    assert rc == 2
    assert "ERROR mutant cap: cc-x-1 used 3 of 3 mutants; this run asks for 1" in out
    assert "Ask the chief (bus ask --to chief)" in out
    assert len(runner.calls) == calls_before  # not even the baseline copy was paid for
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [(r["agent"], r["n"], r["mode"]) for r in rows] == [
        ("cc-x-1", 2, "spec"),
        ("cc-x-1", 1, "file"),
    ]


def test_a_failed_baseline_gives_its_mutants_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No mutant runs when the unmutated tests fail, so the round keeps its cap (twice in job
    20261029-swarm-retro a copy without .git failed the baseline and burned all 3)."""
    ledger = _cap_job(tmp_path, monkeypatch)
    repo = _make_repo(tmp_path)
    _many_lines(repo, 4)

    def failing(cmd: Any, cwd: Path, env: Any, timeout: float) -> Any:
        return mutate.RunResult(1, 0.01)

    spec = ["--repo", str(repo), "--spec", str(_spec_file(tmp_path, 3)), "--", *_CMD]
    assert mutate.main(spec, run=failing) == 2
    assert "ERROR baseline: tests fail without mutation" in capsys.readouterr().out
    assert mutate.mutants_used(ledger, "cc-x-1") == 0
    assert mutate.main(spec, run=_CountingRunner()) == 1  # the same 3 may run again
    assert mutate.mutants_used(ledger, "cc-x-1") == 3


def test_mutant_cap_gives_a_reviewer_five(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cap_job(tmp_path, monkeypatch, agent="rev-x-1")
    repo = _make_repo(tmp_path)
    _many_lines(repo)
    runner = _CountingRunner()
    assert mutate.main(
        ["--repo", str(repo), "--spec", str(_spec_file(tmp_path, 5)), "--", *_CMD], run=runner
    ) == 1
    capsys.readouterr()
    assert mutate.main([*_one_mutant(repo, 5)], run=runner) == 2
    assert "rev-x-1 used 5 of 5 mutants" in capsys.readouterr().out


def test_swarm_mutant_cap_overrides_the_round_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = _cap_job(tmp_path, monkeypatch, cap="4")
    repo = _make_repo(tmp_path)
    _many_lines(repo)
    runner = _CountingRunner()
    assert mutate.main(
        ["--repo", str(repo), "--spec", str(_spec_file(tmp_path, 3)), "--", *_CMD], run=runner
    ) == 1
    assert mutate.main([*_one_mutant(repo, 3)], run=runner) == 1  # the 4th the default refuses
    capsys.readouterr()
    assert mutate.main([*_one_mutant(repo, 4)], run=runner) == 2
    assert "cc-x-1 used 4 of 4 mutants" in capsys.readouterr().out
    assert len(ledger.read_text().splitlines()) == 2


@pytest.mark.parametrize("value", ["lots", "-1"])
def test_invalid_swarm_mutant_cap_is_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    _cap_job(tmp_path, monkeypatch, cap=value)
    repo = _make_repo(tmp_path)
    runner = _CountingRunner()
    assert mutate.main([*_one_mutant(repo, 0)], run=runner) == 2
    out = capsys.readouterr().out
    assert "ERROR mutant cap: SWARM_MUTANT_CAP" in out
    assert runner.calls == []


def test_no_agent_means_no_cap_and_no_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bus alone names no round: the old uncapped behaviour, and nothing is recorded."""
    bus = tmp_path / "job" / "bus"
    bus.mkdir(parents=True)
    monkeypatch.setenv("SWARM_BUS", str(bus))
    repo = _make_repo(tmp_path)
    _many_lines(repo)
    runner = _CountingRunner()
    assert mutate.main(
        ["--repo", str(repo), "--spec", str(_spec_file(tmp_path, 10)), "--", *_CMD], run=runner
    ) == 1
    assert "MUTANTS killed=0 survived=10 errors=0" in capsys.readouterr().out
    assert list(tmp_path.rglob("mutants.jsonl")) == []


def test_no_bus_means_no_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SWARM_AGENT", "cc-x-1")
    repo = _make_repo(tmp_path)
    _many_lines(repo, 4)
    runner = _CountingRunner()
    assert mutate.main(
        ["--repo", str(repo), "--spec", str(_spec_file(tmp_path, 4)), "--", *_CMD], run=runner
    ) == 1
    assert "MUTANTS killed=0 survived=4 errors=0" in capsys.readouterr().out
    assert list(tmp_path.rglob("mutants.jsonl")) == []
