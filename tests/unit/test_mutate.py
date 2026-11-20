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


def test_failing_baseline_aborts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _make_repo(tmp_path)
    failing = [sys.executable, "-c", "raise SystemExit(3)"]
    rc = _run(
        repo, "--file", "vibe_quant/mod.py", "--old", _KILL[0], "--new", _KILL[1], cmd=failing
    )
    out = _lines(capsys)
    assert rc == 2
    assert out == ["ERROR baseline: tests fail without mutation"]


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


# --- the hand-mutant cap: 2 per round -----------------------------------------------------


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


def test_mutant_cap_refuses_a_maker_past_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = _cap_job(tmp_path, monkeypatch)
    repo = _make_repo(tmp_path)
    _many_lines(repo, 4)
    runner = _CountingRunner()

    assert mutate.main(
        ["--repo", str(repo), "--spec", str(_spec_file(tmp_path, 1)), "--", *_CMD], run=runner
    ) == 1  # every mutant survives the stub: exit 1, but the run happened and was counted
    assert mutate.main([*_one_mutant(repo, 1)], run=runner) == 1
    capsys.readouterr()
    calls_before = len(runner.calls)

    rc = mutate.main([*_one_mutant(repo, 2)], run=runner)
    out = capsys.readouterr().out
    assert rc == 2
    assert "ERROR mutant cap: cc-x-1 used 2 of 2 mutants; this run asks for 1" in out
    assert "Ask the chief in your thread ([question])" in out
    assert len(runner.calls) == calls_before  # not even the baseline copy was paid for
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [(r["agent"], r["n"], r["mode"]) for r in rows] == [
        ("cc-x-1", 1, "spec"),
        ("cc-x-1", 1, "file"),
    ]
