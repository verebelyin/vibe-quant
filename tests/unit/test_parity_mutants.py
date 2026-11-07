"""scripts/agents/parity_mutants.py — a test-deleting diff must not LOSE a mutant kill.

Every test drives the CLI in-process against a throwaway git repo, with real pytest,
coverage and git subprocesses in the copies: the deleted-node-id diff, the per-line
candidates, the base/head screen and the "covered elsewhere" confirmation are exactly
the code path the gate's `parity` check runs. The repo is a real git repo (not a
DeclaredGit) because `git archive` and `git diff <base> -- tests/` ARE the feature.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "parity_mutants.py"
assert _SCRIPT.is_file(), f"missing {_SCRIPT}"
_spec = importlib.util.spec_from_file_location("parity_mutants", _SCRIPT)
assert _spec is not None and _spec.loader is not None
parity = importlib.util.module_from_spec(_spec)
sys.modules["parity_mutants"] = parity
_spec.loader.exec_module(parity)


@pytest.fixture(scope="module", autouse=True)
def _fast_copy_pytest() -> Iterator[None]:
    """The copies are bare repos of stdlib-only files: loading the venv's plugin set
    (randomly/xdist/hypothesis/...) only pays pytest start-up, and every test here runs ~8-10
    of those subprocesses. Verified: the same node ids, summaries and verdicts either way.
    Scope: only these tests' scratch repos (they need no plugins); the tool's own env is unchanged."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        yield


@pytest.fixture(autouse=True)
def _enough_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    """--workers is clamped to the host's cores: these rows ask for up to 4 whatever the host has."""
    monkeypatch.setattr(parity.os, "cpu_count", lambda: 64)


_GIT_ENV_OVERRIDES = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "parity",
    "GIT_AUTHOR_EMAIL": "parity@test",
    "GIT_COMMITTER_NAME": "parity",
    "GIT_COMMITTER_EMAIL": "parity@test",
}

# One branch. `>` -> `>=` is the only mutant line_mutants finds (the `return` lines carry
# no operator and line 1 is skipped), so every expectation below has exactly one candidate.
_MOD = """\
def bucket(value: int) -> str:
    if value > 10:
        return "big"
    return "small"
"""

# Two branches with DIFFERENT comparison text (line_mutants refuses a non-unique line), so a
# test that drives `tag` too puts tag's line in a different coverage tier than bucket's.
_MOD_TWO = """\
def bucket(value: int) -> str:
    if value > 10:
        return "big"
    return "small"


def tag(value: int) -> str:
    if value >= 10:
        return "wide"
    return "narrow"
"""

# test_big first: `-x` then stops at test_boundary, which is the killed-at-base node.
_TEST = """\
from vibe_quant.bucketmod import bucket


def test_big() -> None:
    assert bucket(20) == "big"


def test_boundary() -> None:
    assert bucket(10) == "small"


def test_dup() -> None:
    assert bucket(10) == "small"
"""

# test_both is the only test that asserts the boundary both ways; deleting it loses line 2's
# kill too, but keeps test_boundary for the head side. Deleting it alone leaves line 7 lost.
_TEST_TWO = """\
from vibe_quant.bucketmod import bucket, tag


def test_big() -> None:
    assert bucket(20) == "big"


def test_boundary() -> None:
    assert bucket(10) == "small"


def test_both() -> None:
    assert bucket(10) == "small"
    assert tag(10) == "wide"
"""

_OTHER = """\
from vibe_quant.bucketmod import bucket


def test_boundary_other() -> None:
    assert bucket(10) == "small"
"""

# A module-level assignment with no operator: its only mutant is `del` (the line becomes `pass`),
# which makes `from vibe_quant.tokenmod import TOKEN` fail at COLLECTION in any importer.
_TOKEN_MOD = """\
TOKEN = tuple()


def label() -> str:
    return "token"
"""

_TOKEN_TEST = """\
from vibe_quant import tokenmod


def test_label() -> None:
    assert tokenmod.label() == "token"


def test_token_is_empty() -> None:
    assert tokenmod.TOKEN == ()
"""

_TOKEN_TEST_HEAD = _TOKEN_TEST.replace(
    "\n\ndef test_token_is_empty() -> None:\n    assert tokenmod.TOKEN == ()\n", ""
)

_TOKEN_IMPORTER = """\
from vibe_quant.tokenmod import TOKEN


def test_importer_token() -> None:
    assert TOKEN == ()
"""

_BIG_ONLY = (
    'from vibe_quant.bucketmod import bucket\n\n\ndef test_big() -> None:\n'
    '    assert bucket(20) == "big"\n'
)


def _git_env() -> dict[str, str]:
    return {**os.environ, **_GIT_ENV_OVERRIDES}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=_git_env(),
    )


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _init(repo: Path) -> Path:
    """A committed base: `vibe_quant/` production code plus its unittest, ready for edits."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "vibe_quant/__init__.py", "")
    _write(repo, "tests/__init__.py", "")
    _write(repo, "tests/unit/__init__.py", "")
    return repo


def _commit(repo: Path) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")


def _repo(tmp_path: Path, mod: str = _MOD, test: str = _TEST) -> Path:
    repo = _init(tmp_path / "repo")
    _write(repo, "vibe_quant/bucketmod.py", mod)
    _write(repo, "tests/unit/test_bucket.py", test)
    _commit(repo)
    return repo


def _run(repo: Path, *extra: str) -> int:
    # The scratch copies load no plugins (_fast_copy_pytest), so xdist's `-n` is not a
    # recognised option there: these end-to-end rows pin the serial path. The parallel argv is
    # pinned by test_workers_reach_every_non_coverage_pytest_command below.
    return int(parity.main(["--repo", str(repo), "--base", "HEAD", "--workers", "1", *extra]))


def test_deleting_the_only_killer_is_a_lost_kill(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    # Working tree = head: drop both tests that exercise the boundary.
    _write(
        repo,
        "tests/unit/test_bucket.py",
        'from vibe_quant.bucketmod import bucket\n\n\ndef test_big() -> None:\n'
        '    assert bucket(20) == "big"\n',
    )
    assert _run(repo) == 1
    out = capsys.readouterr()
    assert "base=KILLED head=SURVIVED vibe_quant/bucketmod.py:2:cmp" in out.out
    assert "LOST KILL vibe_quant/bucketmod.py:2:cmp (killed at base by " in out.out
    assert "test_bucket.py::test_boundary)" in out.out
    assert out.out.splitlines()[-1] == "PARITY compared=1 lost=1 errors=0"
    assert "covered elsewhere" not in out.out


@pytest.mark.slow(reason="real pytest subprocesses on a scratch repo: base, head and mutant runs")
def test_deleting_a_duplicate_test_loses_no_kill(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    head = _TEST.replace(
        '\n\ndef test_dup() -> None:\n    assert bucket(10) == "small"\n', "\n"
    )
    assert head != _TEST
    # a wall-clock `timing` test in the head set must not run under coverage (it distorts speed)
    head = "import pytest\n" + head + "\n\n@pytest.mark.timing\ndef test_speed_ratio() -> None:\n    assert False\n"
    _write(repo, "tests/unit/test_bucket.py", head)
    assert _run(repo) == 0
    out = capsys.readouterr()
    assert "base=KILLED head=KILLED vibe_quant/bucketmod.py:2:cmp" in out.out
    assert not [ln for ln in out.out.splitlines() if ln.startswith("LOST KILL ")]
    assert out.out.splitlines()[-1] == "PARITY compared=1 lost=0 errors=0"


def test_no_deleted_test_runs_nothing_else(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    _write(repo, "tests/unit/test_bucket.py", _TEST + "\n\ndef test_new() -> None:\n    assert 1\n")
    assert _run(repo) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "PARITY no deleted tests"
    assert out.err.splitlines() == ["parity: collect 1 base file(s), 1 head file(s)"]


@pytest.mark.slow(reason="real pytest subprocesses on a scratch repo: base, head and mutant runs")
def test_a_killer_outside_the_changed_set_is_covered_elsewhere(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One e2e row for the "covered elsewhere" confirmation: the other import spellings of the
    naming test (parenthesised, aliased, `import a.b`, a broken file) are pinned directly on
    `test_files_naming` below, so they cost no copy, collect and coverage run each."""
    repo = _repo(tmp_path)
    _write(repo, "tests/unit/test_other.py", _OTHER)
    _commit(repo)
    # Only test_bucket.py changes; test_other.py still names the mutated module.
    _write(
        repo,
        "tests/unit/test_bucket.py",
        'from vibe_quant.bucketmod import bucket\n\n\ndef test_big() -> None:\n'
        '    assert bucket(20) == "big"\n',
    )
    assert _run(repo) == 0
    out = capsys.readouterr()
    assert "base=KILLED head=SURVIVED vibe_quant/bucketmod.py:2:cmp" in out.out
    assert (
        "covered elsewhere vibe_quant/bucketmod.py:2:cmp "
        "(tests/unit/test_other.py::test_boundary_other)" in out.out
    )
    assert not [ln for ln in out.out.splitlines() if ln.startswith("LOST KILL ")]
    assert out.out.splitlines()[-1] == "PARITY compared=1 lost=0 errors=0"


def test_test_files_naming_reads_imports_and_falls_back_on_a_syntax_error(tmp_path: Path) -> None:
    """Imports are parsed (a comment is not an import); a file that does not parse is matched
    by the single-line regex, because a naming test that cannot be read must still be run."""
    _write(tmp_path, "tests/unit/test_multi.py", "from vibe_quant import (\n    bucketmod,\n)\n")
    _write(tmp_path, "tests/unit/test_alias.py", "from vibe_quant import tagmod, bucketmod as bm\n")
    _write(tmp_path, "tests/unit/test_dotted.py", "import vibe_quant.bucketmod\n")
    _write(tmp_path, "tests/unit/test_comment.py", "from vibe_quant import tagmod  # bucketmod\n")
    _write(tmp_path, "tests/unit/test_other_pkg.py", "from other import bucketmod\n")
    _write(tmp_path, "tests/unit/test_broken.py", "from vibe_quant import bucketmod\ndef (:\n")
    assert parity.test_files_naming(tmp_path, "vibe_quant/bucketmod.py") == [
        "tests/unit/test_alias.py",
        "tests/unit/test_broken.py",
        "tests/unit/test_dotted.py",
        "tests/unit/test_multi.py",
    ]


def test_a_failing_importer_is_no_comparison_not_covered_elsewhere(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """test_other fails with NO mutant applied: its failure under the mutant proves nothing."""
    repo = _repo(tmp_path)
    _write(
        repo,
        "tests/unit/test_other.py",
        "import vibe_quant.bucketmod\n\n\ndef test_unrelated_failure() -> None:\n"
        '    assert False, "PREEXISTING FAILURE"\n',
    )
    _commit(repo)
    _write(repo, "tests/unit/test_bucket.py", _BIG_ONLY)
    assert _run(repo) == 2
    out = capsys.readouterr()
    assert "covered elsewhere" not in out.out
    assert not [ln for ln in out.out.splitlines() if ln.startswith("LOST KILL ")]
    assert "confirm:" in out.out and "tests/unit/test_other.py::test_unrelated_failure" in out.out
    assert out.out.splitlines()[-1] == "PARITY compared=1 lost=0 errors=1"


def test_an_import_breaking_mutant_is_covered_elsewhere_after_a_passing_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _init(tmp_path / "repo")
    _write(repo, "vibe_quant/tokenmod.py", _TOKEN_MOD)
    _write(repo, "tests/unit/test_token.py", _TOKEN_TEST)
    _write(repo, "tests/unit/test_importer.py", _TOKEN_IMPORTER)
    _commit(repo)
    _write(repo, "tests/unit/test_token.py", _TOKEN_TEST_HEAD)
    assert _run(repo) == 0
    out = capsys.readouterr()
    assert "base=KILLED head=SURVIVED vibe_quant/tokenmod.py:1:del" in out.out
    covered = [ln for ln in out.out.splitlines() if ln.startswith("covered elsewhere ")]
    assert len(covered) == 1
    assert covered[0].startswith("covered elsewhere vibe_quant/tokenmod.py:1:del (collection error: ")
    assert "cannot import name 'TOKEN'" in covered[0]
    assert not [ln for ln in out.out.splitlines() if ln.startswith("LOST KILL ")]
    assert out.out.splitlines()[-1] == "PARITY compared=1 lost=0 errors=0"


@pytest.mark.slow(
    reason="real pytest subprocesses on a scratch repo: 4 sides and one importer baseline"
)
def test_two_candidates_sharing_an_importer_set_run_its_baseline_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path, mod=_MOD_TWO, test=_TEST_TWO)
    _write(repo, "tests/unit/test_other.py", _OTHER)
    _commit(repo)
    _write(repo, "tests/unit/test_bucket.py", _BIG_ONLY)
    baselines: list[str] = []

    def run_fn(cmd: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: float) -> object:
        unmutated = (cwd / "vibe_quant" / "bucketmod.py").read_text(encoding="utf-8") == _MOD_TWO
        if cwd.name == "head" and unmutated and "tests/unit/test_other.py" in cmd:
            baselines.append(" ".join(cmd[-2:]))
        return parity.mutate.run_command(cmd, cwd, env, timeout)

    assert parity.main(["--repo", str(repo), "--base", "HEAD", "--workers", "1"], run_fn) == 1
    out = capsys.readouterr()
    # line 2 is killed by test_other (covered elsewhere); line 8's tag() is not (lost)
    assert "covered elsewhere vibe_quant/bucketmod.py:2:cmp" in out.out
    assert "LOST KILL vibe_quant/bucketmod.py:8:cmp (killed at base by " in out.out
    assert out.out.splitlines()[-1] == "PARITY compared=2 lost=1 errors=0"
    assert len(baselines) == 1, baselines


def test_max_fills_from_the_lost_coverage_tier_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Line 8 is executed by the deleted test only (priority 1), line 2 by the head set too
    (priority 2): `--max 1` must pick line 8, not the first candidate line of the file."""
    repo = _repo(tmp_path, mod=_MOD_TWO, test=_TEST_TWO)
    head = _TEST_TWO.replace(
        "\n\ndef test_both() -> None:\n    assert bucket(10) == \"small\"\n"
        '    assert tag(10) == "wide"\n',
        "\n",
    )
    assert head != _TEST_TWO
    _write(repo, "tests/unit/test_bucket.py", head)
    assert _run(repo, "--max", "1") == 1
    out = capsys.readouterr()
    assert "candidate vibe_quant/bucketmod.py:8:cmp (priority 1)" in out.err
    assert "bucketmod.py:2" not in out.err
    assert "base=KILLED head=SURVIVED vibe_quant/bucketmod.py:8:cmp" in out.out
    assert "bucketmod.py:2" not in out.out
    assert out.out.splitlines()[-1] == "PARITY compared=1 lost=1 errors=0"


@pytest.mark.slow(reason="real pytest subprocesses on a scratch repo: base, head and mutant runs")
def test_real_coverage_contexts_pick_the_tests_that_reached_each_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real coverage and pytest: line 2 was reached by test_big and test_boundary at head; line 8
    only by the deleted test_both, so head has no context there and runs the changed files
    naming the module. The base side runs the changed files whole. The verdicts are an
    all-files screen's."""
    repo = _repo(tmp_path, mod=_MOD_TWO, test=_TEST_TWO)
    head = _TEST_TWO.replace(
        "\n\ndef test_both() -> None:\n    assert bucket(10) == \"small\"\n"
        '    assert tag(10) == "wide"\n',
        "\n",
    )
    _write(repo, "tests/unit/test_bucket.py", head)
    assert _run(repo) == 1
    out = capsys.readouterr()
    assert "parity: select vibe_quant/bucketmod.py:2:cmp head=2" in out.err
    assert "parity: select vibe_quant/bucketmod.py:8:cmp head=1" in out.err
    assert "base=KILLED head=KILLED vibe_quant/bucketmod.py:2:cmp" in out.out
    assert "base=KILLED head=SURVIVED vibe_quant/bucketmod.py:8:cmp  LOST KILL" in out.out
    assert "LOST KILL vibe_quant/bucketmod.py:8:cmp (killed at base by " in out.out
    assert out.out.splitlines()[-1] == "PARITY compared=2 lost=1 errors=0"


@pytest.mark.slow(reason="real pytest subprocesses on a scratch repo: base, head and mutant runs")
def test_a_test_weakened_in_place_still_counts_as_a_base_killer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """test_big2 is deleted but kills nothing; test_boundary keeps its name and loses its assert.
    Only the changed-body test killed the `>` mutant at base, so the base side must run it
    (it runs every changed test file, not only the deleted tests)."""
    single = _TEST.replace(
        '\n\ndef test_dup() -> None:\n    assert bucket(10) == "small"\n', "\n"
    )
    base_test = single + '\n\ndef test_big2() -> None:\n    assert bucket(20) == "big"\n'
    repo = _repo(tmp_path, test=base_test)
    head = single.replace('assert bucket(10) == "small"', "assert bucket(10)", 1)
    assert head != single
    _write(repo, "tests/unit/test_bucket.py", head)
    assert _run(repo) == 1
    out = capsys.readouterr()
    assert "base=KILLED head=SURVIVED vibe_quant/bucketmod.py:2:cmp  LOST KILL" in out.out
    assert "parity: select vibe_quant/bucketmod.py:2:cmp head=" in out.err
    assert out.out.splitlines()[-1] == "PARITY compared=1 lost=1 errors=0"


def test_a_unique_line_of_a_changed_production_file_is_still_a_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The deleted tests' own module changed too (the usual prune): its unchanged lines are
    still mutated, so the loss is found instead of a whole-file skip."""
    repo = _repo(tmp_path)
    _write(repo, "vibe_quant/bucketmod.py", _MOD + "\n# head edit\n")
    _write(repo, "tests/unit/test_bucket.py", _BIG_ONLY)
    assert _run(repo) == 1
    out = capsys.readouterr()
    assert "skip vibe_quant/bucketmod.py" not in out.err
    assert "LOST KILL vibe_quant/bucketmod.py:2:cmp (" in out.out
    assert out.out.splitlines()[-1] == "PARITY compared=1 lost=1 errors=0"


def test_a_changed_file_without_a_shared_line_is_named_in_the_empty_proof(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    _write(repo, "vibe_quant/bucketmod.py", _MOD.replace("value > 10", "value > 11"))
    _write(repo, "tests/unit/test_bucket.py", _BIG_ONLY)
    assert _run(repo) == 0
    out = capsys.readouterr()
    assert "skip vibe_quant/bucketmod.py: changed between base and head" in out.err
    assert out.out.splitlines()[-1] == (
        "PARITY compared=0 lost=0 errors=0 "
        "(no reachable candidate: skipped vibe_quant/bucketmod.py)"
    )


def test_import_time_lines_of_a_module_no_test_imports_are_no_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The deleted test only imports `wrapper`, which imports `flagmod`: flagmod's line 1 runs
    at import time for ANY importer, so mutating it proves nothing about the deleted test."""
    repo = _init(tmp_path / "repo")
    _write(repo, "vibe_quant/flagmod.py", "LIMIT = 10 > 5\n")
    _write(
        repo,
        "vibe_quant/wrapper.py",
        "from vibe_quant.flagmod import LIMIT\n\n\ndef on() -> bool:\n    return LIMIT\n",
    )
    kept = "from vibe_quant import wrapper\n\n\ndef test_kept() -> None:\n    assert True\n"
    _write(
        repo,
        "tests/unit/test_wrapper.py",
        kept + "\n\ndef test_import() -> None:\n    assert wrapper is not None\n",
    )
    _commit(repo)
    _write(repo, "tests/unit/test_wrapper.py", kept)
    assert _run(repo) == 0
    out = capsys.readouterr()
    assert "flagmod" not in out.err + out.out
    assert out.out.splitlines()[-1] == (
        "PARITY compared=0 lost=0 errors=0 (no reachable candidate: skipped none)"
    )


def test_lines_a_conftest_autouse_fixture_runs_are_no_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """tests/unit/conftest.py runs `hook.ping()` before EVERY test: coverage marks its line
    for the deleted test too, but no test removal can lose a kill on it."""
    repo = _init(tmp_path / "repo")
    _write(repo, "vibe_quant/hook.py", "def ping(x: int = 2) -> bool:\n    return x > 1\n")
    _write(
        repo,
        "tests/unit/conftest.py",
        "import pytest\n\nfrom vibe_quant import hook\n\n\n@pytest.fixture(autouse=True)\n"
        "def _ping() -> None:\n    hook.ping()\n",
    )
    kept = "def test_kept() -> None:\n    assert True\n"
    _write(repo, "tests/unit/test_plain.py", kept + "\n\ndef test_gone() -> None:\n    assert 1\n")
    _commit(repo)
    _write(repo, "tests/unit/test_plain.py", kept)
    assert _run(repo) == 0
    out = capsys.readouterr()
    assert "hook.py" not in out.err + out.out
    assert out.out.splitlines()[-1] == (
        "PARITY compared=0 lost=0 errors=0 (no reachable candidate: skipped none)"
    )


def test_a_line_a_fixture_and_the_deleted_test_body_both_reach_stays_a_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fixture calls `ping()` at x=2 and ignores it; the deleted test reaches the same line
    through `wrapper.boundary(1)` and asserts the result: `>` -> `>=` is a LOST kill, so
    fixture coverage must not hide the line (rev-parityfix-1 B1: it read PASS)."""
    repo = _init(tmp_path / "repo")
    _write(repo, "vibe_quant/hook.py", "def ping(x: int = 2) -> bool:\n    return x > 1\n")
    _write(
        repo,
        "vibe_quant/wrapper.py",
        "from vibe_quant.hook import ping\n\n\ndef boundary(x: int) -> bool:\n    return ping(x)\n",
    )
    _write(
        repo,
        "tests/unit/conftest.py",
        "import pytest\n\nfrom vibe_quant import hook\n\n\n@pytest.fixture(autouse=True)\n"
        "def _warmup() -> None:\n    hook.ping()\n",
    )
    kept = "from vibe_quant import wrapper\n\n\ndef test_kept() -> None:\n    assert wrapper is not None\n"
    _write(
        repo,
        "tests/unit/test_wrapper.py",
        kept + "\n\ndef test_boundary() -> None:\n    assert wrapper.boundary(1) is False\n",
    )
    _commit(repo)
    _write(repo, "tests/unit/test_wrapper.py", kept)
    assert _run(repo) == 1
    out = capsys.readouterr()
    assert "LOST KILL vibe_quant/hook.py:2:cmp (killed at base by tests/unit/test_wrapper.py::test_boundary)" in out.out


# --- renamed tests are not deleted -------------------------------------------------------


def _rename_sources(
    tmp_path: Path, base: str, head: str, head_name: str = "test_x.py"
) -> tuple[Path, Path]:
    base_dir, head_dir = tmp_path / "base", tmp_path / "head"
    _write(base_dir, "tests/unit/test_x.py", base)
    _write(head_dir, f"tests/unit/{head_name}", head)
    return base_dir, head_dir


_RENAME_BASE = """\
import pytest

LIMIT = 10


def _make() -> int:
    return LIMIT


def test_one() -> None:
    \"\"\"Doc.\"\"\"
    assert _make() == 10  # trailing


@pytest.mark.parametrize("n", [1, 2])
def test_two(n: int) -> None:
    assert n < _make()


class TestGroup:
    def test_three(self) -> None:
        assert _make() == 10
"""

_TWINS = "def test_a() -> None:\n    assert 1\n\n\ndef test_b() -> None:\n    assert 1\n"


def _ids(*names: str) -> list[str]:
    return [f"tests/unit/test_x.py::{name}" for name in names]


_X = {"tests/unit/test_x.py"}  # the file renamed_ids may rename in (the caller proved it unchanged)


def test_a_pure_rename_with_new_docstring_and_comments_is_not_a_deletion(tmp_path: Path) -> None:
    head = (
        _RENAME_BASE.replace("test_one", "test_first")
        .replace("test_two", "test_second")
        .replace("test_three", "test_third")
        .replace("Doc.", "Other doc.")
        .replace("# trailing", "# renamed")
    )
    base_dir, head_dir = _rename_sources(tmp_path, _RENAME_BASE, head)
    base_ids = _ids("test_one", "test_two[1]", "test_two[2]", "TestGroup::test_three")
    head_ids = _ids("test_first", "test_second[1]", "test_second[2]", "TestGroup::test_third")
    got = parity.renamed_ids(base_dir, head_dir, base_ids, base_ids, head_ids, _X)
    assert got == set(base_ids)


def test_a_rename_that_changes_the_assert_is_still_deleted(tmp_path: Path) -> None:
    head = _RENAME_BASE.replace("test_one", "test_first").replace("== 10  # trailing", "== 11")
    base_dir, head_dir = _rename_sources(tmp_path, _RENAME_BASE, head)
    base_ids = _ids("test_one", "TestGroup::test_three")
    head_ids = _ids("test_first", "TestGroup::test_three")
    got = parity.renamed_ids(base_dir, head_dir, _ids("test_one"), base_ids, head_ids, _X)
    assert got == set()


def test_a_rename_whose_helper_changed_is_still_deleted(tmp_path: Path) -> None:
    """Same test text, but `_make()` returns something else at head: it no longer tests the same."""
    head = _RENAME_BASE.replace("test_one", "test_first").replace("return LIMIT", "return LIMIT + 1")
    base_dir, head_dir = _rename_sources(tmp_path, _RENAME_BASE, head)
    got = parity.renamed_ids(
        base_dir, head_dir, _ids("test_one"), _ids("test_one"), _ids("test_first"), _X
    )
    assert got == set()


def test_a_new_param_value_is_a_different_test(tmp_path: Path) -> None:
    head = _RENAME_BASE.replace("test_two", "test_second")
    base_dir, head_dir = _rename_sources(tmp_path, _RENAME_BASE, head)
    head_ids = _ids("test_second[1]", "test_second[3]")
    got = parity.renamed_ids(
        base_dir, head_dir, _ids("test_two[2]"), _ids("test_two[2]"), head_ids, _X
    )
    assert got == set()


def test_a_deleted_duplicate_of_a_surviving_test_is_not_a_rename(tmp_path: Path) -> None:
    """Two identical tests at base, one at head: the head id is not new, so nothing was renamed."""
    head = "def test_a() -> None:\n    assert 1\n"
    base_dir, head_dir = _rename_sources(tmp_path, _TWINS, head)
    got = parity.renamed_ids(
        base_dir, head_dir, _ids("test_b"), _ids("test_a", "test_b"), _ids("test_a"), _X
    )
    assert got == set()


def test_one_new_head_test_answers_one_deleted_twin_only(tmp_path: Path) -> None:
    head = "def test_c() -> None:\n    assert 1\n"
    base_dir, head_dir = _rename_sources(tmp_path, _TWINS, head)
    base_ids = _ids("test_a", "test_b")
    got = parity.renamed_ids(base_dir, head_dir, base_ids, base_ids, _ids("test_c"), _X)
    assert got == {_ids("test_a")[0]}


def test_a_test_moved_to_another_file_is_a_deletion(tmp_path: Path) -> None:
    """A rename is answered within its own file: the base file lost a test, so it is not the same
    file any more and nothing proves the moved test still runs against the same environment."""
    base = "def test_a() -> None:\n    assert 1\n"
    base_dir, head_dir = _rename_sources(
        tmp_path, base, base.replace("test_a", "test_moved"), "test_y.py"
    )
    moved = ["tests/unit/test_y.py::test_moved"]
    got = parity.renamed_ids(base_dir, head_dir, _ids("test_a"), _ids("test_a"), moved, _X)
    assert got == set()


def test_a_deleted_id_of_a_file_not_proven_unchanged_is_never_a_rename(tmp_path: Path) -> None:
    head = _RENAME_BASE.replace("test_one", "test_first")
    base_dir, head_dir = _rename_sources(tmp_path, _RENAME_BASE, head)
    args = (base_dir, head_dir, _ids("test_one"), _ids("test_one"), _ids("test_first"))
    assert parity.renamed_ids(*args, set()) == set()
    assert parity.renamed_ids(*args, _X) == set(_ids("test_one"))


def test_an_unparsable_head_file_leaves_the_id_deleted(tmp_path: Path) -> None:
    base = "def test_a() -> None:\n    assert 1\n"
    base_dir, head_dir = _rename_sources(tmp_path, base, "def test_b(:\n")
    got = parity.renamed_ids(base_dir, head_dir, _ids("test_a"), _ids("test_a"), _ids("test_b"), _X)
    assert got == set()


def test_a_pure_rename_diff_runs_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    _write(
        repo,
        "tests/unit/test_bucket.py",
        _TEST.replace("test_boundary", "test_edge").replace("test_dup", "test_twin"),
    )
    assert _run(repo) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "PARITY no deleted tests"
    assert "2 of 2 missing test id(s) are AST-identical renames" in out.err
    assert "parity: coverage" not in out.err


@pytest.mark.slow(reason="real pytest subprocesses on a scratch repo: base, head and mutant runs")
def test_a_rename_that_also_drops_the_assert_is_still_a_lost_kill(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    head = _TEST.replace("test_boundary", "test_edge").replace("test_dup", "test_twin")
    head = head.replace('assert bucket(10) == "small"', 'assert bucket(20) == "big"')
    _write(repo, "tests/unit/test_bucket.py", head)
    assert _run(repo) == 1
    out = capsys.readouterr()
    assert "LOST KILL vibe_quant/bucketmod.py:2:cmp (" in out.out
    assert out.out.splitlines()[-1] == "PARITY compared=1 lost=1 errors=0"


# --- a rename or a narrowed screen is only taken where it is provably safe ---------------------

_BOUNDARY_HEAD = "from vibe_quant.bucketmod import bucket\n\n\ndef test_kept() -> None:\n    assert bucket(30) == \"big\"\n"
_INPUT_TEST = (
    "from vibe_quant.bucketmod import bucket\n\nINPUT = {n}\n\n\n"
    "def check() -> None:\n    assert bucket(INPUT) == \"small\"\n\n\n"
)


@pytest.mark.slow(reason="real pytest subprocesses on a scratch repo: base, head and mutant runs")
def test_a_rename_whose_module_constant_changed_still_loses_the_kill(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """test_old -> test_new with an identical body, but `INPUT` (read through the unchanged
    `check()`) moved from the boundary 10 to 9: the renamed test no longer kills `>` -> `>=`."""
    repo = _repo(tmp_path, test=_INPUT_TEST.format(n=10) + "def test_old() -> None:\n    check()\n")
    _write(repo, "tests/unit/test_bucket.py", _INPUT_TEST.format(n=9) + "def test_new() -> None:\n    check()\n")
    assert _run(repo) == 1
    out = capsys.readouterr()
    assert "LOST KILL vibe_quant/bucketmod.py:2:cmp (killed at base by tests/unit/test_bucket.py::test_old)" in out.out
    assert out.out.splitlines()[-1] == "PARITY compared=1 lost=1 errors=0"


@pytest.mark.slow(reason="real pytest subprocesses on a scratch repo: base, head and mutant runs")
def test_a_kept_test_whose_helper_constant_changed_is_a_base_killer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """test_kept is untouched, but the module constant its helper reads moved off the boundary
    at head: it killed `>` -> `>=` at base and no longer does. The base screen must run it."""
    base = _INPUT_TEST.format(n=10) + (
        "def test_kept() -> None:\n    check()\n\n\n"
        'def test_deleted() -> None:\n    assert bucket(20) == "big"\n'
    )
    repo = _repo(tmp_path, test=base)
    _write(repo, "tests/unit/test_bucket.py", _INPUT_TEST.format(n=9) + "def test_kept() -> None:\n    check()\n")
    assert _run(repo) == 1
    out = capsys.readouterr()
    assert "LOST KILL vibe_quant/bucketmod.py:2:cmp (killed at base by tests/unit/test_bucket.py::test_kept)" in out.out
    assert out.out.splitlines()[-1] == "PARITY compared=1 lost=1 errors=0"


@pytest.mark.slow(reason="real pytest subprocesses on a scratch repo: base, head and mutant runs")
def test_a_kept_test_stops_killing_when_the_production_it_calls_changed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """test_kept and its whole module are identical across revisions; `sample()` moved off the
    boundary at head. Source identity of a kept test proves nothing while production changed."""
    mod = _MOD + "\n\ndef sample() -> str:\n    return bucket(10)\n"
    base = (
        "from vibe_quant.bucketmod import bucket, sample\n\n\n"
        'def test_kept() -> None:\n    assert sample() == "small"\n\n\n'
        'def test_deleted() -> None:\n    assert bucket(20) == "big"\n'
    )
    repo = _repo(tmp_path, mod=mod, test=base)
    _write(repo, "vibe_quant/bucketmod.py", mod.replace("bucket(10)", "bucket(9)"))
    _write(repo, "tests/unit/test_bucket.py", base.split("\n\ndef test_deleted")[0] + "\n")
    assert _run(repo) == 1
    out = capsys.readouterr()
    assert "LOST KILL vibe_quant/bucketmod.py:2:cmp (killed at base by tests/unit/test_bucket.py::test_kept)" in out.out


@pytest.mark.slow(reason="real pytest subprocesses on a scratch repo: base, head and mutant runs")
def test_a_deleted_test_that_reaches_the_line_only_through_a_fixture_still_loses_its_kill(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """test_fixture reaches the boundary through a conftest fixture (a setup context, no test
    body), test_body reaches the same line with a non-boundary value and kills nothing."""
    repo = _init(tmp_path / "repo")
    _write(repo, "vibe_quant/bucketmod.py", _MOD)
    _write(
        repo,
        "tests/conftest.py",
        "import pytest\nfrom vibe_quant.bucketmod import bucket\n\n\n@pytest.fixture\n"
        "def boundary() -> str:\n    return bucket(10)\n",
    )
    base = (
        "from vibe_quant.bucketmod import bucket\n\n\n"
        'def test_fixture(boundary: str) -> None:\n    assert boundary == "small"\n\n\n'
        'def test_body() -> None:\n    assert bucket(20) == "big"\n\n\n'
        'def test_kept() -> None:\n    assert bucket(30) == "big"\n'
    )
    _write(repo, "tests/unit/test_bucket.py", base)
    _commit(repo)
    _write(repo, "tests/unit/test_bucket.py", _BOUNDARY_HEAD)
    assert _run(repo) == 1
    out = capsys.readouterr()
    assert "LOST KILL vibe_quant/bucketmod.py:2:cmp (killed at base by tests/unit/test_bucket.py::test_fixture)" in out.out


@pytest.mark.slow(reason="real pytest subprocesses on a scratch repo: base, head and mutant runs")
def test_a_rename_next_to_a_changed_production_file_is_still_screened(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The renamed test is AST-identical, but production changed in the same diff: it may no
    longer kill what it killed, so the rename is not taken for granted."""
    repo = _repo(tmp_path)
    _write(repo, "vibe_quant/bucketmod.py", _MOD.replace("value > 10", "value > 11"))
    _write(repo, "tests/unit/test_bucket.py", _TEST.replace("test_boundary", "test_edge").replace("test_dup", "test_twin"))
    assert _run(repo) in (0, 1)
    out = capsys.readouterr()
    assert out.out.strip() != "PARITY no deleted tests"
    assert "parity: coverage 2 deleted test(s)" in out.err


def _two_trees(tmp_path: Path, base: dict[str, str], head: dict[str, str]) -> tuple[Path, Path]:
    base_dir, head_dir = tmp_path / "base", tmp_path / "head"
    for root, files in ((base_dir, base), (head_dir, head)):
        for rel, text in files.items():
            _write(root, rel, text)
    return base_dir, head_dir


def test_only_a_file_that_is_the_same_apart_from_names_and_docs_is_unchanged(tmp_path: Path) -> None:
    same_base = 'X = 1\n\n\ndef test_a() -> None:\n    """Doc."""\n    assert X  # c\n'
    base_dir, head_dir = _two_trees(
        tmp_path,
        {
            "tests/unit/test_same.py": same_base,
            "tests/unit/test_const.py": 'X = 1\n\n\ndef test_a() -> None:\n    assert X\n',
            "tests/unit/test_gone.py": "def test_a() -> None:\n    assert 1\n\n\ndef test_b() -> None:\n    assert 2\n",
            "tests/unit/test_broken.py": "def test_a() -> None:\n    assert 1\n",
        },
        {
            "tests/unit/test_same.py": same_base.replace("test_a", "test_renamed").replace("Doc.", "Other.").replace("# c", "# d"),
            "tests/unit/test_const.py": 'X = 2\n\n\ndef test_a() -> None:\n    assert X\n',
            "tests/unit/test_gone.py": "def test_a() -> None:\n    assert 1\n",
            "tests/unit/test_broken.py": "def test_a(:\n",
        },
    )
    files = sorted(f"tests/unit/test_{n}.py" for n in ("same", "const", "gone", "broken", "newfile"))
    assert parity.unchanged_test_files(base_dir, head_dir, files) == {"tests/unit/test_same.py"}


def test_a_same_file_that_imports_a_changed_test_module_is_not_unchanged(tmp_path: Path) -> None:
    importer = "from tests.unit.test_lib import X\n\n\ndef test_a() -> None:\n    assert X\n"
    base_dir, head_dir = _two_trees(
        tmp_path,
        {"tests/unit/test_imp.py": importer, "tests/unit/test_lib.py": "X = 1\n\n\ndef test_l() -> None:\n    assert X\n"},
        {"tests/unit/test_imp.py": importer, "tests/unit/test_lib.py": "X = 0\n\n\ndef test_l() -> None:\n    assert 1\n"},
    )
    files = ["tests/unit/test_imp.py", "tests/unit/test_lib.py"]
    assert parity.unchanged_test_files(base_dir, head_dir, files) == set()


def test_changed_non_test_paths_name_every_changed_file_but_tests(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo, "tests/unit/test_bucket.py", _TEST + "\n\ndef test_new() -> None:\n    assert 1\n")
    assert parity.changed_non_test_paths(repo, "HEAD", None) == []
    _write(repo, "docs/notes.md", "x\n")  # tests read docs and config too
    _write(repo, "pyproject.toml", "x\n")
    _write(repo, "vibe_quant/bucketmod.py", _MOD + "\n# edit\n")
    _write(repo, "tests/unit/conftest.py", "X = 1\n")  # untracked: a new importable helper
    _write(repo, "scripts/tool.py", "X = 1\n")
    assert parity.changed_non_test_paths(repo, "HEAD", None) == [
        "docs/notes.md",
        "pyproject.toml",
        "scripts/tool.py",
        "tests/unit/conftest.py",
        "vibe_quant/bucketmod.py",
    ]


@pytest.mark.slow(reason="four mutants sleep 0.2 s each so overlap is observable")
def test_workers_never_exceed_the_cores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4 workers asked on a 2-core host: at most 2 mutants run at the same time (a barrier
    would deadlock if 4 did, so the peak is read from a counter)."""
    monkeypatch.setattr(parity.os, "cpu_count", lambda: 2)
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    lock = threading.Lock()
    active = [0]
    peak = [0]

    def on_mutant(_cwd: Path) -> None:
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with lock:
            active[0] -= 1

    fake.on_mutant = on_mutant
    assert _run_four(repo, fake, clock, "--workers", "4") == 1
    assert peak[0] <= 2, peak[0]
    for call in fake.calls:
        if "-n" in call:
            assert int(call[call.index("-n") + 1]) <= 2, call


def test_the_worker_budget_is_one_when_the_host_reports_no_cores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity.os, "cpu_count", lambda: None)
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    assert _run_four(repo, fake, clock, "--workers", "4") == 1
    assert len(fake.mutant_calls) > 8  # the confirm runs happened ...
    assert all("-n" not in call for call in fake.calls)  # ... and none of them is parallel


def test_unknown_base_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    assert parity.main(["--repo", str(repo), "--base", "nope"]) == 2
    out = capsys.readouterr()
    assert out.out.splitlines()[-1].startswith("PARITY error:")
    assert "nope" in out.out


def test_the_script_exits_2_on_its_own(tmp_path: Path) -> None:
    """The gate and a hand run both call the FILE, so the `__main__` exit path is the CLI."""
    repo = _repo(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--repo", str(repo), "--base", "nope"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2
    assert proc.stdout.splitlines()[-1].startswith("PARITY error:")
    assert "unknown base rev: nope" in proc.stderr


def test_candidates_reach_template_string_code_of_an_executed_template(tmp_path: Path) -> None:
    """A strategy template body is exec'd under a synthetic module name, so coverage only sees
    the `X = _lines(` statement: when it ran, the body's code lines are candidates (even when
    no deleted test body reached them), its nested docstring is not."""
    src = (
        "X = _lines(\n"
        "    '''\n"
        "    def size(self, n) -> int:\n"
        '        """Size it, n <= 3 fine."""\n'
        "        if n <= 3:\n"
        "            return 5\n"
        "        return 7\n"
        "    '''\n"
        ")\n"
    )
    (tmp_path / "vibe_quant").mkdir()
    (tmp_path / "vibe_quant" / "tpl.py").write_text(src)
    (tmp_path / "vibe_quant" / "other.py").write_text("def f(x):\n    if x <= 1:\n        return 1\n")
    got, _ = parity.candidate_mutants(
        tmp_path,
        tmp_path,
        {"vibe_quant/tpl.py": {1}, "vibe_quant/other.py": {2}},
        {},
        20,
        body_exec={"vibe_quant/tpl.py": {1}},
        named=lambda _path: False,
    )
    tiers = {m.name: p for m, p in got}
    assert tiers["vibe_quant/tpl.py:5:cmp"] == 2  # no body-reach proof: spare budget only
    assert not any(n.startswith("vibe_quant/other.py:") for n in tiers)  # the filter still applies
    assert not any(n.startswith("vibe_quant/tpl.py:4:") for n in tiers)


def test_candidates_reach_an_operator_in_a_generated_fstring_item(tmp_path: Path) -> None:
    """parity reuses mutate.line_mutants, so a covered compiler.py-style f-string line yields
    its `emit` mutant with the `{dsl.max_bars_in_trade}` field and the `f` prefix untouched."""
    src = (
        "def build(lines, dsl):\n"
        "    lines.extend(\n"
        "        [\n"
        '            f"        if self._bars_in_trade >= {dsl.max_bars_in_trade}:",\n'
        "        ]\n"
        "    )\n"
    )
    (tmp_path / "vibe_quant").mkdir()
    (tmp_path / "vibe_quant" / "emit.py").write_text(src)
    got, skipped = parity.candidate_mutants(
        tmp_path,
        tmp_path,
        {"vibe_quant/emit.py": {4}},
        {},
        20,
        body_exec={"vibe_quant/emit.py": {4}},
    )
    assert skipped == []
    assert [(m.name, m.new) for m, _tier in got] == [
        (
            "vibe_quant/emit.py:4:emit:cmp",
            '            f"        if self._bars_in_trade > {dsl.max_bars_in_trade}:",',
        )
    ]


def _fake_screen(verdicts: dict[str, tuple[str, str]], calls: list[list[str]]) -> object:
    def screen_some(batch: Sequence[object]) -> list[object]:
        calls.append([m.name for m in batch])  # type: ignore[attr-defined]
        return [
            parity.Screened(m, *verdicts[m.name], "", 0)  # type: ignore[attr-defined]
            for m in batch
        ]

    return screen_some


def _m(name: str) -> object:
    return parity.mutate.Mutant(name, "vibe_quant/x.py", "a", "b")


def test_a_mutant_invalid_at_both_sides_is_replaced_by_the_next_candidate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A gate once reported 3 INVALID-at-both mutants as 6 errors with lost=0: no verdict
    either way, so the slot goes to the next candidate."""
    verdicts = {
        "a": ("KILLED", "KILLED"), "b": ("INVALID", "INVALID"),
        "c": ("INVALID", "INVALID"), "d": ("KILLED", "SURVIVED"), "e": ("SURVIVED", "SURVIVED"),
    }  # fmt: skip
    calls: list[list[str]] = []
    kept, dropped = parity.screen_replacing(
        [_m("a"), _m("b")], lambda: [_m("a"), _m("b"), _m("c"), _m("d"), _m("e")],
        _fake_screen(verdicts, calls),
    )  # fmt: skip
    assert [k.mutant.name for k in kept] == ["a", "d"]
    assert dropped == 2
    assert calls == [["a", "b"], ["c"], ["d"]]
    assert "dropped b: INVALID at base and head" in capsys.readouterr().out


def test_a_one_sided_invalid_is_kept_as_an_error_not_replaced() -> None:
    verdicts = {"a": ("INVALID", "SURVIVED"), "b": ("KILLED", "INVALID")}
    calls: list[list[str]] = []
    kept, dropped = parity.screen_replacing(
        [_m("a"), _m("b")], lambda: [], _fake_screen(verdicts, calls)
    )
    assert [k.mutant.name for k in kept] == ["a", "b"]
    assert (dropped, calls) == (0, [["a", "b"]])


def test_replacement_stops_when_the_candidates_run_out() -> None:
    verdicts = {"a": ("INVALID", "INVALID"), "b": ("INVALID", "INVALID")}
    calls: list[list[str]] = []
    kept, dropped = parity.screen_replacing(
        [_m("a")], lambda: [_m("a"), _m("b")], _fake_screen(verdicts, calls)
    )
    assert (kept, dropped, calls) == ([], 2, [["a"], ["b"]])


def test_a_run_whose_every_candidate_was_dropped_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping INVALID-at-both mutants must not turn 'no answer' into a clean 0."""
    repo = _repo(tmp_path)
    _write(
        repo,
        "tests/unit/test_bucket.py",
        'from vibe_quant.bucketmod import bucket\n\n\ndef test_big() -> None:\n'
        '    assert bucket(20) == "big"\n',
    )
    monkeypatch.setattr(
        parity, "screen_replacing", lambda first, more, some, progress: ([], len(first))
    )
    assert _run(repo) == 2
    out = capsys.readouterr().out.splitlines()
    assert out[-2] == "  no valid comparison: all 1 candidates INVALID at base and head"
    assert out[-1] == "PARITY compared=0 lost=0 errors=1"


# --- parallel pytest runs, progress lines and the --budget wall cap -----------------------
# The copies run with plugin autoload off (_fast_copy_pytest), so a real pytest there cannot
# honour `-n`: every subprocess is answered by the fake below and its argv is recorded. The
# clock is a plain float the test drives, so no test ever sleeps.

# Four branches with DIFFERENT comparison text (line_mutants refuses a non-unique line), so
# each line yields exactly one `cmp` mutant: a `--max 4` run has 4 mutants to screen.
_MOD_FOUR = """\
def a(value: int) -> bool:
    return value > 10


def b(value: int) -> bool:
    return value < 10


def c(value: int) -> bool:
    return value != 11


def d(value: int) -> bool:
    return value == 10
"""

_TEST_FOUR = """\
from vibe_quant.bucketmod import a, b, c, d


def test_a() -> None:
    assert a(20) is True


def test_b() -> None:
    assert b(1) is True


def test_c() -> None:
    assert c(10) is True


def test_d() -> None:
    assert d(10) is True
"""

_TEST_FOUR_HEAD = _TEST_FOUR.replace(
    "\n\ndef test_d() -> None:\n    assert d(10) is True\n", "\n"
)
assert _TEST_FOUR_HEAD != _TEST_FOUR

_FOUR_IDS = [f"tests/unit/test_bucket.py::test_{name}" for name in "abcd"]
_LINES = (2, 6, 10, 14)  # the four comparison lines of _MOD_FOUR
_KILLED = "FAILED tests/unit/test_bucket.py::test_a - assert a(20)\n"


class FakeParityRun:
    """A run_fn for the copies: answers collection/coverage, scripts the mutant runs.

    Collection returns `_FOUR_IDS` at base and the same minus test_d at head; the coverage
    report marks the four comparison lines as reached by a test body; a mutant run gets the
    test's `verdict(n, side)` (n counts MUTATED runs) and `after(n)` may move the clock."""

    def __init__(self, clock: list[float]) -> None:
        self.calls: list[list[str]] = []
        self.mutated = 0
        self.clock = clock
        self.verdict: Callable[[int, str], tuple[int, str]] = lambda _n, _side: (
            0,
            "4 passed in 0.1s",
        )
        self.after: Callable[[int], None] = lambda _n: None
        self.timeout_when_clamped = False
        self.on_mutant: Callable[[Path], None] = lambda _cwd: None
        self.selected: list[tuple[str, list[str]]] = []  # (side, ids of an @argsfile run)
        self.mutant_calls: list[list[str]] = []
        self.mutant_sides: list[str] = []  # the copy ("base"/"head") of each mutant_calls entry
        self.verify_calls: list[list[str]] = []

    def __call__(
        self, cmd: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: float
    ) -> object:
        call = [str(arg) for arg in cmd]
        self.calls.append(call)
        if "--collect-only" in call:
            ids = _FOUR_IDS if cwd.name == "base" else _FOUR_IDS[:3]
            return parity.mutate.RunResult(0, 0.0, output="\n".join(ids))
        if call[1] == "-c":  # the coverage-context query: base lines belong to test_d, head to their own
            wanted = json.loads(Path(call[4]).read_text(encoding="utf-8"))
            owner = Path(call[3]).name
            found = {
                path: {
                    str(line): [
                        "tests.unit.test_bucket.test_"
                        + ("d" if owner == "base" else "abcd"[_LINES.index(line)])
                    ]
                    for line in lines
                }
                for path, lines in wanted.items()
            }
            return parity.mutate.RunResult(0, 0.0, output=json.dumps(found))
        if call[call.index("-m") + 1] == "coverage":
            if "json" in call:
                report = Path(call[call.index("-o") + 1])
                files = {
                    "vibe_quant/bucketmod.py": {
                        "executed_lines": list(_LINES),
                        "contexts": {str(line): ["test"] for line in _LINES},
                    }
                }
                report.write_text(json.dumps({"files": files}), encoding="utf-8")
                return parity.mutate.RunResult(0, 0.0)
            return parity.mutate.RunResult(0, 0.0, output="4 passed in 0.1s")
        mutated = (cwd / "vibe_quant" / "bucketmod.py").read_text(encoding="utf-8") != _MOD_FOUR
        if mutated:
            self.mutant_calls.append(call)
            self.mutant_sides.append(cwd.name)
            for arg in call:
                if arg.startswith("@"):
                    ids = Path(arg[1:]).read_text(encoding="utf-8").split()
                    self.selected.append((cwd.name, ids))
            self.on_mutant(cwd)
            self.mutated += 1
            if self.timeout_when_clamped and timeout < parity.TOOL_TIMEOUT:
                self.after(self.mutated)
                return parity.mutate.RunResult(124, 0.0, timed_out=True)
            exit_code, output = self.verdict(self.mutated, cwd.name)
        else:
            if call[-1].startswith("@") and call[-1].endswith("v.txt"):
                self.verify_calls.append(call)  # a KILLED verdict's failures re-run unmutated
            exit_code, output = 0, "4 passed in 0.1s"
        self.after(self.mutated)
        return parity.mutate.RunResult(exit_code, 0.0, output=output)


def _repo_four(tmp_path: Path) -> Path:
    repo = _init(tmp_path / "repo")
    _write(repo, "vibe_quant/bucketmod.py", _MOD_FOUR)
    _write(repo, "tests/unit/test_bucket.py", _TEST_FOUR)
    _commit(repo)
    _write(repo, "tests/unit/test_bucket.py", _TEST_FOUR_HEAD)  # head = working tree
    return repo


def _run_four(
    repo: Path, fake: FakeParityRun, clock: list[float], *extra: str
) -> int:
    if "--workers" not in extra:
        extra = ("--workers", "1", *extra)  # serial slots: the fake's run order is the contract
    return int(
        parity.main(
            ["--repo", str(repo), "--base", "HEAD", "--max", "4", *extra],
            fake,
            lambda: clock[0],
        )
    )


def _screens(fake: FakeParityRun) -> list[list[str]]:
    """The mutated pytest runs of the screen: 2 sides x 4 mutants come before any confirm run."""
    return fake.mutant_calls[:8]


def test_workers_reach_the_confirm_runs_while_the_slots_share_them(tmp_path: Path) -> None:
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    assert _run_four(repo, fake, clock, "--workers", "4") == 1
    screens = _screens(fake)
    assert len(screens) == 8
    assert len(fake.verify_calls) == 4  # one per KILLED base verdict
    parallel = 0
    for call in fake.calls:
        if call[1] == "-c" or call[call.index("-m") + 1] == "coverage" or "--collect-only" in call:
            assert "-n" not in call, call
        elif call in screens or call in fake.verify_calls:
            assert "-n" not in call, call  # 4 slots x 1 worker = the 4 workers
        else:
            assert call[call.index("-n") + 1] == "4", call
            assert call[call.index("--dist") + 1] == "worksteal", call
            parallel += 1
    assert parallel == 5, parallel  # 1 importer baseline + 4 confirms


def test_slots_split_the_workers_between_the_mutants_that_run_together(tmp_path: Path) -> None:
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    assert _run_four(repo, fake, clock, "--workers", "4", "--max", "2") == 1
    screens = fake.mutant_calls[:4]
    assert len(screens) == 4
    for call in [*screens, *fake.verify_calls]:  # 2 mutants in 2 slots: 4 // 2 workers each
        assert call[call.index("-n") + 1] == "2", call


def test_two_workers_are_parallel_and_one_is_serial(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `>= 2` edge: exactly 2 workers add the flags, exactly 1 stays serial. The two
    mutants that move that comparison either way both die on these two runs."""
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    assert _run_four(repo, fake, clock, "--workers", "2") == 1
    screens = _screens(fake)
    flagged = [
        call for call in fake.calls
        if call[1] != "-c"
        and call[call.index("-m") + 1] != "coverage"
        and "--collect-only" not in call
        and call not in screens
        and call not in fake.verify_calls
    ]  # fmt: skip
    assert flagged
    for call in flagged:
        assert call[call.index("-n") + 1] == "2", call
        assert call[call.index("--dist") + 1] == "worksteal", call
    assert all("-n" not in call for call in [*screens, *fake.verify_calls])  # 2 slots x 1 worker
    capsys.readouterr()
    serial = FakeParityRun([0.0])
    serial.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    assert _run_four(repo, serial, [0.0], "--workers", "1") == 1
    assert all("-n" not in call for call in serial.calls), serial.calls


def test_each_mutant_runs_only_the_tests_whose_context_reached_its_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Base always runs the changed files whole; head runs the tests that reached the line, and a
    line whose context names no collected test (test_d is gone at head) every changed file."""
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    assert _run_four(repo, fake, clock) == 1
    assert [ids for side, ids in fake.selected if side == "base"] == []  # whole files, no @argsfile
    base_screens = [
        call for call, side in zip(fake.mutant_calls[:8], fake.mutant_sides[:8], strict=True)
        if side == "base"
    ]  # fmt: skip
    assert len(base_screens) == 4
    assert all(call[-1] == "tests/unit/test_bucket.py" for call in base_screens)
    assert sorted(ids for side, ids in fake.selected if side == "head") == [
        ["tests/unit/test_bucket.py"],  # line 14: its context names a test gone at head
        *[[f"tests/unit/test_bucket.py::test_{name}"] for name in "abc"],
    ]
    err = capsys.readouterr().err
    assert "parity: select vibe_quant/bucketmod.py:2:cmp head=1" in err
    assert "parity: select vibe_quant/bucketmod.py:14:cmp head=1" in err


def test_a_file_that_only_got_renames_is_left_out_of_the_base_screen(tmp_path: Path) -> None:
    """test_same.py changed only by a test rename (no production file changed): it cannot have
    lost a kill, so the base screen runs test_bucket.py alone. Both files stay in the head set."""
    repo = _init(tmp_path / "repo")
    _write(repo, "vibe_quant/bucketmod.py", _MOD_FOUR)
    _write(repo, "tests/unit/test_bucket.py", _TEST_FOUR)
    same = "def test_one() -> None:\n    assert 1\n"
    _write(repo, "tests/unit/test_same.py", same)
    _commit(repo)
    _write(repo, "tests/unit/test_bucket.py", _TEST_FOUR_HEAD)
    _write(repo, "tests/unit/test_same.py", same.replace("test_one", "test_first"))
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    assert _run_four(repo, fake, clock, "--max", "1") == 1
    base_screens = [
        call for call, side in zip(fake.mutant_calls, fake.mutant_sides, strict=True) if side == "base"
    ]
    assert base_screens
    assert all(call[-1] == "tests/unit/test_bucket.py" for call in base_screens)


def test_a_mutant_no_base_test_kills_skips_the_head_side(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)  # every run passes: no mutant is killed at base
    assert _run_four(repo, fake, clock) == 0
    out = capsys.readouterr().out
    assert out.count("base=SURVIVED head=SKIPPED ") == 4
    assert len(fake.mutant_calls) == 4 and not fake.verify_calls
    assert out.splitlines()[-1] == "PARITY compared=4 lost=0 errors=0"


def test_a_kill_whose_failures_also_fail_unmutated_is_a_broken_baseline_not_a_kill(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The base copy is red without any mutant: the failing node reruns unmutated and fails again."""
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, _side: (1, _KILLED)
    original = fake.__call__

    def red_when_rerun(cmd: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: float) -> object:
        if str(cmd[-1]).startswith("@") and str(cmd[-1]).endswith("v.txt"):
            return parity.mutate.RunResult(1, 0.0, output=_KILLED)
        return original(cmd, cwd, env, timeout)

    assert int(
        parity.main(
            ["--repo", str(repo), "--base", "HEAD", "--max", "4", "--workers", "1"],
            red_when_rerun,
            lambda: clock[0],
        )
    ) == 2
    out = capsys.readouterr().out
    assert "PARITY error: base baseline: tests fail without mutation: tests/unit/test_bucket.py::test_a" in out


@pytest.mark.slow(reason="git archive/clone of a scratch repo into 4 slots, plus the slot sleeps")
def test_mutants_screen_in_parallel_slots_and_print_in_candidate_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Four slots: the 1st mutant is held back until the other three finished, yet the lines
    come out 2, 6, 10, 14; two mutants are inside their copies at the same time."""
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    lock = threading.Lock()
    inside: set[Path] = set()
    peak: list[int] = [0]

    def mutated_line(cwd: Path) -> int:
        text = (cwd / "vibe_quant" / "bucketmod.py").read_text(encoding="utf-8").splitlines()
        return next(n for n, (a, b) in enumerate(zip(text, _MOD_FOUR.splitlines(), strict=True), 1) if a != b)

    def on_mutant(cwd: Path) -> None:
        with lock:
            inside.add(cwd)
            peak[0] = max(peak[0], len(inside))
        if mutated_line(cwd) == 2:
            deadline = time.monotonic() + 20
            while fake.mutated < 6 and time.monotonic() < deadline:
                time.sleep(0.01)  # the slow mutant: the other three started both sides first
            time.sleep(0.3)
        else:
            time.sleep(0.05)

    def after(_runs: int) -> None:
        with lock:
            inside.clear()

    fake.on_mutant = on_mutant
    fake.after = after
    assert _run_four(repo, fake, clock, "--workers", "4") == 1
    out = capsys.readouterr().out.splitlines()
    shown = [line.split()[2] for line in out if line.startswith("base=")]
    assert shown == [f"vibe_quant/bucketmod.py:{n}:cmp" for n in (2, 6, 10, 14)]
    assert peak[0] >= 2  # two copies were mutated at once


def test_parallel_slots_never_exceed_the_workers_and_use_their_own_copies(
    tmp_path: Path,
) -> None:
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    lock = threading.Lock()
    active: list[Path] = []
    peak: list[int] = [0]
    copies: set[Path] = set()

    def on_mutant(cwd: Path) -> None:
        with lock:
            assert cwd not in active, "two mutants share one copy"
            active.append(cwd)
            copies.add(cwd)
            peak[0] = max(peak[0], len(active))
        time.sleep(0.05)

    def after(_runs: int) -> None:
        with lock:
            active.clear()

    fake.on_mutant = on_mutant
    fake.after = after
    assert _run_four(repo, fake, clock, "--workers", "3") == 1
    assert 2 <= peak[0] <= 3
    assert len({c for c in copies if c.name == "head"}) == 3  # one head copy per slot


def test_a_budget_stop_inside_a_parallel_screen_is_still_incomplete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, _side: (1, _KILLED)
    fake.after = lambda n: clock.__setitem__(0, 100.0) if n >= 3 else None
    assert _run_four(repo, fake, clock, "--workers", "4", "--budget", "100") == 3
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith("PARITY INCOMPLETE compared=")
    assert last.endswith("of 4 lost=0 errors=0 (budget 100s hit in screen)")


def test_a_negative_budget_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo_four(tmp_path)
    with pytest.raises(SystemExit) as caught:
        parity.main(["--repo", str(repo), "--base", "HEAD", "--budget", "-1"])
    assert caught.value.code == 2
    assert "--budget must be >= 0" in capsys.readouterr().err


def test_a_negative_worker_count_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo_four(tmp_path)
    with pytest.raises(SystemExit) as caught:
        parity.main(["--repo", str(repo), "--base", "HEAD", "--workers", "-1"])
    assert caught.value.code == 2
    assert "--workers must be >= 0" in capsys.readouterr().err


def test_budget_hit_in_screen_reports_incomplete_and_exits_3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, _side: (1, _KILLED)  # killed on both sides: both sides run

    def after(runs: int) -> None:
        if runs == 4:  # the 2nd mutant's head run just finished (its re-check hits the cap)
            clock[0] = 100.0

    fake.after = after
    assert _run_four(repo, fake, clock, "--budget", "100") == 3
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == (
        "PARITY INCOMPLETE compared=1 of 4 lost=0 errors=0 (budget 100s hit in screen)"
    )
    assert len([line for line in out if line.startswith("base=")]) == 1


def test_budget_after_a_confirmed_loss_still_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")

    def after(runs: int) -> None:
        if runs == 9:  # 8 screen runs + the 1st confirm run, which SURVIVED
            clock[0] = 100.0

    fake.after = after
    assert _run_four(repo, fake, clock, "--budget", "100") == 1
    out = capsys.readouterr().out.splitlines()
    assert len([line for line in out if line.startswith("LOST KILL ")]) == 1
    assert "(budget 100s hit in confirm)" in out[-1]


def test_an_error_seen_before_the_budget_cap_is_never_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The base side of the 1st mutant ends in a pytest internal error (exit 3), then the cap
    expires before its head side runs: the error is still counted, so the run FAILS (exit 2)
    instead of landing as an INCOMPLETE warning."""
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda n, _side: (3, "INTERNALERROR injected worker error") if n == 1 else (0, "4 passed")
    fake.after = lambda n: clock.__setitem__(0, 100.0) if n == 1 else None
    assert _run_four(repo, fake, clock, "--budget", "100") == 2
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == (
        "PARITY INCOMPLETE compared=0 of 4 lost=0 errors=1 (budget 100s hit in screen)"
    )


def test_a_completed_comparison_error_counts_once_under_the_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mutant 1 finishes both sides with a base error, then the cap hits mutant 2: errors=1,
    not 2 (the finished comparison is not counted again when it is screened)."""
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda n, side: (
        (3, "INTERNALERROR injected worker error") if (n, side) == (1, "base") else (0, "4 passed")
    )
    fake.after = lambda n: clock.__setitem__(0, 100.0) if n == 2 else None
    assert _run_four(repo, fake, clock, "--budget", "100") == 2
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == (
        "PARITY INCOMPLETE compared=1 of 4 lost=0 errors=1 (budget 100s hit in screen)"
    )


def test_an_error_beats_a_confirmed_loss_under_the_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Precedence errors (2) > confirmed loss (1) > INCOMPLETE (3) also holds on a cut-short
    run: mutant 1 errors at base, mutants 2-4 are lost kills, the cap hits during confirm."""
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)

    def verdict(n: int, side: str) -> tuple[int, str]:
        if n == 1:
            return 3, "INTERNALERROR injected worker error"
        return (1, _KILLED) if side == "base" else (0, "4 passed")

    fake.verdict = verdict

    def after(runs: int) -> None:
        if runs == 9:  # 8 screen runs + the 1st confirm run, which SURVIVED (a lost kill)
            clock[0] = 100.0

    fake.after = after
    assert _run_four(repo, fake, clock, "--budget", "100") == 2
    out = capsys.readouterr().out.splitlines()
    assert "LOST KILL" in "\n".join(out)
    assert "errors=1" in out[-1] and "lost=1" in out[-1]


def test_a_budget_clamped_timeout_is_no_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run cut short by the remaining budget is not a KILLED of anything: the mutant and
    everything after it is uncompared, and the run has no clean verdict (exit 3)."""
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, _side: (1, _KILLED)
    fake.timeout_when_clamped = True

    def after(runs: int) -> None:
        if runs == 2:  # the 1st mutant is done; only 50 s of budget is left
            clock[0] = 350.0

    fake.after = after
    assert _run_four(repo, fake, clock, "--budget", "400") == 3
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == (
        "PARITY INCOMPLETE compared=1 of 4 lost=0 errors=0 (budget 400s hit in screen)"
    )
    assert "TIMEOUT" not in "\n".join(out)
    assert len([line for line in out if line.startswith("base=")]) == 1


def test_progress_lines_mark_every_phase_and_mutant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")

    def after(_runs: int) -> None:
        clock[0] += 1.0

    fake.after = after
    assert _run_four(repo, fake, clock) == 1
    err = capsys.readouterr().err.splitlines()
    for phase in ("collect", "coverage", "screen", "confirm"):
        starts = [line for line in err if re.match(rf"^parity: {phase} (?!\d+/)", line)]
        assert len(starts) == 1, (phase, err)
    for phase in ("screen", "confirm"):
        lines = [
            line
            for line in err
            if re.fullmatch(
                rf"parity: {phase} \d+/4 vibe_quant/bucketmod\.py:\d+:cmp \d+\.\ds", line
            )
        ]
        assert len(lines) == 4, (phase, err)
