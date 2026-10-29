"""scripts/agents/parity_mutants.py — a test-deleting diff must not LOSE a mutant kill.

Every test drives the CLI in-process against a throwaway git repo, with real pytest,
coverage and git subprocesses in the copies: the deleted-node-id diff, the per-line
candidates, the base/head screen and the "covered elsewhere" confirmation are exactly
the code path the gate's `parity` check runs. The repo is a real git repo (not a
DeclaredGit) because `git archive` and `git diff <base> -- tests/` ARE the feature.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "parity_mutants.py"
assert _SCRIPT.is_file(), f"missing {_SCRIPT}"
_spec = importlib.util.spec_from_file_location("parity_mutants", _SCRIPT)
assert _spec is not None and _spec.loader is not None
parity = importlib.util.module_from_spec(_spec)
sys.modules["parity_mutants"] = parity
_spec.loader.exec_module(parity)

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
    return parity.main(["--repo", str(repo), "--base", "HEAD", *extra])


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


def test_deleting_a_duplicate_test_loses_no_kill(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    head = _TEST.replace(
        '\n\ndef test_dup() -> None:\n    assert bucket(10) == "small"\n', "\n"
    )
    assert head != _TEST
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
    assert out.err == ""


def test_a_killer_outside_the_changed_set_is_covered_elsewhere(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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


def test_a_multiline_from_import_names_the_module(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    _write(
        repo,
        "tests/unit/test_other.py",
        "from vibe_quant import (\n    bucketmod,\n)\n\n\n"
        'def test_boundary_other() -> None:\n    assert bucketmod.bucket(10) == "small"\n',
    )
    _commit(repo)
    _write(repo, "tests/unit/test_bucket.py", _BIG_ONLY)
    assert _run(repo) == 0
    out = capsys.readouterr()
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

    assert parity.main(["--repo", str(repo), "--base", "HEAD"], run_fn) == 1
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
