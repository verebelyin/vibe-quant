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
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

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
        self.verdict = lambda _n, _side: (0, "4 passed in 0.1s")  # type: ignore[assignment]
        self.after = lambda _n: None  # type: ignore[assignment]
        self.timeout_when_clamped = False

    def __call__(
        self, cmd: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: float
    ) -> object:
        call = [str(arg) for arg in cmd]
        self.calls.append(call)
        if "--collect-only" in call:
            ids = _FOUR_IDS if cwd.name == "base" else _FOUR_IDS[:3]
            return parity.mutate.RunResult(0, 0.0, output="\n".join(ids))
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
            self.mutated += 1
            if self.timeout_when_clamped and timeout < parity.TOOL_TIMEOUT:
                self.after(self.mutated)
                return parity.mutate.RunResult(124, 0.0, timed_out=True)
            exit_code, output = self.verdict(self.mutated, cwd.name)
        else:
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
    return int(
        parity.main(
            ["--repo", str(repo), "--base", "HEAD", "--max", "4", *extra],
            fake,
            lambda: clock[0],
        )
    )


def test_workers_reach_every_non_coverage_pytest_command(tmp_path: Path) -> None:
    repo = _repo_four(tmp_path)
    clock = [0.0]
    fake = FakeParityRun(clock)
    fake.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    assert _run_four(repo, fake, clock, "--workers", "4") == 1
    parallel = 0
    for call in fake.calls:
        mode = call[call.index("-m") + 1]
        if mode == "coverage" or "--collect-only" in call:
            assert "-n" not in call, call
            continue
        assert call[call.index("-n") + 1] == "4", call
        assert call[call.index("--dist") + 1] == "worksteal", call
        parallel += 1
    assert parallel == 15, parallel  # 2 baselines + 8 screens + 1 importer + 4 confirms


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
    for call in fake.calls:
        if call[call.index("-m") + 1] == "coverage" or "--collect-only" in call:
            assert "-n" not in call, call
        else:
            assert call[call.index("-n") + 1] == "2", call
            assert call[call.index("--dist") + 1] == "worksteal", call
    capsys.readouterr()
    serial = FakeParityRun([0.0])
    serial.verdict = lambda _n, side: (1, _KILLED) if side == "base" else (0, "4 passed")
    assert _run_four(repo, serial, [0.0], "--workers", "1") == 1
    assert all("-n" not in call for call in serial.calls), serial.calls


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

    def after(runs: int) -> None:
        if runs == 4:  # the 2nd mutant's head run just finished
            clock[0] = 100.0

    fake.after = after
    assert _run_four(repo, fake, clock, "--budget", "100") == 3
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == (
        "PARITY INCOMPLETE compared=2 of 4 lost=0 errors=0 (budget 100s hit in screen)"
    )
    assert len([line for line in out if line.startswith("base=")]) == 2


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
