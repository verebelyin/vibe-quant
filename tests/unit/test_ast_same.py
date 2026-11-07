"""Tests for ``scripts/agents/ast_same.py`` — the docstring/name/comment-only refactor proof.

The checker is run as a subprocess against a throwaway git repo so the tests do not
depend on this checkout's history: a change that touches only docstrings, ``#``
comments or ``test_*`` names stays SAME, while an edited assert value or a deleted
test is a DIFF with a node path.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "ast_same.py"

_TEST_FILE = "tests/unit/test_sample.py"

_BASE = '''\
"""Module contract: a module docstring."""


def test_alpha() -> None:
    """The alpha case."""
    assert 1 + 1 == 2  # parity holds


def test_beta() -> None:
    """The beta case."""
    assert 2 * 2 == 4
'''


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tests" / "unit").mkdir(parents=True)
    (repo / _TEST_FILE).write_text(_BASE)
    _git(repo, "init", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")
    return repo


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--repo", str(repo), *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_docstring_comment_and_test_name_changes_are_same(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / _TEST_FILE).write_text(
        '''\
"""A module docstring."""


def test_alpha_case() -> None:
    """The alpha case."""
    assert 1 + 1 == 2  # stays


def test_beta_case() -> None:
    """The beta case."""
    assert 2 * 2 == 4
'''
    )
    done = _run(repo, "--base", "HEAD")
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [f"SAME {_TEST_FILE}"]


def test_changed_assert_value_is_a_diff_with_a_node_path(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / _TEST_FILE).write_text(_BASE.replace("1 + 1 == 2", "1 + 1 == 3"))
    done = _run(repo, "--base", "HEAD")
    assert done.returncode == 1
    lines = done.stdout.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"DIFF {_TEST_FILE}: Module.body[0].body[0]")
    assert lines[0].endswith("(2 != 3)")


def test_deleted_test_is_a_diff(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / _TEST_FILE).write_text(
        '''\
"""Module contract: a module docstring."""


def test_alpha() -> None:
    """The alpha case."""
    assert 1 + 1 == 2  # parity holds
'''
    )
    done = _run(repo, "--base", "HEAD")
    assert done.returncode == 1
    assert done.stdout.splitlines() == [f"DIFF {_TEST_FILE}: 2 test functions at base, 1 at head"]


def test_new_test_file_is_reported_and_not_a_diff(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / _TEST_FILE).write_text(_BASE.replace("The alpha case.", "Alpha."))
    (repo / "tests" / "unit" / "test_new.py").write_text("def test_new() -> None:\n    assert True\n")
    done = _run(repo, "--base", "HEAD")
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == ["NEW tests/unit/test_new.py", f"SAME {_TEST_FILE}"]


def test_deleted_file_is_a_diff(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / _TEST_FILE).unlink()
    done = _run(repo, "--base", "HEAD")
    assert done.returncode == 1
    assert done.stdout.splitlines() == [f"DIFF {_TEST_FILE}: missing at head"]


def test_renamed_file_with_an_edited_assert_is_not_a_pass(tmp_path: Path) -> None:
    """A staged rename looks like a brand-new file at its destination; its source must still
    be checked, so a rename plus a body change exits 1."""
    repo = _repo(tmp_path)
    _git(repo, "mv", _TEST_FILE, "tests/unit/test_moved.py")
    moved = repo / "tests" / "unit" / "test_moved.py"
    moved.write_text(_BASE.replace("2 * 2 == 4", "2 * 2 == 5"))
    _git(repo, "add", "-A")
    done = _run(repo, "--base", "HEAD")
    assert done.returncode == 1
    assert f"DIFF {_TEST_FILE}: missing at head" in done.stdout.splitlines()

def test_explicit_paths_skip_unchanged_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / _TEST_FILE).write_text(_BASE.replace('"""The alpha case."""', '"""Alpha."""'))
    done = _run(repo, "--base", "HEAD", _TEST_FILE)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines() == [f"SAME {_TEST_FILE}"]


def test_missing_base_ref_fails_closed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    done = _run(repo, "--base", "no-such-ref")
    assert done.returncode == 2
    assert "no-such-ref" in done.stderr


def _module() -> object:
    spec = importlib.util.spec_from_file_location("ast_same_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_FP_SOURCE = """\
import os
from pkg import helper as aliased

LIMIT = 3


def _make():
    return LIMIT


class TestGroup:
    base = 1

    def _own(self):
        return self.base

    def test_one(self, fixture_a):
        \"\"\"Doc.\"\"\"
        assert _make() == aliased(os.sep)  # note


def test_two():
    assert _make()
"""


def test_fingerprints_ignore_names_docstrings_and_comments_but_not_bodies() -> None:
    mod = _module()
    base = mod.fingerprint_tests(_FP_SOURCE)  # type: ignore[attr-defined]
    assert set(base) == {"TestGroup::test_one", "test_two"}
    renamed = _FP_SOURCE.replace("test_one", "test_first").replace("Doc.", "Other.")
    head = mod.fingerprint_tests(renamed.replace("# note", ""))  # type: ignore[attr-defined]
    assert head["TestGroup::test_first"] == base["TestGroup::test_one"]
    edited = mod.fingerprint_tests(  # type: ignore[attr-defined]
        _FP_SOURCE.replace("assert _make()\n", "assert not _make()\n")
    )
    assert edited["test_two"].body != base["test_two"].body
    assert edited["TestGroup::test_one"] == base["TestGroup::test_one"]


def test_fingerprint_names_hold_the_fixtures_and_referenced_module_names() -> None:
    mod = _module()
    got = mod.fingerprint_tests(_FP_SOURCE)["TestGroup::test_one"]  # type: ignore[attr-defined]
    assert {"fixture_a", "_make", "aliased", "os"} <= got.names


def test_a_changed_class_member_changes_the_header_not_the_body() -> None:
    mod = _module()
    base = mod.fingerprint_tests(_FP_SOURCE)["TestGroup::test_one"]  # type: ignore[attr-defined]
    changed = _FP_SOURCE.replace("return self.base", "return 7")
    other = mod.fingerprint_tests(changed)["TestGroup::test_one"]  # type: ignore[attr-defined]
    assert (other.body, other.header != base.header) == (base.body, True)


def test_env_fingerprints_track_helpers_imports_and_assignments() -> None:
    mod = _module()
    env = mod.env_fingerprints(_FP_SOURCE)  # type: ignore[attr-defined]
    assert {"os", "aliased", "LIMIT", "_make"} <= set(env)
    assert "TestGroup" not in env and "test_two" not in env
    changed = mod.env_fingerprints(_FP_SOURCE.replace("LIMIT = 3", "LIMIT = 4"))  # type: ignore[attr-defined]
    assert changed["LIMIT"] != env["LIMIT"] and changed["_make"] == env["_make"]
    moved = mod.env_fingerprints(_FP_SOURCE.replace("pkg", "pkg2"))  # type: ignore[attr-defined]
    assert moved["aliased"] != env["aliased"]
    guarded = mod.env_fingerprints(  # type: ignore[attr-defined]
        "try:\n    import fast as impl\nexcept ImportError:\n    impl = None\n"
    )
    assert "impl" in guarded and "fast" not in guarded
