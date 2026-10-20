"""scripts/agents/merge_words.py: word-level 3-way merge of conflict hunks."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/merge_words.py"
_spec = importlib.util.spec_from_file_location("merge_words", _PATH)
assert _spec and _spec.loader
merge_words = importlib.util.module_from_spec(_spec)
sys.modules["merge_words"] = merge_words
_spec.loader.exec_module(merge_words)

_ENV = {
    **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}
LONG = (
    "The engine fills at the next bar, charges taker fees of 4 bps, "
    "applies funding every 8 hours and liquidates at 90 percent margin."
)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check, env=_ENV
    )


def _conflicted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ours: str, theirs: str) -> Path:
    """main edits the SPEC.md paragraph one way, branch `side` another; merge conflicts."""
    for k, v in _ENV.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    spec = repo / "SPEC.md"
    spec.write_text(f"# Spec\n\n{LONG}\n\nend\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "checkout", "-qb", "side")
    spec.write_text(f"# Spec\n\n{theirs}\n\nend\n")
    _git(repo, "commit", "-qam", "side")
    _git(repo, "checkout", "-q", "main")
    spec.write_text(f"# Spec\n\n{ours}\n\nend\n")
    _git(repo, "commit", "-qam", "main")
    assert _git(repo, "merge", "side", check=False).returncode != 0
    assert "<<<<<<<" in spec.read_text()
    return spec


def test_two_edits_in_different_places_of_one_long_line_both_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _conflicted(
        tmp_path,
        monkeypatch,
        LONG.replace("4 bps", "5 bps"),
        LONG.replace("90 percent", "80 percent"),
    )
    assert merge_words.main([str(spec)]) == 0
    assert spec.read_text() == (
        "# Spec\n\n" + LONG.replace("4 bps", "5 bps").replace("90 percent", "80 percent")
        + "\n\nend\n"
    )


def test_overlapping_edits_exit_1_and_leave_the_file_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _conflicted(
        tmp_path,
        monkeypatch,
        LONG.replace("4 bps", "5 bps"),
        LONG.replace("4 bps", "6 bps"),
    )
    before = spec.read_text()
    assert merge_words.main([str(spec)]) == 1
    assert spec.read_text() == before
    assert "overlapping edits" in capsys.readouterr().err


def test_a_file_without_conflicts_is_a_noop(tmp_path: Path) -> None:
    f = tmp_path / "plain.md"
    f.write_text("nothing to merge\n")
    assert merge_words.main([str(f)]) == 0
    assert f.read_text() == "nothing to merge\n"


def test_explicit_base_ref_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _conflicted(
        tmp_path,
        monkeypatch,
        LONG.replace("4 bps", "5 bps"),
        LONG.replace("90 percent", "80 percent"),
    )
    base = _git(spec.parent, "merge-base", "HEAD", "MERGE_HEAD").stdout.strip()
    assert merge_words.main([str(spec), "--base", base]) == 0
    assert "<<<<<<<" not in spec.read_text() and "5 bps" in spec.read_text()


def test_merge_hunk_touching_edits_conflict_and_identical_edits_merge() -> None:
    base = "a b c"
    with pytest.raises(merge_words.MergeError):  # replace "c" vs append at its end: touching
        merge_words.merge_hunk("a b X", base, "a b c!")
    assert merge_words.merge_hunk("x b c", base, "a b c d") == "x b c d"  # apart: both
    assert merge_words.merge_hunk("x b c", base, "x b y") == "x b y"  # same edit once
