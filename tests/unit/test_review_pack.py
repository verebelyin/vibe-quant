"""Tests for scripts/agents/review_pack.py — the one-command review surface.

A throwaway git repo in tmp_path is driven through review_pack.main() in-process, so the
asserted contract is exactly the CLI's stdout/stderr/exit code; the git subprocesses are real.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "review_pack.py"
_spec = importlib.util.spec_from_file_location("review_pack", _SCRIPT)
assert _spec is not None and _spec.loader is not None
review_pack = importlib.util.module_from_spec(_spec)
sys.modules["review_pack"] = review_pack
_spec.loader.exec_module(review_pack)

_GIT_ENV_OVERRIDES = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "review-pack",
    "GIT_AUTHOR_EMAIL": "review-pack@test",
    "GIT_COMMITTER_NAME": "review-pack",
    "GIT_COMMITTER_EMAIL": "review-pack@test",
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
    return out.stdout.strip()


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = review_pack.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "wt"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _write(root, "vibe_quant/mod.py", "A = 1\n")
    _write(root, "uv.lock", "LOCKBODY\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root


def test_merge_base_picks_fork_point_after_parent_moves(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fork = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "feature")
    _write(repo, "vibe_quant/mod.py", "A = 1\nFEATURE = True\n")
    _git(repo, "commit", "-qam", "feature work")
    feature_tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "vibe_quant/other.py", "PARENT = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "parent moves")

    code, out, _ = _run(capsys, str(repo), "--merge-base", "main", "--head", "feature")

    assert code == 0
    assert f"RANGE {fork}..{feature_tip}" in out
    assert "FEATURE = True" in out
    assert "PARENT = 1" not in out
