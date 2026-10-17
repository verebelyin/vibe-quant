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


def test_lockfile_is_skipped_not_printed(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "vibe_quant/mod.py", "A = 1\nB = 2  # SOURCEBODY\n")
    _write(repo, "uv.lock", "LOCKBODY\nLOCKEXTRA\n")
    _git(repo, "commit", "-qam", "change")

    code, out, _ = _run(capsys, str(repo), "--range", f"{base}..HEAD")

    assert code == 0
    assert "SKIPPED uv.lock (+1 -0) generated/lockfile" in out
    assert "SOURCEBODY" in out
    assert "LOCKEXTRA" not in out


def test_nested_lockfiles_and_golden_are_skipped(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "frontend/pnpm-lock.yaml", "NESTEDPNPM\n")
    _write(repo, "sub/x.lock", "SUBX\n")
    _write(repo, "tests/fixtures/a.golden", "GOLD\n")
    _write(repo, "uv.lock", "LOCKBODY\nTOPLOCK\n")
    _write(repo, "src/a.py", "SRCA = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "nested")

    code, out, _ = _run(capsys, str(repo), "--range", f"{base}..HEAD")

    assert code == 0
    assert "SKIPPED frontend/pnpm-lock.yaml (+1 -0) generated/lockfile" in out
    assert "SKIPPED sub/x.lock (+1 -0) generated/lockfile" in out
    assert "SKIPPED tests/fixtures/a.golden (+1 -0) generated/lockfile" in out
    assert "SKIPPED uv.lock (+1 -0) generated/lockfile" in out
    assert out.count("diff --git ") == 1
    assert "SRCA = 1" in out
    for body in ("NESTEDPNPM", "SUBX", "GOLD", "TOPLOCK"):
        assert body not in out


def test_truncation_message_and_full_dump(
    repo: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "vibe_quant/big.py", "\n".join(f"L{i} = {i}" for i in range(60)) + "\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "big")
    head = _git(repo, "rev-parse", "HEAD")
    full = tmp_path / "full.diff"

    code, out, _ = _run(
        capsys, str(repo), "--range", f"{base}..{head}", "--max-lines", "10", "--full", str(full)
    )

    assert code == 0
    assert out.count("diff --git ") == 1
    assert "TRUNCATED:" in out
    assert f"more diff lines in vibe_quant/big.py; read one file with: git -C {repo} diff {base}..{head} -- vibe_quant/big.py" in out
    assert "L59 = 59" not in out

    full_text = full.read_text()
    assert "L59 = 59" in full_text
    assert "TRUNCATED" not in full_text


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


def test_range_warns_and_pins_refs(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _git(repo, "rev-parse", "main")
    _git(repo, "checkout", "-qb", "work")
    _write(repo, "vibe_quant/mod.py", "A = 1\nCHANGED\n")
    _git(repo, "commit", "-qam", "change")
    head = _git(repo, "rev-parse", "HEAD")

    code, out, _ = _run(capsys, str(repo), "--range", "main..work")

    assert code == 0
    assert f"WARN main is a ref; pinned to {base}" in out
    assert f"WARN work is a ref; pinned to {head}" in out
    assert f"RANGE {base}..{head}" in out


def test_empty_range_exits_2(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sha = _git(repo, "rev-parse", "HEAD")

    code, _, err = _run(capsys, str(repo), "--range", f"{sha}..{sha}")

    assert code == 2
    assert "empty range" in err.lower()


def test_unresolvable_ref_exits_2(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = _run(capsys, str(repo), "--range", "nope..HEAD")

    assert code == 2
    assert "nope" in err


def test_diff_sections_ordered_source_then_tests_then_docs(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _write(repo, "docs/guide.md", "DOC\n")
    _write(repo, "tests/unit/test_x.py", "def test_x() -> None:\n    assert True\n")
    _write(repo, "vibe_quant/mod.py", "A = 1\nSRC\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "three")

    code, out, _ = _run(capsys, str(repo), "--range", f"{base}..HEAD")

    assert code == 0
    order = [line.split(" b/")[-1] for line in out.splitlines() if line.startswith("diff --git ")]
    assert order == ["vibe_quant/mod.py", "tests/unit/test_x.py", "docs/guide.md"]
