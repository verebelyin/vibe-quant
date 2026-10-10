"""scripts/agents/worktree.sh — node_modules sharing vs --private-node-modules (vibe-quant-kzbc6).

Driven through bash in a throwaway tmp git repo so the real checkout is never touched
(worktree.sh creates <parent-of-repo>/vq-<slug>)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "worktree.sh"


def _clean_env() -> dict[str, str]:
    """Env without git's repo-pinning vars: when pytest runs inside a git hook or a worktree
    command these would redirect the tmp-repo git calls onto the real checkout."""
    drop = {"GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"}
    return {k: v for k, v in os.environ.items() if k not in drop}


def _git(cwd: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=_clean_env(),
    )
    return res.stdout.strip()


def _make_repo(tmp_path: Path) -> Path:
    main = tmp_path / "main"
    (main / "scripts" / "agents").mkdir(parents=True)
    (main / "frontend" / "node_modules").mkdir(parents=True)
    (main / "frontend" / "package.json").write_text("{}\n", encoding="utf-8")
    (main / "scripts" / "agents" / "worktree.sh").write_text(
        SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _git(main, "init", "-q")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "init")
    return main


def _run(main: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(main / "scripts" / "agents" / "worktree.sh"), *args],
        cwd=main,
        capture_output=True,
        text=True,
        env=_clean_env(),
    )


def test_default_symlinks_node_modules_and_warns(tmp_path: Path) -> None:
    main = _make_repo(tmp_path)
    res = _run(main, "s1")
    assert res.returncode == 0, res.stderr
    link = tmp_path / "vq-s1" / "frontend" / "node_modules"
    assert link.is_symlink()
    assert link.resolve() == (main / "frontend" / "node_modules").resolve()
    out = res.stdout + res.stderr
    assert "WARNING" in out
    assert "--private-node-modules" in out


def test_private_node_modules_skips_symlink_and_hints(tmp_path: Path) -> None:
    main = _make_repo(tmp_path)
    res = _run(main, "--private-node-modules", "s2")  # flag before the slug
    assert res.returncode == 0, res.stderr
    assert not (tmp_path / "vq-s2" / "frontend" / "node_modules").exists()
    out = res.stdout + res.stderr
    assert "pnpm install --frozen-lockfile" in out
    assert "vq-s2/frontend" in out


def test_private_node_modules_flag_after_positionals(tmp_path: Path) -> None:
    main = _make_repo(tmp_path)
    res = _run(main, "s3", "HEAD", "--private-node-modules")
    assert res.returncode == 0, res.stderr
    assert not (tmp_path / "vq-s3" / "frontend" / "node_modules").exists()


def test_positional_base_ref_is_the_worktree_start_point(tmp_path: Path) -> None:
    """`worktree.sh s4 feat` must branch from feat, not from HEAD (the default)."""
    main = _make_repo(tmp_path)
    _git(main, "checkout", "-q", "-b", "feat")
    (main / "feat.txt").write_text("x\n", encoding="utf-8")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "feat work")
    feat_sha = _git(main, "rev-parse", "feat")
    _git(main, "checkout", "-q", "-")  # back to the default branch: HEAD != feat
    assert _git(main, "rev-parse", "HEAD") != feat_sha
    res = _run(main, "s4", "feat")
    assert res.returncode == 0, res.stderr
    wt = tmp_path / "vq-s4"
    assert _git(wt, "rev-parse", "HEAD") == feat_sha
    assert (wt / "feat.txt").exists()
    assert "(from feat)" in res.stdout


def test_unknown_option_rejected_with_rc_2(tmp_path: Path) -> None:
    main = _make_repo(tmp_path)
    res = _run(main, "--foo", "s5")
    assert res.returncode == 2
    assert "unknown option '--foo'" in res.stderr
    assert not (tmp_path / "vq-s5").exists()  # nothing created before the arg check
