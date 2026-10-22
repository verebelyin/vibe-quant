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


def test_venv_is_linked_and_the_link_is_gitignored(tmp_path: Path) -> None:
    """Workers ran `.venv/bin/ruff ... | tail; echo EXIT=$?` in a worktree without .venv and
    read EXIT=0 for a check that never ran; the link must exist and not dirty the worktree."""
    main = _make_repo(tmp_path)
    (main / ".venv" / "bin").mkdir(parents=True)
    repo_gitignore = SCRIPT.parents[2] / ".gitignore"
    (main / ".gitignore").write_text(repo_gitignore.read_text(encoding="utf-8"), encoding="utf-8")
    _git(main, "add", ".gitignore")
    _git(main, "commit", "-qm", "ignore")
    res = _run(main, "s4")
    assert res.returncode == 0, res.stderr
    link = tmp_path / "vq-s4" / ".venv"
    assert link.is_symlink()
    assert link.resolve() == (main / ".venv").resolve()
    assert ".venv" not in _git(tmp_path / "vq-s4", "status", "--porcelain")


def test_records_start_sha_untracked_safe_and_review_range_equals_land_range(
    tmp_path: Path,
) -> None:
    """A task cut from a job branch, merged into it, with the job branch moved on: the review
    range and the land fork are still the task's own commits (merge-base would be empty)."""
    import importlib.util
    import sys

    def _load(name: str) -> object:
        path = SCRIPT.parent / f"{name}.py"
        sys.path.insert(0, str(SCRIPT.parent))
        spec = importlib.util.spec_from_file_location(f"r2_{name}", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    mkbrief, land = _load("mkbrief"), _load("land")
    main = _make_repo(tmp_path)
    _git(main, "checkout", "-qb", "job")
    (main / "earlier.txt").write_text("e\n", encoding="utf-8")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "earlier task")
    start = _git(main, "rev-parse", "job")

    res = _run(main, "t1", "job")
    assert res.returncode == 0, res.stderr
    wt = tmp_path / "vq-t1"
    assert (wt / ".vq-base").read_text(encoding="utf-8").strip() == start
    assert ".vq-base" not in _git(wt, "status", "--porcelain")  # never shows up as untracked
    (wt / "task.txt").write_text("t\n", encoding="utf-8")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-qm", "task work")
    own = _git(wt, "rev-parse", "HEAD")

    _git(main, "merge", "-q", "--no-edit", "swarm/t1")
    (main / "later.txt").write_text("l\n", encoding="utf-8")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "job moves on")

    job_dir = tmp_path / "jobdir"
    (job_dir / "prompts").mkdir(parents=True)
    tpl = tmp_path / "rev.md"
    tpl.write_text("BASE {BASE_SHA}\nHEAD {HEAD_SHA}\n", encoding="utf-8")
    rc = mkbrief.main(  # type: ignore[attr-defined]
        ["--job", str(job_dir), "--id", "rev-t1-1", "--template", str(tpl), "--worktree", str(wt),
         "--review", "swarm/t1", "--parent", "job"]
    )  # fmt: skip
    assert rc == 0
    text = (job_dir / "prompts" / "rev-t1-1.md").read_text(encoding="utf-8")
    assert f"BASE {start}" in text
    task = land.Task("swarm/t1", None, tmp_path / "m.txt")  # type: ignore[attr-defined]
    fork = land.task_fork(main, "job", task)  # type: ignore[attr-defined]
    assert fork == start
    assert _git(main, "rev-list", f"{fork}..swarm/t1").split() == [own]


def test_maker_gate_base_is_the_recorded_start_sha_when_the_job_branch_moves(tmp_path: Path) -> None:
    import importlib.util
    import sys

    path = SCRIPT.parent / "mkbrief.py"
    spec = importlib.util.spec_from_file_location("r2_mkbrief_gate", path)
    assert spec is not None and spec.loader is not None
    mkbrief = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mkbrief
    spec.loader.exec_module(mkbrief)

    main = _make_repo(tmp_path)
    _git(main, "checkout", "-qb", "job")
    start = _git(main, "rev-parse", "job")
    assert _run(main, "t2", "job").returncode == 0
    wt = tmp_path / "vq-t2"
    (main / "moved.txt").write_text("m\n", encoding="utf-8")
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "job moves")
    (wt / "task.txt").write_text("t\n", encoding="utf-8")
    _git(wt, "add", "task.txt")
    _git(wt, "commit", "-qm", "task")

    job_dir = tmp_path / "jobdir"
    (job_dir / "prompts").mkdir(parents=True)
    tpl = tmp_path / "mk.md"
    tpl.write_text("GATE: {GATE}\n", encoding="utf-8")
    argv = ["--job", str(job_dir), "--id", "cc-t2-1", "--template", str(tpl), "--worktree", str(wt),
            "--base", "job", "--scope", "+task.txt"]  # fmt: skip
    assert mkbrief.main(argv) == 0  # type: ignore[attr-defined]
    gate = (job_dir / "prompts" / "cc-t2-1.md").read_text(encoding="utf-8")
    assert f"--base {start}" in gate
    assert _git(main, "rev-parse", "job") not in gate
