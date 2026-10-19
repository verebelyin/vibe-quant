"""Tests for the git guardrails: scripts/git-hooks/* and scripts/agents/install-hooks.sh.

The commit-msg checker is imported through importlib (like test_swarm_check.py) so its
pass/fail cases run in-process; one subprocess test drives the real CLI anyway. The lint
script and the installer run against a throwaway git repo in tmp_path -- never the real
repo -- and the lint script gets the real ruff through its ``VQ_RUFF`` override. Git runs
hermetic (no user/system config, identity from the environment) and the ambient
``GIT_DIR``/``GIT_WORK_TREE``/``GIT_INDEX_FILE`` are stripped, so no test can touch the
checkout it runs from.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
_GIT_HOOKS_DIR = REPO_ROOT / "scripts" / "git-hooks"
CHECK_COMMIT_MSG = _GIT_HOOKS_DIR / "check_commit_msg.py"
PRE_COMMIT_LINT = _GIT_HOOKS_DIR / "pre-commit-lint.sh"
INSTALL_HOOKS = REPO_ROOT / "scripts" / "agents" / "install-hooks.sh"

_spec = importlib.util.spec_from_file_location("check_commit_msg", CHECK_COMMIT_MSG)
assert _spec is not None and _spec.loader is not None
check_commit_msg = importlib.util.module_from_spec(_spec)
sys.modules["check_commit_msg"] = check_commit_msg
_spec.loader.exec_module(check_commit_msg)

_GIT_ENV_OVERRIDES = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "hooks",
    "GIT_AUTHOR_EMAIL": "hooks@test",
    "GIT_COMMITTER_NAME": "hooks",
    "GIT_COMMITTER_EMAIL": "hooks@test",
}

_AMBIENT_GIT_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")

VERBOSE_MESSAGE = """fix(dsl): do a thing (vq-1)

Why:
the loader logged at INFO and the log was noise

What:
it logs at DEBUG now

Verification:
- `pytest -q` -> 1 passed
"""


@pytest.fixture(autouse=True)
def _strip_ambient_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``GIT_DIR`` exported by the caller would point every `git` call at the real repo."""
    for name in _AMBIENT_GIT_VARS:
        monkeypatch.delenv(name, raising=False)


def _git_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _AMBIENT_GIT_VARS}
    env.update(_GIT_ENV_OVERRIDES)
    if extra is not None:
        env.update(extra)
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


def _make_repo(tmp_path: Path, branch: str = "main") -> Path:
    """A throwaway repo with the three scripts copied in at their real relative paths."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "git-hooks").mkdir(parents=True)
    (repo / "scripts" / "agents").mkdir(parents=True)
    shutil.copy2(PRE_COMMIT_LINT, repo / "scripts" / "git-hooks" / "pre-commit-lint.sh")
    shutil.copy2(CHECK_COMMIT_MSG, repo / "scripts" / "git-hooks" / "check_commit_msg.py")
    shutil.copy2(INSTALL_HOOKS, repo / "scripts" / "agents" / "install-hooks.sh")
    _git(repo, "init", "-q", "-b", branch)
    return repo


def _message_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(text, encoding="utf-8")
    return path


def _main_ruff() -> Path:
    """The ruff next to the running interpreter: the main checkout's venv, also from a worktree
    or a non-git scratch copy (mutate.py, git archive), where `git rev-parse` fails."""
    return Path(sys.executable).parent / "ruff"


def _install(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo / "scripts" / "agents" / "install-hooks.sh")],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_git_env(),
    )


# --------------------------------------------------------------- commit-msg checker


def test_message_missing_headings_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(repo)
    message = _message_file(tmp_path, "fix(dsl): do a thing\n\nsome body text\n")

    assert check_commit_msg.main([str(message)]) == 1

    stderr = capsys.readouterr().err
    assert "Why" in stderr
    assert "What" in stderr
    assert "Verification" in stderr
    assert "docs/orchestration/prompts/commit-message.md" in stderr


def test_verbose_message_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(repo)
    message = _message_file(tmp_path, VERBOSE_MESSAGE)
    assert check_commit_msg.main([str(message)]) == 0


def test_markdown_headings_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(repo)
    message = _message_file(
        tmp_path, "fix(dsl): x\n\n# Why:\nbecause\n\n**What**:\nchanged\n\nVerification\n- ok\n"
    )
    assert check_commit_msg.main([str(message)]) == 0


@pytest.mark.parametrize(
    "subject",
    [
        "orchestration: board render + telemetry for job 1",
        "self-improve: cmd-task, verifier — job 1 Land",
        "wip",
        "chore(beads): sync",
        "bd: close vq-1",
        "Merge branch 'main'",
        "Revert \"fix(dsl): x\"",
        "journal: RESULT P-2 — 0 champions",
    ],
)
def test_exempt_subject_passes_on_main(
    subject: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(repo)
    message = _message_file(tmp_path, subject + "\n")
    assert check_commit_msg.main([str(message)]) == 0


def test_other_branches_always_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "feature/thing")
    monkeypatch.chdir(repo)
    message = _message_file(tmp_path, "whatever\n\nterse\n")
    assert check_commit_msg.main([str(message)]) == 0


def test_cli_exit_codes_are_enforced_by_the_script(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    message = _message_file(tmp_path, "fix(dsl): do a thing\n\nbody\n")

    on_main = subprocess.run(
        [sys.executable, str(CHECK_COMMIT_MSG), str(message)],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    assert on_main.returncode == 1
    assert "missing: Why, What, Verification" in on_main.stderr

    _git(repo, "checkout", "-q", "-b", "feature/thing")
    on_feature = subprocess.run(
        [sys.executable, str(CHECK_COMMIT_MSG), str(message)],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    assert on_feature.returncode == 0


# -------------------------------------------------------------------- installer


def test_install_hooks_is_idempotent_and_keeps_beads(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    hooks = repo / ".git" / "hooks"
    pre_commit = hooks / "pre-commit"
    beads = (
        "#!/usr/bin/env sh\n"
        "# --- BEGIN BEADS INTEGRATION v1.3.1 ---\n"
        "bd hooks run pre-commit\n"
        "# --- END BEADS INTEGRATION v1.3.1 ---\n"
    )
    pre_commit.write_text(beads, encoding="utf-8")

    first = _install(repo)
    assert first.returncode == 0, first.stderr
    after_first = pre_commit.read_text(encoding="utf-8")
    first_mtime = pre_commit.stat().st_mtime_ns
    commit_msg = hooks / "commit-msg"
    after_first_msg = commit_msg.read_text(encoding="utf-8")

    assert "bd hooks run pre-commit" in after_first
    assert "# --- BEGIN VIBE-QUANT LINT ---" in after_first
    assert "# --- END VIBE-QUANT LINT ---" in after_first
    assert "scripts/git-hooks/pre-commit-lint.sh" in after_first
    assert "scripts/git-hooks/check_commit_msg.py" in after_first_msg
    assert os.access(pre_commit, os.X_OK)
    assert os.access(commit_msg, os.X_OK)

    second = _install(repo)
    assert second.returncode == 0, second.stderr
    assert pre_commit.read_text(encoding="utf-8") == after_first
    assert pre_commit.stat().st_mtime_ns == first_mtime
    assert commit_msg.read_text(encoding="utf-8") == after_first_msg


def test_install_hooks_gives_a_bare_hook_a_shebang(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    pre_commit = repo / ".git" / "hooks" / "pre-commit"
    pre_commit.write_text("bd hooks run pre-commit\n", encoding="utf-8")

    assert _install(repo).returncode == 0

    text = pre_commit.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env sh\n")
    assert "bd hooks run pre-commit\n" in text


def test_installed_commit_msg_hook_rejects_on_main(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    assert _install(repo).returncode == 0
    hook = repo / ".git" / "hooks" / "commit-msg"
    message = _message_file(tmp_path, "fix(dsl): do a thing\n\nbody\n")

    rejected = subprocess.run([str(hook), str(message)], cwd=repo, capture_output=True, text=True)
    assert rejected.returncode == 1
    assert "missing: Why, What, Verification" in rejected.stderr

    _git(repo, "checkout", "-q", "-b", "feature/thing")
    accepted = subprocess.run([str(hook), str(message)], cwd=repo, capture_output=True, text=True)
    assert accepted.returncode == 0


# ------------------------------------------------------------------ pre-commit lint


def _lint_env(ruff: Path) -> dict[str, str]:
    return _git_env({"VQ_RUFF": str(ruff)})


def _run_lint(repo: Path, ruff: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo / "scripts" / "git-hooks" / "pre-commit-lint.sh")],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_lint_env(ruff),
    )


def test_pre_commit_lint_passes_with_nothing_staged(tmp_path: Path) -> None:
    ruff = _main_ruff()
    if not os.access(ruff, os.X_OK):
        pytest.skip(f"ruff not available at {ruff}")
    repo = _make_repo(tmp_path)
    result = _run_lint(repo, ruff)
    assert result.returncode == 0, result.stderr


def test_pre_commit_lint_fails_on_staged_ruff_error(tmp_path: Path) -> None:
    ruff = _main_ruff()
    if not os.access(ruff, os.X_OK):
        pytest.skip(f"ruff not available at {ruff}")
    repo = _make_repo(tmp_path)
    bad = repo / "bad.py"
    bad.write_text("import os\n\n\nprint('ok')\n", encoding="utf-8")
    _git(repo, "add", "bad.py")

    result = _run_lint(repo, ruff)

    assert result.returncode == 1
    assert "F401" in result.stdout + result.stderr


def test_pre_commit_lint_skip_lint_env(tmp_path: Path) -> None:
    ruff = _main_ruff()
    if not os.access(ruff, os.X_OK):
        pytest.skip(f"ruff not available at {ruff}")
    repo = _make_repo(tmp_path)
    (repo / "bad.py").write_text("import os\n\n\nprint('ok')\n", encoding="utf-8")
    _git(repo, "add", "bad.py")

    result = subprocess.run(
        [str(repo / "scripts" / "git-hooks" / "pre-commit-lint.sh")],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_git_env({"VQ_RUFF": str(ruff), "SKIP_LINT": "1"}),
    )
    assert result.returncode == 0


def _run_lint_budget(repo: Path, budget_cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo / "scripts" / "git-hooks" / "pre-commit-lint.sh")],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_git_env({"VQ_RUFF": "/nonexistent-ruff", "VQ_BUDGET_CMD": budget_cmd}),
    )


@pytest.mark.parametrize(
    "path", ["CLAUDE.md", ".claude/agents/reviewer.md", "docs/orchestration/README.md", "docs/claude/x.md"]
)
def test_pre_commit_runs_budget_check_for_staged_steering_file(tmp_path: Path, path: str) -> None:
    repo = _make_repo(tmp_path)
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("words\n", encoding="utf-8")
    _git(repo, "add", path)

    failing = _run_lint_budget(repo, "false")
    assert failing.returncode == 1
    assert "budget check failed" in failing.stderr
    assert _run_lint_budget(repo, "true").returncode == 0


def test_pre_commit_skips_budget_check_for_other_markdown(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "README.md").write_text("words\n", encoding="utf-8")
    _git(repo, "add", "README.md")

    assert _run_lint_budget(repo, "false").returncode == 0


@pytest.mark.parametrize("branch", ["main", "swarm/task"])
def test_non_claude_co_author_is_rejected_on_every_branch(
    branch: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _make_repo(tmp_path)
    if branch != "main":
        _git(repo, "checkout", "-q", "-b", branch)
    monkeypatch.chdir(repo)
    message = _message_file(
        tmp_path,
        "fix(dsl): x\n\nWhy:\na\n\nWhat:\nb\n\nVerification:\nc\n\n"
        "Co-Authored-By: DeepSeek V4 <noreply@deepseek.com>\n",
    )
    assert check_commit_msg.main([str(message)]) == 1
    assert "DeepSeek V4" in capsys.readouterr().err


def test_claude_co_author_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo(tmp_path)
    monkeypatch.chdir(repo)
    message = _message_file(
        tmp_path,
        "fix(dsl): x\n\nWhy:\na\n\nWhat:\nb\n\nVerification:\nc\n\n"
        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n",
    )
    assert check_commit_msg.main([str(message)]) == 0


def _run_lint_policy(repo: Path, policy_cmd: str) -> subprocess.CompletedProcess[str]:
    ruff = _main_ruff()
    if not os.access(ruff, os.X_OK):
        pytest.skip(f"ruff not available at {ruff}")
    (repo / "ok.py").write_text("X = 1\n", encoding="utf-8")
    _git(repo, "add", "ok.py")
    return subprocess.run(
        [str(repo / "scripts" / "git-hooks" / "pre-commit-lint.sh")],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_git_env({"VQ_RUFF": str(ruff), "VQ_POLICY_CMD": policy_cmd}),
    )


def test_pre_commit_runs_policy_tests_on_main_and_blocks_a_red_one(tmp_path: Path) -> None:
    """The chief commits on main directly; main went red when only swarm-check ran them."""
    repo = _make_repo(tmp_path)
    result = _run_lint_policy(repo, "false")
    assert result.returncode == 1
    assert "policy tests failed on main" in result.stderr
    assert _run_lint_policy(repo, "true").returncode == 0


def test_pre_commit_skips_policy_tests_off_main(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, branch="swarm/task")
    assert _run_lint_policy(repo, "false").returncode == 0
