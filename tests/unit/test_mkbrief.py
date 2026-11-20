"""Tests for scripts/agents/mkbrief.py — the brief generator.

The CLI is driven end-to-end against throwaway git repos in tmp_path (glob
validation walks the --worktree filesystem; review mode runs real `git
merge-base`/`git rev-parse`), so no real project checkout is ever touched.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.unit.conftest import BoardServer

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "mkbrief.py"
_spec = importlib.util.spec_from_file_location("mkbrief", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mkbrief = importlib.util.module_from_spec(_spec)
sys.modules["mkbrief"] = mkbrief
_spec.loader.exec_module(mkbrief)

# Hermetic, non-interactive git so a throwaway repo needs no config and no hooks.
_GIT_ENV = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "mkbrief",
    "GIT_AUTHOR_EMAIL": "mkbrief@test",
    "GIT_COMMITTER_NAME": "mkbrief",
    "GIT_COMMITTER_EMAIL": "mkbrief@test",
}

_TEMPLATE = """# Task: {TASK}
{TELEMETRY}

ID: {ID}
JOB: {JOB}
BUS: {BUS}
WT: {WT}
RUN_AS: {RUN_AS}
SCOPE: {SCOPE}
GATE: {GATE}
"""

_REVIEW_TEMPLATE = """# Review: {TASK}
{TELEMETRY}
BASE_SHA: {BASE_SHA}
HEAD_SHA: {HEAD_SHA}
RANGE: {RANGE}
REVIEW_PACK: {REVIEW_PACK}
"""


def _git(repo: Path, *args: str) -> str:
    drop = {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(_GIT_ENV)
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env
    )
    return out.stdout.strip()


def _build_repo(repo: Path) -> Path:
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("A = 1\n")
    (repo / "tests" / "unit" / "sub").mkdir(parents=True)
    (repo / "tests" / "unit" / "test_x.py").write_text("def test_x() -> None:\n    assert True\n")
    (repo / "tests" / "unit" / "sub" / "test_y.py").write_text(
        "def test_y() -> None:\n    assert True\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


_REPO_TEMPLATE: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def _repo_template_fixture(tmp_path_factory: pytest.TempPathFactory) -> None:
    """One committed base repo; each test gets a private copy (8ms) instead of a fresh
    `git init`+`add`+`commit` (131ms). Tests then branch/commit on their own copy."""
    global _REPO_TEMPLATE
    _REPO_TEMPLATE = _build_repo(tmp_path_factory.mktemp("mkbrief-template") / "wt")


def _repo(tmp_path: Path, name: str = "wt") -> Path:
    assert _REPO_TEMPLATE is not None, "the module template fixture did not run"
    repo = tmp_path / name
    shutil.copytree(_REPO_TEMPLATE, repo)
    return repo


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "job"
    (job / "prompts").mkdir(parents=True)
    return job


def _template(tmp_path: Path, text: str = _TEMPLATE, name: str = "t.md") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _argv(
    job: Path,
    template: Path,
    *,
    request_id: str = "cc-rd-demo-1",
    scope: str = "a.py",
    worktree: Path | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    args = ["--job", str(job), "--id", request_id, "--template", str(template)]
    if worktree is not None:
        args += ["--worktree", str(worktree)]
    if scope is not None:
        args += ["--scope", scope]
    return args + (extra or [])


def test_dispatch_text_carries_telemetry_and_file_exists_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path)

    rc = mkbrief.main(_argv(job, template, request_id="cc-rd-demo-1", scope="a.py", worktree=repo))

    out_path = job / "prompts" / "cc-rd-demo-1.md"
    assert rc == 0
    assert out_path.exists()
    assert capsys.readouterr().out == (
        f"Read your brief at {out_path.resolve()} and follow it exactly.\n"
        "TELEMETRY: task=rd-demo round=1\n"
    )
    assert "TELEMETRY: task=rd-demo round=1" in out_path.read_text()


def test_review_mode_pins_merge_base_after_parent_moves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    fork = _git(repo, "rev-parse", "HEAD")
    _git(repo, "branch", "parent")

    _git(repo, "checkout", "-qb", "feat")
    (repo / "feat.py").write_text("FEAT = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "feat work")
    head = _git(repo, "rev-parse", "HEAD")

    _git(repo, "checkout", "-q", "parent")
    (repo / "parent.py").write_text("PARENT = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "parent moves on")
    parent_tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "feat")

    job = _job(tmp_path)
    template = _template(tmp_path, _REVIEW_TEMPLATE)
    rc = mkbrief.main(
        _argv(
            job,
            template,
            scope="a.py",
            worktree=repo,
            extra=["--review", "feat", "--parent", "parent"],
        )
    )

    assert rc == 0
    capsys.readouterr()
    text = (job / "prompts" / "cc-rd-demo-1.md").read_text()
    assert f"BASE_SHA: {fork}" in text
    assert f"HEAD_SHA: {head}" in text
    assert f"RANGE: {fork}..{head}" in text
    assert f"REVIEW_PACK: python3 scripts/agents/review_pack.py {repo} --range {fork}..{head}" in text
    assert parent_tip not in text


def test_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _repo(tmp_path)
    job = _job(tmp_path)
    template = _template(tmp_path)
    argv = _argv(job, template, worktree=repo)

    assert mkbrief.main(argv) == 0
    original = (job / "prompts" / "cc-rd-demo-1.md").read_text()
    capsys.readouterr()

    rc = mkbrief.main(argv)
    assert rc == 2
    assert "exists" in capsys.readouterr().err
    assert (job / "prompts" / "cc-rd-demo-1.md").read_text() == original

    assert mkbrief.main(argv + ["--force"]) == 0
    capsys.readouterr()


def _commit(repo: Path, fname: str) -> str:
    (repo / fname).write_text("X = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", fname)
    return _git(repo, "rev-parse", "HEAD")


def test_stacked_worktree_without_base_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    _git(repo, "checkout", "-qb", "swarm/a")
    a_tip = _commit(repo, "a1.py")
    _git(repo, "checkout", "-qb", "swarm/b")
    _commit(repo, "b1.py")
    rc = mkbrief.main(_argv(_job(tmp_path), _template(tmp_path), worktree=repo))
    err = capsys.readouterr().err
    assert rc == 2
    assert "stacked worktree without .vq-base: pass --base <fork point>" in err
    assert a_tip in err
    assert not (tmp_path / "job" / "prompts" / "cc-rd-demo-1.md").exists()


# --- the verifier and research brief templates render ------------------------------------------

_PROMPTS = Path(__file__).resolve().parents[2] / "docs" / "orchestration" / "prompts"
_PLACEHOLDER = re.compile(r"\{[A-Z][A-Z0-9_]*\}")


@pytest.mark.parametrize(
    ("name", "request_id", "sets"),
    [
        ("verify-task.md", "ver-demo-1", {"EXTRA_CHECKS": "none"}),
        (
            "research-eval.md",
            "cc-demo-eval-1",
            {"GOAL": "evaluate S1", "FACTS": "none", "STEPS": "1. run it", "DONE_WHEN": "one run"},
        ),
    ],
)
def test_the_verifier_and_research_templates_render_without_placeholders(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    request_id: str,
    sets: dict[str, str],
) -> None:
    job = _job(tmp_path)
    argv = ["--job", str(job), "--id", request_id, "--template", str(_PROMPTS / name)]
    for key, value in sets.items():
        argv += ["--set", f"{key}={value}"]

    assert mkbrief.main(argv) == 0, capsys.readouterr().err
    text = (job / "prompts" / f"{request_id}.md").read_text()
    assert _PLACEHOLDER.findall(text) == []
    assert "Render: mkbrief.py" not in text


# --- board threads: one thread per task in job/<job> (R18-R21 of job 20261120-board) ------------

_BOARD_PROTO = _PROMPTS / "board-protocol.md"
_GOAL_TEMPLATE = _TEMPLATE + "GOAL: {GOAL}\n"
_MAIN_BIN = "/Users/verebelyin/projects/vibe-quant/scripts/agents/bin"
_OWN_BIN = str(Path(__file__).resolve().parents[2] / "scripts" / "agents" / "bin")


@pytest.fixture
def board(board_server: BoardServer, monkeypatch: pytest.MonkeyPatch) -> BoardServer:
    """The fixture board for in-process renders, in a channel namespace of its own."""
    for key in ("BOARD_URL", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, board_server.env[key])
    monkeypatch.setenv("SWARM_BOARD_NS", "t" + uuid.uuid4().hex[:12])
    return board_server


def _board_render(
    tmp_path: Path, request_id: str, goal: str = "ship the board", extra: list[str] | None = None,
    template: str = _GOAL_TEMPLATE,
) -> int:
    argv = _argv(_job(tmp_path) if not (tmp_path / "job").exists() else tmp_path / "job",
                 _template(tmp_path, template), request_id=request_id)
    argv += ["--set", f"GOAL={goal}", "--append", str(_BOARD_PROTO), *(extra or [])]
    rc: int = mkbrief.main(argv)
    return rc


def _brief(tmp_path: Path, request_id: str) -> str:
    return (tmp_path / "job" / "prompts" / f"{request_id}.md").read_text()


def _board_block(text: str) -> str:
    return text[text.index("BOARD (") :]


def _block_commands(text: str) -> tuple[str, list[str]]:
    """(the export line, every other `vb`/`q=` command of the rendered BOARD block), runnable here."""
    spans = [s.replace(_MAIN_BIN, _OWN_BIN) for s in re.findall(r"`([^`]+)`", _board_block(text))]
    export = next(s for s in spans if s.startswith("export BOARD_URL="))
    return export, [s for s in spans if s.startswith(("vb ", "q=$("))]


def _ask_and_wait(commands: list[str]) -> tuple[str, str, str]:
    """(the whole one-call ask, its `q=... && echo` half, its `vb wait` half)."""
    whole = next(c for c in commands if c.startswith("q="))
    ask, wait = whole.rsplit(" && vb wait ", 1)
    return whole, ask, "vb wait " + wait


def _sh(shell: str, script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([shell, "-c", script], env=env, capture_output=True, text=True, timeout=120, check=False)


def _seed_kb(board: BoardServer) -> None:
    """`vb read kb/ ...` needs a kb channel: the real board has one from the import."""
    out = board.vb("post", "kb/gotchas", "RULE: seed", "--label", "state=live", "--idempotency-key", "seed-kb", profile="chief")
    assert out.returncode == 0, out.stderr


@pytest.mark.slow(reason="every block command as a real vb process (~0.2 s each) in a real bash/zsh")
@pytest.mark.parametrize("shell", ["/bin/bash", "/bin/zsh"])
def test_every_board_block_command_runs_unchanged(
    tmp_path: Path, board: BoardServer, shell: str
) -> None:
    """One shell run of every command, as written; the chief answers between the ask and the wait."""
    _seed_kb(board)
    assert _board_render(tmp_path, "cc-dash-1") == 0
    export, commands = _block_commands(_brief(tmp_path, "cc-dash-1"))
    whole, ask, wait = _ask_and_wait(commands)
    assert len(commands) == 6, commands
    lines = [export, *(f'{c} >/dev/null; echo "rc=$?"' for c in commands if c != whole)]
    lines += [ask, "BOARD_PROFILE=chief vb reply \"$q\" 'take a' >/dev/null", f'{wait}; echo "rc=$?"']

    out = _sh(shell, "\n".join(lines), board.env)

    assert re.findall(r"^rc=\d+$", out.stdout, re.M) == ["rc=0"] * 6, (out.stdout, out.stderr)
    assert "take a" in out.stdout
