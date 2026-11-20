"""Tests for scripts/agents/handoff_lint.py — the handoff verdict parser.

handoff_lint.main() is driven in-process (file path and `-` stdin) and the asserted contract
is exactly its printed OK/MISSING/BAD lines and exit code.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.unit.conftest import BoardServer

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "handoff_lint.py"
_spec = importlib.util.spec_from_file_location("handoff_lint", _SCRIPT)
assert _spec is not None and _spec.loader is not None
handoff_lint = importlib.util.module_from_spec(_spec)
sys.modules["handoff_lint"] = handoff_lint
_spec.loader.exec_module(handoff_lint)


def _run(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    text: str,
    *argv: str,
    stdin: bool = False,
) -> tuple[int, str]:
    if stdin:
        sys.stdin = io.StringIO(text)
        target = "-"
    else:
        target = str(tmp_path / "handoff.md")
        Path(target).write_text(text)
    code = handoff_lint.main([*argv, target])
    captured = capsys.readouterr()
    sys.stdin = sys.__stdin__
    return code, captured.out


_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=_GIT_ENV
    ).stdout.strip()


def _commit(repo: Path, content: str, message: str = "feat: x") -> str:
    (repo / "a.py").write_text(content)
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _gate_log(repo: Path, sha: str, verdict: str = "GATE PASS", tree: str | None = None) -> Path:
    """A LOG as swarm_check writes it: transcript, then `HEAD <sha> TREE <tree>`, then the verdict."""
    log = repo / "data" / "swarm" / "_gate-logs" / f"wt-{sha[:12]}-20261029T000000Z.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    tree = tree or _git(repo, "rev-parse", f"{sha}^{{tree}}")
    log.write_text(f"$ ruff check\nAll checks passed!\n[exit 0]\nHEAD {sha} TREE {tree}\n{verdict}\n")
    return log


def _maker(sha: str, log: Path | str) -> str:
    return f"""\
## Handoff
status: done
task: make the thing
artifacts:
  - commit {sha[:8]} feat: x
evidence:
  - `pytest -q` -> 3 passed
  - `swarm-check.sh wt --scope x` -> `GATE PASS` (LOG {log})
improve: none
"""


_REPO = Path()
_SHA = ""
_MAKER_OK = ""


@pytest.fixture(scope="module", autouse=True)
def _gated_repo(tmp_path_factory: pytest.TempPathFactory) -> None:
    """One real repo + PASS LOG: the lint resolves the cited commit's tree with git."""
    global _REPO, _SHA, _MAKER_OK
    _REPO = tmp_path_factory.mktemp("gated")
    _git(_REPO, "init", "-q")
    _SHA = _commit(_REPO, "x = 1\n")
    _MAKER_OK = _maker(_SHA, _gate_log(_REPO, _SHA))


def test_valid_maker(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, out = _run(capsys, tmp_path, _MAKER_OK)
    assert code == 0
    assert out.strip() == "OK"


def test_maker_without_gate_pass_or_log_fails(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    text = _MAKER_OK.replace("GATE PASS", "see the run below").replace("_gate-logs/", "logs/")
    code, out = _run(capsys, tmp_path, text)
    assert code == 1
    assert "MISSING gate: no 'GATE PASS' line" in out
    assert "MISSING gate: no _gate-logs/ LOG path" in out


def _lint(text: str, root: Path | None = None) -> list[str]:
    problems: list[str] = handoff_lint.check(text, None, root=root)
    return problems


def test_a_failed_gate_log_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    sha = _commit(repo, "y = 1\n")
    log = _gate_log(repo, sha, verdict="GATE FAIL (1 failed)")
    assert _lint(_maker(sha, log)) == ["BAD gate: no cited LOG ends with 'GATE PASS'"]


def test_the_passing_tree_must_be_a_cited_commits_tree(tmp_path: Path) -> None:
    """A PASS LOG of an earlier commit does not cover a later one; a message-only amend does."""
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    gated = _commit(repo, "y = 1\n")
    log = _gate_log(repo, gated)
    later = _commit(repo, "y = 2\n")
    assert _lint(_maker(later, log)) == [handoff_lint.NO_GATED_COMMIT]
    _git(repo, "reset", "-q", "--hard", gated)
    _git(repo, "commit", "-q", "--amend", "-m", "feat: reworded")
    reworded = _git(repo, "rev-parse", "HEAD")
    assert reworded != gated
    assert _lint(_maker(reworded, log)) == []


def test_a_gate_over_uncommitted_edits_proves_no_commit(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    sha = _commit(repo, "y = 1\n")
    assert _lint(_maker(sha, _gate_log(repo, sha, tree="dirty"))) == [handoff_lint.NO_GATED_COMMIT]


# ------------------------------------------------------------------ absorbed rulings (job bus)


_RULING_ID = "18e49c1b6b3b81a0-4370"


def _absorbed_maker(sha: str, log: Path, ruling: str = _RULING_ID) -> str:
    return f"""\
## Handoff
status: done
task: make the thing
artifacts:
  - commit {sha[:8]} feat: x
evidence:
  - `swarm-check.sh wt --scope x` -> GATE FAIL (1 failed): stale-reads only (LOG {log})
  - absorbed: {ruling}
improve: none
"""


def _bus(repo: Path, *messages: dict[str, str]) -> None:
    bus = repo / "data" / "swarm" / "job-x" / "bus" / "messages.jsonl"
    bus.parent.mkdir(parents=True, exist_ok=True)
    bus.write_text("".join(json.dumps(m) + "\n" for m in messages))


def _failed_repo(tmp_path: Path) -> tuple[Path, str, Path]:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    sha = _commit(repo, "y = 1\n")
    return repo, sha, _gate_log(repo, sha, verdict="GATE FAIL (1 failed)")


def test_an_absorbed_ruling_must_exist_and_come_from_the_chief(tmp_path: Path) -> None:
    repo, sha, log = _failed_repo(tmp_path)
    assert _lint(_absorbed_maker(sha, log), root=repo) == [
        f"BAD gate: absorbed ruling {_RULING_ID} is not on any job bus"
    ]
    _bus(repo, {"id": _RULING_ID, "from": "cc-x-1", "body": "absorbed please"})
    not_chief = f"BAD gate: bus message {_RULING_ID} is not a chief ruling saying 'absorbed'"
    assert _lint(_absorbed_maker(sha, log), root=repo) == [not_chief]
    _bus(repo, {"id": _RULING_ID, "from": "chief", "body": "fix the stale read and re-run"})
    assert _lint(_absorbed_maker(sha, log), root=repo) == [not_chief]


def test_an_absorbed_fail_must_still_be_on_a_cited_commits_tree(tmp_path: Path) -> None:
    repo, sha, log = _failed_repo(tmp_path)
    _bus(repo, {"id": _RULING_ID, "from": "chief", "body": "absorbed"})
    later = _commit(repo, "y = 2\n")
    assert _lint(_absorbed_maker(later, log), root=repo) == [handoff_lint.NO_GATED_COMMIT]


# ------------------------------------------------------------------ absorbed rulings on the board store


@pytest.fixture
def board_ns(board_server: BoardServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """SWARM_STORE=board on the fixture board, a fresh channel namespace (returned)."""
    for key in ("BOARD_URL", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, board_server.env[key])
    monkeypatch.setenv("SWARM_STORE", "board")
    ns = "h" + uuid.uuid4().hex[:12]
    monkeypatch.setenv("SWARM_BOARD_NS", ns)
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path / "global"))
    return ns


def _board_ruling(board_server: BoardServer, ns: str, text: str, *labels: str, profile: str = "chief") -> str:
    args = [arg for label in labels for arg in ("--label", label)]
    proc = board_server.vb("post", f"{ns}/job/20261120-x", "-", *args, "--json", profile=profile, stdin=text)
    assert proc.returncode == 0, proc.stderr
    return str(json.loads(proc.stdout)["message"]["id"])


def test_a_board_message_by_a_worker_is_not_a_chief_ruling(
    board_server: BoardServer, board_ns: str, tmp_path: Path
) -> None:
    repo, sha, log = _failed_repo(tmp_path)
    board_id = _board_ruling(board_server, board_ns, "absorbed please", profile="cc-x-1")
    assert _lint(_absorbed_maker(sha, log, ruling=board_id), root=repo) == [
        f"BAD gate: bus message {board_id} is not a chief ruling saying 'absorbed'"
    ]
