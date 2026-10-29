"""Tests for scripts/agents/handoff_lint.py — the handoff verdict parser.

handoff_lint.main() is driven in-process (file path and `-` stdin) and the asserted contract
is exactly its printed OK/MISSING/BAD lines and exit code.
"""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_valid_maker_from_stdin(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, out = _run(capsys, tmp_path, _MAKER_OK, stdin=True)
    assert code == 0
    assert out.strip() == "OK"


def test_blocked_status_needs_no_evidence(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text = "## Handoff\nstatus: blocked\nevidence:\nimprove: none\n"
    code, out = _run(capsys, tmp_path, text)
    assert code == 0
    assert out.strip() == "OK"


def test_missing_improve(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text = "## Handoff\nstatus: done\nevidence:\n  - `x` -> y\n"
    code, out = _run(capsys, tmp_path, text)
    assert code == 1
    assert "MISSING improve:" in out


def test_missing_evidence(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text = "## Handoff\nstatus: done\nevidence:\nimprove: none\n"
    code, out = _run(capsys, tmp_path, text)
    assert code == 1
    assert "MISSING evidence:" in out


def test_bad_status_and_bad_verdict(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text = "## Handoff\nstatus: Done\nrole: reviewer\nevidence:\n  - x\nimprove: none\nverdict: MAYBE\nfix-route: claude\n"
    code, out = _run(capsys, tmp_path, text)
    assert code == 1
    assert "BAD status:" in out
    assert "BAD verdict:" in out
    assert "BAD fix-route:" in out
    assert "MISSING" not in out


def test_fix_route_just_above_the_block(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text = (
        "fix-route: deepseek\n"
        "## Handoff\n"
        "status: done\n"
        "role: reviewer\n"
        "evidence:\n"
        "  - `cmd` -> out\n"
        "improve: none\n"
        "verdict: SHOULD-FIX\n"
    )
    code, out = _run(capsys, tmp_path, text)
    assert code == 0
    assert out.strip() == "OK"


def test_reviewer_role_detected_without_flag(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text = "## Handoff\nstatus: done\nrole: reviewer\nevidence:\n  - x\nimprove: none\n"
    code, out = _run(capsys, tmp_path, text)
    assert code == 1
    assert "MISSING verdict:" in out
    assert "MISSING fix-route:" in out


@pytest.mark.parametrize("role", ["risk-officer", "overfit-auditor"])
def test_checker_role_is_detected_without_flag(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, role: str
) -> None:
    text = f"## Handoff\nstatus: done\nrole: {role}\nevidence:\n  - x\nimprove: none\n"
    code, out = _run(capsys, tmp_path, text)
    assert code == 1
    assert "MISSING verdict:" in out
    assert "MISSING fix-route:" in out
    assert "MISSING gate:" not in out


_REAL_CHECKER_ROLES = (
    "risk-midnight-1",  # plain risk-<slug>: no checker marker in the name
    "risk-officer (risk-paper2-1)",
    "risk-closeall-1 (risk-officer, code-change review)",
)


@pytest.mark.parametrize("role", _REAL_CHECKER_ROLES)
def test_real_checker_role_forms_need_a_verdict(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, role: str
) -> None:
    """The role strings copied from data/swarm/20261027-taker/handoffs/risk-*.md."""
    text = f"## Handoff\nstatus: done\nrole: {role}\nevidence:\n  - x\nimprove: none\n"
    code, out = _run(capsys, tmp_path, text)
    assert code == 1
    assert "MISSING verdict:" in out
    assert "MISSING fix-route:" in out
    assert "MISSING gate:" not in out


@pytest.mark.parametrize(
    "verdict", ["PASS", "SHOULD-FIX", "BLOCK", "FAIL", "VETO", "REJECT", "INSUFFICIENT"]
)
def test_checker_verdict_vocabulary_is_accepted(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, verdict: str
) -> None:
    """--role reviewer isolates the vocabulary: risk-officer.md uses PASS/FAIL/BLOCK, diff VETO; overfit-auditor.md PASS/REJECT/INSUFFICIENT."""
    text = (
        f"fix-route: none\n## Handoff\nstatus: done\nrole: risk-recalert-1\n"
        f"verdict: {verdict}\nevidence:\n  - read it\nimprove: none\n"
    )
    code, out = _run(capsys, tmp_path, text, "--role", "reviewer")
    assert (code, out.strip()) == (0, "OK")


def test_a_qualified_verdict_is_not_one_token(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """risk-midnight-1 and risk-paper2-1 wrote `PASS (money safety)`: a BAD verdict, with a hint."""
    text = (
        "fix-route: none\n## Handoff\nstatus: done\nrole: risk-midnight-1\n"
        "verdict: PASS (money safety)\nevidence:\n  - read it\nimprove: none\n"
    )
    code, out = _run(capsys, tmp_path, text)
    assert code == 1
    assert "BAD verdict: 'PASS (money safety)'" in out
    assert "one token; put qualifiers in claims" in out


@pytest.mark.parametrize("role", ["risk-officer"])
def test_checker_role_block_with_verdict_needs_no_gate_line(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, role: str
) -> None:
    text = (
        f"fix-route: none\n## Handoff\nstatus: done\nrole: {role}\nverdict: PASS\n"
        "evidence:\n  - read it\nimprove: none\n"
    )
    code, out = _run(capsys, tmp_path, text)
    assert (code, out.strip()) == (0, "OK")


def test_maker_role_still_needs_the_gate_line(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    text = "## Handoff\nstatus: done\nrole: maker\nevidence:\n  - x\nimprove: none\n"
    code, out = _run(capsys, tmp_path, text)
    assert code == 1
    assert "MISSING gate: no 'GATE PASS' line" in out
    assert "MISSING verdict:" not in out


def test_last_handoff_block_wins(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text = "## Handoff\nstatus: done\nevidence:\nimprove:\n\n" + _MAKER_OK
    code, out = _run(capsys, tmp_path, text)
    assert code == 0
    assert out.strip() == "OK"


def test_no_handoff_block(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, out = _run(capsys, tmp_path, "just prose\n")
    assert code == 1
    assert "MISSING handoff:" in out


def test_maker_without_gate_pass_or_log_fails(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    text = _MAKER_OK.replace("GATE PASS", "see the run below").replace("_gate-logs/", "logs/")
    code, out = _run(capsys, tmp_path, text)
    assert code == 1
    assert "MISSING gate: no 'GATE PASS' line" in out
    assert "MISSING gate: no _gate-logs/ LOG path" in out


def test_reviewer_needs_no_gate_line(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text = "fix-route: none\n## Handoff\nstatus: done\nverdict: PASS\nevidence:\n  - read it\nimprove: none\n"
    code, out = _run(capsys, tmp_path, text, "--role", "reviewer")
    assert (code, out.strip()) == (0, "OK")


@pytest.mark.parametrize(
    ("summary", "ok"),
    [
        ("MUTANTS compared=3 lost=0 errors=0", True),
        ("MUTANTS compared=3 lost=1 errors=0", False),
        ("MUTANTS compared=3 lost=0 errors=1", False),
        ("MUTANTS compared=0 lost=0 errors=0", False),
        ("MUTANTS killed=3 survived=0 errors=0", False),  # a single-copy run proves no base kill
    ],
)
def test_mutants_flag_needs_a_clean_mutant_summary(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, summary: str, ok: bool
) -> None:
    """With --mutants the maker's done block must carry a clean `--compare-rev` summary line;
    without the flag the same block needs none."""
    handoff = tmp_path / "cc-x-1.md"
    handoff.write_text(_MAKER_OK.replace("improve: none", f"  - {summary}\nimprove: none"))
    assert handoff_lint.main([str(handoff)]) == 0
    code = handoff_lint.main(["--mutants", str(handoff)])
    out = capsys.readouterr().out
    assert code == (0 if ok else 1), out
    assert ("MISSING mutants" in out) is not ok


def test_a_brief_naming_compare_rev_no_longer_demands_a_mutant_summary(tmp_path: Path) -> None:
    """The brief-text trigger fired on any brief that merely NAMED --compare-rev (a caps brief
    citing it as a fact made its maker run a 4th mutant past its own cap); the gate's parity
    check in --full is the proof that a test-deleting round lost no kill."""
    job = tmp_path / "job"
    (job / "handoffs").mkdir(parents=True)
    (job / "prompts").mkdir()
    handoff = job / "handoffs" / "cc-x-1.md"
    handoff.write_text(_MAKER_OK)
    (job / "prompts" / "cc-x-1.md").write_text("8. `mutate.py --compare-rev 3aee9a89 --spec m.json -- pytest`\n")
    assert handoff_lint.main([str(handoff)]) == 0


def test_mutants_flag_requires_the_summary_from_stdin(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, out = _run(capsys, tmp_path, _MAKER_OK, "--mutants", stdin=True)
    assert code == 1
    assert "MISSING mutants: no 'MUTANTS compared=N lost=0 errors=0' line" in out


def _lint(text: str, root: Path | None = None) -> list[str]:
    problems: list[str] = handoff_lint.check(text, None, root=root)
    return problems


def test_a_cited_gate_log_must_exist() -> None:
    missing = _REPO / "data" / "swarm" / "_gate-logs" / "wt-000000000000-x.log"
    assert _lint(_maker(_SHA, missing)) == [f"BAD gate: cited LOG {missing} does not exist"]


def test_a_log_without_the_verdict_trailer_is_refused(tmp_path: Path) -> None:
    """LOGs written before the gate recorded HEAD/TREE + verdict prove nothing about this commit."""
    old = tmp_path / "_gate-logs" / "wt-old.log"
    old.parent.mkdir()
    old.write_text("$ ruff check\nAll checks passed!\n[exit 0]\n")
    (problem,) = _lint(_maker(_SHA, old))
    assert problem.startswith(f"BAD gate: LOG {old} does not end with 'HEAD <sha> TREE <tree>'")


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


def test_a_relative_log_path_resolves_against_the_main_checkout() -> None:
    log = _gate_log(_REPO, _SHA)
    relative = log.relative_to(_REPO)
    assert _lint(_maker(_SHA, relative), root=_REPO) == []
    assert _lint(_maker(_SHA, relative), root=_REPO / "elsewhere") == [
        f"BAD gate: cited LOG {relative} does not exist"
    ]


def test_a_gate_logs_mention_without_a_log_file_is_refused() -> None:
    """`_gate-logs/` alone (a dir, a name without .log) cites nothing the lint can open."""
    for cited in ("data/swarm/_gate-logs/vq-x-6395257e59ee-20261028T153131Z", "data/swarm/_gate-logs/"):
        assert _lint(_maker(_SHA, cited), root=_REPO) == [
            "BAD gate: '_gate-logs/' appears but no '_gate-logs/<name>.log' path is cited"
        ]
