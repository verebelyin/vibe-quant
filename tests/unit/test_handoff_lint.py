"""Tests for scripts/agents/handoff_lint.py — the handoff verdict parser.

handoff_lint.main() is driven in-process (file path and `-` stdin) and the asserted contract
is exactly its printed OK/MISSING/BAD lines and exit code.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest  # noqa: TC002  # annotations only; no runtime pytest API in this file

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


_MAKER_OK = """\
## Handoff
status: done
task: make the thing
evidence:
  - `pytest -q` -> 3 passed
  - `swarm-check.sh wt --scope x` -> `GATE PASS` (LOG data/swarm/_gate-logs/vq-x-1.log)
improve: none
"""


def test_valid_maker(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, out = _run(capsys, tmp_path, _MAKER_OK)
    assert code == 0
    assert out.strip() == "OK"


def test_valid_maker_from_stdin(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, out = _run(capsys, tmp_path, _MAKER_OK, stdin=True)
    assert code == 0
    assert out.strip() == "OK"


def test_valid_reviewer(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text = (
        _MAKER_OK
        + "role: reviewer\nverdict: PASS\nfix-route: none\n"
    )
    code, out = _run(capsys, tmp_path, text, "--role", "reviewer")
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
