"""ask_lint.py: a question to the user must be plain words, with a concrete example and a
one-line summary. The chief re-asked three times in Oct-Nov 2026 because the question carried
bead ids, file paths and no example the user could check.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/ask_lint.py"
_spec = importlib.util.spec_from_file_location("ask_lint_under_test", _PATH)
assert _spec and _spec.loader
ask_lint = importlib.util.module_from_spec(_spec)
sys.modules["ask_lint_under_test"] = ask_lint
_spec.loader.exec_module(ask_lint)

BAD = (
    "We hit vibe-quant-abcde while running scripts/agents/chief.py. "
    "Should we change it?"
)
GOOD = (
    "You hold BTC and ETH. ETH fails to save at 14:32 and the screen shows 0 positions.\n"
    "\n"
    "1 = keep BTC and ETH?"
)
CLEAN_BODY = "You hold BTC and ETH. ETH fails to save at 14:32 and the screen shows 0 positions."
CLEAN_TAIL = "\n\n1 = keep BTC and ETH?"


def _question(body: str = CLEAN_BODY, tail: str = CLEAN_TAIL) -> str:
    return body + tail


# --- CLI: exit 1 with one line per problem, exit 0 clean --------------------------------


def test_bad_question_exits_1_and_names_every_problem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "q.txt"
    path.write_text(BAD)
    assert ask_lint.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "bead id in question: vibe-quant-abcde" in out
    assert "file path in question: scripts/agents/chief.py" in out
    assert "no concrete example: add a coin (e.g. BTC) and a number" in out
    assert "missing final summary line (e.g. '1 = A or B?')" in out
    assert len(out.strip().splitlines()) == 4


def test_good_question_exits_0_and_prints_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "q.txt"
    path.write_text(GOOD)
    assert ask_lint.main([str(path)]) == 0
    assert capsys.readouterr().out == ""


def test_reads_stdin_when_no_file_arg(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(BAD))
    assert ask_lint.main([]) == 1
    assert "bead id in question: vibe-quant-abcde" in capsys.readouterr().out


# --- one rule at a time -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (_question(f"See vibe-quant-abcde. {CLEAN_BODY}"), "bead id in question: vibe-quant-abcde"),
        (_question(f"Ask cc-a-1 about it. {CLEAN_BODY}"), "agent/request id in question: cc-a-1"),
        (_question(f"Ask rev-b-2 about it. {CLEAN_BODY}"), "agent/request id in question: rev-b-2"),
        (
            _question(f"See scripts/agents/chief.py. {CLEAN_BODY}"),
            "file path in question: scripts/agents/chief.py",
        ),
        (_question(f"Rule R12 covers it. {CLEAN_BODY}"), "contract-row id in question: R12"),
    ],
)
def test_lint_flags_internal_names(text: str, problem: str) -> None:
    assert ask_lint.lint(text) == [problem]


def test_concrete_example_needs_a_coin() -> None:
    body = "You hold 2 assets. They fail to save at 14:32 and the screen shows 0 positions."
    assert ask_lint.lint(_question(body)) == [
        "no concrete example: add a coin (e.g. BTC) and a number"
    ]


def test_concrete_example_needs_a_number_outside_the_summary() -> None:
    body = "You hold BTC and ETH. ETH fails to save and the screen shows no positions."
    assert ask_lint.lint(_question(body)) == [
        "no concrete example: add a coin (e.g. BTC) and a number"
    ]


def test_an_usdt_pair_counts_as_a_coin() -> None:
    body = "You hold ETHUSDT and it drops to 2000.00 at 09:05."
    assert ask_lint.lint(_question(body)) == []


def test_missing_final_summary_line_is_flagged() -> None:
    assert ask_lint.lint(CLEAN_BODY + "\n\nShould we change this?") == [
        "missing final summary line (e.g. '1 = A or B?')"
    ]


def test_the_summary_line_must_be_last() -> None:
    text = "1 = keep both?\n\n" + CLEAN_BODY
    assert ask_lint.lint(text) == ["missing final summary line (e.g. '1 = A or B?')"]


def test_a_second_numbered_summary_line_is_fine() -> None:
    assert ask_lint.lint(CLEAN_BODY + "\n\n1 = keep both? 2 = OK?") == []
