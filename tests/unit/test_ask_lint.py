"""ask_lint.py: a question to the user must be plain words, with a concrete example and a
one-line summary. The chief re-asked three times in Oct-Nov 2026 because the question carried
bead ids, file paths and no example the user could check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
