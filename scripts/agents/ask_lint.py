#!/usr/bin/env python3
"""Lint a question before it goes to the user (orchestrate SKILL.md § Asking the user).

    scripts/agents/ask_lint.py QUESTION_FILE
    echo '<question>' | scripts/agents/ask_lint.py
    scripts/agents/ask_lint.py --status MESSAGE_FILE   # a status message's "Waiting on you"

The chief re-asked three times in Oct-Nov 2026: the first wording carried a bead id, a file
path and no example the user could picture. A question must be plain words, show one real
example with a coin and a number, and end with a one-line summary ("1 = A or B?"). This lints
that shape: exit 0 when clean, exit 1 with one line per problem. Stdlib only.

`--status` checks a whole status message instead: every number its "Waiting on you" part points
at ("9 = A or B?") must be asked in the same message. Nov 2026: the chief linted the questions
file, then ended status messages with "9 = A or B? 10 = A or B?" alone; the user asked 8 times
to have the questions restated. The Stop hook `hooks/ask_guard.py` runs this check.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

COINS = frozenset(
    (
        "ADA", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "ETC", "ETH",
        "FIL", "HBAR", "HYPE", "ICP", "LINK", "LTC", "MATIC", "NEAR", "OP", "SHIB", "SOL", "SUI", "TON",
        "TRX", "UNI", "XLM", "XRP", "ZEC",
    )
)  # fmt: skip

_BEAD = re.compile(r"vibe-quant-\w+")
_AGENT = re.compile(r"\b(?:cc|rev|impl)-[A-Za-z0-9][A-Za-z0-9_-]*")
_PATH = re.compile(r"\bscripts/\S+|\b\w+/\w+\.\w+")
_CONTRACT = re.compile(r"\bR\d+\b")
_COIN = re.compile(r"\b(?:" + "|".join(sorted(COINS)) + r")\b|\b[A-Z]{2,10}USDT\b")
_NUMBER = re.compile(r"\d")
_SUMMARY = re.compile(r"^\s*\d+\s*=\s*.+\?$")


def _matches(pattern: re.Pattern[str], text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in pattern.finditer(text)))


def _paths(text: str) -> list[str]:
    """Path tokens without the sentence punctuation that follows them."""
    return list(dict.fromkeys(match.group(0).rstrip(".,;:!?)]") for match in _PATH.finditer(text)))


def _body(text: str) -> str:
    """The question without its final summary line: the summary's own digits are not the
    concrete number the example must carry."""
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and _SUMMARY.match(lines[-1]):
        return "\n".join(lines[:-1])
    return text


def lint(text: str) -> list[str]:
    """Every problem with a question to the user, in a stable order; [] when clean."""
    problems = [f"bead id in question: {token}" for token in _matches(_BEAD, text)]
    problems += [f"agent/request id in question: {token}" for token in _matches(_AGENT, text)]
    problems += [f"file path in question: {token}" for token in _paths(text)]
    problems += [f"contract-row id in question: {token}" for token in _matches(_CONTRACT, text)]
    body = _body(text)
    if not _COIN.search(body) or not _NUMBER.search(body):
        problems.append("no concrete example: add a coin (e.g. BTC) and a number")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not _SUMMARY.match(lines[-1]):
        problems.append("missing final summary line (e.g. '1 = A or B?')")
    return problems


_WAITING = re.compile(r"waiting on you", re.IGNORECASE)
_REFERENCED = re.compile(r"\b(\d{1,3})\s*=\s*[^=?\n]*\?")


def _asked(text: str, number: str) -> bool:
    """A line labelled with the number ("9.", "**9)**", "Q9:", "Question 9:") that carries text."""
    label = rf"(?:q(?:uestion)?\s*)?{number}\s*[.):\u2014-]"
    line = re.compile(rf"^[\s*#>_-]*(?:\*\*)?{label}(?:\*\*)?\s*\S", re.IGNORECASE | re.MULTILINE)
    inline = re.compile(rf"\bquestion\s+{number}\s*[.:]\s*\S", re.IGNORECASE)
    return bool(line.search(text) or inline.search(text))


def status_problems(text: str) -> list[str]:
    """Every number the last "Waiting on you" part references that the message never asks."""
    starts = [match.start() for match in _WAITING.finditer(text)]
    if not starts:
        return []
    tail = text[starts[-1] :]
    numbers = list(dict.fromkeys(match.group(1) for match in _REFERENCED.finditer(tail)))
    return [
        f"question {n} is referenced but not asked in this message: restate it (plain words, "
        "example, options)"
        for n in numbers
        if not _asked(text, n)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ask_lint.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", nargs="?", type=Path, help="question file (default: stdin)")
    parser.add_argument("--status", action="store_true", help="lint a whole status message")
    args = parser.parse_args(argv)
    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    problems = status_problems(text) if args.status else lint(text)
    if problems:
        print("\n".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
