"""Enforce the vibe-quant commit-message format on ``main``.

Usage::

    scripts/git-hooks/check_commit_msg.py <commit-msg-file>

``git commit`` passes the message file as ``$1`` (see ``githooks(5)``). On any branch
other than ``main`` every message is accepted, so worker branches and scratch commits
are unaffected. On ``main``:

* a subject matching ``self-improve``, ``orchestration``, ``Merge``, ``Revert``,
  ``wip``, ``chore(beads)`` or ``bd: `` passes as-is (board/telemetry and prompt
  commits keep their own one-line style);
* every other message must carry the ``Why``, ``What`` and ``Verification`` heading
  lines from ``docs/orchestration/prompts/commit-message.md`` (``Why``/``**Why**:``/
  ``# Why`` all count). Missing headings are named on stderr.

Exit status: 0 accepted, 1 rejected, 2 usage error.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_BRANCH = "main"
DOC_PATH = "docs/orchestration/prompts/commit-message.md"

EXEMPT_SUBJECT = re.compile(
    r"^(self-improve|orchestration|Merge|Revert|wip|chore\(beads\)|bd: )"
)
REQUIRED_HEADINGS = ("Why", "What", "Verification")
HEADING = re.compile(r"^(?:#{1,6}\s*)?\*{0,2}(Why|What|Verification)\b")


def current_branch() -> str:
    """Return the current branch name, or ``""`` when git cannot say (detached etc.).

    ``symbolic-ref`` also answers on an unborn branch (a fresh ``git init`` with no
    commit), where ``rev-parse --abbrev-ref HEAD`` does not.
    """
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def message_lines(text: str) -> list[str]:
    """Return the message's lines with git's ``#`` comment lines removed.

    A ``#`` line that introduces a required heading stays: the documented format
    allows ``# Why``, and git's template comments never look like one.
    """
    return [
        line
        for line in text.splitlines()
        if not line.startswith("#") or HEADING.match(line.strip()) is not None
    ]


def missing_headings(text: str) -> list[str]:
    """Return the required headings that no line of ``text`` introduces."""
    found = {
        match.group(1)
        for line in message_lines(text)
        if (match := HEADING.match(line.strip())) is not None
    }
    return [heading for heading in REQUIRED_HEADINGS if heading not in found]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: check_commit_msg.py <commit-msg-file>", file=sys.stderr)
        return 2

    if current_branch() != DEFAULT_BRANCH:
        return 0

    text = Path(args[0]).read_text(encoding="utf-8", errors="replace")
    subject = next((line.strip() for line in message_lines(text) if line.strip()), "")
    if EXEMPT_SUBJECT.match(subject):
        return 0

    missing = missing_headings(text)
    if not missing:
        return 0

    print(f"commit message on {DEFAULT_BRANCH} is missing: {', '.join(missing)}", file=sys.stderr)
    print(f"see {DOC_PATH} for the required Why / What / Verification format", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
