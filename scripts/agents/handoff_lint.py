#!/usr/bin/env python3
"""Check a swarm handoff block for the fields the orchestrator counts.

    handoff_lint.py [--role maker|reviewer] <file | ->

Finds the LAST `## Handoff` block and prints `OK` (exit 0), or one
`MISSING <field>: <detail>` / `BAD <field>: <detail>` line per problem (exit 1). Field keys are
matched case-insensitively; values are compared case-sensitively after stripping.

Always required:
    status:   done | blocked | failed | needs-decision
    evidence: at least one following `- ` item (not required when status is blocked)
    improve:  present, any value (a bare `none` is fine)
Reviewer blocks also require (with `--role reviewer`, or when the block's `role:` contains
"review"):
    verdict:    PASS | SHOULD-FIX | BLOCK
    fix-route:  deepseek | sonnet | none   (the line may sit just above the block)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HEADING = re.compile(r"^##\s+handoff\s*$", re.IGNORECASE)
_ANY_HEADING = re.compile(r"^#{1,6}\s")
_FIELD = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*:[ \t]*(.*)$")
_BULLET = re.compile(r"^\s*-\s")

_STATUSES = ("done", "blocked", "failed", "needs-decision")
_VERDICTS = ("PASS", "SHOULD-FIX", "BLOCK")
_FIX_ROUTES = ("deepseek", "sonnet", "none")

_BLOCK = "the last '## Handoff' block"


def _fields(block: list[str]) -> dict[str, tuple[int, str]]:
    found: dict[str, tuple[int, str]] = {}
    for index, line in enumerate(block):
        match = _FIELD.match(line)
        if match is not None:
            found.setdefault(match.group(1).lower(), (index, match.group(2).strip()))
    return found


def _has_bullet(block: list[str], evidence: tuple[int, str] | None) -> bool:
    if evidence is None:
        return False
    for line in block[evidence[0] + 1 :]:
        if _BULLET.match(line):
            return True
        if line.strip():
            return False
    return False


def _fix_route(fields: dict[str, tuple[int, str]], above: list[str]) -> str | None:
    if "fix-route" in fields:
        return fields["fix-route"][1]
    for line in reversed(above):
        match = _FIELD.match(line)
        if match is not None and match.group(1).lower() == "fix-route":
            return match.group(2).strip()
    return None


def _enum(problems: list[str], name: str, value: str | None, allowed: tuple[str, ...]) -> None:
    if value is None:
        problems.append(f"MISSING {name}: no '{name}:' line in or just above {_BLOCK}")
    elif value not in allowed:
        problems.append(f"BAD {name}: {value!r} is not one of {', '.join(allowed)}")


def check(text: str, role: str | None) -> list[str]:
    lines = text.splitlines()
    headings = [i for i, line in enumerate(lines) if _HEADING.match(line)]
    if not headings:
        return ["MISSING handoff: no '## Handoff' block found"]
    start = headings[-1]
    end = next((i for i in range(start + 1, len(lines)) if _ANY_HEADING.match(lines[i])), len(lines))
    block = lines[start + 1 : end]
    fields = _fields(block)

    problems: list[str] = []
    status = fields.get("status")
    _enum(problems, "status", None if status is None else status[1], _STATUSES)

    blocked = status is not None and status[1] == "blocked"
    if not blocked and not _has_bullet(block, fields.get("evidence")):
        problems.append(f"MISSING evidence: no '- ' item follows 'evidence:' in {_BLOCK}")

    if "improve" not in fields:
        problems.append(f"MISSING improve: no 'improve:' line in {_BLOCK}")

    if role == "reviewer" or _is_reviewer(fields):
        verdict = fields.get("verdict")
        _enum(problems, "verdict", None if verdict is None else verdict[1], _VERDICTS)
        _enum(problems, "fix-route", _fix_route(fields, lines[:start]), _FIX_ROUTES)

    return problems


def _is_reviewer(fields: dict[str, tuple[int, str]]) -> bool:
    role = fields.get("role")
    return role is not None and "review" in role[1].lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="handoff_lint.py", description="Lint a swarm handoff block.")
    parser.add_argument("--role", choices=("maker", "reviewer"), help="force the required field set")
    parser.add_argument("path", metavar="file|-", help="handoff file, or - for stdin")
    args = parser.parse_args(argv)

    if args.path == "-":
        text = sys.stdin.read()
    else:
        try:
            text = Path(args.path).read_text()
        except OSError as exc:
            print(f"MISSING handoff: cannot read {args.path}: {exc}")
            return 1

    problems = check(text, args.role)
    if problems:
        print("\n".join(problems))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
