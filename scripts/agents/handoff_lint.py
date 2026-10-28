#!/usr/bin/env python3
"""Check a swarm handoff block for the fields the orchestrator counts.

    handoff_lint.py [--role maker|reviewer] [--mutants] <file | ->

Finds the LAST `## Handoff` block and prints `OK` (exit 0), or one
`MISSING <field>: <detail>` / `BAD <field>: <detail>` line per problem (exit 1). Field keys are
matched case-insensitively; values are compared case-sensitively after stripping.

Always required:
    status:   done | blocked | failed | needs-decision
    evidence: at least one following `- ` item (not required when status is blocked)
    improve:  present, any value (a bare `none` is fine)
Maker blocks with `status: done` (not reviewer) also require, anywhere in the block, a `GATE PASS` line and a
`_gate-logs/` LOG path (a maker once reported done with "GATE: see the swarm-check run below").
With `--mutants`, or when the brief beside the handoff (`<job>/prompts/<name>.md`) mentions
`--compare-rev`, such a maker block also needs mutate.py's summary `MUTANTS compared=N lost=0
errors=0` with N >= 1 (13 of 16 prune reviews found a kill the round had lost).
Checker blocks also require (with `--role reviewer`, or when the block's `role:` contains
"review", starts with "risk-", or contains "risk-officer" / "overfit-auditor" — so the real
`risk-midnight-1`, `risk-officer (risk-paper2-1)` and `risk-closeall-1 (risk-officer, ...)` forms count):
    verdict:    PASS | SHOULD-FIX | BLOCK | FAIL | VETO | REJECT | INSUFFICIENT   (exactly one token: qualifiers go in claims)
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
_MUTANTS_CLEAN = re.compile(r"MUTANTS compared=([1-9]\d*) lost=0 errors=0\b")

_STATUSES = ("done", "blocked", "failed", "needs-decision")
_VERDICTS = ("PASS", "SHOULD-FIX", "BLOCK", "FAIL", "VETO", "REJECT", "INSUFFICIENT")
_FIX_ROUTES = ("deepseek", "sonnet", "none")
_VERDICT_HINT = "one token; put qualifiers in claims"

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


def brief_requires_mutants(handoff: Path) -> bool:
    """The task's brief (``<job>/prompts/<handoff name>``) asks for a ``mutate.py --compare-rev`` run."""
    brief = handoff.parent.parent / "prompts" / handoff.name
    return brief.is_file() and "--compare-rev" in brief.read_text()


def check(text: str, role: str | None, mutants: bool = False) -> list[str]:
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

    reviewer = role == "reviewer" or _is_reviewer(fields)
    if not reviewer and status is not None and status[1] == "done":
        body = "\n".join(block)
        if "GATE PASS" not in body:
            problems.append(f"MISSING gate: no 'GATE PASS' line in {_BLOCK}")
        if "_gate-logs/" not in body:
            problems.append(f"MISSING gate: no _gate-logs/ LOG path in {_BLOCK}")
        if mutants and _MUTANTS_CLEAN.search(body) is None:
            problems.append(
                f"MISSING mutants: no 'MUTANTS compared=N lost=0 errors=0' line (mutate.py --compare-rev) in {_BLOCK}"
            )

    if reviewer:
        verdict = fields.get("verdict")
        if verdict is None:
            problems.append(f"MISSING verdict: no 'verdict:' line in or just above {_BLOCK}")
        elif verdict[1] not in _VERDICTS:
            problems.append(
                f"BAD verdict: {verdict[1]!r} is not one of {', '.join(_VERDICTS)}"
                f" ({_VERDICT_HINT})"
            )
        _enum(problems, "fix-route", _fix_route(fields, lines[:start]), _FIX_ROUTES)

    return problems


_CHECKER_ROLES = ("risk-officer", "overfit-auditor")


def _is_reviewer(fields: dict[str, tuple[int, str]]) -> bool:
    """The block's own ``role:`` marks a checker verdict (reviewer, risk officer, overfit auditor)."""
    role = fields.get("role")
    if role is None:
        return False
    value = role[1].lower()
    return (
        "review" in value
        or value.startswith("risk-")
        or any(marker in value for marker in _CHECKER_ROLES)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="handoff_lint.py", description="Lint a swarm handoff block.")
    parser.add_argument("--role", choices=("maker", "reviewer"), help="force the required field set")
    parser.add_argument("--mutants", action="store_true", help="require a clean mutate.py --compare-rev summary")
    parser.add_argument("path", metavar="file|-", help="handoff file, or - for stdin")
    args = parser.parse_args(argv)

    mutants = args.mutants
    if args.path == "-":
        text = sys.stdin.read()
    else:
        try:
            text = Path(args.path).read_text()
        except OSError as exc:
            print(f"MISSING handoff: cannot read {args.path}: {exc}")
            return 1
        mutants = mutants or brief_requires_mutants(Path(args.path))

    problems = check(text, args.role, mutants)
    if problems:
        print("\n".join(problems))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
