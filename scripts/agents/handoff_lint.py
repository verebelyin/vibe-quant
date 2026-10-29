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
Every cited LOG must exist (a relative path is read from the main checkout) and end with swarm_check's
`HEAD <sha> TREE <tree>` + GATE lines; at least one must end `GATE PASS` with the tree of a commit sha
cited in the block (`git rev-parse <sha>^{tree}`, so a message-only amend still matches; a TREE `dirty`
gate never does). 61 of 204 LOGs cited by makers in Oct 14-28 did not exist.
A `done` maker block reporting findings (`fixed: <id> ...`) needs, for each id, a
`sibling-scan: <id> <rg command> -> <N> sites (...)` line in the block — 13 of 16 prune
reviews found a bug whose siblings the fix round missed, so a fixer says where it looked.
With `--mutants` such a maker block also needs mutate.py's summary `MUTANTS compared=N lost=0
errors=0` with N >= 1 (13 of 16 prune reviews found a kill the round had lost). Brief text no
longer triggers it (any brief naming `--compare-rev` did): swarm-check's `parity` check in
`--full` proves a test-deleting round lost no kill.
Checker blocks also require (with `--role reviewer`, or when the block's `role:` contains
"review", starts with "risk-", or contains "risk-officer" / "overfit-auditor" — so the real
`risk-midnight-1`, `risk-officer (risk-paper2-1)` and `risk-closeall-1 (risk-officer, ...)` forms count):
    verdict:    PASS | SHOULD-FIX | BLOCK | FAIL | VETO | REJECT | INSUFFICIENT   (exactly one token: qualifiers go in claims)
    fix-route:  deepseek | sonnet | none   (the line may sit just above the block)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_HEADING = re.compile(r"^##\s+handoff\s*$", re.IGNORECASE)
_ANY_HEADING = re.compile(r"^#{1,6}\s")
_FIELD = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*:[ \t]*(.*)$")
_BULLET = re.compile(r"^\s*-\s")
_MUTANTS_CLEAN = re.compile(r"MUTANTS compared=([1-9]\d*) lost=0 errors=0\b")
_FIXED = re.compile(r"^\s*(?:[-*]\s+)?fixed\s*:\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
_SIBLING_SCAN = r"^\s*(?:[-*]\s+)?sibling-scan\s*:\s*{name}(?![A-Za-z0-9_.-]).*->\s*\d+\s+site"
_GATE_LOG = re.compile(r"[^\s`'\"(]*_gate-logs/[^\s`'\")\]]+\.log")
_SHA = re.compile(r"(?<![0-9A-Za-z_])[0-9a-f]{7,40}(?![0-9A-Za-z_])")
_TRAILER = re.compile(r"^HEAD (\S+) TREE (\S+)$")

_STATUSES = ("done", "blocked", "failed", "needs-decision")
_VERDICTS = ("PASS", "SHOULD-FIX", "BLOCK", "FAIL", "VETO", "REJECT", "INSUFFICIENT")
_FIX_ROUTES = ("deepseek", "sonnet", "none")
_VERDICT_HINT = "one token; put qualifiers in claims"

_BLOCK = "the last '## Handoff' block"
NO_GATED_COMMIT = (
    f"BAD gate: no commit cited in {_BLOCK} has the tree a PASSing LOG gated (cite the gated"
    " commit sha; TREE 'dirty'/'moved' = the gate read edits no commit holds)"
)


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


def _fixed_ids(block: list[str]) -> list[str]:
    """Finding ids the block reports as fixed, in first-mention order."""
    ids: list[str] = []
    for line in block:
        match = _FIXED.match(line)
        if match is not None and match.group(1) not in ids:
            ids.append(match.group(1))
    return ids


def _has_sibling_scan(block: list[str], finding: str) -> bool:
    """A ``sibling-scan: <finding> ... -> <N> sites`` line for exactly this id."""
    pattern = re.compile(_SIBLING_SCAN.format(name=re.escape(finding)), re.IGNORECASE)
    return any(pattern.match(line) for line in block)


def _main_root() -> Path:
    """The main checkout (gate LOGs live there), also when this script runs from a worktree."""
    here = Path(__file__).resolve().parent
    proc = subprocess.run(
        ["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    )
    common = proc.stdout.strip()
    return Path(common).parent if proc.returncode == 0 and common else here.parents[1]


def _tree(sha: str, cwd: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--verify", "--quiet", f"{sha}^{{tree}}"],
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _trailer(log: Path) -> tuple[str, str] | None:
    """(TREE, GATE line) from the LOG's last two non-blank lines, or None without them."""
    lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
    match = _TRAILER.match(lines[-2]) if len(lines) >= 2 else None
    return (match.group(2), lines[-1]) if match else None


def gate_log_problems(body: str, root: Path | None = None) -> list[str]:
    """The cited gate LOGs exist and one PASSed on the tree of a commit the block cites."""
    cited = list(dict.fromkeys(_GATE_LOG.findall(body)))
    if not cited:
        return ["BAD gate: '_gate-logs/' appears but no '_gate-logs/<name>.log' path is cited"]
    root = _main_root() if root is None else root
    problems: list[str] = []
    passed: dict[str, Path] = {}
    for raw in cited:
        log = Path(raw).expanduser()
        log = log if log.is_absolute() else root / log
        if not log.is_file():
            problems.append(f"BAD gate: cited LOG {raw} does not exist")
            continue
        trailer = _trailer(log)
        if trailer is None:
            problems.append(
                f"BAD gate: LOG {raw} does not end with 'HEAD <sha> TREE <tree>' and a GATE line"
                " (written before the gate recorded them: re-run the gate)"
            )
        elif trailer[1] == "GATE PASS":
            passed.setdefault(trailer[0], log)
    if problems:
        return problems
    if not passed:
        return ["BAD gate: no cited LOG ends with 'GATE PASS'"]
    cwd = next(iter(passed.values())).parent
    shas = dict.fromkeys(_SHA.findall(_GATE_LOG.sub(" ", body)))
    if any(_tree(sha, cwd) in passed for sha in shas):
        return []
    return [NO_GATED_COMMIT]


def check(
    text: str, role: str | None, mutants: bool = False, root: Path | None = None
) -> list[str]:
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
        else:
            problems.extend(gate_log_problems(body, root))
        for finding in _fixed_ids(block):
            if not _has_sibling_scan(block, finding):
                problems.append(f"MISSING sibling-scan: {finding}")
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

    if args.path == "-":
        text = sys.stdin.read()
    else:
        try:
            text = Path(args.path).read_text()
        except OSError as exc:
            print(f"MISSING handoff: cannot read {args.path}: {exc}")
            return 1

    problems = check(text, args.role, args.mutants)
    if problems:
        print("\n".join(problems))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
