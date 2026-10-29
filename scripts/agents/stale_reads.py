#!/usr/bin/env python3
"""The commands an agent ran against the WRONG tree: main instead of its worktree.

    stale_reads.py --agent <id> --worktree <dir> [--db <t3 state.sqlite>]

A T3 child starts its shell in the MAIN checkout, and 59 of 342 children in Oct 14-28
never `cd <worktree>`: 104+25 relative reads (grep/sed/cat of scripts/, vibe_quant/,
tests/) and 116 test/gate runs read or tested main's STALE code (job 20261029-swarm-retro:
cc-facts-1 read scripts/agents/handoff_lint.py from main while its branch had changed
that file). The gate FAILs such a round and names the command (user decision: no PATH
wrapper).

`stale_commands(db, agent, worktree, main)` reads the T3 history DB read-only
(`orchestration_v2_projection_turn_items`, rows with `type = 'command_execution'`), takes
the agent's `payload_json.input` commands in ordinal order, and keeps the ones that:

  * in some shell SEGMENT (split on `;` `&&` `||` `|` newline) mention a REPO-RELATIVE code
    path (`scripts/`, `vibe_quant/`, `tests/`, `frontend/src` at a word start, not inside an
    absolute path), or an absolute path under MAIN into the same roots, or START with a
    pytest/mypy/ruff run (`.venv/bin/python -m pytest`, `.venv/bin/mypy`, `uv run ruff`), or
    run python with `PYTHONPATH=$PWD` (imports would resolve from the cwd);
  * contain no absolute path under the WORKTREE, and no `cd` into it.

So `cd /tmp/probe && cat scripts/x.py`, `sed -n 1p <worktree>/scripts/x.py` and
`cd <main> && cat docs/orchestration/README.md` are fine: a /tmp copy, an absolute
worktree path, and a design-doc read never test the wrong tree. Chief ruling R16: RUNNING the
shared swarm tooling from main (`python3 scripts/agents/<x>.py`, the bare
`scripts/agents/swarm-check.sh`) is exempt, and the whole segment is data (its operands are
not reads, even with `PYTHONPATH=$PWD` in front); a `sed`/`cat`/`rg`/`head` READ of scripts/
(relative or `<main>/scripts/`) and a pytest/mypy/ruff run from main stay stale. Data is not
a read: `--path`/`--body` values, `echo`/`printf` text and a `for p in <list>` list (unless
the loop body uses `$p` outside exempt tooling) are skipped. The default DB is
`~/.t3/userdata/statev2.sqlite`; `$SWARM_T3_DB` (or `--db`) points it elsewhere.

CLI: prints each stale command (first 160 characters) then `STALE n of m commands`;
exit 1 when n > 0, 0 otherwise, 2 when the DB is missing. The main checkout comes from
`--main`, else from the worktree's git common dir; when the worktree is deleted and no
`--main` is given, absolute main paths are not checked (only relative paths and runners).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

DB_ENV = "SWARM_T3_DB"
DEFAULT_DB = Path.home() / ".t3" / "userdata" / "statev2.sqlite"
TURN_ITEMS = "orchestration_v2_projection_turn_items"
# Repo code the worktree may have changed: a read of these from main is a false source.
REPO_ROOTS = ("scripts/", "vibe_quant/", "tests/", "frontend/src")
# Turtle threshold: the first characters of a stale command the CLI and the gate show.
SHOWN_CHARS = 160
PYTHONPATH_PWD = re.compile(r"PYTHONPATH=[\"']?\$PWD\b")
# python (any form): with PYTHONPATH=$PWD its imports follow the cwd.
RUNNER = re.compile(r"(?:^|[\s;&|('\"=/])python[0-9.]*\b")
# A segment that STARTS with a test/lint run: pytest, mypy, ruff, `python -m <one>`, `uv run <one>`.
TEST_RUN = re.compile(
    r"(?:uv\s+run\s+)?(?:\S*/)?(?:python[0-9.]*\s+-m\s+(?:\S*/)?)?(?:pytest|mypy|ruff)(?=\s|$)"
)
CD = re.compile(r"(?:^|&&|;|\|\|)\s*cd\s+(?:\"([^\"]*)\"|'([^']*)'|([^\s;&|]+))")
# A `--path <file>` / `--body <text>` VALUE is a search needle or a message body, not a file
# the command reads: briefs order `bus board search --path 'scripts/agents/bus.py'` and DMs
# quote paths in prose. Stripped before the stale triggers are read.
PATH_ARG = re.compile(r"--(?:path|body)[= ]+(?:\"[^\"]*\"|'[^']*'|\S+)")
# Chief ruling R16: RUNNING the shared swarm tooling from main is what the bus protocol orders
# (`python3 scripts/agents/<x>.py`, the bare `scripts/agents/swarm-check.sh`); a segment that
# starts with such a run is exempt. Reads (sed/cat/rg/head) and pytest/mypy/ruff runs of main's
# code do not start with it, so they stay stale.
SWARM_TOOL_RUN = re.compile(
    r"(?:(?:\S*/)?(?:python[0-9.]*|bash|sh)\s+)?(?:-\S+\s+)*(?:\S*/)?scripts/agents/[\w.-]+(?=\s|$)"
)
ENV_PREFIX = re.compile(r"(?:[A-Za-z_]\w*=(?:\"[^\"]*\"|'[^']*'|\S*)\s+)*")
KEYWORD_PREFIX = re.compile(r"(?:(?:do|then|else|\{|\(|\w+\(\)\s*\{)\s+)+")
SEGMENT_SPLIT = re.compile(r"&&|\|\||[;&|\n]")
FOR_LIST = re.compile(r"for\s+(\w+)\s+in\s+(.*)", re.S)
DATA_ONLY = re.compile(r"(?:echo|printf)\b")
# A scratch copy of the repo never tests main: its own code is the one under test.
SCRATCH_ROOTS = ("/tmp/", "/private/tmp/", "/var/folders/")
NO_MAIN = Path("/nonexistent-t3-main")


def resolve_db(path: str | Path | None = None) -> Path:
    """The T3 state DB: an explicit path, else $SWARM_T3_DB, else the local T3 state."""
    raw = str(path) if path is not None else os.environ.get(DB_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_DB


def agent_commands(db: Path, agent: str) -> list[str]:
    """Every command the agent ran, in ordinal order (its child thread id names the agent)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            f"select payload_json from {TURN_ITEMS} "
            "where type = 'command_execution' and thread_id like ? order by ordinal",
            (f"%delegate-task%3A{agent}",),
        ).fetchall()
    finally:
        con.close()
    commands: list[str] = []
    for (payload,) in rows:
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            continue
        command = data.get("input") if isinstance(data, dict) else None
        if isinstance(command, str) and command.strip():
            commands.append(command)
    return commands


def _mentions_repo_path(command: str) -> bool:
    """A repo-relative code path at a word start (`.`/`..` prefixes keep it relative)."""
    for root in REPO_ROOTS:
        start = 0
        while (i := command.find(root, start)) != -1:
            start = i + 1
            if i == 0:
                return True
            prev = command[i - 1]
            if prev.isalnum() or prev == "_":
                continue  # inside a longer word (e.g. "mytests/")
            if prev in "/-.":
                if command[:i].endswith(("./", "../")):
                    return True
                continue  # inside an absolute or dotted path: <tree>/scripts/...
            return True
    return False


def _mentions_main_code(command: str, main: Path | None) -> bool:
    """An absolute path under MAIN that holds code under test (see REPO_ROOTS)."""
    if main is None:
        return False
    prefix = str(main).rstrip("/") + "/"
    return any(prefix + root in command for root in REPO_ROOTS)


def _cd_targets(command: str) -> list[str]:
    targets: list[str] = []
    for match in CD.finditer(command):
        target = next((g for g in match.groups() if g is not None), "")
        if target:
            targets.append(target)
    return targets


def _under(path: str, root: Path) -> bool:
    bare = str(root).rstrip("/")
    norm = path.rstrip("/")
    return bool(bare) and (norm == bare or norm.startswith(bare + "/"))


def _in_worktree(command: str, worktree: Path) -> bool:
    bare = str(worktree).rstrip("/")
    return bool(bare) and re.search(re.escape(bare) + r"(?![\w.-])", command) is not None


def _segment_is_candidate(segment: str, loops: dict[str, str], main: Path | None) -> bool:
    """Whether one shell segment reads/runs repo code; loop headers and data never do."""
    text = KEYWORD_PREFIX.sub("", segment.strip() + " ", count=1).strip()
    loop = FOR_LIST.fullmatch(text)
    if loop:
        loops[loop.group(1)] = loop.group(2)
        return False
    env = ENV_PREFIX.match(text)
    rest = text[env.end() :] if env else text
    if not rest or DATA_ONLY.match(rest) or SWARM_TOOL_RUN.match(rest):
        return False
    for var, items in loops.items():
        if re.search(r"\$\{?" + re.escape(var) + r"\b", text):
            text += " " + items
    return bool(
        _mentions_repo_path(text)
        or _mentions_main_code(text, main)
        or TEST_RUN.match(rest)
        or (PYTHONPATH_PWD.search(text) and RUNNER.search(rest))
    )


def is_stale(command: str, worktree: Path, main: Path | None) -> bool:
    """Whether the command reads or runs repo code from a tree other than the worktree."""
    loops: dict[str, str] = {}
    segments = SEGMENT_SPLIT.split(PATH_ARG.sub("--path", command))
    # a list, not a generator: every segment must run so loop headers register their lists
    if not any([_segment_is_candidate(seg, loops, main) for seg in segments]):
        return False
    if _in_worktree(command, worktree):
        return False
    absolute = [t for t in _cd_targets(command) if t.startswith(("/", "~"))]
    targets = [str(Path(t).expanduser()) for t in absolute]
    if any(_under(t, worktree) for t in targets):
        return False
    return not (targets and all(t.startswith(SCRATCH_ROOTS) for t in targets))


def stale_report(
    db: Path, agent: str, worktree: Path, main: Path | None
) -> tuple[list[str], int]:
    """(the agent's stale commands in ordinal order, how many commands it ran)."""
    commands = agent_commands(db, agent)
    return [c for c in commands if is_stale(c, worktree, main)], len(commands)


def stale_commands(db: Path, agent: str, worktree: Path, main: Path | None) -> list[str]:
    """The agent's command_execution inputs that used the wrong tree, in ordinal order."""
    return stale_report(db, agent, worktree, main)[0]


def derive_main_root(worktree: Path) -> Path | None:
    """The main checkout a worktree belongs to (its git common dir's parent), or None."""
    proc = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    )
    common = proc.stdout.strip()
    if proc.returncode != 0 or not common:
        return None
    return Path(common).parent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Commands an agent ran against main instead of its worktree"
    )
    parser.add_argument("--agent", required=True, help="the child's clientRequestId (e.g. cc-x-1)")
    parser.add_argument("--worktree", required=True, help="the worktree the child should have used")
    parser.add_argument("--main", default=None, help="main checkout (else the worktree's git root)")
    parser.add_argument("--db", default=None, help=f"T3 state DB (else ${DB_ENV})")
    args = parser.parse_args(argv)

    db = resolve_db(args.db)
    if not db.is_file():
        print(f"no T3 history db at {db} (pass --db or set {DB_ENV})", file=sys.stderr)
        return 2
    worktree = Path(args.worktree).expanduser().resolve()
    main_root = Path(args.main).expanduser().resolve() if args.main else None
    stale, total = stale_report(
        db, args.agent, worktree, main_root or derive_main_root(worktree) or NO_MAIN
    )
    for command in stale:
        print(" ".join(command.split())[:SHOWN_CHARS])
    print(f"STALE {len(stale)} of {total} commands")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
