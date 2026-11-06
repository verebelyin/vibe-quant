#!/usr/bin/env python3
"""The commands an agent ran against the WRONG tree: main instead of its worktree.

    stale_reads.py --agent <id> --worktree <dir> [--base <rev>] [--db <t3 state.sqlite>]

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
a read: `--path`/`--body`/`--scope` values, `echo`/`printf` text and a `for p in <list>` list
(unless the loop body uses `$p` outside exempt tooling) are skipped. A heredoc whose body a
`cat >file`/`tee file` WRITES is text, not a read, so its body is skipped; one fed to an
interpreter (`bash <<EOF`, `python3 - <<EOF`) is code that runs, so its body stays. A heredoc
belongs to its own simple command (`cat /tmp/x; bash <<EOF` is bash's) and ends at the exact
delimiter line (`'END-SCRIPT'` counts; a space-indented `  EOF` does not; `<<-` strips only
leading TABs). A tooling
run whose ARGUMENTS name repo code (`mutate.py --file vibe_quant/x.py`, `... -- pytest
tests/unit/x.py`) acts on main's copy, so it is stale. A plain read (sed/cat/rg/head ...) of
paths the branch leaves as they were at `--base` (`unchanged_paths`) is not stale: main's copy
IS the branch's copy. `ls` compares only the NAMES under a path, so a file added, deleted or
renamed relative to `--base` (or untracked) makes it changed while a content edit does not.
Segments split on `;` `&&` `||` `|` `&` newline OUTSIDE quotes (`grep "a|b" x.py` is one
segment); an unterminated quote falls back to the plain split. The default DB is
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
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
# A `--path <file>` / `--body <text>` / `--scope <globs>` VALUE is a search needle, a message
# body or the gate's list of paths the diff may touch, not a file the command reads: briefs
# order `bus board search --path 'scripts/agents/bus.py'`, DMs quote paths in prose and every
# gate run carries `--scope scripts/...`. Stripped before the stale triggers are read.
PATH_ARG = re.compile(r"--(?:path|body|scope)[= ]+(?:\"[^\"]*\"|'[^']*'|\S+)")
# Chief ruling R16: RUNNING the shared swarm tooling from main is what the bus protocol orders
# (`python3 scripts/agents/<x>.py`, the bare `scripts/agents/swarm-check.sh`); a segment that
# starts with such a run is exempt. Reads (sed/cat/rg/head) and pytest/mypy/ruff runs of main's
# code do not start with it, so they stay stale.
SWARM_TOOL_RUN = re.compile(
    r"(?:(?:\S*/)?(?:python[0-9.]*|bash|sh)\s+)?(?:-\S+\s+)*(?:\S*/)?scripts/agents/[\w.-]+(?=\s|$)"
)
# `export NAME=VALUE ...` only sets variables: the bus protocol's first line exports
# PATH=<main>/scripts/agents/bin, a value, not a read (job 20261029-gaps-speed cc-kufxs-1).
EXPORT_ONLY = re.compile(r"export\s+(?:[A-Za-z_]\w*=(?:\"[^\"]*\"|'[^']*'|\S*)\s*)+")
ENV_PREFIX = re.compile(r"(?:[A-Za-z_]\w*=(?:\"[^\"]*\"|'[^']*'|\S*)\s+)*")
KEYWORD_PREFIX = re.compile(r"(?:(?:do|then|else|\{|\(|\w+\(\)\s*\{)\s+)+")
SEGMENT_SPLIT = re.compile(r"&&|\|\||[;&|\n]")
FOR_LIST = re.compile(r"for\s+(\w+)\s+in\s+(.*)", re.S)
DATA_ONLY = re.compile(r"(?:echo|printf)\b")
# A scratch copy of the repo never tests main: its own code is the one under test.
SCRATCH_ROOTS = ("/tmp/", "/private/tmp/", "/var/folders/")
NO_MAIN = Path("/nonexistent-t3-main")
# A heredoc whose body `cat >file` / `tee file` WRITES is text (a script being written out),
# not a read of repo code: its body is dropped before segmenting. One fed to an interpreter
# (`bash <<EOF`, `python3 - <<EOF`) is code that runs, so its body stays a candidate.
ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=")
REDIRECT_CHARS = frozenset("<>&|;()")


# A read-only look at a path the branch leaves as it was at the branch point is harmless:
# main's copy IS the branch's copy (four makers in one job were ruled false positives on
# sed/rg of main's swarm tooling). Test/lint runs and reads of changed paths stay stale.
READ_TOOL = re.compile(r"(?:\S*/)?(?:sed|cat|head|tail|rg|grep|less|wc|nl|ls)(?=\s|$)")
PATH_TOKEN = re.compile(r"[^\s'\"=;&|()<>]+")


def _repo_paths(text: str, main: Path | None) -> list[str]:
    """Repo-relative code paths a segment names (main-absolute ones made relative)."""
    prefix = (str(main).rstrip("/") + "/") if main is not None else None
    found: list[str] = []
    for token in PATH_TOKEN.findall(text):
        if prefix and token.startswith(prefix):
            token = token[len(prefix) :]
        token = token.removeprefix("./")
        if token.startswith(REPO_ROOTS):
            found.append(token)
    return found


def _split_spans(command: str) -> list[tuple[int, int]] | None:
    """(start, end) of each shell segment, split on ; && || | & newline OUTSIDE quotes.

    None when a quote is left open (callers fall back to the plain split).
    """
    spans: list[tuple[int, int]] = []
    start = 0
    quote: str | None = None
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if quote is not None:
            if c == "\\" and quote == '"' and i + 1 < n:
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c in "'\"":
            quote = c
        elif command.startswith(("&&", "||"), i):
            spans.append((start, i))
            i += 2
            start = i
            continue
        elif c in ";&|\n":
            spans.append((start, i))
            start = i + 1
        i += 1
    if quote is not None:
        return None
    spans.append((start, n))
    return spans


def _delimiter_word(text: str, i: int) -> tuple[str, int] | None:
    """The heredoc delimiter WORD starting at `text[i]`, quotes removed, and where it ends."""
    n = len(text)
    tag: list[str] = []
    while i < n and text[i] not in " \t\n" and text[i] not in REDIRECT_CHARS:
        c = text[i]
        if c in "'\"":
            close = text.find(c, i + 1)
            if close < 0:
                return None
            tag.append(text[i + 1 : close])
            i = close + 1
        elif c == "\\" and i + 1 < n:
            tag.append(text[i + 1])
            i += 2
        else:
            tag.append(c)
            i += 1
    return ("".join(tag), i) if tag else None


def _heredocs_in(span: str) -> list[tuple[str, bool]]:
    """(delimiter, strips leading tabs) of each heredoc `<<[-]WORD` in one segment, in order.

    Quote-aware: a `<<` inside quotes and a here-string `<<<` are not heredocs.
    """
    found: list[tuple[str, bool]] = []
    quote: str | None = None
    i, n = 0, len(span)
    while i < n:
        c = span[i]
        if quote is not None:
            if c == "\\" and quote == '"' and i + 1 < n:
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c in "'\"":
            quote = c
            i += 1
            continue
        if span.startswith("<<<", i):
            i += 3
            continue
        if span.startswith("<<", i):
            i += 2
            dash = span.startswith("-", i)
            i += 1 if dash else 0
            while i < n and span[i] in " \t":
                i += 1
            word = _delimiter_word(span, i)
            if word is not None:
                found.append((word[0], dash))
                i = word[1]
            continue
        i += 1
    return found


def _owner_writes_a_file(span: str) -> bool:
    """Whether the ONE simple command in `span` is `cat >file` / `tee file` (a writer).

    Only that command's own words and redirects count: a word quoted into another command's
    arguments (`bash <<EOF "tee x"`) or a sibling command on the line never makes it one.
    """
    lexer = shlex.shlex(span, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    words: list[str] = []
    stdout_redirected = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if set(tok) <= REDIRECT_CHARS:
            other_fd = i > 0 and tokens[i - 1].isdigit() and tokens[i - 1] != "1"
            if tok in (">", ">>", ">|") and not other_fd:
                stdout_redirected = True
            if tok.startswith(("<", ">")):
                i += 1  # skip the redirect's operand (file or delimiter)
        else:
            words.append(tok)
        i += 1
    while words and ASSIGNMENT.match(words[0]):
        words.pop(0)
    if not words:
        return False
    command = words[0].rsplit("/", 1)[-1]
    if command == "cat":
        return stdout_redirected
    return command == "tee" and any(not w.startswith("-") for w in words[1:])


def _drop_written_heredocs(command: str) -> str:
    """Drop the body of every heredoc a `cat >file`/`tee file` writes; keep its opener line.

    A heredoc belongs to ONE simple command (its segment on the opener line); only that
    command being a writer makes the body text. The terminator is the exact delimiter line
    (`<<-` also strips leading TABs, nothing else). An interpreter's heredoc keeps its body,
    which is code.
    """
    if "<<" not in command:
        return command
    lines = command.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        spans = _split_spans(line)
        if spans is None:
            continue
        for start, end in spans:
            owner = line[start:end]
            writer: bool | None = None
            for tag, dash in _heredocs_in(owner):
                if writer is None:
                    writer = _owner_writes_a_file(owner)
                stop = next(
                    (
                        j
                        for j in range(i, len(lines))
                        if (lines[j].lstrip("\t") if dash else lines[j]) == tag
                    ),
                    -1,
                )
                if stop < 0:
                    break
                if not writer:
                    out.extend(lines[i : stop + 1])
                i = stop + 1
    return "\n".join(out)


def _segments(command: str) -> list[str]:
    """Shell segments split on ; && || | & newline OUTSIDE quotes (plain split if unbalanced)."""
    command = _drop_written_heredocs(command)
    spans = _split_spans(command)
    if spans is None:
        return SEGMENT_SPLIT.split(command)
    return [command[a:b] for a, b in spans]


def _pure_read_of_unchanged(
    command: str, main: Path | None, unchanged: Callable[..., bool]
) -> bool:
    """Every code-touching segment is a plain read, and every path it names is unchanged.

    With `ls` the predicate compares only the NAMES under the path, since `ls` reveals no
    contents; every other read tool keeps the byte-identical rule.
    """
    loops: dict[str, str] = {}
    saw = False
    for segment in _segments(PATH_ARG.sub("--path", command)):
        if not _segment_is_candidate(segment, loops, main):
            continue
        text = KEYWORD_PREFIX.sub("", segment.strip() + " ", count=1).strip()
        env = ENV_PREFIX.match(text)
        rest = text[env.end() :] if env else text
        paths = _repo_paths(text, main)
        tool = READ_TOOL.match(rest)
        if tool is None or not paths:
            return False
        names_only = tool.group(0).rsplit("/", 1)[-1] == "ls"
        if not all(unchanged(path, names_only=names_only) for path in paths):
            return False
        saw = True
    return saw


def unchanged_paths(
    worktree: Path, main: Path | None, base: str | None = None
) -> Callable[..., bool]:
    """A predicate: is this repo-relative path unchanged relative to the branch point?

    With `base` (the revision the branch started from) a path is unchanged when the worktree's
    working tree equals `base` for it (`git diff --quiet <base> -- <path>` is empty, no
    untracked file under it) and main has no uncommitted change under it. Without `base`,
    main's HEAD is the reference (a path main moved after the branch point then counts as
    changed). `names_only` (used by `ls`) compares only the NAMES under a path: a file added,
    deleted or renamed relative to the reference, or untracked, makes it changed while a
    content edit does not. Any git error, a missing tree or a doubt answers False (stale), so
    the exemption fails closed.
    """
    ref = base
    if ref is None and main is not None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(main), "rev-parse", "HEAD"], capture_output=True, text=True
            )
            ref = proc.stdout.strip() if proc.returncode == 0 else None
        except OSError:
            ref = None
    cache: dict[tuple[str, bool], bool] = {}

    def same(path: str, names_only: bool = False) -> bool:
        if ref is None or main is None:
            return False
        key = (path, names_only)
        if key not in cache:
            cache[key] = _same_in_both(worktree, main, ref, path, names_only=names_only)
        return cache[key]

    return same


def _same_in_both(
    worktree: Path, main: Path, ref: str, path: str, *, names_only: bool = False
) -> bool:
    try:
        if names_only:
            diff = subprocess.run(
                [
                    "git", "-C", str(worktree), "diff", "--name-only", "--diff-filter=ADR",
                    ref, "--", path,
                ],  # fmt: skip
                capture_output=True,
                text=True,
            )
        else:
            diff = subprocess.run(
                ["git", "-C", str(worktree), "diff", "--quiet", ref, "--", path],
                capture_output=True,
                text=True,
            )
        untracked = subprocess.run(
            ["git", "-C", str(worktree), "ls-files", "--others", "--exclude-standard", "--", path],
            capture_output=True,
            text=True,
        )
        main_dirty = subprocess.run(
            ["git", "-C", str(main), "status", "--porcelain", "--", path],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    if diff.returncode != 0 or (names_only and diff.stdout.strip()):
        return False
    return (
        untracked.returncode == 0
        and not untracked.stdout.strip()
        and main_dirty.returncode == 0
        and not main_dirty.stdout.strip()
    )


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
    if EXPORT_ONLY.fullmatch(text):
        return False
    loop = FOR_LIST.fullmatch(text)
    if loop:
        loops[loop.group(1)] = loop.group(2)
        return False
    env = ENV_PREFIX.match(text)
    rest = text[env.end() :] if env else text
    if not rest or DATA_ONLY.match(rest):
        return False
    tool = SWARM_TOOL_RUN.match(rest)
    if tool:
        # running the shared tooling is exempt; its arguments naming repo code are not
        return bool(_repo_paths(rest[tool.end() :], main))
    for var, items in loops.items():
        if re.search(r"\$\{?" + re.escape(var) + r"\b", text):
            text += " " + items
    return bool(
        _mentions_repo_path(text)
        or _mentions_main_code(text, main)
        or TEST_RUN.match(rest)
        or (PYTHONPATH_PWD.search(text) and RUNNER.search(rest))
    )


def is_stale(
    command: str,
    worktree: Path,
    main: Path | None,
    *,
    unchanged: Callable[[str], bool] | None = None,
) -> bool:
    """Whether the command reads or runs repo code from a tree other than the worktree."""
    loops: dict[str, str] = {}
    segments = _segments(PATH_ARG.sub("--path", command))
    # a list, not a generator: every segment must run so loop headers register their lists
    if not any([_segment_is_candidate(seg, loops, main) for seg in segments]):
        return False
    if unchanged is not None and _pure_read_of_unchanged(command, main, unchanged):
        return False
    if _in_worktree(command, worktree):
        return False
    absolute = [t for t in _cd_targets(command) if t.startswith(("/", "~"))]
    targets = [str(Path(t).expanduser()) for t in absolute]
    if any(_under(t, worktree) for t in targets):
        return False
    return not (targets and all(any(_under(t, Path(r)) for r in SCRATCH_ROOTS) for t in targets))


def stale_report(
    db: Path,
    agent: str,
    worktree: Path,
    main: Path | None,
    *,
    unchanged: Callable[[str], bool] | None = None,
) -> tuple[list[str], int]:
    """(the agent's stale commands in ordinal order, how many commands it ran)."""
    commands = agent_commands(db, agent)
    stale = [c for c in commands if is_stale(c, worktree, main, unchanged=unchanged)]
    return stale, len(commands)


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
    parser.add_argument(
        "--base", default=None, help="the revision the branch started from (else main HEAD)"
    )
    parser.add_argument("--db", default=None, help=f"T3 state DB (else ${DB_ENV})")
    args = parser.parse_args(argv)

    db = resolve_db(args.db)
    if not db.is_file():
        print(f"no T3 history db at {db} (pass --db or set {DB_ENV})", file=sys.stderr)
        return 2
    worktree = Path(args.worktree).expanduser().resolve()
    main_root = Path(args.main).expanduser().resolve() if args.main else None
    root = main_root or derive_main_root(worktree) or NO_MAIN
    stale, total = stale_report(
        db,
        args.agent,
        worktree,
        root,
        unchanged=unchanged_paths(worktree, root, base=args.base),
    )
    for command in stale:
        print(" ".join(command.split())[:SHOWN_CHARS])
    print(f"STALE {len(stale)} of {total} commands")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
