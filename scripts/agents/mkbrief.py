#!/usr/bin/env python3
"""Generate a swarm brief from a template, validate it, and print the dispatch text.

    mkbrief.py --job JOB_DIR --id CLIENT_REQUEST_ID --template FILE
               [--worktree ABS] [--scope 'glob,glob'] [--base REF] [--new NAME[,NAME] ...]
               [--review BRANCH --parent REF] [--deletes NAME,..] [--accept-outside FILE ...]
               [--set KEY=VALUE ...] [--set-file KEY=PATH ...] [--prior-from FILE[,FILE] ...]
               [--facts-from BEAD[,BEAD] ...] [--no-input-table REASON]
               [--append FILE ...] [--force]

The template uses ``{UPPER_SNAKE}`` placeholders. Auto-filled keys: ``ID``,
``JOB`` (abs job dir), ``BUS`` (abs ``<job>/bus``), ``TASK`` + ``ROUND`` (from
the id: a trailing ``-<digits>`` is the round, default 1; the task is the id
without the ``cc-``/``rev-``/``revlite-``/``impl-``/``ver-`` prefix and without the round
suffix), ``TELEMETRY`` (``TELEMETRY: task=<TASK> round=<ROUND>``; a ``rev-``/``revlite-`` id
reports ``task=review:<TASK>`` and ``ver-`` ``task=verify:<TASK>``, so a checker's run is never
a maker attempt in telemetry), ``WT``,
``RUN_AS``, ``SCOPE`` and ``GATE``.

``--review BRANCH --parent REF`` (``--parent`` defaults to ``<worktree>/.vq-base``, refused when the branch merged its parent; an explicit one always wins) additionally fills ``BASE_SHA`` (``git
merge-base REF BRANCH``), ``HEAD_SHA`` (``git rev-parse BRANCH``), ``RANGE``
and ``REVIEW_PACK``; git runs in ``--worktree`` (else the job dir).

``--base REF`` with ``--worktree`` is resolved (a valid ``<worktree>/.vq-base`` start SHA wins over it) to a commit sha at render time
(``git rev-parse --verify REF^{commit}``, in ``--worktree``), so the rendered
GATE diffs against the commit that was current when the brief was written, not
a branch that moved since. When the given ref is not already that sha, the GATE
line keeps it in a trailing ``# base REF @ render`` comment; passing the sha
itself omits the redundant comment. Without ``--worktree`` the ref is passed
through verbatim.

With ``--worktree``, ``--base`` and a non-empty ``--scope`` (not review mode),
every path changed in ``<base_sha>...HEAD`` must match a ``--scope`` glob, or the
brief exits 2 and lists the unmatched paths (a scope that omits a file the branch
already changed would fail the gate's own scope check). With no ``--scope`` the
gate SKIPs its scope check (``swarm_check.py`` ``elif cfg.scope``), so this branch
check is skipped too. The gate's rule is reused as-is
(``swarm_check.reviewable_paths`` + ``scope_offenders``), so the bus's
board/telemetry mirror files are never offenders here either.

Validation (exit 2, every problem printed to stderr, nothing written):
  * any ``{UPPER_SNAKE}`` in the raw template or ``--append``ed file text that
    has no value (values from ``--set``/``--set-file`` are never scanned, so a
    pasted code excerpt with ``{UPPER}`` in it renders fine);
  * a ``--scope`` glob that matches no file in ``--worktree`` (a glob prefixed
    with ``+`` means "new file, may not exist yet" and is skipped; the ``+`` is
    stripped from the rendered ``SCOPE``; a path the task deletes — present in
    the ``--base`` tree and gone from the worktree — or named in ``--deletes``
    is accepted);
  * ``tests/unit/**/*.py`` in ``--scope`` without ``tests/unit/*.py`` (the
    gate's fnmatch needs a subdirectory for ``**``);
  * ``--worktree`` without ``--base`` (and no usable ``<worktree>/.vq-base``, which is taken as
    the base) whose HEAD is stacked on another local branch or holds merge commits relative to
    ``main`` (the gate's default base would count that work as ours);
  * ``--worktree`` with a ``--base`` that does not resolve to a commit in it
    (the git error is printed);
  * a ``--scope`` path that differs from a worktree file only by case (macOS
    filesystems accept it, the gate's fnmatch does not), including a ``+`` new file;
  * the output file already exists (unless ``--force``).

  * A fact checked against the worktree (maker/fix briefs with ``--worktree``, never
    ``--review``): every backticked span of the pasted ``FACTS``, ``STEPS``,
    ``PRIOR_FINDINGS`` and ``RULINGS`` values that looks like a repo path (``/``, a
    source suffix, optionally ``:line``) must exist — a relative one in the worktree
    file list or as a ``--scope`` file the task creates; a backticked symbol (``name``,
    ``name()``, ``mod.name``, ``Class.method()``) containing ``_``, ``.`` or ``()``
    must appear as a whole word in a tracked ``.py/.ts/.tsx/.md/.sh/.json`` file
    (``--new NAME[,NAME]``: the task creates it, skip the hit check). The template
    text is never scanned, only the values.
  * SCOPE seeds: a symbol a non-test ``--scope`` file defines (``def``/``class``/``=``
    at column 0) whose user is a test outside SCOPE gets that test appended to SCOPE
    and the gate (``SCOPE seed: <file> (uses <name>)``); more than 8 such files adds
    none and says so. A name the task deletes is never seeded.
  * The top 5 board posts naming a literal ``--scope`` path (same ranking as
    ``bus.py board search --path``) are appended to ``FACTS`` (a fix brief: ``RULINGS``)
    as information. Appended after the checks, so their own backticks are not checked.

  * An input-classes table: a maker/fix brief whose ``--scope`` (globs expanded
    against the worktree) holds a non-test .py/.ts/.tsx file whose stem matches
    ``pars|valid|load|rout|schema|config`` must carry a Markdown table (delimiter row;
    outer pipes optional) with an ``input class`` header cell and >= 3 body rows in
    ``FACTS`` or ``STEPS``; checked only when the template has {FACTS} or {STEPS}.
    ``--no-input-table REASON`` skips it and writes ``- No input table: REASON`` to FACTS;
    it and ``--facts-from`` exit 2 on a template without {FACTS}.

  * ``--facts-from BEAD[,BEAD]`` prepends, per bead, ``- Bead <id> (<title>):`` and the
    ``bd show <id> --json`` description lines indented by 2 spaces (verbatim) to
    ``FACTS``, before a ``--set-file FACTS``. ``bd`` missing or failing exits 2 with the
    stderr tail.

  * ``--deletes NAME[,NAME]`` (also ``delete `NAME` `` in FACTS): a test file outside
    ``--scope`` that uses one of those exact names, unless ``--accept-outside FILE`` lists it.
    A scoped def/class of that name holding a swallow (``find_swallows`` of
    ``tests/unit/test_exception_hygiene.py``) adds that test to ``SCOPE`` (the ratchet moves).

``--review`` without ``--parent`` uses the start SHA ``worktree.sh`` wrote to ``<worktree>/.vq-base``
as the range base; it exits 2 when that SHA is not an ancestor of the branch or the branch merged
its parent since (rebase or pass ``--parent``). An explicit ``--parent`` gives
``merge-base REF BRANCH`` (the recorded SHA only when that parent already contains the branch). Exit 2 on an empty range.

A leading ``<!-- ... -->`` usage comment (and the blank lines after it) is
stripped from the template and every appended file before rendering, so the
generator's own usage notes never land in a brief.

A line ending in ``<!-- if TAG -->`` in the template or an appended file is kept
(tag removed) only when the brief uses it: ``frontend`` when a rendered SCOPE
glob starts with ``frontend/`` or ``e2e/``, ``peers`` when ``PEERS`` is set to
something other than ``none``/empty. An unknown tag exits 2, so a typo is never
a silently dropped rule.

``--review`` cuts a ``SPEC`` that is a maker/fixer brief (a ``ROLE: maker`` or
``ROLE: fixer`` line) down to its GOAL/SCOPE/FACTS/STEPS/DONE WHEN/PRIOR
FINDINGS/RULINGS sections, dropping RULES, RETURN, GATE, WORKTREE and any SWARM
BUS block: the reviewer has its own. A plain spec file passes through unchanged.

``--prior-from FILE[,FILE]`` (repeatable; reviewer handoff files) fills
``PRIOR_FINDINGS`` with, per file, a ``From <basename> (verdict <V>):`` header
and only the ``open:`` field's bullets, verbatim and with their fenced repro
blocks — ``claims:``, ``evidence:``, ``artifacts:`` and ``improve:`` are dropped
(a pasted whole handoff reached 13.7k chars a fixer did not need). The total is
capped at 6000 chars: a bullet that does not fit is dropped WHOLE (never cut
inside one, so a fence never breaks the rest of the brief) and the text ends
with ``(+N findings cut: read <file>)``. Exit 2 when the first bullet alone is
over 6000, when a file's fences are unbalanced, or when ``--set-file
PRIOR_FINDINGS=`` is also given.

On success it writes ``<job>/prompts/<id>.md`` and prints, on stdout:

    Read your brief at <abs path> and follow it exactly.
    TELEMETRY: task=<task> round=<round>

so the file always exists before the dispatch that points at it, and the
TELEMETRY line is always in the dispatched text. Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

_PREFIXES = ("cc-", "rev-", "revlite-", "impl-", "ver-")
# a checker's run is telemetry task `review:<t>` / `verify:<t>`, never a maker attempt of <t>
_CHECKER_TASK = {"rev-": "review:", "revlite-": "review:", "ver-": "verify:"}
_ROUND_RE = re.compile(r"-(\d+)$")
_PLACEHOLDER_RE = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
_LEADING_COMMENT_RE = re.compile(r"\A\s*<!--.*?-->[ \t]*", re.DOTALL)
_BLANK_LINES_RE = re.compile(r"\A(?:[ \t]*\n)+")
_CONDITIONAL_RE = re.compile(
    r"^(?P<body>.*?)[ \t]*<!--[ \t]*if[ \t]+(?P<tag>[\w-]+)[ \t]*-->[ \t]*$"
)
_CONDITIONAL_TAGS = ("frontend", "peers")
_SPEC_ROLE_RE = re.compile(r"^ROLE:\s+(?:maker|fixer)\b", re.MULTILINE)
_SPEC_SECTION_RE = re.compile(
    r"^(GOAL|SCOPE|FACTS|STEPS|DONE WHEN|PRIOR FINDINGS|RULINGS|RULES|RETURN|GATE|WORKTREE"
    r"|SWARM BUS)(?=:|,|\s*$|\s+\(|\s+\u2264)"  # a header, not prose that starts with the word
)
_SPEC_KEEP = frozenset(
    {"GOAL", "SCOPE", "FACTS", "STEPS", "DONE WHEN", "PRIOR FINDINGS", "RULINGS"}
)
_BROAD_UNIT_GLOB = "tests/unit/**/*.py"
_UNIT_TOP_GLOB = "tests/unit/*.py"
_UNIT_HINT = f"glob {_BROAD_UNIT_GLOB} does not match top-level tests/unit files; add {_UNIT_TOP_GLOB}"

_PRIOR_CAP = 6000
_HANDOFF_HEADING = re.compile(r"^##\s+handoff\s*$", re.IGNORECASE)
_ANY_HEADING = re.compile(r"^#{1,6}\s")
_FIELD_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*\s*:")
_OPEN_LINE = re.compile(r"^open\s*:\s*(.*)$", re.IGNORECASE)
_VERDICT_LINE = re.compile(r"^verdict\s*:\s*(.*)$", re.IGNORECASE)
_BULLET_LINE = re.compile(r"^[ \t]*-[ \t]")

_INPUT_STEM_RE = re.compile(r"pars|valid|load|rout|schema|config", re.IGNORECASE)
_INPUT_HEADER_RE = re.compile(r"\binput\s+class\b", re.IGNORECASE)
_TABLE_CELL_SEP_RE = re.compile(r":?-+:?")
_INPUT_CODE_SUFFIXES = (".py", ".ts", ".tsx")
_INPUT_MIN_ROWS = 3
_BD_TIMEOUT = 20
_BD_TAIL = 400

_PYTHON = "/Users/verebelyin/projects/vibe-quant/.venv/bin/python"
_GATE = "scripts/agents/swarm-check.sh"


def parse_id(client_id: str) -> tuple[str, int]:
    """Split a client request id into (task, round). See the module docstring."""
    task = client_id
    round_no = 1
    match = _ROUND_RE.search(task)
    if match:
        round_no = int(match.group(1))
        task = task[: match.start()]
    for prefix in _PREFIXES:
        if task.startswith(prefix):
            task = task[len(prefix) :]
            break
    return task, round_no


def _checker_label(client_id: str) -> str:
    return next((label for prefix, label in _CHECKER_TASK.items() if client_id.startswith(prefix)), "")


def render(text: str, values: dict[str, str]) -> str:
    """Replace every ``{UPPER_SNAKE}`` known in ``values``; leave the rest as-is."""
    return _PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def unfilled(sources: Iterable[str], values: dict[str, str]) -> list[str]:
    """``{UPPER_SNAKE}`` keys present in the raw template/appends with no value.

    Only the raw template text and raw ``--append``ed file text are scanned —
    never the values substituted for ``--set``/``--set-file``, which are code
    excerpts and may legitimately contain ``{UPPER}`` text.
    """
    keys = {m.group(1) for source in sources for m in _PLACEHOLDER_RE.finditer(source)}
    return sorted(key for key in keys if key not in values)


def strip_leading_comment(text: str) -> str:
    """Drop a leading ``<!-- ... -->`` block and the blank lines after it."""
    match = _LEADING_COMMENT_RE.match(text)
    if not match:
        return text
    return _BLANK_LINES_RE.sub("", text[match.end() :])


def conditional_lines(
    text: str, enabled: dict[str, bool]
) -> tuple[str, list[str]]:
    """Keep or drop lines ending in ``<!-- if TAG -->``; return (text, unknown tags).

    A kept line loses the tag. An unknown tag is returned rather than guessed at,
    so a typo in a template is refused instead of silently dropping the rule."""
    kept: list[str] = []
    unknown: list[str] = []
    for line in text.splitlines():
        match = _CONDITIONAL_RE.match(line)
        if not match:
            kept.append(line)
            continue
        tag = match.group("tag")
        if tag not in _CONDITIONAL_TAGS:
            unknown.append(tag)
            kept.append(line)
        elif enabled.get(tag):
            kept.append(match.group("body").rstrip())
    rendered = "\n".join(kept)
    return (rendered + "\n", unknown) if text.endswith("\n") else (rendered, unknown)


def maker_spec_excerpt(text: str) -> str:
    """A maker/fixer brief pasted as a review ``SPEC``: only its task sections.

    The reviewer already has its own RULES/RETURN/GATE/WORKTREE and, when it is
    dispatched with the bus protocol, its own SWARM BUS block: repeating the
    maker's was half of a review brief. Anything that is not a maker/fixer brief
    (``ROLE: maker``/``ROLE: fixer``) passes through unchanged."""
    if not _SPEC_ROLE_RE.search(text):
        return text
    kept: list[str] = []
    section: str | None = None
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            match = _SPEC_SECTION_RE.match(line)
            if match:
                section = match.group(1)
        if section in _SPEC_KEEP:
            kept.append(line)
    return "\n".join(kept).strip() + "\n"


def handoff_block(text: str) -> list[str]:
    """Lines of the last ``## Handoff`` block (empty when the file has none)."""
    lines = text.splitlines()
    prose: list[bool] = []  # outside a ``` fence: a `# comment` in a bash repro is no heading
    in_fence = False
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            prose.append(False)
        else:
            prose.append(not in_fence)
    starts = [i for i, line in enumerate(lines) if prose[i] and _HANDOFF_HEADING.match(line)]
    if not starts:
        return []
    start = starts[-1] + 1
    end = next(
        (i for i in range(start, len(lines)) if prose[i] and _ANY_HEADING.match(lines[i])),
        len(lines),
    )
    return lines[start:end]


def open_bullets(block: list[str]) -> list[str]:
    """The ``open:`` field's bullets, each with its fenced repro block, verbatim.

    A bullet starts at a ``- `` line outside a fence and runs to the next one; the
    ``open:`` field ends at the next top-level field or heading outside a fence."""
    section: list[str] = []
    started = in_fence = False
    for line in block:
        if started and not in_fence and (_FIELD_LINE.match(line) or _ANY_HEADING.match(line)):
            break
        if line.startswith("```"):
            in_fence = not in_fence
        if started:
            section.append(line)
        elif match := _OPEN_LINE.match(line):
            started = True
            inline = match.group(1).strip()
            if inline and inline.lower() != "none":
                section.append("  - " + inline)
    bullets: list[str] = []
    current: list[str] | None = None
    in_fence = False
    for line in section:
        if not in_fence and _BULLET_LINE.match(line):
            if current is not None:
                bullets.append("\n".join(_without_blank_tail(current)))
            current = [line]
        elif current is not None:
            current.append(line)
        if line.startswith("```"):
            in_fence = not in_fence
    if current is not None:
        bullets.append("\n".join(_without_blank_tail(current)))
    return bullets


def _without_blank_tail(lines: list[str]) -> list[str]:
    end = len(lines)
    while end and not lines[end - 1].strip():
        end -= 1
    return lines[:end]


def prior_piece(path: Path) -> tuple[list[str], list[str]]:
    """(header + open bullets, []) for one reviewer handoff, or ([], [problem]).

    Only ``open:`` survives: ``claims:``/``evidence:``/``artifacts:``/``improve:`` are
    the reviewer's reasoning, and pasting them made a fixer wade through 13.7k chars."""
    try:
        text = path.read_text()
    except OSError:
        return [], [f"prior-from: cannot read {path}"]
    block = handoff_block(text)
    if not block:
        return [], [f"prior-from: no '## Handoff' block in {path}"]
    verdict = next(
        (match.group(1).strip() for line in block if (match := _VERDICT_LINE.match(line))),
        "none",
    )
    lines = [f"From {path.name} (verdict {verdict}):", *open_bullets(block)]
    if sum(1 for line in "\n".join(lines).splitlines() if line.startswith("```")) % 2:
        return [], [f"prior-from: unbalanced code fence in {path}"]
    return lines, []


def prior_findings(paths: list[str]) -> tuple[str, list[str]]:
    """``PRIOR_FINDINGS`` from reviewer handoffs: open bullets only, capped at 6000.

    Bullets are dropped from the tail WHOLE (a half fence breaks the rest of the
    brief), and the surviving text ends with ``(+N findings cut: read <file>)``."""
    entries: list[tuple[str, str, bool]] = []  # (text, source file, is a finding)
    for raw in paths:
        path = Path(raw)
        lines, problems = prior_piece(path)
        if problems:
            return "", problems
        entries.append((lines[0], str(path), False))
        entries.extend((bullet, str(path), True) for bullet in lines[1:])
    first = next((entry for entry in entries if entry[2]), None)
    if first is not None and len(first[0]) > _PRIOR_CAP:
        return "", [
            f"prior-from: first finding of {first[1]} is {len(first[0])} chars"
            f" (> {_PRIOR_CAP}); trim by hand with --set-file"
        ]
    kept = entries
    dropped: list[tuple[str, str, bool]] = []
    while True:
        text = "\n".join(entry[0] for entry in kept)
        cut = sum(1 for entry in dropped if entry[2])
        if cut:
            sources = sorted({e[1] for e in dropped if e[2]}, key=[str(Path(raw)) for raw in paths].index)
            text += f"\n(+{cut} findings cut: read {', '.join(sources)})"
        if len(text) <= _PRIOR_CAP or not any(entry[2] for entry in kept):
            return text, []
        dropped.append(kept.pop())


def parse_scope(raw: str) -> tuple[list[str], list[str]]:
    """Split a ``--scope`` string into (rendered globs, globs to check).

    A ``+`` prefix marks a not-yet-existing file: it is stripped from the
    rendered glob and excluded from the "matches a file" check.
    """
    rendered: list[str] = []
    checked: list[str] = []
    for part in raw.split(","):
        glob = part.strip()
        if not glob:
            continue
        if glob.startswith("+"):
            rendered.append(glob[1:])
        else:
            rendered.append(glob)
            checked.append(glob)
    return rendered, checked


def worktree_files(root: Path) -> list[str]:
    """Tracked + untracked-not-ignored files (``git ls-files``); a non-git dir is walked.

    A full walk of the main checkout (data/, node_modules/) made one brief take 23.3 s; this 0.25 s."""
    key = (str(root), ("ls-files", "-co", "--exclude-standard"))
    if _MEMO is not None and key in _MEMO:
        return list(_MEMO[key])
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    files: list[str] = []
    if proc.returncode == 0:
        # split, never strip: a file name may begin or end with a space
        files = [line for line in proc.stdout.splitlines() if line]
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for name in filenames:
                files.append(Path(dirpath, name).relative_to(root).as_posix())
    if _MEMO is not None:
        _MEMO[key] = list(files)
    return files


def base_tree_files(root: Path, base: str) -> list[str]:
    """Paths in ``base``'s tree (``git ls-tree``); [] when git cannot read it."""
    key = (str(root), ("ls-tree", "-r", "--name-only", "-z", base))
    if _MEMO is not None and key in _MEMO:
        return list(_MEMO[key])
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "-z", base],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    paths = [line for line in proc.stdout.split("\0") if line] if proc.returncode == 0 else []
    if _MEMO is not None:
        _MEMO[key] = list(paths)
    return paths


def scope_problems(
    rendered: Iterable[str],
    checked: Iterable[str],
    worktree: str,
    base: str = "",
    deletes: Iterable[str] = (),
) -> list[str]:
    """Glob problems: nothing matched, or the broad top-level tests glob footgun.

    A path the task deletes cannot match a worktree file, so it also matches a
    ``base`` tree path that is gone from the worktree, or a ``--deletes`` name.
    """
    rendered = list(rendered)
    problems: list[str] = []
    if _BROAD_UNIT_GLOB in rendered and _UNIT_TOP_GLOB not in rendered:
        problems.append(_UNIT_HINT)
    if worktree:
        root = Path(worktree)
        files = worktree_files(root)
        gone = list(deletes)
        deleted: list[str] | None = None
        for glob in checked:
            if glob == _BROAD_UNIT_GLOB:
                continue  # covered by the dedicated hint above
            if any(fnmatch.fnmatch(path, glob) for path in files):
                continue
            if any(fnmatch.fnmatch(name, glob) for name in gone):
                continue
            if base and deleted is None:
                present = set(files)
                deleted = [path for path in base_tree_files(root, base) if path not in present]
            if deleted and any(fnmatch.fnmatch(path, glob) for path in deleted):
                continue
            problems.append(
                f"glob {glob} matches no file in {worktree} (prefix + for a file you create)"
            )
    return problems


def case_problems(rendered: Iterable[str], worktree: str) -> list[str]:
    """Literal scope paths that exist only under another case (sizingChip.test.ts vs SizingChip)."""
    if not worktree:
        return []
    by_lower: dict[str, str] = {}
    for path in worktree_files(Path(worktree)):
        by_lower.setdefault(path.lower(), path)
    problems: list[str] = []
    files = set(by_lower.values())
    for glob in rendered:
        if any(c in glob for c in "*?["):
            continue
        actual = by_lower.get(glob.lower())
        if actual is not None and actual != glob and glob not in files:
            problems.append(f"scope path {glob} differs only by case from {actual}")
    return problems


_DELETE_FACT_RE = re.compile(r"\bdelete[sd]?\s+`(\w+)`")
_HYGIENE_TEST = "tests/unit/test_exception_hygiene.py"


def deleted_names(deletes: str, facts: str) -> set[str]:
    """Names from ``--deletes`` plus ``delete `NAME` `` mentions in the FACTS text."""
    names = {n.strip() for n in deletes.split(",") if n.strip()}
    return names | set(_DELETE_FACT_RE.findall(facts))


def tests_using_deleted(
    worktree: str, rendered: Iterable[str], names: Iterable[str], accepted: Iterable[str]
) -> list[tuple[str, str]]:
    """(test file, name) for tests outside scope (and not accepted) that use a deleted name exactly."""
    names = sorted(names)
    if not worktree or not names:
        return []
    root = Path(worktree)
    globs, ok = list(rendered), set(accepted)
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b")
    hits: list[tuple[str, str]] = []
    for rel in worktree_files(root):
        is_test = (rel.startswith("tests/") and rel.endswith(".py")) or rel.endswith(
            (".test.ts", ".test.tsx")
        )
        if not is_test or rel in ok or "node_modules/" in rel:
            continue
        if any(fnmatch.fnmatch(rel, g) for g in globs):
            continue
        found = pattern.search((root / rel).read_text(encoding="utf-8", errors="replace"))
        if found:
            hits.append((rel, found.group(1)))
    return sorted(hits)


def moves_swallow(worktree: Path, rendered: Iterable[str], names: Iterable[str]) -> bool:
    """True when a scoped source def/class named in ``names`` contains a swallow the exception
    ratchet counts (``find_swallows`` of tests/unit/test_exception_hygiene.py, loaded from the worktree)."""
    hygiene = worktree / _HYGIENE_TEST
    names = set(names)
    if not names or not hygiene.is_file():
        return False
    spec = importlib.util.spec_from_file_location("_mkbrief_hygiene", hygiene)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # NamedTuple/dataclass resolve via sys.modules
    spec.loader.exec_module(module)
    globs = list(rendered)
    for rel in worktree_files(worktree):
        if not rel.endswith(".py") or "test" in Path(rel).name:
            continue
        if not any(fnmatch.fnmatch(rel, g) for g in globs):
            continue
        text = (worktree / rel).read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name in names
            ):
                span = textwrap.dedent("\n".join(lines[node.lineno - 1 : node.end_lineno]))
                if module.find_swallows(span):
                    return True
    return False


_FACT_KEYS = ("FACTS", "STEPS", "PRIOR_FINDINGS", "RULINGS")
_FACT_SPAN_RE = re.compile(r"`([^`\n]+)`")
# A line that says its spans exist nowhere BY DESIGN (a mutant label, a name the tests must not
# have, a file a fixture builds in tmp, an output the run writes) is not checked (rev-facts-1).
_FACT_CONTEXT_SKIP_RE = re.compile(
    r"\b(?:delete[sd]?|removed?|absent|renamed?|was|were|old|other than|histor\w*|mutants?"
    r"|labels?|fixtures?|tmp|temp\w*|builds?|writes?|creates?|new)\b",
    re.IGNORECASE,
)
_FACT_ASSIGN_RE = re.compile(r"^[A-Z][A-Z0-9_]*=(.+)$")
_FACT_LINE_SUFFIX_RE = re.compile(r":\d+(?:-\d+)?$")
_FACT_PATH_SUFFIXES = (".py", ".md", ".ts", ".tsx", ".json", ".jsonl", ".sh", ".toml", ".yaml", ".yml")
_FACT_PATH_FORBIDDEN = frozenset("*?[<{ \t")
_NAME_SPAN_RE = re.compile(r"[A-Za-z_]\w*(?:\.\w+)*(?:\(\))?")
_NAME_FILE_RE = re.compile(r"\.(?:py|md|ts|tsx|json|jsonl|sh|toml|ya?ml)$")
_NAME_SCAN_SUFFIXES = (".py", ".ts", ".tsx", ".md", ".sh", ".json")
_SEED_MAX = 8
_BOARD_TOP = 5
_BOARD_BODY = 160


def fact_spans(values: dict[str, str]) -> list[str]:
    """Every distinct backtick span of the pasted FACTS/STEPS/PRIOR_FINDINGS/RULINGS text.

    Only the values, never the template: a span the maker is told about must be real."""
    spans: list[str] = []
    for key in _FACT_KEYS:
        for line in values.get(key, "").splitlines():
            if _FACT_CONTEXT_SKIP_RE.search(line):
                continue
            for span in _FACT_SPAN_RE.findall(line):
                if span not in spans:
                    spans.append(span)
    return spans


def fact_path(span: str) -> str | None:
    """The repo path a span names (``a/b.py``, ``a/b.py:12-30``), or None for a non-path.

    A path needs a ``/`` and a source suffix; a glob span and a bare file name do not
    qualify (the maker cites a file the task touches, or something to create in SCOPE)."""
    text = _FACT_LINE_SUFFIX_RE.sub("", span.strip())
    assigned = _FACT_ASSIGN_RE.match(text)
    if assigned:  # `ENV=path`: the value is the path
        text = assigned.group(1)
    if text.startswith(("/tmp/", "/private/tmp/")):  # scratch output the run writes
        return None
    if "/" not in text or any(char in text for char in _FACT_PATH_FORBIDDEN):
        return None
    return text if text.endswith(_FACT_PATH_SUFFIXES) else None


def fact_path_problems(spans: Iterable[str], worktree: str, rendered: Iterable[str]) -> list[str]:
    """A fact's backticked path must exist, or be a file the brief's SCOPE creates."""
    root = Path(worktree)
    files = set(worktree_files(root))
    globs = list(rendered)
    problems: list[str] = []
    for span in spans:
        path = fact_path(span)
        if path is None:
            continue
        found = (
            Path(path).exists()
            if Path(path).is_absolute()
            else path in files or any(fnmatch.fnmatch(path, glob) for glob in globs)
        )
        if not found:
            problems.append(
                f"fact path {span} not found in {worktree}"
                " (a file the task creates goes into --scope with +)"
            )
    return problems


def fact_names(spans: Iterable[str], new: Iterable[str]) -> list[str]:
    """The symbol each fact span names, minus the ones ``--new`` says the task creates."""
    skip = set(new)
    names: list[str] = []
    for span in spans:
        text = span.strip()
        if not _NAME_SPAN_RE.fullmatch(text) or _NAME_FILE_RE.search(text):
            continue
        if not ("_" in text or "." in text or text.endswith("()")):
            continue
        name = text.rsplit(".", 1)[-1].removesuffix("()")
        if name and name not in skip and name not in names:
            names.append(name)
    return names


def name_hits(root: Path, names: Iterable[str]) -> dict[str, list[str]]:
    """{name: [files that use it as a whole word]} in ONE pass over the worktree's sources.

    A per-name ``rg`` costs a spawn each; one alternation regex over the tracked
    .py/.ts/.tsx/.md/.sh/.json files answers every name at once."""
    wanted = sorted(set(names), key=len, reverse=True)
    hits: dict[str, list[str]] = {name: [] for name in wanted}
    if not wanted:
        return hits
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, wanted)) + r")\b")
    for rel in worktree_files(root):
        if not rel.endswith(_NAME_SCAN_SUFFIXES) or "node_modules/" in rel:
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name in set(pattern.findall(text)):
            if name in hits:
                hits[name].append(rel)
    return hits


def is_test_path(rel: str) -> bool:
    """A test file: under tests/ (Python) or a ``*.test.ts``/``*.test.tsx``."""
    return (rel.startswith("tests/") and rel.endswith(".py")) or rel.endswith(
        (".test.ts", ".test.tsx")
    )


def _input_table_scope_paths(rendered: Iterable[str], files: Iterable[str] = ()) -> list[str]:
    """Non-test .py/.ts/.tsx SCOPE paths whose stem names a parser/validator/loader/router.

    A glob is expanded against ``files`` (the worktree's); a literal path (also a ``+new`` one,
    which no worktree file matches yet) counts as itself."""
    files = list(files)
    paths: list[str] = []
    for glob in rendered:
        if any(char in glob for char in "*?["):
            matches = [rel for rel in files if fnmatch.fnmatch(rel, glob)]
        else:
            matches = [glob]
        for rel in matches:
            if (
                rel.endswith(_INPUT_CODE_SUFFIXES)
                and not is_test_path(rel)
                and _INPUT_STEM_RE.search(Path(rel).stem)
                and rel not in paths
            ):
                paths.append(rel)
    return paths


def _table_cells(line: str) -> list[str] | None:
    """The cells of a Markdown table row (outer pipes optional); None for a line without ``|``."""
    if "|" not in line:
        return None
    text = line.strip()
    text = text.removeprefix("|").removesuffix("|")
    return [cell.strip() for cell in text.split("|")]


def _is_table_delimiter(cells: list[str] | None) -> bool:
    return cells is not None and all(_TABLE_CELL_SEP_RE.fullmatch(cell) for cell in cells)


def has_input_table(text: str) -> bool:
    """True when ``text`` holds a Markdown table (header, delimiter row, body rows; outer pipes
    optional) with an ``input class`` header cell and at least three body rows."""
    lines = text.splitlines()
    for index in range(len(lines) - 1):
        header = _table_cells(lines[index])
        if header is None or not any(_INPUT_HEADER_RE.search(cell) for cell in header):
            continue
        if not _is_table_delimiter(_table_cells(lines[index + 1])):
            continue
        body = 0
        row = index + 2
        while row < len(lines) and _table_cells(lines[row]) is not None:
            body += 1
            row += 1
        if body >= _INPUT_MIN_ROWS:
            return True
    return False


def input_table_problems(
    rendered: Iterable[str], values: dict[str, str], files: Iterable[str] = ()
) -> list[str]:
    """A parser/validator/loader/router brief must list its input classes in FACTS or STEPS."""
    text = values.get("FACTS", "") + "\n" + values.get("STEPS", "")
    if has_input_table(text):
        return []
    return [
        f'SCOPE touches {path} (parser/validator/loader/router): add an "input class | expected"'
        f" table with >= 3 rows to FACTS or STEPS"
        for path in _input_table_scope_paths(rendered, files)
    ]


def _bd_show(bead: str) -> tuple[dict[str, Any], str]:
    """One bead from ``bd show <id> --json``: (object, "") or ({}, problem with stderr tail)."""
    try:
        proc = subprocess.run(
            ["bd", "show", bead, "--json"], capture_output=True, text=True,
            timeout=_BD_TIMEOUT, check=False,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, f"facts-from: bd show {bead} failed: {str(exc).strip()[-_BD_TAIL:]}"
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-_BD_TAIL:]
        return {}, f"facts-from: bd show {bead} failed: {tail}"
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {}, f"facts-from: bd show {bead} printed no JSON: {exc}"
    objects = data if isinstance(data, list) else [data]
    if not objects or not isinstance(objects[0], dict):
        return {}, f"facts-from: bd show {bead} printed no bead"
    return objects[0], ""


def bead_facts(bead: str) -> tuple[list[str], list[str]]:
    """(FACTS lines, problems) for one bead: a header and its description, verbatim."""
    obj, problem = _bd_show(bead)
    if problem:
        return [], [problem]
    block = [f"- Bead {bead} ({obj.get('title', '')}):"]
    block.extend(
        f"  {line}" if line else "" for line in str(obj.get("description", "")).splitlines()
    )
    return block, []


def facts_from_beads(beads: Iterable[str]) -> tuple[str, list[str]]:
    """Bead text to PREPEND to FACTS (before ``--set-file FACTS``), one block per bead."""
    lines: list[str] = []
    for bead in beads:
        block, problems = bead_facts(bead)
        if problems:
            return "", problems
        lines.extend(block)
    return "\n".join(lines), []


def defines(text: str, name: str) -> bool:
    """True when ``text`` defines ``name`` at column 0 (``def``/``class``/``name =``)."""
    escaped = re.escape(name)
    return bool(
        re.search(rf"^(?:def|class)\s+{escaped}\b", text, re.MULTILINE)
        or re.search(rf"^{escaped}\s*=", text, re.MULTILINE)
    )


def scope_seeds(
    worktree: str,
    rendered: Iterable[str],
    names: Iterable[str],
    hits: dict[str, list[str]],
    excluded: Iterable[str],
) -> tuple[list[str], list[str]]:
    """(test files to append to SCOPE, stderr lines) for a changed symbol's outside users.

    A maker asked for a SCOPE extension for such a test 20% of the time; the seed answers
    it. More than ``_SEED_MAX`` users is too much noise, so none is added. A name the
    task deletes is never seeded: its users stay a deliberate SCOPE decision."""
    root = Path(worktree)
    skip = set(excluded)
    globs = list(rendered)
    in_scope = [rel for rel in worktree_files(root) if any(fnmatch.fnmatch(rel, g) for g in globs)]
    in_scope_set = set(in_scope)
    checked = [name for name in names if name not in skip]
    # name -> (module stem, is python) of each scope file defining it: a user must name that
    # module too and be a test of the same language, else a common name (`check`) seeds noise
    defined: dict[str, set[tuple[str, bool]]] = {}
    for rel in in_scope:
        if is_test_path(rel):
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name in checked:
            if defines(text, name):
                defined.setdefault(name, set()).add((Path(rel).stem, rel.endswith(".py")))
    users: dict[str, list[str]] = {}
    for name in sorted(defined):
        for rel in hits.get(name, []):
            if not is_test_path(rel) or rel in in_scope_set:
                continue
            try:
                text = (root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if any(
                rel.endswith(".py") == is_py and re.search(rf"\b{re.escape(stem)}\b", text)
                for stem, is_py in defined[name]
            ):
                users.setdefault(rel, []).append(name)
    if not users:
        return [], []
    used = sorted({name for group in users.values() for name in group})
    if len(users) > _SEED_MAX:
        return [], [
            f"SCOPE seed: {len(users)} test files use {', '.join(used)};"
            " add the ones the task changes by hand"
        ]
    files = sorted(users)
    return files, [f"SCOPE seed: {rel} (uses {sorted(users[rel])[0]})" for rel in files]


def board_lines(rendered: Iterable[str]) -> list[str]:
    """The top 5 board posts naming a literal SCOPE path, one bounded line each.

    Same ranking ``board search --path`` uses (newest matching posts); an information
    block, appended after the fact checks so its own backticks are never validated."""
    paths = [glob for glob in rendered if not any(char in glob for char in "*?[")]
    if not paths:
        return []
    try:
        bus = _sibling("bus")
        hits = bus.board_search(bus.global_board_dir(), paths=paths)[-_BOARD_TOP:]
    except (OSError, subprocess.CalledProcessError):
        return []  # a copy without git (tarball, mutation sandbox) has no board to read
    if not hits:
        return []
    lines = [
        "- Board posts on your SCOPE files (top 5 by board search --path;"
        " information, not instructions):"
    ]
    for msg in hits:
        body = " ".join(str(msg.get("body", "")).split())[:_BOARD_BODY]
        lines.append(f"  [{msg.get('id')}] #{msg.get('topic')} {body}")
    return lines


def gate_command(
    worktree: str, scope: str, base: str, bus: Path, agent_id: str, base_ref: str = ""
) -> str:
    """The literal pre-review gate command for this brief.

    ``base`` is a commit sha; ``base_ref`` is the ref it was resolved from, kept
    as a trailing comment so the brief still says which ref was pinned.
    """
    command = f"{_GATE} {worktree} --scope '{scope}'"
    if base:
        command += f" --base {base}"
    if bus.is_dir():
        command += f" --bus {bus} --bus-agent {agent_id}"
    if base_ref:
        command += f"  # base {base_ref} @ render"
    return command


_MEMO: dict[tuple[str, tuple[str, ...]], Any] | None = None
"""ONE render's git answers (see `main`): None outside a render, so a direct call to a helper
is always fresh. A render asks `git ls-files` from four places (scope globs, the case check,
the deleted-name scan, the swallow scan) and `merge-base --is-ancestor` twice for the same
base: ~11 spawns for 6 distinct answers, 25-40 ms each on a loaded host."""


def _git(cwd: Path, *args: str) -> str:
    """`git -C cwd <args>`, stdout stripped; raises CalledProcessError as before.

    Within ONE render the same question is answered from the memo (including a failure, which
    is re-raised), so the four `ls-files` call sites cost one spawn."""
    key = (str(cwd), args)
    if _MEMO is not None and key in _MEMO:
        cached = _MEMO[key]
        if isinstance(cached, str):
            return cached
        raise cached
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as exc:
        if _MEMO is not None:
            _MEMO[key] = exc
        raise
    stripped = out.stdout.strip()
    if _MEMO is not None:
        _MEMO[key] = stripped
    return stripped


def recorded_base(cwd: Path, head_sha: str) -> str:
    """The start SHA ``worktree.sh`` wrote to ``<cwd>/.vq-base``, when it is an ancestor of
    ``head_sha`` (a moving parent branch would give the wrong or an empty range); else ""."""
    try:
        sha = (cwd / ".vq-base").read_text().strip()
        _git(cwd, "merge-base", "--is-ancestor", sha, head_sha)
    except (subprocess.CalledProcessError, OSError):
        return ""
    return sha


def _gate_module() -> Any:
    """``swarm_check.py`` (same directory) loaded by path — the gate's own scope rule, never a copy."""
    name = "_mkbrief_swarm_check"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name("swarm_check.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its @dataclass decorators resolve via sys.modules
    spec.loader.exec_module(module)
    return module


def branch_scope_offenders(worktree: Path, base_sha: str, globs: Iterable[str]) -> list[str]:
    """Paths changed in ``base_sha...HEAD`` that no ``--scope`` glob covers.

    Uses the gate's own rule (``swarm_check.scope_offenders`` over
    ``reviewable_paths``, the same pair the gate applies to its diff), so board
    and telemetry mirror files are excluded and a brief is never rejected for a
    path the gate itself would ignore.
    """
    gate = _gate_module()
    changed = [
        path
        for path in _git(
            worktree, "diff", "--name-only", "--no-renames", "-z", f"{base_sha}...HEAD"
        ).split("\0")
        if path
    ]
    return list(gate.scope_offenders(gate.reviewable_paths(changed), list(globs)))


def stacked_candidates(worktree: Path) -> list[str]:
    """Fork-point candidates when ``worktree`` HEAD is not a plain branch cut from main, else [].

    Stacked = main..HEAD holds a merge commit, or the tip of another local branch, also one at HEAD (work the
    gate's default base would count as ours). Candidates are ``<branch> <sha>`` / merge-base.
    Not a git checkout, or no ``main`` ref: [] (nothing to compare against).
    """
    try:
        own = _git(worktree, "rev-parse", "--abbrev-ref", "HEAD")
        _git(worktree, "rev-parse", "--verify", "-q", "main")
        merges = _git(worktree, "rev-list", "--merges", "main..HEAD")
        inside = set(_git(worktree, "rev-list", "main..HEAD").split())
        refs = _git(worktree, "for-each-ref", "refs/heads", "--format=%(refname:short) %(objectname)")
        fork = _git(worktree, "merge-base", "main", "HEAD")
    except (subprocess.CalledProcessError, OSError):
        return []
    found = []
    for ref in refs.splitlines():
        name, _, sha = ref.partition(" ")
        if sha in inside and name not in (own, "main"):
            found.append(f"{name} {sha}")
    if found:
        return found
    return [f"main {fork}"] if merges else []


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mkbrief.py", description="Generate and validate a swarm brief."
    )
    parser.add_argument("--job", required=True, help="job directory (data/swarm/<job>)")
    parser.add_argument("--id", required=True, help="client request id (also the brief filename)")
    parser.add_argument("--template", required=True, help="template file with {UPPER_SNAKE} keys")
    parser.add_argument("--worktree", default="", help="worktree the brief works in")
    parser.add_argument("--scope", default="", help="comma-separated SCOPE globs")
    parser.add_argument(
        "--base", default="", help="gate --base ref (pinned to a sha when --worktree is given)"
    )
    parser.add_argument("--review", default="", help="review mode: the branch under review")
    parser.add_argument("--parent", default="", help="review mode: the ref to merge-base against")
    parser.add_argument(
        "--deletes", default="", metavar="NAME[,NAME]",
        help="symbols the task deletes or moves: tests outside SCOPE using one fail rendering "
        "(also read from FACTS 'delete `NAME`'); adds test_exception_hygiene.py when one holds a swallow",
    )  # fmt: skip
    parser.add_argument(
        "--accept-outside", action="append", default=[], metavar="FILE",
        help="test file outside SCOPE that uses a deleted name and is knowingly left alone",
    )  # fmt: skip
    parser.add_argument(
        "--new", action="append", default=[], metavar="NAME[,NAME]",
        help="symbols the task creates: their fact-name check is skipped (repeatable)",
    )  # fmt: skip
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--set-file", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument(
        "--prior-from", action="append", default=[], metavar="FILE[,FILE]",
        help="reviewer handoff file(s): PRIOR_FINDINGS = their open: bullets only "
        "(claims/evidence/artifacts/improve dropped), whole-bullet 6000-char cap",
    )  # fmt: skip
    parser.add_argument(
        "--facts-from", action="append", default=[], metavar="BEAD[,BEAD]",
        help="bead ids: their title and description (verbatim) are prepended to FACTS",
    )  # fmt: skip
    parser.add_argument(
        "--no-input-table", default="", metavar="REASON",
        help="skip the input-classes table check for a parser/validator/loader/router scope "
        "and record the reason in FACTS",
    )  # fmt: skip
    parser.add_argument("--append", action="append", default=[], metavar="FILE")
    parser.add_argument("--force", action="store_true", help="overwrite an existing brief")
    parser.add_argument(
        "--route",
        default="",
        help="models.py role: print the delegate_task JSON (T3) or the route line, write "
        "<id>.delegate.json for route_guard.py, and append a ledger row",
    )
    parser.add_argument("--bead", default="", help="bead id for the --route ledger row")
    return parser.parse_args(argv)


def _sibling(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"mkbrief_{name}", Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dispatch(job: Path, args: argparse.Namespace, task: str, dispatch_text: str) -> int:
    """--route: one ready-to-send dispatch line + a "rendered" ledger row (exit 2: no route).

    Both runtimes record the route in ``<job>/prompts/<id>.delegate.json``: the T3 call
    itself, or ``{"role", "model"}`` for a Claude role, so ``route_guard.py`` can refuse a
    dispatch of another model. The row says rendered, not dispatched: the chief records the
    real dispatch (taskId) with ledger.py once delegate_task returns."""
    models = _sibling("models")
    default = Path(__file__).resolve().parents[2] / "docs" / "orchestration" / "models.json"
    data = models.load(Path(os.environ.get("VQ_MODELS_JSON") or default))
    routes = data.get("routes")
    if not isinstance(routes, dict) or args.route not in routes:
        valid = ", ".join(routes) if isinstance(routes, dict) else ""
        print(f"route {args.route}: unknown role; valid roles: {valid}", file=sys.stderr)
        return 2
    chosen, skipped = models.resolve(data, args.route)
    if chosen is None:
        why = "; ".join(f"{k}: {r}" for k, r in skipped)
        print(f"route {args.route}: none available ({why})", file=sys.stderr)
        return 2
    key, model = chosen
    runtime = str(model.get("runtime"))
    if runtime == "t3":
        call = {"task": dispatch_text, "target": model.get("target"), "mode": "async",
                "clientRequestId": args.id}  # fmt: skip
        print("DELEGATE_TASK " + json.dumps(call))
        (job / "prompts" / f"{args.id}.delegate.json").write_text(json.dumps(call) + "\n")
    else:
        target = model.get("target")
        recorded = target.get("model") if isinstance(target, dict) else None
        print(f"ROUTE {args.route} -> {key} ({runtime}) {json.dumps(target)}")
        (job / "prompts" / f"{args.id}.delegate.json").write_text(
            json.dumps({"role": args.route, "model": str(recorded or key)}) + "\n"
        )
    ledger = _sibling("ledger")
    ledger.append_row(job, {
        "agent": args.id, "runtime": runtime, "model": key, "task": task, "bead": args.bead,
        "worktree": args.worktree, "status": f"{ledger.now_hm()} rendered",
    })  # fmt: skip
    return 0


def _key_value(items: list[str], flag: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {item!r}")
        pairs.append((key, value))
    return pairs


def main(argv: list[str] | None = None) -> int:
    """One render. The git answers that cannot change DURING it are memoised (see `_MEMO`):
    the memo is installed here and removed in a finally, so a second render, and any direct
    call to a helper, sees the repository as it is now."""
    global _MEMO
    previous, _MEMO = _MEMO, {}
    try:
        return _render_main(argv)
    finally:
        _MEMO = previous


def _render_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    task, round_no = parse_id(args.id)
    job = Path(args.job).resolve()
    bus = job / "bus"

    values: dict[str, str] = {
        "ID": args.id,
        "JOB": str(job),
        "BUS": str(bus),
        "TASK": task,
        "ROUND": str(round_no),
        "TELEMETRY": f"TELEMETRY: task={_checker_label(args.id)}{task} round={round_no}",
        "WT": args.worktree,
        "RUN_AS": f"cd {args.worktree} && PYTHONPATH=$PWD {_PYTHON}",
        "PEERS": "none",  # bus-protocol.md; --set PEERS=<ids> overrides
    }
    rendered_globs, checked_globs = parse_scope(args.scope)
    sources = [strip_leading_comment(Path(args.template).read_text())]
    sources += [strip_leading_comment(Path(path).read_text()) for path in args.append]
    template_keys = {key for source in sources for key in _PLACEHOLDER_RE.findall(source)}
    flag_problems = [
        f"{flag} needs a template with a {{FACTS}} placeholder ({args.template} has none): its text would be dropped"
        for flag, given in (("--facts-from", args.facts_from), ("--no-input-table", args.no_input_table))
        if given and "FACTS" not in template_keys
    ]
    if flag_problems:
        for problem in flag_problems:
            print(problem, file=sys.stderr)
        return 2
    set_pairs = _key_value(list(args.set), "--set")
    file_pairs = _key_value(list(args.set_file), "--set-file")
    if args.prior_from:
        if any(key == "PRIOR_FINDINGS" for key, _ in [*set_pairs, *file_pairs]):
            print(
                "--prior-from and --set/--set-file PRIOR_FINDINGS= are mutually exclusive",
                file=sys.stderr,
            )
            return 2
        prior_text, prior_problems = prior_findings(
            [part.strip() for item in args.prior_from for part in item.split(",") if part.strip()]
        )
        if prior_problems:
            for problem in prior_problems:
                print(problem, file=sys.stderr)
            return 2
        values["PRIOR_FINDINGS"] = prior_text
    for key, value in set_pairs:
        values[key] = value
    for key, path in file_pairs:
        values[key] = Path(path).read_text()
    if args.review and values.get("SPEC"):
        values["SPEC"] = maker_spec_excerpt(values["SPEC"])
    if args.facts_from:
        beads = [
            bead.strip()
            for item in args.facts_from
            for bead in item.split(",")
            if bead.strip()
        ]
        bead_text, bead_problems = facts_from_beads(beads)
        if bead_problems:
            for problem in bead_problems:
                print(problem, file=sys.stderr)
            return 2
        if bead_text:
            current = values.get("FACTS", "")
            values["FACTS"] = f"{bead_text}\n{current}" if current else bead_text
    gone = deleted_names(args.deletes, values.get("FACTS", ""))
    if (
        args.worktree
        and not args.review
        and _HYGIENE_TEST not in rendered_globs
        and moves_swallow(Path(args.worktree), rendered_globs, gone)
    ):
        rendered_globs.append(_HYGIENE_TEST)
    fact_problems: list[str] = []
    seed_lines: list[str] = []
    if args.worktree and not args.review:
        spans = fact_spans(values)
        fact_problems = fact_path_problems(spans, args.worktree, rendered_globs)
        created = {name.strip() for item in args.new for name in item.split(",") if name.strip()}
        checked = fact_names(spans, created)
        hits = name_hits(Path(args.worktree), checked)
        fact_problems += [
            f"fact name {name} has no hit in {args.worktree}"
            f" (pass --new {name} if the task creates it)"
            for name in checked
            if not hits[name]
        ]
        seeds, seed_lines = scope_seeds(args.worktree, rendered_globs, checked, hits, gone)
        rendered_globs.extend(seeds)
    input_problems: list[str] = []
    if not args.review and template_keys & {"FACTS", "STEPS"}:
        if args.no_input_table:
            note = f"- No input table: {args.no_input_table}"
            current = values.get("FACTS", "")
            values["FACTS"] = f"{current}\n{note}" if current else note
        else:
            input_problems = input_table_problems(
                rendered_globs, values, worktree_files(Path(args.worktree)) if args.worktree else ()
            )
    scope_out = ",".join(rendered_globs)
    if args.worktree and not args.base and not args.review:
        args.base = recorded_base(Path(args.worktree).resolve(), "HEAD")  # worktree.sh's fork point
    base, base_ref = args.base, ""
    if args.worktree and base:
        cwd = Path(args.worktree).resolve()
        try:
            base = _git(cwd, "rev-parse", "--verify", f"{args.base}^{{commit}}")
        except (subprocess.CalledProcessError, OSError) as exc:
            stderr = getattr(exc, "stderr", "") or str(exc)
            print(
                f"--base {args.base} does not resolve in {cwd}: {stderr.strip()}",
                file=sys.stderr,
            )
            return 2
        base = recorded_base(cwd, "HEAD") or base  # worktree.sh's start SHA beats a moved branch
        base_ref = "" if base == args.base else args.base
    values["SCOPE"] = scope_out
    values["GATE"] = gate_command(args.worktree, scope_out, base, bus, args.id, base_ref)

    if args.review and not args.parent and args.worktree:
        wt = Path(args.worktree).resolve()
        if (wt / ".vq-base").exists():
            try:
                head = _git(wt, "rev-parse", args.review)
                recorded = recorded_base(wt, head)
                merged = bool(recorded) and bool(
                    _git(wt, "rev-list", "--merges", f"{recorded}..{head}")
                )
            except (subprocess.CalledProcessError, OSError) as exc:
                print(f"git failed in {wt}: {exc}", file=sys.stderr)
                return 2
            if not recorded:
                print(
                    f"worktree .vq-base is not an ancestor of {args.review}; pass --parent",
                    file=sys.stderr,
                )
                return 2
            if merged:
                print(
                    f"{args.review} merged its parent since .vq-base; rebase or pass --parent",
                    file=sys.stderr,
                )
                return 2
            args.parent = recorded  # worktree.sh's start SHA
    if bool(args.review) != bool(args.parent):
        print("--review and --parent must be given together", file=sys.stderr)
        return 2
    if args.review:
        cwd = Path(args.worktree).resolve() if args.worktree else job
        try:
            head_sha = _git(cwd, "rev-parse", args.review)
            base_sha = _git(cwd, "merge-base", args.parent, args.review)  # --parent wins ...
            if base_sha == head_sha:  # ... unless it already contains the branch (merged, empty)
                base_sha = recorded_base(cwd, head_sha) or base_sha
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"git failed in {cwd}: {exc}", file=sys.stderr)
            return 2
        if base_sha == head_sha:
            print(
                f"empty review range: {args.review} has no commits past {base_sha[:12]}",
                file=sys.stderr,
            )
            return 2
        values["BASE_SHA"] = base_sha
        values["HEAD_SHA"] = head_sha
        values["RANGE"] = f"{base_sha}..{head_sha}"
        values["REVIEW_PACK"] = (
            f"python3 scripts/agents/review_pack.py {args.worktree} --range {values['RANGE']}"
        )

    if args.worktree and not args.base and not args.review:
        candidates = stacked_candidates(Path(args.worktree).resolve())
        if candidates:
            print(f"stacked worktree without .vq-base: pass --base <fork point> (e.g. {candidates[0].split()[1]})", file=sys.stderr)
            for candidate in candidates:
                print(f"  git merge-base candidate: {candidate}", file=sys.stderr)
            return 2

    branch_problems: list[str] = []
    if args.worktree and base and rendered_globs and not args.review:
        try:
            offenders = branch_scope_offenders(
                Path(args.worktree).resolve(), base, rendered_globs
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            stderr = getattr(exc, "stderr", "") or str(exc)
            print(f"git diff {base}...HEAD failed in {args.worktree}: {stderr.strip()}", file=sys.stderr)
            return 2
        branch_problems = [
            f"changed vs base {base} but matches no --scope glob: {path}" for path in offenders
        ]

    board_target = next((key for key in ("FACTS", "RULINGS") if key in values), "")
    if board_target and not args.review:
        block = board_lines(rendered_globs)
        if block:
            current = values[board_target].rstrip()
            joined = "\n".join(block)
            values[board_target] = f"{current}\n{joined}" if current else joined

    enabled = {
        "frontend": any(glob.startswith(("frontend/", "e2e/")) for glob in rendered_globs),
        "peers": values.get("PEERS", "").strip().lower() not in ("", "none"),
    }
    sources: list[str] = []
    tag_problems: list[str] = []
    for path in [args.template, *args.append]:
        body, unknown = conditional_lines(strip_leading_comment(Path(path).read_text()), enabled)
        sources.append(body)
        tag_problems += [f"{path}: unknown conditional tag 'if {tag}'" for tag in unknown]
    if tag_problems:
        for problem in tag_problems:
            print(problem, file=sys.stderr)
        return 2
    text = render(sources[0], values).rstrip("\n")
    for source in sources[1:]:
        text += "\n\n" + render(source, values).rstrip("\n")
    text += "\n"

    out_path = job / "prompts" / f"{args.id}.md"
    problems = [f"unfilled placeholder {{{key}}}" for key in unfilled(sources, values)]
    problems += scope_problems(rendered_globs, checked_globs, args.worktree, base, gone)
    problems += case_problems(rendered_globs, args.worktree)
    if not args.review:
        problems += [
            f"{rel} uses deleted {name} and is outside --scope (add it to --scope or pass --accept-outside {rel})"
            for rel, name in tests_using_deleted(
                args.worktree, rendered_globs, gone, args.accept_outside
            )
        ]
    problems += branch_problems
    problems += fact_problems
    problems += input_problems
    if out_path.exists() and not args.force:
        problems.append(f"output file already exists: {out_path} (pass --force to overwrite)")
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 2

    for line in seed_lines:
        print(line, file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    dispatch_text = f"Read your brief at {out_path} and follow it exactly.\n{values['TELEMETRY']}"
    print(dispatch_text)
    if args.route:
        return dispatch(job, args, task, dispatch_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
