#!/usr/bin/env python3
"""Generate a swarm brief from a template, validate it, and print the dispatch text.

    mkbrief.py --job JOB_DIR --id CLIENT_REQUEST_ID --template FILE
               [--worktree ABS] [--scope 'glob,glob'] [--base REF]
               [--review BRANCH --parent REF]
               [--set KEY=VALUE ...] [--set-file KEY=PATH ...] [--append FILE ...]
               [--force]

The template uses ``{UPPER_SNAKE}`` placeholders. Auto-filled keys: ``ID``,
``JOB`` (abs job dir), ``BUS`` (abs ``<job>/bus``), ``TASK`` + ``ROUND`` (from
the id: a trailing ``-<digits>`` is the round, default 1; the task is the id
without the ``cc-``/``rev-``/``impl-``/``ver-`` prefix and without the round
suffix), ``TELEMETRY`` (``TELEMETRY: task=<TASK> round=<ROUND>``), ``WT``,
``RUN_AS``, ``SCOPE`` and ``GATE``.

``--review BRANCH --parent REF`` additionally fills ``BASE_SHA`` (``git
merge-base REF BRANCH``), ``HEAD_SHA`` (``git rev-parse BRANCH``), ``RANGE``
and ``REVIEW_PACK``; git runs in ``--worktree`` (else the job dir).

Validation (exit 2, every problem printed to stderr, nothing written):
  * any ``{UPPER_SNAKE}`` in the raw template or ``--append``ed file text that
    has no value (values from ``--set``/``--set-file`` are never scanned, so a
    pasted code excerpt with ``{UPPER}`` in it renders fine);
  * a ``--scope`` glob that matches no file in ``--worktree`` (a glob prefixed
    with ``+`` means "new file, may not exist yet" and is skipped; the ``+`` is
    stripped from the rendered ``SCOPE``);
  * ``tests/unit/**/*.py`` in ``--scope`` without ``tests/unit/*.py`` (the
    gate's fnmatch needs a subdirectory for ``**``);
  * the output file already exists (unless ``--force``).

A leading ``<!-- ... -->`` usage comment (and the blank lines after it) is
stripped from the template and every appended file before rendering, so the
generator's own usage notes never land in a brief.

On success it writes ``<job>/prompts/<id>.md`` and prints, on stdout:

    Read your brief at <abs path> and follow it exactly.
    TELEMETRY: task=<task> round=<round>

so the file always exists before the dispatch that points at it, and the
TELEMETRY line is always in the dispatched text. Stdlib only.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_PREFIXES = ("cc-", "rev-", "impl-", "ver-")
_ROUND_RE = re.compile(r"-(\d+)$")
_PLACEHOLDER_RE = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
_LEADING_COMMENT_RE = re.compile(r"\A\s*<!--.*?-->[ \t]*", re.DOTALL)
_BLANK_LINES_RE = re.compile(r"\A(?:[ \t]*\n)+")
_BROAD_UNIT_GLOB = "tests/unit/**/*.py"
_UNIT_TOP_GLOB = "tests/unit/*.py"
_UNIT_HINT = f"glob {_BROAD_UNIT_GLOB} does not match top-level tests/unit files; add {_UNIT_TOP_GLOB}"

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
    """Every file under ``root`` (POSIX-relative), minus ``.git``; symlinks not followed."""
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            files.append(Path(dirpath, name).relative_to(root).as_posix())
    return files


def scope_problems(rendered: Iterable[str], checked: Iterable[str], worktree: str) -> list[str]:
    """Glob problems: nothing matched, or the broad top-level tests glob footgun."""
    rendered = list(rendered)
    problems: list[str] = []
    if _BROAD_UNIT_GLOB in rendered and _UNIT_TOP_GLOB not in rendered:
        problems.append(_UNIT_HINT)
    if worktree:
        files = worktree_files(Path(worktree))
        for glob in checked:
            if glob == _BROAD_UNIT_GLOB:
                continue  # covered by the dedicated hint above
            if not any(fnmatch.fnmatch(path, glob) for path in files):
                problems.append(f"glob {glob} matches no file in {worktree}")
    return problems


def gate_command(worktree: str, scope: str, base: str, bus: Path, agent_id: str) -> str:
    """The literal pre-review gate command for this brief."""
    command = f"{_GATE} {worktree} --scope '{scope}'"
    if base:
        command += f" --base {base}"
    if bus.is_dir():
        command += f" --bus {bus} --bus-agent {agent_id}"
    return command


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mkbrief.py", description="Generate and validate a swarm brief."
    )
    parser.add_argument("--job", required=True, help="job directory (data/swarm/<job>)")
    parser.add_argument("--id", required=True, help="client request id (also the brief filename)")
    parser.add_argument("--template", required=True, help="template file with {UPPER_SNAKE} keys")
    parser.add_argument("--worktree", default="", help="worktree the brief works in")
    parser.add_argument("--scope", default="", help="comma-separated SCOPE globs")
    parser.add_argument("--base", default="", help="gate --base ref")
    parser.add_argument("--review", default="", help="review mode: the branch under review")
    parser.add_argument("--parent", default="", help="review mode: the ref to merge-base against")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--set-file", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--append", action="append", default=[], metavar="FILE")
    parser.add_argument("--force", action="store_true", help="overwrite an existing brief")
    return parser.parse_args(argv)


def _key_value(items: list[str], flag: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {item!r}")
        pairs.append((key, value))
    return pairs


def main(argv: list[str] | None = None) -> int:
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
        "TELEMETRY": f"TELEMETRY: task={task} round={round_no}",
        "WT": args.worktree,
        "RUN_AS": f"cd {args.worktree} && PYTHONPATH=$PWD {_PYTHON}",
    }
    rendered_globs, checked_globs = parse_scope(args.scope)
    scope_out = ",".join(rendered_globs)
    values["SCOPE"] = scope_out
    values["GATE"] = gate_command(args.worktree, scope_out, args.base, bus, args.id)

    if bool(args.review) != bool(args.parent):
        print("--review and --parent must be given together", file=sys.stderr)
        return 2
    if args.review:
        cwd = Path(args.worktree).resolve() if args.worktree else job
        try:
            base_sha = _git(cwd, "merge-base", args.parent, args.review)
            head_sha = _git(cwd, "rev-parse", args.review)
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"git failed in {cwd}: {exc}", file=sys.stderr)
            return 2
        values["BASE_SHA"] = base_sha
        values["HEAD_SHA"] = head_sha
        values["RANGE"] = f"{base_sha}..{head_sha}"
        values["REVIEW_PACK"] = (
            f"scripts/agents/review_pack.py {args.worktree} --range {values['RANGE']}"
        )

    for key, value in _key_value(list(args.set), "--set"):
        values[key] = value
    for key, path in _key_value(list(args.set_file), "--set-file"):
        values[key] = Path(path).read_text()

    sources = [strip_leading_comment(Path(args.template).read_text())]
    sources += [strip_leading_comment(Path(path).read_text()) for path in args.append]
    text = render(sources[0], values).rstrip("\n")
    for source in sources[1:]:
        text += "\n\n" + render(source, values).rstrip("\n")
    text += "\n"

    out_path = job / "prompts" / f"{args.id}.md"
    problems = [f"unfilled placeholder {{{key}}}" for key in unfilled(sources, values)]
    problems += scope_problems(rendered_globs, checked_globs, args.worktree)
    if out_path.exists() and not args.force:
        problems.append(f"output file already exists: {out_path} (pass --force to overwrite)")
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"Read your brief at {out_path} and follow it exactly.")
    print(values["TELEMETRY"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
