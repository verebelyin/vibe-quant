#!/usr/bin/env python3
"""Generate a swarm brief from a template, validate it, and print the dispatch text.

    mkbrief.py --job JOB_DIR --id CLIENT_REQUEST_ID --template FILE
               [--worktree ABS] [--scope 'glob,glob'] [--base REF]
               [--review BRANCH --parent REF] [--deletes NAME,..] [--accept-outside FILE ...]
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
    stripped from the rendered ``SCOPE``);
  * ``tests/unit/**/*.py`` in ``--scope`` without ``tests/unit/*.py`` (the
    gate's fnmatch needs a subdirectory for ``**``);
  * ``--worktree`` without ``--base`` whose HEAD is stacked on another local branch or holds
    merge commits relative to ``main`` (the gate's default base would count that work as ours);
  * ``--worktree`` with a ``--base`` that does not resolve to a commit in it
    (the git error is printed);
  * a ``--scope`` path that differs from a worktree file only by case (macOS
    filesystems accept it, the gate's fnmatch does not), including a ``+`` new file;
  * the output file already exists (unless ``--force``).

  * ``--deletes NAME[,NAME]`` (also ``delete `NAME` `` in FACTS): a test file outside
    ``--scope`` that uses one of those exact names, unless ``--accept-outside FILE`` lists it.
    A scoped def/class of that name holding a swallow (``find_swallows`` of
    ``tests/unit/test_exception_hygiene.py``) adds that test to ``SCOPE`` (the ratchet moves).

``--review`` uses the start SHA ``worktree.sh`` wrote to ``<worktree>/.vq-base`` as the range base when it
is an ancestor of the branch (else ``merge-base REF BRANCH``), and exits 2 on an empty range.

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
    """Tracked + untracked-not-ignored files (``git ls-files``); a non-git dir is walked.

    A full walk of the main checkout (data/, node_modules/) made one brief take 23.3 s; this 0.25 s."""
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    if proc.returncode == 0:
        return [line for line in proc.stdout.splitlines() if line]
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


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


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
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--set-file", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--append", action="append", default=[], metavar="FILE")
    parser.add_argument("--force", action="store_true", help="overwrite an existing brief")
    parser.add_argument(
        "--route",
        default="",
        help="models.py role: print the delegate_task JSON (T3) and append a ledger row",
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

    The row says rendered, not dispatched: the chief records the real dispatch (taskId)
    with ledger.py once delegate_task returns."""
    models = _sibling("models")
    default = Path(__file__).resolve().parents[2] / "docs" / "orchestration" / "models.json"
    data = models.load(Path(os.environ.get("VQ_MODELS_JSON") or default))
    chosen, skipped = models.resolve(data, args.route)
    if chosen is None:
        why = "; ".join(f"{k}: {r}" for k, r in skipped) or "unknown role"
        print(f"route {args.route}: none available ({why})", file=sys.stderr)
        return 2
    key, model = chosen
    runtime = str(model.get("runtime"))
    if runtime == "t3":
        call = {"task": dispatch_text, "target": model.get("target"), "mode": "async",
                "clientRequestId": args.id}  # fmt: skip
        print("DELEGATE_TASK " + json.dumps(call))
    else:
        print(f"ROUTE {args.route} -> {key} ({runtime}) {json.dumps(model.get('target'))}")
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
    facts = "".join(v for k, v in _key_value(args.set, "--set") if k == "FACTS")
    facts += "".join(
        Path(v).read_text() for k, v in _key_value(args.set_file, "--set-file") if k == "FACTS"
    )
    gone = deleted_names(args.deletes, facts)
    if (
        args.worktree
        and not args.review
        and _HYGIENE_TEST not in rendered_globs
        and moves_swallow(Path(args.worktree), rendered_globs, gone)
    ):
        rendered_globs.append(_HYGIENE_TEST)
    scope_out = ",".join(rendered_globs)
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

    if bool(args.review) != bool(args.parent):
        print("--review and --parent must be given together", file=sys.stderr)
        return 2
    if args.review:
        cwd = Path(args.worktree).resolve() if args.worktree else job
        try:
            head_sha = _git(cwd, "rev-parse", args.review)
            base_sha = recorded_base(cwd, head_sha) or _git(
                cwd, "merge-base", args.parent, args.review
            )
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
            print(f"stacked worktree: pass --base <fork point> (e.g. {candidates[0].split()[1]})", file=sys.stderr)
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
    problems += case_problems(rendered_globs, args.worktree)
    if not args.review:
        problems += [
            f"{rel} uses deleted {name} and is outside --scope (add it to --scope or pass --accept-outside {rel})"
            for rel, name in tests_using_deleted(
                args.worktree, rendered_globs, gone, args.accept_outside
            )
        ]
    problems += branch_problems
    if out_path.exists() and not args.force:
        problems.append(f"output file already exists: {out_path} (pass --force to overwrite)")
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    dispatch_text = f"Read your brief at {out_path} and follow it exactly.\n{values['TELEMETRY']}"
    print(dispatch_text)
    if args.route:
        return dispatch(job, args, task, dispatch_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
