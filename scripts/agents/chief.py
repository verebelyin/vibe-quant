#!/usr/bin/env python3
"""One chief command: create a job, ledger a status row, lint a done handoff, print bus status + ETA.

    scripts/agents/chief.py init --job <dir | job name>
    scripts/agents/chief.py step --job J --id AGENT --status TEXT
        [--runtime R] [--model M] [--task T] [--bead B] [--worktree W] [--handoff FILE]
    scripts/agents/chief.py status --job J
    scripts/agents/chief.py handoff --job J [--next STEP]...
    scripts/agents/chief.py resume --job J

In Oct 14-28 chiefs ran ledger.py 1,349 times, then handoff_lint and `bus.py status` by hand
after each handoff, and the user asked "status? ETA?" 10+ times. `step` does all three: a
status starting with `done` is linted against its handoff (refused, with no row, when it is
not clean), the row is appended exactly as ledger.py writes it, and the status block is
printed. `init` creates the job dir mkbrief writes into (`bus/`, `handoffs/`, `prompts/`), a
`brief.md` skeleton and the header-only `ledger.md` ledger.py appends to, so a chief never
hand-writes either. `status` prints the same block read-only.

The status block is `bus.py status` for the job, then `Running:` — one line per agent whose
latest ledger row is not finished. Each line says in plain words what the item does (the
ledger bead's `bd show` title, else its task cell) plus elapsed minutes and an ETA from the
median wall time of that model's done runs (cross-job telemetry; < 3 runs = unknown); the
agent id stays in brackets at the end. The user asked "explain simply" and "status? ETA?"
repeatedly.

`handoff` writes `<job>/HANDOFF.md` with the same five sections every session switch reads:
Main, Running children, Ledger, Open user questions, Next 3 steps. `resume` prints it and
exits 1 when a section is missing or empty (the Next-steps placeholder counts as empty).
Ledger timestamps carry no year: the current UTC year is assumed, the previous one when that
would land in the future. Stdlib only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COLUMNS = ("agent", "runtime", "model", "task", "bead", "worktree", "status")
RUNNING_PREFIXES = ("dispatched", "rendered", "in-progress", "running")
MIN_ETA_RUNS = 3
OVERDUE_FACTOR = 1.5
HANDOFF_SECTIONS = ("Main", "Running children", "Ledger", "Open user questions", "Next 3 steps")
NEXT_PLACEHOLDER = "- TODO: fill in the next 3 steps"
_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_MODELS = _REPO / "docs" / "orchestration" / "models.json"
_TS = re.compile(r"^(\d{2}-\d{2} \d{2}:\d{2}Z)\s+(.+)$")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_BRIEF_SECTIONS = (
    ("## Goal", ""),
    ("## Constraints", ""),
    ("## Profile", "lean"),
    ("## Finish condition", ""),
    ("## Tasks", ""),
    ("## Answers", ""),
)


def _sibling(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"chief_{name}", Path(__file__).with_name(f"{name}.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SWARM_ROOT = _sibling("ledger").SWARM_ROOT


def _now(now: datetime | None) -> datetime:
    now = now or datetime.now(UTC)
    return now if now.tzinfo is not None else now.replace(tzinfo=UTC)


def model_ids(path: Path | None = None) -> dict[str, str]:
    """Ledger model key -> the telemetry model id it records (models.json `target.model`)."""
    models = _sibling("models")
    data = models.load(path or _DEFAULT_MODELS)
    out: dict[str, str] = {}
    for key, entry in models._models(data).items():
        if not isinstance(entry, dict):
            continue
        target = entry.get("target")
        model = target.get("model") if isinstance(target, dict) else None
        model = model or entry.get("model")
        if isinstance(model, str) and model:
            out[key] = model
    return out


def latest_rows(path: Path) -> dict[str, list[str]]:
    """Latest ledger row per agent (the ledger is append-only; a later row is the state)."""
    rows: dict[str, list[str]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip().replace("\\|", "|") for cell in _CELL_SPLIT.split(stripped)[1:-1]]
        if len(cells) != len(COLUMNS) or cells == list(COLUMNS):
            continue
        agent = cells[0]
        if not agent or set(agent) <= {"-"}:
            continue
        rows[agent] = cells
    return rows


def elapsed_seconds(ts: str, now: datetime) -> float | None:
    """Seconds since a `%m-%d %H:%MZ` ledger stamp, or None when it does not parse.

    The stamp has no year: assume the current UTC year, and the previous one when that lands
    in the future (a job that spans New Year).
    """
    m = _TS.match(ts)
    if m is None:
        return None
    try:
        when = datetime.strptime(f"{now.year}-{m.group(1)}", "%Y-%m-%d %H:%MZ").replace(tzinfo=UTC)
    except ValueError:
        return None
    if when > now:
        when = when.replace(year=when.year - 1)
    return (now - when).total_seconds()


def model_wall(records: list[dict[str, Any]], model_id: str) -> tuple[float | None, int]:
    """(median wall_s, runs) over done telemetry records of ``model_id`` with a wall time."""
    walls: list[float] = []
    for rec in records:
        raw = rec.get("wall_s")
        if (
            str(rec.get("model") or "").lower() != model_id.lower()
            or rec.get("outcome") != "done"
            or not isinstance(raw, (int, float))
            or isinstance(raw, bool)
        ):
            continue
        try:
            wall = float(raw)
        except OverflowError:  # a hand-recorded huge int
            continue
        if math.isfinite(wall):
            walls.append(wall)
    return (statistics.median(walls) if walls else None, len(walls))


def bead_title(bead: str) -> str | None:
    """The bead's title from ``bd show <id> --json``, or None (a screen never fails on bd)."""
    try:
        proc = subprocess.run(
            ["bd", "show", bead, "--json"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    objects = data if isinstance(data, list) else [data]
    if not objects or not isinstance(objects[0], dict):
        return None
    title = objects[0].get("title")
    return " ".join(title.split()) if isinstance(title, str) and title.strip() else None


def bead_titles(rows: dict[str, list[str]]) -> dict[str, str]:
    """bead -> plain-words title, looked up once per bead that a running row cites."""
    found: dict[str, str | None] = {}
    for cells in rows.values():
        bead = cells[4]
        stamp = _TS.match(cells[6])
        if bead in ("", "-") or stamp is None or not stamp.group(2).startswith(RUNNING_PREFIXES):
            continue
        if bead not in found:
            found[bead] = bead_title(bead)
    return {bead: title for bead, title in found.items() if title}


def running_lines(
    rows: dict[str, list[str]],
    records: list[dict[str, Any]],
    models: dict[str, str],
    now: datetime,
    titles: dict[str, str] | None = None,
) -> list[str]:
    """One line per agent whose latest row is not finished: plain words, elapsed and median ETA."""
    titles = titles or {}
    lines: list[str] = []
    running_beads = [
        row[4] for row in rows.values()
        if (m := _TS.match(row[6])) is not None and m.group(2).startswith(RUNNING_PREFIXES)
    ]
    for agent in sorted(rows):
        _, _, key, task, bead, _, status = rows[agent]
        stamp = _TS.match(status)
        if stamp is None or not stamp.group(2).startswith(RUNNING_PREFIXES):
            continue
        elapsed = elapsed_seconds(status, now)
        if elapsed is None:
            continue
        model_id = models.get(key, key)
        median, runs = model_wall(records, model_id)
        minutes = int(elapsed // 60)
        if median is None or runs < MIN_ETA_RUNS:
            eta = f"ETA unknown ({runs} runs)"
        elif elapsed > OVERDUE_FACTOR * median:
            eta = "ETA overdue"
        else:
            eta = f"ETA ~{int(round(max(median - elapsed, 0.0) / 60))}m"
        task = task.split(" taskId ", 1)[0]  # the ledger keeps the T3 id; the screen does not
        words = titles.get(bead) or task
        if bead in titles and running_beads.count(bead) > 1 and task not in ("", "-"):
            words = f"{titles[bead]}: {task}"  # one bead, several tasks: say which
        words = " ".join(words.split()) if words and words != "-" else "unlabelled item"
        line = f"  {words} — started {stamp.group(1)}, elapsed {minutes}m, {eta}"
        if median is not None and runs >= MIN_ETA_RUNS:
            line += f" (median {int(round(median / 60))}m of {runs} runs, {model_id})"
        line += f" [{agent} {key}]"
        lines.append(line)
    return lines


def status_block(
    job: Path,
    now: datetime | None = None,
    records: list[dict[str, Any]] | None = None,
    models: dict[str, str] | None = None,
    titles: dict[str, str] | None = None,
) -> list[str]:
    """The bus screen for ``job``, then `Running:` with a plain-words line and ETA per agent."""
    when = _now(now)
    if records is None:
        records = _sibling("telemetry").read_records()
    if models is None:
        models = model_ids()
    rows = latest_rows(job / "ledger.md")
    if titles is None:
        titles = bead_titles(rows)
    lines: list[str] = _sibling("bus").status_report(job / "bus", "chief")
    running = running_lines(rows, records, models, when, titles)
    lines.append("Running:" if running else "Running: none")
    return lines + running


def job_path(job: Path) -> Path:
    """A bare job name means <repo>/data/swarm/<name>; a path with a '/' is used as is."""
    return SWARM_ROOT / job if len(job.parts) == 1 else job


def brief_text(name: str) -> str:
    """The brief.md skeleton: six headings, the profile line, an empty line under each."""
    lines = [f"# Job {name}", ""]
    for heading, body in _BRIEF_SECTIONS:
        lines += [heading, body, ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def cmd_init(args: argparse.Namespace) -> int:
    """Create the job dir a chief and mkbrief write into; refuse to touch an existing brief."""
    job = job_path(args.job).resolve()
    brief = job / "brief.md"
    if brief.exists():
        print(f"init refused: {brief} already exists (edit it; init writes a skeleton only)",
              file=sys.stderr)  # fmt: skip
        return 2
    for sub in ("bus", "handoffs", "prompts"):
        (job / sub).mkdir(parents=True, exist_ok=True)
    brief.write_text(brief_text(job.name), encoding="utf-8")
    (job / "ledger.md").write_text(_sibling("ledger").header_text(job.name), encoding="utf-8")
    print(f"created {job}")
    print(f"export SWARM_BUS={job / 'bus'}")
    return 0


def cmd_step(args: argparse.Namespace) -> int:
    job = job_path(args.job)
    handoff: Path = Path(args.handoff) if args.handoff else job / "handoffs" / f"{args.agent}.md"
    if args.status.startswith("done") and handoff.is_file():
        problems = _sibling("handoff_lint").check(handoff.read_text(encoding="utf-8"), None)
        if problems:
            print("\n".join(problems))
            print("step refused: handoff not lint-clean")
            return 1
    values = {column: getattr(args, column) for column in COLUMNS if column != "status"}
    # a status-only step keeps the agent's earlier cells (runtime, model, task, ...): the
    # screen reads the latest row, so a bare `--status` would blank its label and ETA
    prior = latest_rows(job / "ledger.md").get(args.agent)
    if prior is not None:
        for index, column in enumerate(COLUMNS[1:-1], start=1):
            if values.get(column) in (None, "", "-") and prior[index] not in ("", "-"):
                values[column] = prior[index]
    values["status"] = f"{_sibling('ledger').now_hm()} {args.status}"
    _sibling("ledger").append_row(job, values)
    print("\n".join(status_block(job)))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print("\n".join(status_block(job_path(args.job))))
    return 0


def _git(*args: str) -> str | None:
    """One git answer from the repo root, or None (a handoff never fails on git)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO), *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def main_lines() -> list[str]:
    """Main SHA vs origin/main with the ahead/behind counts, from git."""
    head = _git("rev-parse", "main")
    origin = _git("rev-parse", "origin/main")
    counts = _git("rev-list", "--left-right", "--count", "main...origin/main")
    ahead = behind = "?"
    if counts:
        parts = counts.split()
        if len(parts) == 2:
            ahead, behind = parts
    return [
        f"- main: {head or 'unknown'}",
        f"- origin/main: {origin or 'unknown'}",
        f"- ahead {ahead}, behind {behind}",
    ]


def _task_ref(task: str) -> str:
    """The T3 taskId kept in a ledger task cell (`P1-A taskId node:x`), else `-`."""
    return task.split(" taskId ", 1)[1].strip() if " taskId " in task else "-"


def running_children_lines(rows: dict[str, list[str]]) -> list[str]:
    """One line per unfinished ledger row: id, runtime, model, clientRequestId and taskId."""
    lines: list[str] = []
    for agent in sorted(rows):
        _, runtime, key, task, _, _, status = rows[agent]
        stamp = _TS.match(status)
        if stamp is None or not stamp.group(2).startswith(RUNNING_PREFIXES):
            continue
        lines.append(
            f"- {agent} | runtime {runtime} | model {key} "
            f"| clientRequestId {agent} | taskId {_task_ref(task)}"
        )
    return lines or ["- (none)"]


def ledger_lines(rows: dict[str, list[str]]) -> list[str]:
    """Every ledger row's latest status, one line per agent."""
    return [f"- {agent} — {cells[6]}" for agent, cells in sorted(rows.items())] or ["- (no rows)"]


def open_question_lines(job: Path) -> list[str]:
    """Open needs-user questions plus asks to the chief, from the job's bus, bodies in full."""
    bus = _sibling("bus")
    items: dict[str, dict[str, object]] = {}
    for msg in bus.needs_user_open(job / "bus"):
        items[str(msg["id"])] = msg
    for msg in bus.pending(job / "bus", "chief"):
        items.setdefault(str(msg["id"]), msg)
    lines = []
    for msg_id, msg in items.items():
        body = str(msg.get("body", "")).strip().splitlines() or [""]
        lines.append(f"- [{msg_id}] {msg.get('from')} ({msg.get('kind')}): {body[0].strip()}")
        # indented, so no body line can pass for a new `## section` on resume
        lines += [f"  {line}".rstrip() for line in body[1:]]
    return lines or ["- (none)"]


def next_lines(steps: list[str]) -> list[str]:
    """The next 3 steps, or the placeholder resume rejects when none were named."""
    lines = [f"- {step.strip()}" for step in steps if step.strip()][:3]
    return lines or [NEXT_PLACEHOLDER]


def handoff_text(job: Path, next_steps: list[str] | None = None) -> str:
    """HANDOFF.md: the five fixed sections, in order, from the ledger, git and the bus."""
    rows = latest_rows(job / "ledger.md")
    blocks = (
        ("Main", main_lines()),
        ("Running children", running_children_lines(rows)),
        ("Ledger", ledger_lines(rows)),
        ("Open user questions", open_question_lines(job)),
        ("Next 3 steps", next_lines(list(next_steps or []))),
    )
    out = [f"# Handoff {job.name}", ""]
    for title, lines in blocks:
        out += [f"## {title}", *lines, ""]
    return "\n".join(out).rstrip("\n") + "\n"


def _sections(text: str) -> dict[str, str]:
    """`## Title` -> its body, for every heading in the handoff."""
    found: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                found[current] = "\n".join(buf).strip()
            current = line[3:].strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        found[current] = "\n".join(buf).strip()
    return found


def _empty(body: str) -> bool:
    """Blank, or holding nothing but the Next-steps placeholder."""
    lines = [line for line in body.splitlines() if line.strip()]
    return not lines or all(line.startswith("- TODO:") for line in lines)


def resume_problems(text: str) -> list[str]:
    """Why a HANDOFF.md cannot be resumed: a missing or empty section; [] when complete."""
    found = _sections(text)
    problems: list[str] = []
    for title in HANDOFF_SECTIONS:
        body = found.get(title)
        if body is None:
            problems.append(f"MISSING section: ## {title}")
        elif _empty(body):
            problems.append(f"EMPTY section: ## {title}")
    return problems


def cmd_handoff(args: argparse.Namespace) -> int:
    job = job_path(args.job)
    job.mkdir(parents=True, exist_ok=True)
    path = job / "HANDOFF.md"
    path.write_text(handoff_text(job, args.next), encoding="utf-8")
    print(path)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    path = job_path(args.job) / "HANDOFF.md"
    if not path.is_file():
        print(f"resume: {path} is missing; write it with chief.py handoff --job {args.job}")
        return 1
    text = path.read_text(encoding="utf-8")
    problems = resume_problems(text)
    if problems:
        print("\n".join(problems))
        return 1
    print(text.rstrip("\n"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="chief.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    init_p = sub.add_parser("init", help="create a job dir (bus/, handoffs/, prompts/, brief.md, ledger.md)")
    init_p.add_argument("--job", required=True, type=Path, help="job dir, or a bare name under data/swarm")
    step_p = sub.add_parser("step", help="lint a done handoff, append the row, print status")
    step_p.add_argument("--job", required=True, type=Path, help="job dir, or a bare name under data/swarm")
    step_p.add_argument("--id", required=True, dest="agent")
    step_p.add_argument("--status", required=True)
    for column in ("runtime", "model", "task", "bead", "worktree"):
        step_p.add_argument(f"--{column}")
    step_p.add_argument("--handoff", help="default: <job>/handoffs/<id>.md when it exists")
    status_p = sub.add_parser("status", help="print the status block read-only")
    status_p.add_argument("--job", required=True, type=Path, help="job dir, or a bare name under data/swarm")
    handoff_p = sub.add_parser("handoff", help="write HANDOFF.md with its five fixed sections")
    handoff_p.add_argument("--job", required=True, type=Path, help="job dir, or a bare name under data/swarm")
    handoff_p.add_argument("--next", action="append", help="a next step (repeat up to 3)")
    resume_p = sub.add_parser("resume", help="print HANDOFF.md; exit 1 when a section is missing or empty")
    resume_p.add_argument("--job", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.command == "init":
        return cmd_init(args)
    if args.command == "step":
        return cmd_step(args)
    if args.command == "handoff":
        return cmd_handoff(args)
    if args.command == "resume":
        return cmd_resume(args)
    return cmd_status(args)


if __name__ == "__main__":
    sys.exit(main())
