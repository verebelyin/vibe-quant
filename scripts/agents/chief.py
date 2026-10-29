#!/usr/bin/env python3
"""One chief command: ledger a status row, lint a done handoff, print bus status + an ETA.

    scripts/agents/chief.py step --job J --id AGENT --status TEXT
        [--runtime R] [--model M] [--task T] [--bead B] [--worktree W] [--handoff FILE]
    scripts/agents/chief.py status --job J

In Oct 14-28 chiefs ran ledger.py 1,349 times, then handoff_lint and `bus.py status` by hand
after each handoff, and the user asked "status? ETA?" 10+ times. `step` does all three: a
status starting with `done` is linted against its handoff (refused, with no row, when it is
not clean), the row is appended exactly as ledger.py writes it, and the status block is
printed. `status` prints the same block read-only.

The status block is `bus.py status` for the job, then `Running:` — one line per agent whose
latest ledger row is not finished — with elapsed minutes and an ETA from the median wall time
of that model's done runs (cross-job telemetry; < 3 runs = unknown). Ledger timestamps carry
no year: the current UTC year is assumed, the previous one when that would land in the future.
Stdlib only.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COLUMNS = ("agent", "runtime", "model", "task", "bead", "worktree", "status")
RUNNING_PREFIXES = ("dispatched", "rendered", "in-progress")
MIN_ETA_RUNS = 3
OVERDUE_FACTOR = 1.5
_DEFAULT_MODELS = Path(__file__).resolve().parents[2] / "docs" / "orchestration" / "models.json"
_TS = re.compile(r"^(\d{2}-\d{2} \d{2}:\d{2}Z)\s+(.+)$")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def _sibling(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"chief_{name}", Path(__file__).with_name(f"{name}.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        if len(cells) != len(COLUMNS):
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


def running_lines(
    rows: dict[str, list[str]],
    records: list[dict[str, Any]],
    models: dict[str, str],
    now: datetime,
) -> list[str]:
    """One line per agent whose latest row is not finished, with elapsed and a median ETA."""
    lines: list[str] = []
    for agent in sorted(rows):
        _, _, key, task, _, _, status = rows[agent]
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
        line = f"  {agent} {key} {task} started {stamp.group(1)} elapsed {minutes}m {eta}"
        if median is not None and runs >= MIN_ETA_RUNS:
            line += f" (median {int(round(median / 60))}m of {runs} runs, {model_id})"
        lines.append(line)
    return lines


def status_block(
    job: Path,
    now: datetime | None = None,
    records: list[dict[str, Any]] | None = None,
    models: dict[str, str] | None = None,
) -> list[str]:
    """The bus screen for ``job``, then `Running:` with an ETA per unfinished agent."""
    when = _now(now)
    if records is None:
        records = _sibling("telemetry").read_records()
    if models is None:
        models = model_ids()
    lines = _sibling("bus").status_report(job / "bus", "chief")
    running = running_lines(latest_rows(job / "ledger.md"), records, models, when)
    lines.append("Running:" if running else "Running: none")
    return lines + running


def cmd_step(args: argparse.Namespace) -> int:
    job: Path = args.job
    handoff: Path = Path(args.handoff) if args.handoff else job / "handoffs" / f"{args.agent}.md"
    if args.status.startswith("done") and handoff.is_file():
        problems = _sibling("handoff_lint").check(handoff.read_text(encoding="utf-8"), None)
        if problems:
            print("\n".join(problems))
            print("step refused: handoff not lint-clean")
            return 1
    values = {column: getattr(args, column) for column in COLUMNS if column != "status"}
    values["status"] = f"{_sibling('ledger').now_hm()} {args.status}"
    _sibling("ledger").append_row(job, values)
    print("\n".join(status_block(job)))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print("\n".join(status_block(args.job)))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="chief.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    step_p = sub.add_parser("step", help="lint a done handoff, append the row, print status")
    step_p.add_argument("--job", required=True, type=Path)
    step_p.add_argument("--id", required=True, dest="agent")
    step_p.add_argument("--status", required=True)
    for column in ("runtime", "model", "task", "bead", "worktree"):
        step_p.add_argument(f"--{column}")
    step_p.add_argument("--handoff", help="default: <job>/handoffs/<id>.md when it exists")
    status_p = sub.add_parser("status", help="print the status block read-only")
    status_p.add_argument("--job", required=True, type=Path)
    args = ap.parse_args(argv)
    return cmd_step(args) if args.command == "step" else cmd_status(args)


if __name__ == "__main__":
    sys.exit(main())
