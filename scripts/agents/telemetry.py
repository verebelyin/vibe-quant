#!/usr/bin/env python3
"""Cross-job telemetry ledger: every swarm task as one JSONL record, so model routing can be decided from data.

Append-only ledger at $SWARM_TELEMETRY, else <main checkout>/docs/orchestration/telemetry/tasks.jsonl
(resolved through git's common dir like bus.py's global_board_dir, so worktree agents share one file).
The default ledger is COMMITTED to git (like the board) — do not gitignore it. Timestamps are UTC
ISO with a trailing "Z". Models are normalised to lowercase once, in make_record.

Record schema (all keys present, null when unknown):
    ts, job, task, agent, runtime ("cmd"|"claude"), model, round (int, 1 = first attempt),
    outcome ("done"|"failed"|"stalled"|"error"), review ("PASS"|"SHOULD-FIX"|"BLOCK"|null),
    tokens_in, tokens_out, wall_s, bead, note

    telemetry.py record --job J --task T --agent A --runtime cmd|claude --model M
                        [--round N] [--outcome X] [--review R] [--tokens-in N] [--tokens-out N]
                        [--wall-s S] [--bead B] [--note TEXT]
    telemetry.py ingest-cmd-log <log.ndjson> --job J --task T --agent A [--round N | --auto-round]
                        [--model M] [--force]
    telemetry.py review --job J --task T --review PASS|SHOULD-FIX|BLOCK [--round N]
    telemetry.py report [--by model|runtime|agent] [--since YYYY-MM-DD] [--markdown]

The ledger is append-only: `review` appends a copy of the record for (job, task, round) (default:
the latest round) with the review set and a FRESH ts; the pass-1 rate and rounds-to-done read
the latest record per (job, task, round), the other columns count attempts (below). `--since`
therefore filters on the last update of a record, not its first run.

Attempts: a retry/relaunch of the same (job, task, round) — e.g. a stalled model replaced by
another — is a NEW attempt and never overwrites the earlier one. Reports keep it and count its
tasks, stalls/errors, tokens and wall under its own group (model), so a stalled MiMo attempt stays
visible next to the DeepSeek run that replaced it. Every plain record is an attempt, even an
identical relaunch. Only a `review` copy (review set) shares the payload of the record it updates
and supersedes that attempt instead of adding one; the pass-1 rate and rounds-to-done still use
only the LATEST attempt per (job, task, round).

Rounds: an explicit `--round N` with `--task <base>` is authoritative. `--auto-round` (ingest only,
ignored when --round is given; opt-in, so a bare "-r" maps to round 2) derives (base, round) from the task name with
^(.+?)-(fix|r)(\\d*)$ — "-fix"/"-r" -> 2, "-fixN" -> N+1, "-rN" -> N — stores the base name as task,
sets note "round derived from name" and prints the derived pair. For names the parser cannot link
(q-ui -> qui-fix, sc -> sc2, m2 -> m2b) use explicit `--task <base> --round N`. Re-ingesting the
same (job, task, round) after a review adds an unreviewed latest record: re-run `review` after any
--force re-ingest.

`ingest-cmd-log` parses a cmd-task.sh NDJSON transcript: the LAST "type":"result" event decides
outcome ("done" iff subtype=="success", else "error"; stopReason "max_turns" goes to note), tokens
come from result.usage.inputTokens/outputTokens and wall_s from durationMs/1000. No result event is
outcome "stalled" with note "no result event (running or killed)". Logs modified < 120 s ago are
refused (probably still running) unless --force. Outcome "done" means the cmd run finished, not
that the work was accepted — acceptance is the review. Model resolution order:
  (1) --model flag; (2) a "model" field on any NDJSON event (top level or the nested event object);
  (3) the log filename (cmd-task.sh logs <ts>-<brief>-<model with / replaced by _>.ndjson): the
      suffix after the last "-<vendor>_" among KNOWN_VENDORS becomes "<vendor>/<rest>";
  (4) else model null with note "model unknown".

`report` shows per group: tasks (distinct job+task), first-pass PASS rate (round-1 records with
review PASS / round-1 records with any review), mean rounds to done (per (job, task) across ALL
records: the earliest round whose latest record has review PASS, charged to the group of the task's
round-1 record; tasks with no PASS or no round-1 record are excluded; --since keeps only tasks with
a record in the window but never changes the attribution), median tokens_in, median wall_s, and the
count of round-records with outcome stalled/error. Tasks, stalls/errors, tokens, wall and the
`attempts` column (printed for every --by) count EVERY attempt (latest_attempts); only the pass-1
rate and rounds-to-done follow the LATEST attempt per (job, task, round).
Known limitation (accepted): `ingest-cmd-log` is the legacy cmd-task.sh path. A log idle >= 120 s
with no result event is recorded as stalled; ingesting the same log again after it finished records
a second attempt (done), so that task counts one extra attempt and one stall. Stdlib only; runs
with any python3.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

SCHEMA = (
    "ts",
    "job",
    "task",
    "agent",
    "runtime",
    "model",
    "round",
    "outcome",
    "review",
    "tokens_in",
    "tokens_out",
    "wall_s",
    "bead",
    "note",
)
OUTCOMES = ("done", "failed", "stalled", "error")
REVIEWS = ("PASS", "SHOULD-FIX", "BLOCK")
RUNTIMES = ("cmd", "claude")
GROUPS = ("model", "runtime", "agent")
KNOWN_VENDORS = (
    "xiaomi",
    "deepseek",
    "qwen",
    "moonshotai",
    "zai-org",
    "z-ai",
    "minimaxai",
    "anthropic",
)


def ledger_path() -> Path:
    """$SWARM_TELEMETRY, else <main checkout>/docs/orchestration/telemetry/tasks.jsonl."""
    raw = os.environ.get("SWARM_TELEMETRY")
    if raw:
        p = Path(raw)
    else:
        here = Path(__file__).resolve().parent
        try:
            common = subprocess.run(
                ["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            sys.exit("telemetry: not inside a git repo; set SWARM_TELEMETRY to a ledger path")
        p = Path(common).parent / "docs" / "orchestration" / "telemetry" / "tasks.jsonl"
    return p


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


STALE_LOG_S = 120
_ROUND_RE = re.compile(r"^(?P<base>.+?)-(?:fix|r)(?P<n>\d*)$")


def derive_round(name: str) -> tuple[str, int] | None:
    """(base, round) from a "<base>-fix", "<base>-fixN" (N+1) or "<base>-rN" (N) name, else None."""
    m = _ROUND_RE.match(name)
    if m is None:
        return None
    n = m.group("n")
    if not n:
        return m.group("base"), 2
    is_fix = "-fix" in name[len(m.group("base")) :]
    rnd = int(n) + (1 if is_fix else 0)
    # "-r0" / "-fix0" would be round 0 / collide with the base task: treat as no match
    return (m.group("base"), rnd) if int(n) >= 1 else None


def _iso_date(raw: str) -> str:
    return datetime.date.fromisoformat(raw).isoformat()


def make_record(
    job: str,
    task: str,
    agent: str,
    runtime: str,
    model: str | None,
    round: int = 1,
    outcome: str | None = None,
    review: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    wall_s: float | None = None,
    bead: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "ts": _now_ts(),
        "job": job,
        "task": task,
        "agent": agent,
        "runtime": runtime,
        "model": model.strip().lower() if model else None,
        "round": round,
        "outcome": outcome,
        "review": review,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_s": wall_s,
        "bead": bead,
        "note": note,
    }


def append_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Append one record to the ledger (flock-guarded like bus.py) and return it."""
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return rec


def read_records() -> list[dict[str, Any]]:
    path = ledger_path()
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def model_from_logname(name: str) -> str | None:
    """Model id from a cmd-task.sh log name: last "-<vendor>_" splits vendor from the rest.

    cmd-task.sh logs <ts>-<brief>-<model with / replaced by _>.ndjson and both brief and model
    may contain "-" and "_"; KNOWN_VENDORS anchors the split (chief's rule, 2026-10-09).
    """
    stem = name[: -len(".ndjson")] if name.endswith(".ndjson") else name
    best: tuple[int, str, str] | None = None
    for vendor in KNOWN_VENDORS:
        token = f"-{vendor}_"
        pos = stem.rfind(token)
        rest = stem[pos + len(token) :] if pos >= 0 else ""
        if rest and (best is None or pos > best[0]):
            best = (pos, vendor, rest)
    if best is None:
        return None
    _, vendor, rest = best
    return f"{vendor}/{rest}"


def _event_model(events: list[Any]) -> str | None:
    """First "model" field found on any event (top level or the nested event object)."""
    for ev in events:
        if not isinstance(ev, dict):
            continue
        for container in (ev, ev.get("event")):
            if isinstance(container, dict):
                m = container.get("model")
                if isinstance(m, str) and m:
                    return m
    return None


def ingest_cmd_log(
    log: Path,
    job: str,
    task: str,
    agent: str,
    round: int = 1,
    model: str | None = None,
    force: bool = False,
    auto_round: bool = False,
) -> dict[str, Any]:
    """Parse one cmd NDJSON transcript and append its ledger record (runtime "cmd")."""
    if not log.exists():
        sys.exit(f"telemetry: log not found: {log}")
    age = time.time() - log.stat().st_mtime
    if age < STALE_LOG_S and not force:
        sys.exit(
            f"telemetry: {log.name} modified {age:.0f}s ago (< {STALE_LOG_S}s): probably still "
            "running; rerun later or pass --force"
        )
    events: list[Any] = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    result: dict[str, Any] | None = None
    for ev in events:
        if isinstance(ev, dict) and ev.get("type") == "result":
            result = ev

    note_parts: list[str] = []
    if auto_round:
        derived = derive_round(task)
        if derived is not None:
            task, round = derived
            note_parts.append("round derived from name")
            print(f"telemetry: derived task={task!r} round={round}", file=sys.stderr)
    tokens_in: int | None = None
    tokens_out: int | None = None
    wall_s: float | None = None
    if result is None:
        outcome = "stalled"
        note_parts.append("no result event (running or killed)")
    else:
        outcome = "done" if result.get("subtype") == "success" else "error"
        if result.get("stopReason") == "max_turns":
            note_parts.append("stopReason: max_turns")
        usage = result.get("usage")
        if isinstance(usage, dict):
            tokens_in = usage.get("inputTokens")
            tokens_out = usage.get("outputTokens")
        duration = result.get("durationMs")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            wall_s = float(duration) / 1000

    if model is None:
        model = _event_model(events) or model_from_logname(log.name)
    if model is None:
        note_parts.append("model unknown")

    return append_record(
        make_record(
            job=job,
            task=task,
            agent=agent,
            runtime="cmd",
            model=model,
            round=round,
            outcome=outcome,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            wall_s=wall_s,
            note="; ".join(note_parts) or None,
        )
    )


def apply_review(job: str, task: str, review: str, round: int | None = None) -> dict[str, Any]:
    """Append a copy of the record for (job, task, round) with the review set (ledger stays append-only).

    round None = the latest round. The copy gets a fresh ts.
    """
    matching = [r for r in read_records() if r.get("job") == job and r.get("task") == task]
    if round is not None:
        matching = [r for r in matching if r.get("round") == round]
    latest: dict[str, Any] | None = None
    top = -1
    for rec in matching:  # last record of the highest round (file order breaks ties)
        rnd = rec.get("round")
        rnd = rnd if isinstance(rnd, int) else 0
        if rnd >= top:
            latest, top = rec, rnd
    if latest is None:
        sys.exit(f"telemetry: no record for job={job!r} task={task!r} round={round or 'any'}")
    new = {key: latest.get(key) for key in SCHEMA}
    new["ts"] = _now_ts()
    new["review"] = review
    return append_record(new)


def latest_per_round(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Latest record per (job, task, round) — review updates supersede the record they copy."""
    by_key: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for rec in records:
        by_key[(rec.get("job"), rec.get("task"), rec.get("round"))] = rec
    return list(by_key.values())


def _attempt_signature(rec: dict[str, Any]) -> tuple[Any, ...]:
    """Payload identifying one attempt: every schema field except ts and review.

    A `review` copy has the same payload as the record it updates, so it collapses onto that
    attempt (see latest_attempts).
    """
    return tuple(rec.get(key) for key in SCHEMA if key not in ("ts", "review"))


def latest_attempts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per attempt, in ledger order.

    Every plain record (review null) is its own attempt — even a byte-identical relaunch, e.g.
    a model stalling twice the same way. Only a review copy (review set) collapses: it replaces
    the latest earlier attempt with the same payload (_attempt_signature); with no such source
    it is kept as an attempt of its own.
    """
    attempts: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("review") is not None:
            sig = _attempt_signature(rec)
            for i in range(len(attempts) - 1, -1, -1):
                if _attempt_signature(attempts[i]) == sig:
                    attempts[i] = rec
                    break
            else:
                attempts.append(rec)
        else:
            attempts.append(rec)
    return attempts


def _windowed(records: list[dict[str, Any]], since: str | None) -> list[dict[str, Any]]:
    if not since:
        return records
    return [rec for rec in records if str(rec.get("ts") or "") >= since]


def _group_key(rec: dict[str, Any], by: str) -> str:
    key = rec.get(by)
    text = key if isinstance(key, str) else str(key)
    return text.lower() if by == "model" else text


def attempt_counts(
    records: list[dict[str, Any]], by: str = "model", since: str | None = None
) -> dict[str, int]:
    """Number of attempts (see latest_attempts) per group — the `report` `attempts` column."""
    counts: dict[str, int] = {}
    for rec in latest_attempts(_windowed(records, since)):
        key = _group_key(rec, by)
        counts[key] = counts.get(key, 0) + 1
    return counts


def report_rows(
    records: list[dict[str, Any]], by: str = "model", since: str | None = None
) -> list[dict[str, Any]]:
    """Per-group task stats (see module docstring for the definitions)."""
    if by not in GROUPS:
        sys.exit(f"telemetry: --by must be one of {GROUPS}")

    full = latest_per_round(records)  # rounds-to-done always sees the UNFILTERED ledger
    deduped = latest_per_round(_windowed(records, since))  # pass-1: latest attempt per round
    attempts = latest_attempts(_windowed(records, since))  # counts: EVERY attempt

    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in attempts:
        groups.setdefault(_group_key(rec, by), []).append(rec)
    canonical: dict[str, list[dict[str, Any]]] = {}
    for rec in deduped:
        canonical.setdefault(_group_key(rec, by), []).append(rec)
    in_window = {(rec.get("job"), rec.get("task")) for rec in deduped}

    # rounds-to-done per (job, task), charged to the round-1 record's group; tasks without a
    # round-1 record are excluded, tasks with no record inside the --since window are dropped
    by_task: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for rec in full:
        if isinstance(rec.get("round"), int) and not isinstance(rec.get("round"), bool):
            by_task.setdefault((rec.get("job"), rec.get("task")), []).append(rec)
    task_rounds: dict[str, list[float]] = {}
    charged: dict[str, set[tuple[Any, Any]]] = {}
    for tkey, trecs in by_task.items():
        trecs.sort(key=lambda r: r["round"])
        passed = [r["round"] for r in trecs if r.get("review") == "PASS"]
        if passed and trecs[0]["round"] == 1 and tkey in in_window:
            task_rounds.setdefault(_group_key(trecs[0], by), []).append(float(min(passed)))
            charged.setdefault(_group_key(trecs[0], by), set()).add(tkey)
            groups.setdefault(_group_key(trecs[0], by), [])

    rows: list[dict[str, Any]] = []
    for name in sorted(groups):
        recs = groups[name]  # every attempt of this group
        canon = canonical.get(name, [])  # latest attempt per round (pass-1 / rounds)
        tasks = {(rec.get("job"), rec.get("task")) for rec in recs} | charged.get(name, set())
        first_pass = [
            rec for rec in canon if rec.get("round") == 1 and rec.get("review") is not None
        ]
        passes = [rec for rec in first_pass if rec.get("review") == "PASS"]
        rounds_to_done = task_rounds.get(name, [])
        tokens = [
            float(rec["tokens_in"])
            for rec in recs
            if isinstance(rec.get("tokens_in"), (int, float))
            and not isinstance(rec.get("tokens_in"), bool)
        ]
        walls = [
            float(rec["wall_s"])
            for rec in recs
            if isinstance(rec.get("wall_s"), (int, float))
            and not isinstance(rec.get("wall_s"), bool)
        ]
        rows.append(
            {
                "group": name,
                "tasks": len(tasks),
                "pass1_passes": len(passes),
                "pass1_attempts": len(first_pass),
                "pass1_rate": (len(passes) / len(first_pass)) if first_pass else None,
                "mean_rounds": statistics.fmean(rounds_to_done) if rounds_to_done else None,
                "med_tokens_in": statistics.median(tokens) if tokens else None,
                "med_wall_s": statistics.median(walls) if walls else None,
                "stalls_errors": sum(
                    1 for rec in recs if rec.get("outcome") in ("stalled", "error")
                ),
            }
        )
    return rows


def format_table(
    rows: list[dict[str, Any]],
    markdown: bool = False,
    attempts: dict[str, int] | None = None,
) -> str:
    headers = [
        "group",
        "tasks",
        "pass1",
        "mean_rounds",
        "med_tokens_in",
        "med_wall_s",
        "stalls+errors",
    ]
    if attempts is not None:
        headers.append("attempts")

    def cells(rec: dict[str, Any]) -> list[str]:
        rate = rec["pass1_rate"]
        out = [
            str(rec["group"]),
            str(rec["tasks"]),
            f"{rec['pass1_passes']}/{rec['pass1_attempts']} ({rate:.2f})"
            if rate is not None
            else "n/a",
            "n/a" if rec["mean_rounds"] is None else f"{rec['mean_rounds']:.2f}",
            "n/a" if rec["med_tokens_in"] is None else f"{rec['med_tokens_in']:g}",
            "n/a" if rec["med_wall_s"] is None else f"{rec['med_wall_s']:g}",
            str(rec["stalls_errors"]),
        ]
        if attempts is not None:
            out.append(str(attempts.get(rec["group"], 0)))
        return out

    body = [cells(rec) for rec in rows]
    if markdown:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines += ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join(lines)
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in body)) if body else len(headers[i])
        for i in range(len(headers))
    ]
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines += ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in body]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="telemetry.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec_p = sub.add_parser("record", help="append one ledger record")
    rec_p.add_argument("--job", required=True)
    rec_p.add_argument("--task", required=True)
    rec_p.add_argument("--agent", required=True)
    rec_p.add_argument("--runtime", required=True, choices=RUNTIMES)
    rec_p.add_argument("--model")
    rec_p.add_argument("--round", dest="round", type=int, default=1)
    rec_p.add_argument("--outcome", choices=OUTCOMES)
    rec_p.add_argument("--review", choices=REVIEWS)
    rec_p.add_argument("--tokens-in", dest="tokens_in", type=int)
    rec_p.add_argument("--tokens-out", dest="tokens_out", type=int)
    rec_p.add_argument("--wall-s", dest="wall_s", type=float)
    rec_p.add_argument("--bead")
    rec_p.add_argument("--note")
    ing_p = sub.add_parser("ingest-cmd-log", help="parse a cmd NDJSON transcript into a record")
    ing_p.add_argument("log")
    ing_p.add_argument("--job", required=True)
    ing_p.add_argument("--task", required=True)
    ing_p.add_argument("--agent", required=True)
    ing_p.add_argument("--round", dest="round", type=int)
    ing_p.add_argument("--auto-round", dest="auto_round", action="store_true")
    ing_p.add_argument(
        "--force", action="store_true", help="ingest even a log modified < 120 s ago"
    )
    ing_p.add_argument("--model")
    rev_p = sub.add_parser("review", help="append a review update for the latest record")
    rev_p.add_argument("--job", required=True)
    rev_p.add_argument("--task", required=True)
    rev_p.add_argument("--review", required=True, choices=REVIEWS)
    rev_p.add_argument("--round", dest="round", type=int, help="default: latest round")
    rep_p = sub.add_parser("report", help="per-group task stats for model routing")
    rep_p.add_argument("--by", default="model", choices=GROUPS)
    rep_p.add_argument("--since", metavar="YYYY-MM-DD", type=_iso_date)
    rep_p.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "record":
        rec = append_record(
            make_record(
                job=args.job,
                task=args.task,
                agent=args.agent,
                runtime=args.runtime,
                model=args.model,
                round=args.round,
                outcome=args.outcome,
                review=args.review,
                tokens_in=args.tokens_in,
                tokens_out=args.tokens_out,
                wall_s=args.wall_s,
                bead=args.bead,
                note=args.note,
            )
        )
    elif args.cmd == "ingest-cmd-log":
        rec = ingest_cmd_log(
            Path(args.log),
            job=args.job,
            task=args.task,
            agent=args.agent,
            round=1 if args.round is None else args.round,
            model=args.model,
            force=args.force,
            auto_round=args.auto_round and args.round is None,
        )
    elif args.cmd == "review":
        rec = apply_review(args.job, args.task, args.review, args.round)
    else:
        records = read_records()
        rows = report_rows(records, by=args.by, since=args.since)
        print(format_table(rows, args.markdown, attempt_counts(records, args.by, args.since)))
        return 0
    print(json.dumps(rec, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
