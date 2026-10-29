#!/usr/bin/env python3
"""Cross-job telemetry ledger: every swarm task as one JSONL record, so model routing can be decided from data.

Append-only ledger at $SWARM_TELEMETRY, else <main checkout>/docs/orchestration/telemetry/tasks.jsonl
(resolved through git's common dir like bus.py's global_board_dir, so worktree agents share one file).
The default ledger is COMMITTED to git (like the board) — do not gitignore it. Timestamps are UTC
ISO with a trailing "Z". Models are normalised to lowercase once, in make_record.

Record schema (all keys present, null when unknown):
    ts, job, task, agent, runtime ("cmd"|"claude"), model, round (int, 1 = first attempt),
    outcome ("done"|"failed"|"stalled"|"error"), review ("PASS"|"SHOULD-FIX"|"BLOCK"|null),
    tokens_in, tokens_out, wall_s, span_s, bead, note,
    source ("t3:<clientRequestId>" | "claude:<agentId>" | null for manual/legacy records),
    comms_asked, comms_dms, comms_replies, board_relied (from the handoff line
    `comms: asked=N dms=N replies=N; board posts relied on: <ids|none>`; board_relied = ids named)

    telemetry.py record --job J --task T --agent A --runtime cmd|claude --model M
                        [--round N] [--outcome X] [--review R] [--tokens-in N] [--tokens-out N]
                        [--wall-s S] [--bead B] [--note TEXT]
    telemetry.py ingest-cmd-log <log.ndjson> --job J --task T --agent A [--round N | --auto-round]
                        [--model M] [--force]
    telemetry.py review --job J --task T --review PASS|SHOULD-FIX|BLOCK [--round N]  (override)
    telemetry.py ingest-runs --job J [--t3-db P] [--cmd-dir P] [--claude-dir P] [--dry-run] [--force]
    telemetry.py report [--by model|runtime|agent|board|asked] [--since YYYY-MM-DD] [--markdown]

Jobs are stored by BASENAME: `--job data/swarm/20261027-taker` (an absolute path and a trailing
slash included) records and matches `20261027-taker`, and a legacy row that stored the path is
normalised on read, so its lookups, review copies and reports join the basename (the file itself
is never rewritten). Any subcommand that takes a job applies the same normalisation.

wall_s is ACTIVE time: the sum of consecutive gaps <= 600 s (10 min) over the run's event stamps,
so an idle stretch is not charged to a task. span_s keeps the old first-to-last span beside it
(null on legacy records; `record` leaves it null, while `ingest-cmd-log` sets it to the run's
duration, which IS its span). T3 stamps are started_at, every turn_items.updated_at of the child
thread and completed_at; a run with no turn items keeps wall_s = span_s and is noted "active time
unknown". Claude stamps are its transcript records.

`ingest-runs` appends one record per FINISHED run of job J (the job id appears in its brief):
T3 delegated tasks from ~/.t3/userdata/statev2.sqlite (runtime "cmd" for Command Code, "codex"
for Codex; Command Code tokens are summed from the matching ~/.commandcode session file, because
T3's own number is a last-context approximation; Codex tokens come from T3 usage) and Claude
subagent transcripts under ~/.claude/projects/<main checkout>/*/subagents (tokens_in = input +
cache creation + cache read, once per API message; transcripts modified < 120 s ago are skipped
unless --force). Identity: a `TELEMETRY: task=<t> round=<n> [agent=<a>]` line in the brief wins
for a maker run only — a reviewer/verifier run (`rev-*`, `revlite-*`, `ver-*`) takes its identity
from its own id, because its brief quotes the maker's line; otherwise identity_from_name
(cc-<slug>-<n>, -rN, -fixN, a bare -N, -opusN; reviewer/verifier/architect
runs get task "review:<t>"/"verify:<t>"/"design:<t>"). Idempotent: a known source is skipped, and
so is a run the chief already recorded by hand (same job, agent, round, no source).
`--by board` / `--by asked` group by the round's comms fields: relied>0 | relied=0 | unknown.

The ledger is append-only: `review` appends a copy of the record for (job, task, round) (default:
the latest round) with the review set and a FRESH ts; the pass-1 rate and rounds-to-done read
the latest record per (job, task, round), the other columns count attempts (below). `--since`
therefore filters on the last update of a record, not its first run.

`ingest-runs` fills the verdicts itself: after the runs it reads
<main checkout>/data/swarm/<job>/handoffs/{rev,revlite}-*.md, maps a reviewer id
`rev-<task>-<round>` to the maker (task, round), reads its `verdict:` token and, when that token
is PASS/SHOULD-FIX/BLOCK and the latest record for (job, task, round) has no review, appends the
review copy exactly as `review` does — the maker's fields unchanged, including its `note`, so the
copy is the same attempt — and prints one provenance line per applied verdict:
`review <VERDICT> <job> <task> r<round> from handoffs/<file>`. A verdict with no maker record is
printed to stderr and skipped; --dry-run reports these without writing. A re-ingest appends
nothing, and a set review is never overwritten — so the `review` subcommand is now an OVERRIDE
for manual corrections only, not a step the chief must remember.

Attempts: a retry/relaunch of the same (job, task, round) — e.g. a stalled model replaced by
another — is a NEW attempt and never overwrites the earlier one. Reports keep it and count its
tasks, stalls/errors, tokens and wall under its own group (model), so a stalled MiMo attempt stays
visible next to the DeepSeek run that replaced it. Every plain record is an attempt, even an
identical relaunch. Only a `review` copy (review set) shares the payload of the record it updates
and supersedes that attempt instead of adding one; the pass-1 rate and rounds-to-done still use
only the LATEST attempt per (job, task, round).

Rounds: an explicit `--round N` with `--task <base>` is authoritative. `--auto-round` (ingest only,
ignored when --round is given; opt-in, so a bare "-r" maps to round 2) derives (base, round) from the task name with
^(.+?)-(?:(fix|r)(\\d*)|(\\d{1,2}))$ — "-fix"/"-r" -> 2, "-fixN" -> N+1, "-rN"/"-N" -> N — stores the base
name as task, sets note "round derived from name" and prints the derived pair. The bare "-N" form is what
a T3 clientRequestId or agent id carries (cc-e6-sweep-2 = round 2); "-opusN"/"-sonnetN" model suffixes are
recognised before it, so a model never reads as a round. For names the parser cannot link
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
a second attempt (done), so that task counts one extra attempt and one stall. Re-ingesting a log
that finished twice (done -> done, a manual re-ingest only) records two attempts, so tokens and
wall are double-counted. Stdlib only; runs with any python3.
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
    from collections.abc import Iterator, Sequence
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
    "span_s",
    "bead",
    "note",
    "source",
    "comms_asked",
    "comms_dms",
    "comms_replies",
    "board_relied",
)
OUTCOMES = ("done", "failed", "stalled", "error")
REVIEWS = ("PASS", "SHOULD-FIX", "BLOCK")
RUNTIMES = ("cmd", "claude", "codex")
GROUPS = ("model", "runtime", "agent", "board", "asked")
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


def _main_checkout() -> Path:
    """The main checkout (not a worktree), via git's common dir."""
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
    return Path(common).parent


def ledger_path() -> Path:
    """$SWARM_TELEMETRY, else <main checkout>/docs/orchestration/telemetry/tasks.jsonl."""
    raw = os.environ.get("SWARM_TELEMETRY")
    if raw:
        return Path(raw)
    return _main_checkout() / "docs" / "orchestration" / "telemetry" / "tasks.jsonl"


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def job_basename(job: str) -> str:
    """The job id from a job given as a name, a path or a path with a trailing slash.

    ``ingest-runs --job data/swarm/20261027-taker`` must store and match ``20261027-taker``:
    a path-form id never groups with the basename rows of the same job.
    """
    return Path(str(job).replace("\\", "/")).name or str(job)


STALE_LOG_S = 120
_ROUND_RE = re.compile(r"^(?P<base>.+?)-(?:(?P<word>fix|r)(?P<n>\d*)|(?P<plain>\d{1,2}))$")


def derive_round(name: str) -> tuple[str, int] | None:
    """(base, round) from a "<base>-fix", "<base>-fixN" (N+1), "<base>-rN"/"<base>-N" (N) name, else None.

    The bare ``-N`` form is how a T3 clientRequestId or a swarm agent id carries its round
    (``cc-e6-sweep-2`` = round 2 of ``cc-e6-sweep``); without it such a run is recorded as a
    round-1 task of its own and a job's rounds-to-done reads 1.0. Only 1-2 digits count, so a
    trailing date (``-20261014``) is not mistaken for a round.
    """
    m = _ROUND_RE.match(name)
    if m is None:
        return None
    plain = m.group("plain")
    if plain is not None:
        return (m.group("base"), int(plain)) if int(plain) >= 1 else None
    n = m.group("n") or ""
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
    span_s: float | None = None,
    bead: str | None = None,
    note: str | None = None,
    source: str | None = None,
    comms: dict[str, int] | None = None,
) -> dict[str, Any]:
    comms = comms or {}
    return {
        "ts": _now_ts(),
        "job": job_basename(job),
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
        "span_s": span_s,
        "bead": bead,
        "note": note,
        "source": source,
        "comms_asked": comms.get("comms_asked"),
        "comms_dms": comms.get("comms_dms"),
        "comms_replies": comms.get("comms_replies"),
        "board_relied": comms.get("board_relied"),
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
            if isinstance(rec.get("job"), str):  # legacy rows stored the job as a path
                rec["job"] = job_basename(rec["job"])
            if _is_legacy_reviewer_row(rec):
                # an old ingest stored a reviewer run under the maker's task
                _, rec["task"], rec["round"] = identity_from_name(rec["agent"])
            out.append(rec)
    return out


def _is_legacy_reviewer_row(rec: dict[str, Any]) -> bool:
    """A reviewer/verifier run stored as a maker task by an ingest that honoured its brief's
    TELEMETRY line. Re-mapped on read (never in the file), so it stops counting as an attempt."""
    agent = rec.get("agent")
    task = rec.get("task")
    return (
        isinstance(agent, str)
        and _REVIEWER_AGENT_RE.match(agent.strip()) is not None
        and isinstance(task, str)
        and not task.startswith(("review:", "verify:"))
    )


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
            # take identity_from_name's task label too, so a cmd-log row joins the T3 row for the
            # same task (cc-e6-sweep-2 -> task e6-sweep, not cc-e6-sweep).
            _, task, round = identity_from_name(task)
            note_parts.append("round derived from name")
            print(f"telemetry: derived task={task!r} round={round}", file=sys.stderr)
    tokens_in: int | None = None
    tokens_out: int | None = None
    wall_s: float | None = None
    span_s: float | None = None
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
            wall_s = span_s = float(duration) / 1000  # a cmd run's duration IS its span

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
            span_s=span_s,
            note="; ".join(note_parts) or None,
        )
    )


def _latest_for(
    records: list[dict[str, Any]], job: str, task: str, round: int | None = None
) -> dict[str, Any] | None:
    """Latest record for (job, task[, round]) — last record of the highest round."""
    matching = [r for r in records if r.get("job") == job and r.get("task") == task]
    if round is not None:
        matching = [r for r in matching if r.get("round") == round]
    latest: dict[str, Any] | None = None
    top = -1
    for rec in matching:  # last record of the highest round (file order breaks ties)
        rnd = rec.get("round")
        rnd = rnd if isinstance(rnd, int) else 0
        if rnd >= top:
            latest, top = rec, rnd
    return latest


def apply_review(job: str, task: str, review: str, round: int | None = None) -> dict[str, Any]:
    """Append a copy of the record for (job, task, round) with the review set (ledger stays append-only).

    round None = the latest round. The copy gets a fresh ts. This is now an override: ingest-runs
    fills the verdicts from the reviewer handoffs, and no automatic step overwrites a set review.
    """
    job = job_basename(job)
    latest = _latest_for(read_records(), job, task, round)
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
    it is kept as an attempt of its own. A manual `record --review X` whose payload matches an
    earlier record therefore collapses onto that earlier attempt, not onto the newest one.
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
    if by in ("board", "asked"):  # derived from the handoff comms line
        field, label = ("board_relied", "relied") if by == "board" else ("comms_asked", "asked")
        n = rec.get(field)
        if not isinstance(n, int) or isinstance(n, bool):
            return "unknown"
        return f"{label}>0" if n > 0 else f"{label}=0"
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


_COMMS_RE = re.compile(
    r"comms:\s*asked=(\d+)\s*dms=(\d+)\s*replies=(\d+)\s*;?\s*board posts relied on:\s*([^\n]*)",
    re.IGNORECASE,
)
_POST_ID_RE = re.compile(r"\b[0-9a-f]{12,16}-\d{3,4}\b")
_TELEMETRY_LINE_RE = re.compile(r"^TELEMETRY:\s*(.+)$", re.MULTILINE)
_VERDICT_LINE_RE = re.compile(r"^verdict:\s*(\S+)\s*$", re.MULTILINE)
_ROLE_TASK_PREFIX = (  # longest first: "revlite-" before "rev-"
    ("architect-", "design:"), ("arch-", "design:"),
    ("reviewer-", "review:"), ("revlite-", "review:"), ("rev-", "review:"),
    ("verifier-", "verify:"), ("ver-", "verify:"),
    ("impl-", ""), ("cmd-", ""), ("cc-", ""),
)
_MODEL_SUFFIX_RE = re.compile(r"^(?P<base>.+?)-(?:opus|sonnet|haiku)(?P<n>\d*)$")
_T3_ROUND_RE = re.compile(r"^(?P<base>cc-.+)-(?P<n>\d+)$")
_REVIEWER_AGENT_RE = re.compile(r"^(?:rev|revlite|ver)-")
STATUS_OUTCOME = {"completed": "done", "failed": "error", "interrupted": "stalled", "cancelled": "stalled"}


def parse_comms(text: str | None) -> dict[str, int] | None:
    """The handoff line `comms: asked=N dms=N replies=N; board posts relied on: <ids|none>`.

    board_relied = number of board post ids named (0 for "none"); None when the line is absent.
    """
    m = _COMMS_RE.search(text or "")
    if m is None:
        return None
    return {
        "comms_asked": int(m.group(1)),
        "comms_dms": int(m.group(2)),
        "comms_replies": int(m.group(3)),
        "board_relied": len(set(_POST_ID_RE.findall(m.group(4)))),
    }


def identity_from_name(name: str) -> tuple[str, str, int]:
    """(agent, task, round) from a swarm agent name or T3 clientRequestId.

    "cc-<slug>-<n>" (T3 id) -> agent cc-<slug>, round n; "-rN"/"-fixN"/a bare "-N" -> round as
    derive_round; "-opusN"/"-sonnet" model suffixes -> round N (default 1) and are checked
    BEFORE the bare "-N" rule, so "-opus5" stays a model and is not read as round 5; a dated
    Codex id loses its "-YYYYMMDD". The task drops the maker prefix (cc-/impl-/cmd-) and maps
    checker roles to "review:", "verify:" or "design:" so they never mix with maker tasks.
    Briefs can override all of this with a `TELEMETRY: task=<t> round=<n>` line (see
    identity_from_prompt).
    """
    name = re.sub(r"-20\d{6}$", "", name.strip())
    agent, base, rnd = name, name, 1
    t3 = _T3_ROUND_RE.match(name)
    if t3 is not None:
        agent = base = t3.group("base")
        rnd = int(t3.group("n"))
        if (inner := derive_round(base)) is not None:  # cc-x-r4-1: the -r4 is the round
            base, rnd = inner
    elif (ms := _MODEL_SUFFIX_RE.match(name)) is not None:
        base, rnd = ms.group("base"), int(ms.group("n") or 1)
    elif (derived := derive_round(name)) is not None:
        base, rnd = derived
    task = base
    for prefix, label in _ROLE_TASK_PREFIX:
        if base.startswith(prefix):
            task = label + base[len(prefix) :]
            break
    return agent, task, rnd


def identity_from_prompt(prompt: str, name: str) -> tuple[str, str, int]:
    """identity_from_name, overridden by `TELEMETRY: task=<t> round=<n> [agent=<a>]` in the brief.

    A reviewer/verifier run (agent id `rev-*`, `revlite-*`, `ver-*`) never takes the line: its
    brief quotes the MAKER's TELEMETRY line, and honouring it would store the review as a maker
    attempt of the maker's task under the reviewer's model. Its own id yields `review:<t>` /
    `verify:<t>` from identity_from_name.
    """
    agent, task, rnd = identity_from_name(name)
    if _REVIEWER_AGENT_RE.match(name.strip()):
        return agent, task, rnd
    m = _TELEMETRY_LINE_RE.search(prompt or "")
    if m is not None:
        kv = dict(re.findall(r"(\w+)=(\S+)", m.group(1)))
        task = kv.get("task", task)
        agent = kv.get("agent", agent)
        if kv.get("round", "").isdigit():
            rnd = int(kv["round"])
    return agent, task, rnd


def _iso_s(raw: object) -> float | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def active_seconds(stamps: list[float], gap: float = 600.0) -> float | None:
    """Sum of consecutive gaps <= ``gap`` over the sorted stamps (None when there are none).

    An idle gap longer than ``gap`` (10 min by default) is not active time: a run parked for
    three days is not a three-day run. A single stamp yields 0.0 — it proves the run happened,
    not how long it was active.
    """
    if not stamps:
        return None
    ordered = sorted(stamps)
    return float(sum(b - a for a, b in zip(ordered, ordered[1:], strict=False) if b - a <= gap))


def _span(stamps: list[float]) -> float | None:
    """First-to-last span of the stamps (the old wall_s)."""
    return (max(stamps) - min(stamps)) if stamps else None


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            yield rec


def _cmd_session_usage(
    cmd_dir: Path, start: float, prompt: str, claimed: set[Path]
) -> tuple[int, int] | None:
    """(input, output) tokens summed over the Command Code session that ran this T3 task.

    T3's own usage for Command Code is absent or a last-context approximation, so the session
    file is the truth. The WHOLE stripped dispatch prompt must appear in the session's first user
    message: a dispatch is a short pointer, and its first 80 chars are identical for every child
    of a job, so a prefix match would hand siblings the same file. Candidates must also start
    within [-5 s, +120 s] of the T3 start; among them the closest start wins. ``claimed`` holds the
    sessions already taken by an earlier run of this ingest, so no two runs share one file.
    Returns None when no session matches.
    """
    needle = prompt.strip()
    best: tuple[float, Path, list[dict[str, Any]]] | None = None
    for path in sorted(cmd_dir.glob("*/*.jsonl")):
        if path.name.endswith(".checkpoints.jsonl") or path in claimed:
            continue
        recs = list(_jsonl(path))
        t0 = _iso_s(recs[0].get("timestamp")) if recs else None
        if t0 is None or not -5 <= t0 - start <= 120:
            continue
        first_user = next((r for r in recs if (r.get("message") or {}).get("role") == "user"), None)
        content = ((first_user or {}).get("message") or {}).get("content")
        text = " ".join(b.get("text", "") for b in content if isinstance(b, dict)) if isinstance(content, list) else str(content or "")
        if needle not in text:
            continue
        if best is None or abs(t0 - start) < abs(best[0] - start):
            best = (t0, path, recs)
    if best is None:
        return None
    _, path, recs = best
    claimed.add(path)
    seen: set[object] = set()
    tin = tout = 0
    for r in recs:
        usage = r.get("usage")
        if (r.get("message") or {}).get("role") == "assistant" and isinstance(usage, dict) and r.get("id") not in seen:
            seen.add(r.get("id"))
            tin += int(usage.get("inputTokens") or 0)
            tout += int(usage.get("outputTokens") or 0)
    return tin, tout


def _turn_item_stamps(con: Any, thread_id: object) -> list[float]:
    """The timestamps of a child thread's turn items: the events that show it was active."""
    if not isinstance(thread_id, str) or not thread_id:
        return []
    rows = con.execute(
        "select updated_at from orchestration_v2_projection_turn_items where thread_id = ?",
        (thread_id,),
    ).fetchall()
    return [t for (raw,) in rows if (t := _iso_s(raw)) is not None]


def _t3_runs(job: str, db: Path, cmd_dir: Path) -> list[dict[str, Any]]:
    """Finished T3 delegated tasks (Command Code + Codex) whose brief names the job."""
    import sqlite3
    from urllib.parse import unquote

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    claimed: set[Path] = set()  # Command Code session files already given to a run in this ingest
    try:
        rows = con.execute(
            "select provider, status, started_at, completed_at, payload_json, child_thread_id "
            "from orchestration_v2_projection_subagents where origin = 'app_owned'"
        ).fetchall()
        for provider, status, started, completed, payload_json, child_thread_id in rows:
            if status not in STATUS_OUTCOME:
                continue  # still running
            payload = json.loads(payload_json)
            prompt = str(payload.get("prompt") or "")
            if job not in prompt:
                continue
            cid = unquote(str(payload.get("id") or "")).rsplit("delegate-task:", 1)[-1]
            agent, task, rnd = identity_from_prompt(prompt, cid)
            start, end = _iso_s(started), _iso_s(completed)
            turn = _turn_item_stamps(con, child_thread_id)
            stamps = [t for t in (start, end, *turn) if t is not None]
            result = payload.get("result") if isinstance(payload.get("result"), str) else ""
            outcome = STATUS_OUTCOME[status]
            notes = [] if outcome == "done" else [str(result)[:160]]
            if not turn:
                notes.append("active time unknown")  # only the raw span is known
            tokens: tuple[int, int] | None = None
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            runtime = "codex" if "codex" in str(provider) else "cmd"
            if runtime == "codex":
                if usage.get("inputTokens") is not None:
                    tokens = (int(usage["inputTokens"]), int(usage.get("outputTokens") or 0))
            elif start is not None:
                tokens = _cmd_session_usage(cmd_dir, start, prompt, claimed)
            if tokens is None:
                notes.append("tokens unknown")
            out.append(make_record(
                job=job, task=task, agent=agent, runtime=runtime, model=payload.get("model"), round=rnd,
                outcome=outcome, tokens_in=tokens[0] if tokens else None, tokens_out=tokens[1] if tokens else None,
                wall_s=active_seconds(stamps) if turn else _span(stamps), span_s=_span(stamps),
                note="; ".join(notes) or None, source=f"t3:{cid}", comms=parse_comms(result),
            ))
    finally:
        con.close()
    return out


def _claude_runs(job: str, claude_dir: Path, force: bool) -> list[dict[str, Any]]:
    """Finished Claude subagents (Agent tool) whose first prompt names the job."""
    out = []
    for path in sorted(claude_dir.glob("*/subagents/agent-*.jsonl")):
        if not force and time.time() - path.stat().st_mtime < STALE_LOG_S:
            continue  # probably still running
        recs = list(_jsonl(path))
        first = next((r for r in recs if r.get("type") == "user"), None)
        content = ((first or {}).get("message") or {}).get("content")
        prompt = content if isinstance(content, str) else " ".join(
            b.get("text", "") for b in content or [] if isinstance(b, dict)
        )
        if job not in prompt:
            continue
        agent_id = path.stem.removeprefix("agent-")
        meta_path = path.with_name(f"agent-{agent_id}.meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        usage_by_id: dict[object, tuple[str, dict[str, Any]]] = {}
        stamps: list[float] = []
        final, error = "", ""
        for r in recs:
            t = _iso_s(r.get("timestamp"))
            if t is not None:
                stamps.append(t)
            msg = r.get("message") or {}
            if r.get("type") != "assistant":
                continue
            text = " ".join(b.get("text", "") for b in msg.get("content") or [] if isinstance(b, dict) and b.get("type") == "text")
            if r.get("isApiErrorMessage") or msg.get("model") == "<synthetic>":
                error = text
                continue
            if isinstance(msg.get("usage"), dict):
                usage_by_id[msg.get("id")] = (str(msg.get("model")), msg["usage"])
            if text:
                final = text
        models: dict[str, int] = {}
        tin = tout = 0
        for model, u in usage_by_id.values():  # one entry per API message (logged once per block)
            models[model] = models.get(model, 0) + 1
            tin += sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            tout += int(u.get("output_tokens") or 0)
        agent, task, rnd = identity_from_prompt(prompt, str(meta.get("name") or agent_id))
        out.append(make_record(
            job=job, task=task, agent=agent, runtime="claude",
            model=max(models, key=lambda k: models[k]) if models else None, round=rnd,
            outcome="error" if error else "done", tokens_in=tin if usage_by_id else None,
            tokens_out=tout if usage_by_id else None,
            wall_s=active_seconds(stamps), span_s=_span(stamps),
            note=error[:160] or None, source=f"claude:{agent_id}", comms=parse_comms(final),
        ))
    return out


def _default_claude_dir() -> Path:
    """~/.claude/projects/<main checkout path with / -> ->, where subagent transcripts live."""
    return Path.home() / ".claude" / "projects" / str(_main_checkout()).replace("/", "-")


def _handoff_dir(job: str) -> Path | None:
    """<main checkout>/data/swarm/<job>/handoffs, or None outside a git checkout.

    The ledger path may come from SWARM_TELEMETRY in a non-git copy (a mutant scratch dir, an
    installed script): there is nowhere to read handoffs from, which is not an ingest error.
    """
    try:
        return _main_checkout() / "data" / "swarm" / job / "handoffs"
    except SystemExit:
        return None


def _handoff_reviews(job: str) -> Iterator[tuple[str, int, str, str]]:
    """(task, round, verdict, filename) for every reviewer handoff of ``job``.

    Reviewers write ``<job dir>/handoffs/rev-<task>-<round>.md`` (or ``revlite-...``); the round
    is the maker round under review. Only a ``verdict:`` token in REVIEWS is a telemetry review:
    FAIL, VETO and the other handoff-lint tokens are ignored.
    """
    base = _handoff_dir(job)
    if base is None or not base.is_dir():
        return
    for path in sorted([*base.glob("rev-*.md"), *base.glob("revlite-*.md")]):
        _, task, rnd = identity_from_name(path.stem)
        if task.startswith("review:"):
            task = task[len("review:") :]
        m = _VERDICT_LINE_RE.search(path.read_text(encoding="utf-8"))
        if m is None or m.group(1) not in REVIEWS:
            continue
        yield task, rnd, m.group(1), path.name


def _apply_handoff_reviews(
    job: str, dry_run: bool, pending: Sequence[dict[str, Any]] = ()
) -> list[dict[str, Any]]:
    """Fill a maker record's null review from its reviewer handoff.

    The copy is exact: every SCHEMA field of the MAKER record including its ``note``, a fresh ts,
    the review — so it collapses onto that attempt in ``latest_attempts`` instead of counting as
    a second one. Provenance is one stdout line per applied verdict. A review already set (by
    hand, or by an earlier ingest) wins, so a re-ingest appends nothing.

    A reviewer run in the ledger is never the maker record: read_records re-maps a legacy row
    stored under the maker's task to its own review:<t>/verify:<t> task, so only a maker row can
    match a handoff's (task, round). A key with no maker row is skipped, never credited to the
    reviewer's model. ``pending`` holds this ingest's new runs: under dry_run they are not on
    disk yet, and a verdict for a maker discovered by this very ingest must still be reported.
    """
    latest: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for rec in [*read_records(), *pending]:
        latest[(rec.get("job"), rec.get("task"), rec.get("round"))] = rec
    added: list[dict[str, Any]] = []
    for task, rnd, verdict, name in _handoff_reviews(job):
        key = (job, task, rnd)
        source = latest.get(key)
        if source is None:
            print(
                f"telemetry: handoffs/{name}: no maker record for job={job!r} task={task!r}"
                f" round={rnd}; skipped",
                file=sys.stderr,
            )
            continue
        if source.get("review") is not None:
            continue
        new = {field: source.get(field) for field in SCHEMA}
        new["ts"] = _now_ts()
        new["review"] = verdict
        latest[key] = new
        if dry_run:
            print(
                f"telemetry: would record {verdict} for job={job!r} task={task!r}"
                f" round={rnd} from handoffs/{name}",
                file=sys.stderr,
            )
            added.append(new)
        else:
            added.append(append_record(new))
            print(f"review {verdict} {job} {task} r{rnd} from handoffs/{name}")
    return added


def ingest_runs(
    job: str,
    t3_db: Path | None = None,
    cmd_dir: Path | None = None,
    claude_dir: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Append one record per finished run of ``job`` found in T3, Command Code and Claude logs.

    Idempotent: a run whose ``source`` is already in the ledger is skipped, and so is a run that
    the chief already recorded by hand (same job, agent and round, no source). Returns the new
    records (with dry_run, the ones it would append).
    """
    job = job_basename(job)
    t3_db = t3_db or Path.home() / ".t3" / "userdata" / "statev2.sqlite"
    cmd_dir = cmd_dir or Path.home() / ".commandcode" / "projects"
    claude_dir = claude_dir or _default_claude_dir()
    found: list[dict[str, Any]] = []
    if t3_db.exists():
        found += _t3_runs(job, t3_db, cmd_dir)
    else:
        print(f"telemetry: no T3 database at {t3_db}; skipping T3 runs", file=sys.stderr)
    if claude_dir.exists():
        found += _claude_runs(job, claude_dir, force)
    existing = read_records()
    sources = {r.get("source") for r in existing if r.get("source")}
    manual = {(r.get("job"), r.get("agent"), r.get("round")) for r in existing if not r.get("source")}
    added = []
    for rec in sorted(found, key=lambda r: str(r["source"])):
        if rec["source"] in sources or (rec["job"], rec["agent"], rec["round"]) in manual:
            continue
        sources.add(rec["source"])
        added.append(rec if dry_run else append_record(rec))
    added += _apply_handoff_reviews(job, dry_run, added)
    return added


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
    run_p = sub.add_parser("ingest-runs", help="records for every finished run of a job (T3 + Claude)")
    run_p.add_argument("--job", required=True)
    run_p.add_argument("--t3-db", dest="t3_db", type=Path)
    run_p.add_argument("--cmd-dir", dest="cmd_dir", type=Path)
    run_p.add_argument("--claude-dir", dest="claude_dir", type=Path)
    run_p.add_argument("--dry-run", dest="dry_run", action="store_true")
    run_p.add_argument("--force", action="store_true", help="include transcripts modified < 120 s ago")
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
    elif args.cmd == "ingest-runs":
        added = ingest_runs(args.job, args.t3_db, args.cmd_dir, args.claude_dir, args.dry_run, args.force)
        for new in added:
            print(json.dumps(new, ensure_ascii=False))
        print(f"telemetry: {len(added)} run(s) {'found' if args.dry_run else 'recorded'} for {job_basename(args.job)}", file=sys.stderr)
        return 0
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
