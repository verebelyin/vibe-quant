#!/usr/bin/env python3
"""Watch discovery runs and print one actionable line per event, then exit.

Chiefs hand-built Monitor pipelines over the discovery logs and guessed the wrong
patterns ("Gen best" instead of the real `=== GEN n/m`); a `| cut` stage buffered
the output so completions arrived late and a 30-minute deadline was missed. This
watches the run's own log from the start (so a late start misses nothing), prints
only the lines worth acting on — flushed per line — and exits on its own once every
watched run has finished:

    Monitor(command="scripts/agents/disc_watch.py 938 940", timeout=1800000)
    Monitor(command="scripts/agents/disc_watch.py 938 --all-gens")

    disc_watch.py RUN_ID [RUN_ID ...] [--logs DIR] [--db PATH] [--interval SEC]
                  [--all-gens] [--kill-after SEC [--api URL]]

`--kill-after SEC` cancels every run still going SEC seconds after the watcher started
(`DELETE <api>/api/discovery/jobs/<id>`, default api http://127.0.0.1:8000) and keeps
watching until the DB says it ended; it prints `KILL after <n>s (HTTP <code>)`. Discovery
operators hand-wrote this kill loop in every wave of one job (8 operators, ~35 calls).

Every printed line is prefixed `run <id>: `:

  * `gen 1/3 best=0.2022 zero=7/8 (139s, ETA 278s)` — gen 1 and any gen that beat
    the previous gen; `--all-gens` prints every gen;
  * `COMPLETE 2 gens, 14 evaluated, 253s` for the pipeline's DISCOVERY COMPLETE line;
  * champion summaries: `champion` at INFO/WARNING from the pipeline, trimmed after
    the logger name;
  * anything at ERROR/CRITICAL, plus a `Traceback` and every line after it down to
    and including the exception line that ends it (a chained traceback prints each
    exception line, and a block is capped at 25 lines);
  * `waiting for log` (once) while the run's log file has not appeared yet;
  * `STATUS completed|failed|cancelled` (once) when the state DB says the run ended.

The log is `logs/discovery_<run_id>_<YYYYMMDD_HHMMSS>.log`, newest file per run id.
Exit codes: 0 when every run is terminal and its log is drained, 1 if any run ended
failed/cancelled, 2 for a run id absent from the DB or an unreadable DB.

The state DB is opened READ-ONLY (`mode=ro`): opening it normally would run
migrations. Defaults resolve against the MAIN checkout (the parent of this script's
`git rev-parse --git-common-dir`), so the watcher works from a worktree or any other
cwd: `--db`, else `$VIBE_QUANT_DB`, else `data/state/vibe_quant.db`; `--logs`, else
`logs/`. Stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import TextIO

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOGS = Path("logs")
DEFAULT_DB = Path("data/state/vibe_quant.db")
DEFAULT_INTERVAL = 5.0
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
BAD_STATUSES = frozenset({"failed", "cancelled"})
ERROR_LEVELS = frozenset({"ERROR", "CRITICAL"})
SUMMARY_LEVELS = frozenset({"INFO", "WARNING"})
TRACEBACK_MAX_LINES = 25
OK_RC = 0
BAD_RC = 1
UNKNOWN_RC = 2

HEADER_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} +\[(?P<level>[A-Z]+)\] "
    r"(?P<logger>[A-Za-z_][\w.]*): (?P<msg>.*)$"
)
GEN_RE = re.compile(
    r"=== GEN (?P<gen>\d+)/(?P<gens>\d+) === best=(?P<best>[-+]?[0-9.]+).*?"
    r"zero_score=(?P<zero>\d+)/(?P<zero_total>\d+) \| gen_time=(?P<gen_time>[0-9.]+)s"
    r".*?ETA=(?P<eta>[0-9.]+)s"
)
COMPLETE_RE = re.compile(
    r"=== DISCOVERY COMPLETE: (?P<total>[0-9.]+)s total .*?\| "
    r"(?P<gens>\d+) gens \| (?P<evaluated>\d+) evaluated"
)


@dataclass
class Run:
    """Watch state for one run id: how far its log is read and what was printed."""

    run_id: str
    log: Path | None = None
    offset: int = 0
    pending: bytes = b""
    prev_best: float | None = None
    waiting_printed: bool = False
    status: str = ""
    status_printed: bool = False
    traceback_left: int = 0
    killed: bool = False


def main_checkout_root(script_dir: Path = SCRIPT_DIR) -> Path:
    """Root of the main checkout that owns ``script_dir``.

    ``--git-common-dir`` names the main checkout's `.git` even when this script lives in
    a linked worktree, so default paths never depend on the cwd. Falls back to the
    script's own repo root when git is missing or ``script_dir`` is not in a repo.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(script_dir),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return script_dir.parents[1]
    common_dir = result.stdout.strip()
    return Path(common_dir).resolve().parent if common_dir else script_dir.parents[1]


def open_db(path: Path) -> sqlite3.Connection:
    """Read-only connection to the state DB.

    ``mode=ro`` matters: a normal ``sqlite3.connect`` runs the repo's schema
    migrations against the live DB, which a watcher must never do.
    """
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def run_status(conn: sqlite3.Connection, run_id: str) -> str | None:
    """Status of a discovery run, or None when the DB has no such run."""
    row = conn.execute(
        "SELECT status FROM backtest_runs WHERE id = ? AND run_mode = 'discovery'", (run_id,)
    ).fetchone()
    return None if row is None else str(row[0])


def newest_log(logs_dir: Path, run_id: str) -> Path | None:
    """Newest `discovery_<run_id>_<timestamp>.log` under ``logs_dir`` (None if none yet)."""
    try:
        matches = list(logs_dir.glob(f"discovery_{glob.escape(run_id)}_*.log"))
    except OSError:
        return None
    return max(matches, key=lambda path: path.name) if matches else None


def _emit(out: TextIO, run_id: str, text: str) -> None:
    print(f"run {run_id}: {text}", file=out, flush=True)


def _read_new_lines(run: Run, *, terminal: bool) -> list[str]:
    """Complete lines appended since the last read, plus a final partial line when done.

    Bytes are buffered so a split multi-byte character can never corrupt a line; the
    trailing partial line is held back until the run is terminal, when it is flushed
    rather than being waited on forever.
    """
    if run.log is None:
        return []
    try:
        size = run.log.stat().st_size
    except OSError:
        return []
    if size < run.offset:  # log replaced or truncated: start over
        run.offset, run.pending = 0, b""
    if size > run.offset:
        with run.log.open("rb") as fh:
            fh.seek(run.offset)
            run.pending += fh.read()
        run.offset = size
    lines: list[str] = []
    while b"\n" in run.pending:
        raw, run.pending = run.pending.split(b"\n", 1)
        lines.append(raw.decode("utf-8", "replace"))
    if terminal and run.pending:
        lines.append(run.pending.decode("utf-8", "replace"))
        run.pending = b""
    return lines


def _handle_gen(run: Run, match: re.Match[str], *, out: TextIO, all_gens: bool) -> None:
    best = float(match.group("best"))
    previous = run.prev_best
    run.prev_best = best
    if not (all_gens or previous is None or best > previous):
        return
    line = (
        f"gen {match.group('gen')}/{match.group('gens')} best={best:.4f} "
        f"zero={match.group('zero')}/{match.group('zero_total')} "
        f"({float(match.group('gen_time')):.0f}s, ETA {float(match.group('eta')):.0f}s)"
    )
    _emit(out, run.run_id, line)


def _is_exception_line(line: str) -> bool:
    """The decisive `ValueError: ...` line that ends one traceback block.

    Unindented (a frame or a `raise ...` continuation is indented) and not one of the
    connectors that introduce another block in a chain.
    """
    if not line.strip() or line[:1].isspace():
        return False
    return not line.startswith(("Traceback", "During handling", "The above exception"))


def _handle_line(run: Run, line: str, *, out: TextIO, all_gens: bool) -> None:
    raw = line.rstrip("\r\n")
    if not raw.strip():
        return
    if run.traceback_left:
        run.traceback_left -= 1
        _emit(out, run.run_id, raw.strip())
        if _is_exception_line(raw):
            run.traceback_left = 0
        return
    if "Traceback" in raw:
        run.traceback_left = TRACEBACK_MAX_LINES
        _emit(out, run.run_id, raw.strip())
        return
    match = HEADER_RE.match(raw)
    if match is None:
        return
    level = match.group("level")
    msg = match.group("msg")
    if level in ERROR_LEVELS:
        _emit(out, run.run_id, msg)
        return
    if "=== GEN " in msg:
        gen = GEN_RE.search(msg)
        if gen is not None:
            _handle_gen(run, gen, out=out, all_gens=all_gens)
        return
    if "=== DISCOVERY COMPLETE" in msg:
        done = COMPLETE_RE.search(msg)
        if done is not None:
            _emit(
                out,
                run.run_id,
                f"COMPLETE {done.group('gens')} gens, {done.group('evaluated')} evaluated, "
                f"{float(done.group('total')):.0f}s",
            )
        return
    if "champion" in msg and level in SUMMARY_LEVELS and match.group("logger").endswith("pipeline"):
        _emit(out, run.run_id, msg)


def _drained(run: Run) -> bool:
    if run.pending:
        return False
    if run.log is None:
        return run.status in TERMINAL_STATUSES
    try:
        return run.log.stat().st_size <= run.offset
    except OSError:
        return True


def watch(
    run_ids: Sequence[str],
    *,
    logs_dir: Path = DEFAULT_LOGS,
    db_path: Path = DEFAULT_DB,
    interval: float = DEFAULT_INTERVAL,
    all_gens: bool = False,
    out: TextIO = sys.stdout,
    sleep: Callable[[float], None] = time.sleep,
    kill_after: float | None = None,
    kill: Callable[[str], str] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Poll the runs' logs and the state DB, printing as events arrive.

    Returns the exit code (0 all done, 1 any run failed/cancelled, 2 unknown run id
    or unreadable DB). ``sleep`` is injectable so tests advance passes without waiting.
    """
    try:
        conn = open_db(db_path)
    except sqlite3.Error as exc:
        print(f"disc_watch: cannot open state DB {db_path}: {exc}", file=sys.stderr, flush=True)
        return UNKNOWN_RC
    try:
        runs = [Run(run_id=run_id) for run_id in run_ids]
        unknown = [run for run in runs if run_status(conn, run.run_id) is None]
        if unknown:
            for run in unknown:
                _emit(out, run.run_id, "UNKNOWN run id")
            return UNKNOWN_RC
        started = clock()
        while True:
            for run in runs:
                status = run_status(conn, run.run_id)
                if status is not None:
                    run.status = status
                terminal = run.status in TERMINAL_STATUSES
                if (
                    kill_after is not None and kill is not None and not terminal
                    and not run.killed and clock() - started >= kill_after
                ):  # fmt: skip
                    run.killed = True
                    _emit(out, run.run_id, f"KILL after {clock() - started:.0f}s ({kill(run.run_id)})")
                if run.log is None:
                    run.log = newest_log(logs_dir, run.run_id)
                    if run.log is None and not run.waiting_printed:
                        _emit(out, run.run_id, "waiting for log")
                        run.waiting_printed = True
                for line in _read_new_lines(run, terminal=terminal):
                    _handle_line(run, line, out=out, all_gens=all_gens)
                if terminal and not run.status_printed:
                    _emit(out, run.run_id, f"STATUS {run.status}")
                    run.status_printed = True
            if all(run.status in TERMINAL_STATUSES and _drained(run) for run in runs):
                break
            sleep(interval)
    finally:
        conn.close()
    return BAD_RC if any(run.status in BAD_STATUSES for run in runs) else OK_RC


def http_cancel(api: str) -> Callable[[str], str]:
    """DELETE /api/discovery/jobs/<id>; returns "HTTP <code>" or the error text (never raises)."""
    import urllib.error
    import urllib.request

    def cancel(run_id: str) -> str:
        req = urllib.request.Request(f"{api.rstrip('/')}/api/discovery/jobs/{run_id}", method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            return f"HTTP {exc.code}"
        except (urllib.error.URLError, OSError) as exc:
            return f"cancel failed: {exc}"

    return cancel


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="disc_watch.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    parser.add_argument("--logs", type=Path, default=None, metavar="DIR")
    parser.add_argument("--db", type=Path, default=None, metavar="PATH")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, metavar="SEC")
    parser.add_argument("--all-gens", action="store_true", help="print every gen, not just improving ones")
    parser.add_argument("--kill-after", type=float, default=None, metavar="SEC",
                        help="cancel runs still going SEC seconds after the watcher started")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="backend for --kill-after")
    args = parser.parse_args(argv)
    root = main_checkout_root()
    env_db = os.environ.get("VIBE_QUANT_DB")
    if args.db is not None:
        db_path = args.db
    elif env_db:
        db_path = Path(env_db)
    else:
        db_path = root / DEFAULT_DB
    return watch(
        args.run_ids,
        logs_dir=args.logs if args.logs is not None else root / DEFAULT_LOGS,
        db_path=db_path,
        interval=args.interval,
        all_gens=args.all_gens,
        kill_after=args.kill_after,
        kill=http_cancel(args.api) if args.kill_after is not None else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
