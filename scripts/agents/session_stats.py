#!/usr/bin/env python3
"""Retro numbers for Claude Code sessions of this repo, plus their T3 worker threads.

    scripts/agents/session_stats.py [--since DAYS] [--session ID ...] [--big CHARS] [--t3-db PATH]

Per chief session (``~/.claude/projects/<repo slug>/<id>.jsonl``, modified within --since
days, default 3): wall span, real user prompts, tool calls by name, tool errors (and how
many were "no active turn" permission failures), tool results over --big chars (default
20000) with the command that produced them, the most repeated Bash commands, the peak
context (input + cache tokens), and the Claude subagents under ``<id>/subagents/``.
Then the T3 delegated worker threads created in the same window (``statev2.sqlite``,
opened read-only): provider, status counts and "Command timed out" hits.

Each retro so far rebuilt these parsers in /tmp (10-17, 10-19). Read-only; stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PROJECTS = Path.home() / ".claude" / "projects" / str(REPO).replace("/", "-")
T3_DB = Path.home() / ".t3" / "userdata" / "statev2.sqlite"


def _blocks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    content = (entry.get("message") or {}).get("content")
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def session_stats(path: Path, big: int) -> dict[str, Any]:
    """The numbers for one transcript (a chief session or a subagent)."""
    tools: Counter[str] = Counter()
    bash: Counter[str] = Counter()
    uses: dict[str, dict[str, Any]] = {}
    errors = no_turn = prompts = peak = 0
    first = last = ""
    large: list[tuple[int, str]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            ts = str(entry.get("timestamp") or "")
            if ts:
                first, last = first or ts, ts
            content = (entry.get("message") or {}).get("content")
            if entry.get("type") == "user" and isinstance(content, str) and not content.startswith("<"):
                prompts += 1
            usage = (entry.get("message") or {}).get("usage")
            if isinstance(usage, dict):
                ctx = sum(int(usage.get(k) or 0) for k in
                          ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))  # fmt: skip
                peak = max(peak, ctx)
            for block in _blocks(entry):
                if block.get("type") == "tool_use":
                    uses[str(block.get("id"))] = block
                    tools[str(block.get("name"))] += 1
                    if block.get("name") == "Bash":
                        bash[str((block.get("input") or {}).get("command", ""))[:80]] += 1
                elif block.get("type") == "tool_result":
                    text = json.dumps(block.get("content"))
                    if block.get("is_error"):
                        errors += 1
                        no_turn += "no active turn" in text
                    if len(text) > big:
                        use = uses.get(str(block.get("tool_use_id")), {})
                        what = (use.get("input") or {}).get("command") or json.dumps(use.get("input"))
                        large.append((len(text), f"{use.get('name')}: {str(what)[:90]}"))
    return {
        "span": f"{first[:16]} -> {last[:16]}", "prompts": prompts, "tools": tools, "errors": errors,
        "no_active_turn": no_turn, "large": sorted(large, reverse=True), "repeats": bash,
        "peak_context": peak,
    }  # fmt: skip


def t3_workers(db: Path, since_iso: str) -> list[tuple[str, str, int]]:
    """(provider, status, count) of delegated worker threads created since ``since_iso``."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT s.provider, s.status, COUNT(*) FROM orchestration_v2_projection_subagents s "
            "WHERE s.updated_at >= ? GROUP BY 1, 2 ORDER BY 3 DESC",
            (since_iso,),
        ).fetchall()
    finally:
        conn.close()


def t3_timeouts(db: Path, since_iso: str) -> int:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM orchestration_v2_projection_turn_items "
            "WHERE updated_at >= ? AND payload_json LIKE '%Command timed out%'",
            (since_iso,),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def report(path: Path, big: int) -> list[str]:
    st = session_stats(path, big)
    lines = [
        f"== {path.stem} {st['span']} prompts={st['prompts']} peak_context={st['peak_context']:,}",
        "  tools: " + ", ".join(f"{n}={c}" for n, c in st["tools"].most_common(8)),
        f"  tool errors: {st['errors']} (no active turn: {st['no_active_turn']})",
    ]
    lines += [f"  big result {n:,} chars <- {what}" for n, what in st["large"][:5]]
    lines += [f"  repeated x{c}: {cmd}" for cmd, c in st["repeats"].most_common(3) if c > 2]
    subs = sorted((path.parent / path.stem / "subagents").glob("*.jsonl"))
    if subs:
        sub_stats = [session_stats(s, big) for s in subs]
        lines.append(
            f"  subagents: {len(subs)}, tool errors {sum(s['errors'] for s in sub_stats)} "
            f"(no active turn {sum(s['no_active_turn'] for s in sub_stats)}), "
            f"big results {sum(len(s['large']) for s in sub_stats)}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", type=float, default=3.0, help="days (default 3)")
    ap.add_argument("--session", nargs="*", default=[], help="session ids (default: all in --since)")
    ap.add_argument("--big", type=int, default=20_000)
    ap.add_argument("--projects", type=Path, default=PROJECTS)
    ap.add_argument("--t3-db", type=Path, default=T3_DB)
    args = ap.parse_args(argv)
    cutoff = time.time() - args.since * 86400
    if args.session:
        paths = [args.projects / f"{s}.jsonl" for s in args.session]
    else:
        paths = sorted(p for p in args.projects.glob("*.jsonl") if p.stat().st_mtime >= cutoff)
    for path in paths:
        for line in report(path, args.big):
            print(line)
    since_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(cutoff))
    if args.t3_db.exists():
        try:
            rows = t3_workers(args.t3_db, since_iso)
            print("== T3 workers since " + since_iso + ": " + ", ".join(f"{p}/{s}={n}" for p, s, n in rows))
            print(f"  'Command timed out' in worker tool calls: {t3_timeouts(args.t3_db, since_iso)}")
        except sqlite3.Error as exc:
            print(f"== T3 db unreadable: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    sys.exit(main())
