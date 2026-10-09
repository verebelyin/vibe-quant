#!/usr/bin/env python3
"""Swarm message bus: agents <-> agents <-> chief (orchestrator), over a shared JSONL file.

Any agent with a shell can use it (Claude subagents, `cmd` workers). One bus per job:
``data/swarm/<job>/bus/messages.jsonl`` (append-only, flock-guarded) plus per-agent read cursors.

    bus.py post  --to chief|all|<agent> --kind info|question|blocker|answer|claim|done --body TEXT [--ref MSG_ID]
    bus.py inbox [--peek]            # new messages for me (to me or to all), advances my cursor
    bus.py ask   --to chief --body TEXT [--timeout 900]   # post a question, block until an answer refs it
    bus.py wait  [--timeout 600]     # block until my inbox has something new
    bus.py tail  [--to chief] [--since-start]  # follow messages (chief: run under a Monitor)
    bus.py roster                    # agents seen, message counts, last activity
  Message board (persistent topics everyone can browse; threads via --ref):
    bus.py board post   --topic design --body TEXT [--kind info|question|claim|...]
    bus.py board read   [--topic design] [--all]   # unread posts (per-agent, per-topic cursor); --all = full history
    bus.py board topics                            # topics with post counts + latest post
    bus.py reply  --ref MSG_ID --body TEXT          # threads under a board post, or answers a DM
    bus.py thread MSG_ID                            # a post and all replies, indented
    bus.py board render [--out board.md]            # Markdown view of the whole board (for humans)

Identity/location come from env (set by cmd-task.sh / worktree briefs) or flags:
    SWARM_BUS=<dir> (or --bus)   SWARM_AGENT=<name> (or --as)
Stdlib only; runs with any python3.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

KINDS = ("info", "question", "blocker", "answer", "claim", "done")
POLL_S = 2.0


def _bus_dir(args: argparse.Namespace) -> Path:
    raw = args.bus or os.environ.get("SWARM_BUS")
    if not raw:
        sys.exit("bus: set SWARM_BUS or pass --bus <dir>")
    d = Path(raw)
    (d / "cursors").mkdir(parents=True, exist_ok=True)
    return d


def _me(args: argparse.Namespace) -> str:
    me = args.as_ or os.environ.get("SWARM_AGENT")
    if not me:
        sys.exit("bus: set SWARM_AGENT or pass --as <name>")
    return str(me)


def _append(bus: Path, msg: dict[str, object]) -> None:
    path = bus / "messages.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _read_from(bus: Path, offset: int) -> tuple[list[dict[str, object]], int]:
    """Messages after byte ``offset`` (complete lines only) and the new offset."""
    path = bus / "messages.jsonl"
    if not path.exists():
        return [], offset
    with path.open("rb") as fh:
        fh.seek(offset)
        data = fh.read()
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    chunk = data[: end + 1]
    msgs = []
    for line in chunk.splitlines():
        try:
            msgs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return msgs, offset + len(chunk)


def _for_me(msg: dict[str, object], me: str) -> bool:
    return msg.get("from") != me and msg.get("to") in (me, "all")


def _cursor_path(bus: Path, me: str) -> Path:
    return bus / "cursors" / f"{me}.offset"


def _get_cursor(bus: Path, me: str) -> int:
    p = _cursor_path(bus, me)
    return int(p.read_text()) if p.exists() else 0


def _set_cursor(bus: Path, me: str, off: int) -> None:
    p = _cursor_path(bus, me)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(off))


def fmt(msg: dict[str, object]) -> str:
    ref = f" re:{msg['ref']}" if msg.get("ref") else ""
    if msg.get("to") == "board":
        return f"[{msg['id']}] #{msg.get('topic')} {msg['from']} ({msg['kind']}{ref}): {msg['body']}"
    return f"[{msg['id']}] {msg['from']} -> {msg['to']} ({msg['kind']}{ref}): {msg['body']}"


def post(
    bus: Path, me: str, to: str, kind: str, body: str, ref: str | None = None, topic: str | None = None
) -> dict[str, object]:
    if kind not in KINDS:
        sys.exit(f"bus: kind must be one of {KINDS}")
    msg: dict[str, object] = {
        "id": f"{time.time_ns():x}-{os.getpid() % 10000:04d}",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "from": me,
        "to": to,
        "kind": kind,
        "body": body,
    }
    if ref:
        msg["ref"] = ref
    if to == "board":
        if not topic:
            sys.exit("bus: board posts need --topic")
        msg["topic"] = topic.lstrip("#").lower()
    _append(bus, msg)
    return msg


def _all(bus: Path) -> list[dict[str, object]]:
    return _read_from(bus, 0)[0]


def inbox(bus: Path, me: str, peek: bool = False) -> list[dict[str, object]]:
    """Direct messages, broadcasts, and replies to anything I posted (incl. board threads)."""
    msgs, off = _read_from(bus, _get_cursor(bus, me))
    if not peek:
        _set_cursor(bus, me, off)
    mine = {m["id"] for m in _all(bus) if m.get("from") == me}
    return [m for m in msgs if _for_me(m, me) or (m.get("from") != me and m.get("ref") in mine)]


def board_read(bus: Path, me: str, topic: str | None = None, everything: bool = False) -> list[dict[str, object]]:
    key = f"{me}.board.{(topic or '_all').lstrip('#').lower()}"
    msgs, off = _read_from(bus, 0 if everything else _get_cursor(bus, key))
    if not everything:
        _set_cursor(bus, key, off)
    out = [m for m in msgs if m.get("to") == "board"]
    if topic:
        out = [m for m in out if m.get("topic") == topic.lstrip("#").lower()]
    return out


def topics(bus: Path) -> dict[str, dict[str, object]]:
    seen: dict[str, dict[str, object]] = {}
    for m in _all(bus):
        if m.get("to") != "board":
            continue
        t = seen.setdefault(str(m["topic"]), {"posts": 0})
        t["posts"] = int(t["posts"]) + 1  # type: ignore[call-overload]
        t["last"] = m
    return seen


def reply(bus: Path, me: str, ref: str, body: str, kind: str = "info") -> dict[str, object]:
    parent = next((m for m in _all(bus) if m["id"] == ref), None)
    if parent is None:
        sys.exit(f"bus: no message {ref}")
    if parent.get("to") == "board":
        return post(bus, me, "board", kind, body, ref=ref, topic=str(parent["topic"]))
    target = parent["from"] if parent["from"] != me else parent["to"]
    if kind == "info" and parent.get("kind") == "question":
        kind = "answer"
    return post(bus, me, str(target), kind, body, ref=ref)


def thread(bus: Path, msg_id: str) -> list[tuple[int, dict[str, object]]]:
    """(depth, message) for the thread containing ``msg_id``, root first."""
    msgs = _all(bus)
    by_id = {m["id"]: m for m in msgs}
    root = by_id.get(msg_id)
    if root is None:
        return []
    while root.get("ref") in by_id:
        root = by_id[root["ref"]]
    children: dict[object, list[dict[str, object]]] = {}
    for m in msgs:
        if m.get("ref"):
            children.setdefault(m["ref"], []).append(m)
    out: list[tuple[int, dict[str, object]]] = []

    def walk(m: dict[str, object], depth: int) -> None:
        out.append((depth, m))
        for c in children.get(m["id"], []):
            walk(c, depth + 1)

    walk(root, 0)
    return out


def render(bus: Path) -> str:
    msgs = [m for m in _all(bus) if m.get("to") == "board"]
    ids = {m["id"] for m in msgs}
    lines = ["# Swarm board", ""]
    for t in sorted({str(m["topic"]) for m in msgs}):
        lines += [f"## #{t}", ""]
        for root in (m for m in msgs if m.get("topic") == t and m.get("ref") not in ids):
            for depth, m in thread(bus, str(root["id"])):
                tag = "" if m["kind"] == "info" else f" **{m['kind']}**"
                lines.append(f"{'  ' * depth}- `{m['ts']}` **{m['from']}**{tag}: {m['body']}  <sub>{m['id']}</sub>")
        lines.append("")
    return "\n".join(lines)


def _follow(bus: Path, offset: int, timeout: float | None) -> Iterator[dict[str, object]]:
    deadline = None if timeout is None else time.monotonic() + timeout
    while deadline is None or time.monotonic() < deadline:
        msgs, offset = _read_from(bus, offset)
        yield from msgs
        time.sleep(POLL_S)


def ask(bus: Path, me: str, to: str, body: str, timeout: float) -> dict[str, object] | None:
    start = (bus / "messages.jsonl").stat().st_size if (bus / "messages.jsonl").exists() else 0
    q = post(bus, me, to, "question", body)
    for m in _follow(bus, start, timeout):
        if m.get("ref") == q["id"] and m.get("from") != me:
            return m
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bus.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus")
    ap.add_argument("--as", dest="as_")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("post")
    p.add_argument("--to", required=True)
    p.add_argument("--kind", default="info", choices=KINDS)
    p.add_argument("--body", required=True)
    p.add_argument("--ref")
    i = sub.add_parser("inbox")
    i.add_argument("--peek", action="store_true")
    a = sub.add_parser("ask")
    a.add_argument("--to", default="chief")
    a.add_argument("--body", required=True)
    a.add_argument("--timeout", type=float, default=900)
    w = sub.add_parser("wait")
    w.add_argument("--timeout", type=float, default=600)
    t = sub.add_parser("tail")
    t.add_argument("--to", help="only messages addressed to this name (plus 'all')")
    t.add_argument("--since-start", action="store_true", help="replay history first")
    sub.add_parser("roster")
    r = sub.add_parser("reply")
    r.add_argument("--ref", required=True)
    r.add_argument("--body", required=True)
    r.add_argument("--kind", default="info", choices=KINDS)
    th = sub.add_parser("thread")
    th.add_argument("msg_id")
    bd = sub.add_parser("board")
    bsub = bd.add_subparsers(dest="bcmd", required=True)
    bp = bsub.add_parser("post")
    bp.add_argument("--topic", required=True)
    bp.add_argument("--body", required=True)
    bp.add_argument("--kind", default="info", choices=KINDS)
    br = bsub.add_parser("read")
    br.add_argument("--topic")
    br.add_argument("--all", action="store_true")
    bsub.add_parser("topics")
    bre = bsub.add_parser("render")
    bre.add_argument("--out")
    args = ap.parse_args(argv)
    bus = _bus_dir(args)

    if args.cmd == "post":
        print(fmt(post(bus, _me(args), args.to, args.kind, args.body, args.ref)))
    elif args.cmd == "inbox":
        msgs = inbox(bus, _me(args), args.peek)
        for m in msgs:
            print(fmt(m))
        if not msgs:
            print("(inbox empty)")
    elif args.cmd == "ask":
        ans = ask(bus, _me(args), args.to, args.body, args.timeout)
        if ans is None:
            print(f"(no answer within {args.timeout:.0f}s — proceed with your best judgement and say so in your report)")
            return 3
        print(fmt(ans))
    elif args.cmd == "wait":
        me = _me(args)
        for m in _follow(bus, _get_cursor(bus, me), args.timeout):
            if _for_me(m, me):
                for mm in inbox(bus, me):
                    print(fmt(mm))
                return 0
        print("(nothing new)")
        return 3
    elif args.cmd == "tail":
        path = bus / "messages.jsonl"
        start = 0 if args.since_start or not path.exists() else path.stat().st_size
        for m in _follow(bus, start, None):
            if args.to and m.get("to") not in (args.to, "all", "board"):
                continue
            print(fmt(m), flush=True)
    elif args.cmd == "reply":
        print(fmt(reply(bus, _me(args), args.ref, args.body, args.kind)))
    elif args.cmd == "thread":
        for depth, m in thread(bus, args.msg_id):
            print("  " * depth + fmt(m))
    elif args.cmd == "board":
        if args.bcmd == "post":
            print(fmt(post(bus, _me(args), "board", args.kind, args.body, topic=args.topic)))
        elif args.bcmd == "read":
            msgs = board_read(bus, _me(args), args.topic, args.all)
            for m in msgs:
                print(fmt(m))
            if not msgs:
                print("(no new board posts)")
        elif args.bcmd == "topics":
            for name, info in sorted(topics(bus).items()):
                last = info["last"]
                assert isinstance(last, dict)
                print(f"#{name:16} posts={info['posts']:<4} last={last['ts']} {last['from']}: {str(last['body'])[:60]}")
        elif args.bcmd == "render":
            md = render(bus)
            if args.out:
                Path(args.out).write_text(md + "\n", encoding="utf-8")
                print(f"wrote {args.out}")
            else:
                print(md)
    elif args.cmd == "roster":
        msgs, _ = _read_from(bus, 0)
        seen: dict[str, list[object]] = {}
        for m in msgs:
            s = seen.setdefault(str(m["from"]), [0, ""])
            s[0] = int(s[0]) + 1  # type: ignore[call-overload]
            s[1] = m["ts"]
        for name, (n, ts) in sorted(seen.items()):
            print(f"{name:24} msgs={n:<4} last={ts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
