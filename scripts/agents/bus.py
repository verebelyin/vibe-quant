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
  Decisions only the human user can make (topic needs-user; surfaced by board brief + digest):
    bus.py ask-user --body "<question + options + recommendation>" [--job-only]  # queue one; prints the message id
    bus.py needs-user open [--global]              # unanswered questions, oldest first, with age
    bus.py needs-user answer --ref MSG_ID --body "<decision>" --user   # the USER's answer (from='user'; 'by' = who relayed it);
                                                                      # refuses (exit 2) without --user: only the user closes decisions
  Persistent board (survives jobs; committed at docs/orchestration/board/, override with SWARM_BOARD):
    posts on PERSISTENT_TOPICS (findings, gotchas, decisions, model-notes) are mirrored there automatically,
    tagged with the job; add --global to board read/topics/render/thread (and --recent N to board read).
  Job buses live in <repo>/data/swarm (found via git's common dir); with SWARM_BOARD set, SWARM_JOBS
  points at the job root and digest/brief skip job buses when it is absent.

Identity/location come from env (set by cmd-task.sh / worktree briefs) or flags:
    SWARM_BUS=<dir> (or --bus)   SWARM_AGENT=<name> (or --as)
  Without a job bus everything goes to the persistent board (the lobby), so any agent can use it any time.
  Overseer:  bus.py --as chief digest   # everything new on the persistent board + every job bus
             bus.py board brief        # onboarding text (printed by the SessionStart hook)
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
    from typing import NoReturn

KINDS = ("info", "question", "blocker", "answer", "claim", "done")
PERSISTENT_TOPICS = ("findings", "gotchas", "decisions", "model-notes", "thoughts", "self-improvement", "needs-user")


def _repo_root() -> Path:
    """The main checkout (not a worktree), via git's common dir."""
    import subprocess

    here = Path(__file__).resolve().parent
    common = subprocess.run(
        ["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(common).parent


def global_board_dir() -> Path:
    """The cross-job board: $SWARM_BOARD, else <main checkout>/docs/orchestration/board.

    Resolved through git's common dir so agents running in worktrees all write the
    main checkout's copy (one file, committed by the orchestrator at landing).
    """
    raw = os.environ.get("SWARM_BOARD")
    d = Path(raw) if raw else _repo_root() / "docs" / "orchestration" / "board"
    d.mkdir(parents=True, exist_ok=True)
    return d
POLL_S = 2.0


def _bus_dir(args: argparse.Namespace) -> Path:
    """Job bus if given (--bus / $SWARM_BUS), else the persistent board = the shared lobby."""
    raw = args.bus or os.environ.get("SWARM_BUS")
    d = Path(raw) if raw else global_board_dir()
    (d / "cursors").mkdir(parents=True, exist_ok=True)
    return d


def _me(args: argparse.Namespace, default: str = "anon") -> str:
    """--as / $SWARM_AGENT; reading works anonymously, but posts should carry a real name."""
    return str(args.as_ or os.environ.get("SWARM_AGENT") or default)


def _refuse(msg: str) -> NoReturn:
    """A refused policy violation: state the rule, exit 2 (argparse's usage-error code)."""
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _append(bus: Path, msg: dict[str, object]) -> None:
    path = bus / "messages.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
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
        job = f" @{msg['job']}" if msg.get("job") else ""
        return f"[{msg['id']}] #{msg.get('topic')} {msg['from']}{job} ({msg['kind']}{ref}): {msg['body']}"
    return f"[{msg['id']}] {msg['from']} -> {msg['to']} ({msg['kind']}{ref}): {msg['body']}"


def post(
    bus: Path, me: str, to: str, kind: str, body: str, ref: str | None = None, topic: str | None = None,
    mirror: bool = True, extra: dict[str, object] | None = None,
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
    if extra:
        msg.update(extra)
    if to == "board":
        if not topic:
            sys.exit("bus: board posts need --topic")
        msg["topic"] = topic.lstrip("#").lower()
    _append(bus, msg)
    if mirror and to == "board" and msg["topic"] in PERSISTENT_TOPICS:
        gdir = global_board_dir()
        if gdir.resolve() != bus.resolve():
            _append(gdir, {**msg, "job": bus.resolve().parent.name})
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


def board_read(
    bus: Path, me: str, topic: str | None = None, everything: bool = False, recent: int | None = None
) -> list[dict[str, object]]:
    key = f"{me}.board.{(topic or '_all').lstrip('#').lower()}"
    full = everything or recent is not None
    msgs, off = _read_from(bus, 0 if full else _get_cursor(bus, key))
    if not everything:
        _set_cursor(bus, key, off)
    out = [m for m in msgs if m.get("to") == "board"]
    if topic:
        out = [m for m in out if m.get("topic") == topic.lstrip("#").lower()]
    return out[-recent:] if recent else out


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


def _age(ts: str) -> str:
    """Compact age of a message timestamp: 30s / 12m / 5h / 3d."""
    s = int(time.time() - time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S")))
    if s < 60:
        return f"{max(s, 0)}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


def ask_user(bus: Path, me: str, body: str, job_only: bool = False) -> dict[str, object]:
    """Queue a decision only the human can make (topic needs-user; mirrored to the global board)."""
    return post(bus, me, "board", "question", body, topic="needs-user", mirror=not job_only)


def needs_user_answer(bus: Path, by: str, ref: str, body: str) -> dict[str, object]:
    """Record the USER's decision on a needs-user question — the only way to close one.

    ``from`` is always "user"; ``by`` names who relayed it (--as/$SWARM_AGENT, or "user").
    Refuses (exit 2) when ``ref`` is not a needs-user question. When it is not on ``bus``,
    the global board and job buses are searched and the answer is posted on the bus that
    holds the question. Mirrored to the global board only when the question itself was.
    """
    parent = next((m for m in _all(bus) if m["id"] == ref), None)
    if parent is None:  # not on the selected bus: look where the questions live
        for src in [global_board_dir(), *_job_buses()]:
            parent = next((m for m in _all(src) if m["id"] == ref), None)
            if parent is not None:
                bus = src
                break
    if parent is None:
        sys.exit(f"bus: no message {ref}")
    if parent.get("kind") != "question" or parent.get("topic") != "needs-user":
        _refuse(f"bus: {ref} is not a needs-user question")
    mirrored = any(m["id"] == ref for m in _all(global_board_dir()))
    return post(
        bus, "user", "board", "answer", body, ref=ref, topic="needs-user",
        mirror=mirrored, extra={"user_answer": True, "by": by},
    )


def needs_user_open(bus: Path) -> list[dict[str, object]]:
    """needs-user questions with no user decision yet, oldest first.

    Only a reply with ``user_answer`` (from `needs-user answer`) closes a question;
    agent replies of any kind are discussion.
    """
    out = [m for m in _all(bus) if m.get("to") == "board" and m.get("topic") == "needs-user" and m.get("kind") == "question"]
    open_ = [m for m in out if not any(r.get("user_answer") for _, r in thread(bus, str(m["id"])))]
    return sorted(open_, key=lambda m: str(m["ts"]))


def needs_user_open_all() -> list[dict[str, object]]:
    """Open needs-user questions everywhere: the global board plus job-only ones, tagged with their job."""
    g = global_board_dir()
    seen = {m["id"] for m in _all(g)}
    out = needs_user_open(g)
    for jb in _job_buses():
        for m in needs_user_open(jb):
            if m["id"] not in seen:  # a mirrored copy on the global board is authoritative
                out.append({**m, "job": jb.parent.name})
    return sorted(out, key=lambda m: str(m["ts"]))


def _needs_user_section(msgs: list[dict[str, object]], max_items: int = 5) -> list[str]:
    """'OPEN DECISIONS FOR THE USER' block for brief/digest; empty when there are none."""
    if not msgs:
        return []
    lines = [f"OPEN DECISIONS FOR THE USER ({len(msgs)}):"]
    return lines + [f"  [OPEN] {fmt(m)} ({_age(str(m['ts']))} old)" for m in msgs[:max_items]]


def render(bus: Path) -> str:
    msgs = [m for m in _all(bus) if m.get("to") == "board"]
    ids = {m["id"] for m in msgs}
    lines = ["# Swarm board", ""]
    for t in sorted({str(m["topic"]) for m in msgs}):
        lines += [f"## #{t}", ""]
        for root in (m for m in msgs if m.get("topic") == t and m.get("ref") not in ids):
            for depth, m in thread(bus, str(root["id"])):
                tag = "" if m["kind"] == "info" else f" **{m['kind']}**"
                job = f" @{m['job']}" if m.get("job") else ""
                lines.append(f"{'  ' * depth}- `{m['ts']}` **{m['from']}**{job}{tag}: {m['body']}  <sub>{m['id']}</sub>")
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


def _job_buses() -> list[Path]:
    """Job buses: $SWARM_JOBS root, else <repo>/data/swarm via git's common dir.

    With $SWARM_BOARD set (a scratch/shallow board) there is no repo layout to search,
    so job buses are skipped unless $SWARM_JOBS says where they are.
    """
    raw = os.environ.get("SWARM_JOBS")
    if raw:
        root = Path(raw)
    elif os.environ.get("SWARM_BOARD"):
        return []
    else:
        root = _repo_root() / "data" / "swarm"
    return sorted(p for p in root.glob("*/bus") if (p / "messages.jsonl").exists())


def digest(me: str, recent: int | None = None) -> list[tuple[str, list[dict[str, object]]]]:
    """Everything new for an overseer: the persistent board + every job bus (all messages, not just mine)."""
    out: list[tuple[str, list[dict[str, object]]]] = []
    for src in [global_board_dir(), *_job_buses()]:
        key = f"{me}.digest"
        if recent:
            msgs = _all(src)[-recent:]
        else:
            msgs, off = _read_from(src, _get_cursor(src, key))
            _set_cursor(src, key, off)
        if msgs:
            label = "persistent board" if src == global_board_dir() else f"job {src.parent.name}"
            out.append((label, msgs))
    return out


def brief(per_topic: int = 3) -> str:
    """Compact onboarding text: how to use the board + the latest posts per persistent topic."""
    g = global_board_dir()
    lines = [
        "SWARM BOARD — shared memory + chat for every agent (persistent across sessions/jobs).",
        "  python3 scripts/agents/bus.py --as <you> board read --global --recent 30   # catch up",
        "  python3 scripts/agents/bus.py --as <you> board post --topic findings|gotchas|decisions|model-notes|thoughts|chat --body '...'",
        "  python3 scripts/agents/bus.py --as <you> reply --ref <id> --body '...'   |   inbox   |   ask --to chief --body '...'",
        "Post what future agents should know (findings, gotchas, decisions, thoughts); chat freely on #chat.",
    ]
    msgs = [m for m in _all(g) if m.get("to") == "board"]
    by_topic: dict[str, list[dict[str, object]]] = {}
    for m in msgs:
        by_topic.setdefault(str(m["topic"]), []).append(m)
    lines += _needs_user_section(needs_user_open_all())
    by_topic = {t: ms for t, ms in by_topic.items() if t != "needs-user"}  # shown in the OPEN section above
    if by_topic:
        lines.append("Latest posts:")
        for t in sorted(by_topic):
            for m in by_topic[t][-per_topic:]:
                body = str(m["body"])
                lines.append(f"  #{t} {m['from']}: {body[:170]}{'…' if len(body) > 170 else ''}")
    return "\n".join(lines)


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
    au = sub.add_parser("ask-user")
    au.add_argument("--body", required=True)
    au.add_argument("--job-only", action="store_true", help="keep it on the job bus (don't mirror to the global board)")
    nu = sub.add_parser("needs-user")
    nusub = nu.add_subparsers(dest="ncmd", required=True)
    nuo = nusub.add_parser("open")
    nuo.add_argument("--global", dest="glob", action="store_true")
    nua = nusub.add_parser("answer")
    nua.add_argument("--ref", required=True)
    nua.add_argument("--body", required=True)
    nua.add_argument(
        "--user", action="store_true",
        help="required: only the user closes decisions. --body must be the user's verbatim words — "
             "typed by them, or relayed by the chief. Without --user this refuses (exit 2).",
    )
    t = sub.add_parser("tail")
    t.add_argument("--to", help="only messages addressed to this name (plus 'all')")
    t.add_argument("--since-start", action="store_true", help="replay history first")
    sub.add_parser("roster")
    dg = sub.add_parser("digest")
    dg.add_argument("--recent", type=int)
    r = sub.add_parser("reply")
    r.add_argument("--ref", required=True)
    r.add_argument("--body", required=True)
    r.add_argument("--kind", default="info", choices=KINDS)
    th = sub.add_parser("thread")
    th.add_argument("msg_id")
    th.add_argument("--global", dest="glob", action="store_true")
    bd = sub.add_parser("board")
    bsub = bd.add_subparsers(dest="bcmd", required=True)
    bp = bsub.add_parser("post")
    bp.add_argument("--topic", required=True)
    bp.add_argument("--body", required=True)
    bp.add_argument("--kind", default="info", choices=KINDS)
    br = bsub.add_parser("read")
    br.add_argument("--topic")
    br.add_argument("--all", action="store_true")
    br.add_argument("--recent", type=int)
    br.add_argument("--global", dest="glob", action="store_true")
    bt = bsub.add_parser("topics")
    bt.add_argument("--global", dest="glob", action="store_true")
    bbr = bsub.add_parser("brief")
    bbr.add_argument("--per-topic", type=int, default=3)
    bre = bsub.add_parser("render")
    bre.add_argument("--out")
    bre.add_argument("--global", dest="glob", action="store_true")
    args = ap.parse_args(argv)
    if getattr(args, "glob", False):
        args.bus = str(global_board_dir())
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
    elif args.cmd == "ask-user":
        print(ask_user(bus, _me(args), args.body, args.job_only)["id"])
    elif args.cmd == "needs-user":
        if args.ncmd == "open":
            open_qs = needs_user_open(bus)
            for m in open_qs:
                print(f"{fmt(m)} ({_age(str(m['ts']))} old)")
            if not open_qs:
                print("(no open decisions for the user)")
        else:
            if not args.user:
                _refuse("needs-user answer requires --user: only the user closes decisions; relay the user's verbatim words")
            print(fmt(needs_user_answer(bus, _me(args, "user"), args.ref, args.body)))
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
            msgs = board_read(bus, _me(args), args.topic, args.all, args.recent)
            for m in msgs:
                print(fmt(m))
            if not msgs:
                print("(no new board posts)")
        elif args.bcmd == "topics":
            for name, info in sorted(topics(bus).items()):
                last = info["last"]
                assert isinstance(last, dict)
                print(f"#{name:16} posts={info['posts']:<4} last={last['ts']} {last['from']}: {str(last['body'])[:60]}")
        elif args.bcmd == "brief":
            print(brief(args.per_topic))
        elif args.bcmd == "render":
            md = render(bus)
            if args.out:
                Path(args.out).write_text(md + "\n", encoding="utf-8")
                print(f"wrote {args.out}")
            else:
                print(md)
    elif args.cmd == "digest":
        for line in _needs_user_section(needs_user_open_all()):
            print(line)
        groups = digest(_me(args), args.recent)
        for label, msgs in groups:
            print(f"== {label} ({len(msgs)} new)")
            for m in msgs:
                print("  " + fmt(m))
        if not groups:
            print("(nothing new on any board)")
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
