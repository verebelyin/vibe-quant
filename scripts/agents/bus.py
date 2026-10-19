#!/usr/bin/env python3
"""Swarm message bus: agents <-> agents <-> chief (orchestrator), over a shared JSONL file.

Any agent with a shell can use it (Claude subagents, `cmd` workers). One bus per job:
``data/swarm/<job>/bus/messages.jsonl`` (append-only, flock-guarded) plus per-agent read cursors.

    bus.py post  --to chief|all|<agent> --kind info|question|blocker|answer|claim|done --body TEXT [--ref MSG_ID] [--client-id TEXT] [--no-handoff-check]
    bus.py inbox [--peek]            # new messages for me (to me or to all), advances my cursor
    bus.py pending [--global]        # questions/blockers addressed to me with no reply yet, oldest first,
                                     # one bounded line each (the body is cut to 60 chars); --global =
                                     # persistent board + every job bus (job-tagged), as digest does
    bus.py ask   --to chief --body TEXT [--timeout 900] [--client-id TEXT]   # post a question, block until an answer refs it
    bus.py wait  [--timeout 600]     # block until my inbox has something new
    bus.py tail  [--to chief] [--since-start] [--durable] [--timeout SECONDS]  # follow messages (chief: run under a Monitor)
                                     # --durable persists its own delivery cursor, so a restart
                                     # replays everything that arrived while the watcher was dead;
                                     # one durable tail per (--as, --to): a second exits with an error.
                                     # Deleting its cursor loses the backlog: re-run with --since-start
                                     # --kinds question,blocker emits only those kinds, so a Monitor wakes
                                     # on what it can act on (the cursor still advances past the rest)
    bus.py roster                    # agents seen, message counts, last activity
    bus.py status                    # one screen (<= 20 lines) for the chief (or --as): unanswered
                                     # questions/blockers to it, unread DMs it sent per agent, whether its
                                     # durable tail is alive, the last message, open needs-user items
  Message board (persistent topics everyone can browse; threads via --ref):
    bus.py board post   --topic design --body TEXT [--kind info|question|claim|...] [--client-id TEXT]
    bus.py board read   [--topic design] [--all]   # unread posts (per-agent, per-topic cursor); --all = full history
    bus.py board topics                            # topics with post counts + latest post
    bus.py board search [--topic T ...] [--path P ...] [--grep W ...] [--limit N] [--all]   # no cursor change;
                                     # topics OR, greps AND (case-insensitive); a file matches by path or
                                     # basename, a dir/glob by '<dir>/'; add --global for the persistent board;
                                     # retired/superseded posts are hidden unless --all (then tagged)
    bus.py board retire --ref MSG_ID --reason TEXT # a stale/wrong post: hidden from brief + search, kept in the log
    bus.py board post ... --supersedes MSG_ID      # a corrected post replaces MSG_ID (which is then hidden)
    bus.py reply  --ref MSG_ID --body TEXT          # threads under a post, or answers a DM;
                                     # --ref accepts ANY id `board search --global` prints — the
                                     # reply lands on the bus that HOLDS the post (see it with
                                     # `bus.py thread --global MSG_ID`)
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
import contextlib
import fcntl
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import NoReturn, TypeGuard

_MSG_KEYS = frozenset(("id", "from", "to", "kind", "body"))  # a readable record carries all of these
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


def _append(bus: Path, msg: dict[str, object]) -> dict[str, object]:
    """Append ``msg`` under an exclusive lock; return the stored message.

    With a ``client_id`` the append is idempotent per ``(from, client_id)``: an existing
    message with the same sender and client id is returned instead of a duplicate line.
    """
    path = bus / "messages.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    client_id = msg.get("client_id")
    with path.open("ab+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            data = fh.read()
            if data and not data.endswith(b"\n"):
                # a crashed writer left a torn tail: terminate it, so it is its own line (an
                # undecodable fragment is skipped by readers, a complete record is delivered)
                # instead of swallowing the record appended below
                fh.write(b"\n")
                data += b"\n"
            if client_id:
                for line in data.splitlines():
                    try:
                        existing = json.loads(line)
                    except ValueError:  # torn fragments (incl. a cut multibyte character)
                        continue
                    if _is_message(existing) and existing.get("client_id") == client_id and existing.get("from") == msg.get("from"):
                        fh.flush()
                        os.fsync(fh.fileno())
                        return existing
            fh.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
            return msg
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _beyond_eof(path: Path, offset: int, size: int) -> NoReturn:
    """Refuse an offset past the end of the log: never reset it or wait for the file to grow back."""
    sys.exit(
        f"bus: offset {offset} is beyond the end of {path} ({size} bytes): the log shrank or was replaced "
        "under a reader; delete its cursor file to reset (a durable tail also needs --since-start "
        "to replay its backlog)"
    )


def _is_message(rec: object) -> TypeGuard[dict[str, object]]:
    """A record a reader would deliver: a dict carrying every message key.

    Shared by readers and the client-id dedupe, so a record counts as already posted only
    if some reader can deliver it.
    """
    return isinstance(rec, dict) and rec.keys() >= _MSG_KEYS


def _read_from(bus: Path, offset: int) -> tuple[list[dict[str, object]], int]:
    """Messages after byte ``offset`` (complete lines only) and the new offset."""
    path = bus / "messages.jsonl"
    if not path.exists():
        if offset > 0:
            _beyond_eof(path, offset, 0)
        return [], offset
    with path.open("rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        if offset > size:  # the file shrank under a reader (git checkout/rebase of a tracked board)
            _beyond_eof(path, offset, size)
        fh.seek(offset)
        data = fh.read()
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    chunk = data[: end + 1]
    msgs = []
    for line in chunk.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:  # JSONDecodeError and UnicodeDecodeError (a terminated torn fragment)
            continue
        # a cursor inside a torn fragment can decode its suffix as a scalar/list/partial dict
        # (e.g. ``123``): skip it like an undecodable line so the cursor still advances
        if _is_message(rec):
            msgs.append(rec)
    return msgs, offset + len(chunk)


def _complete_end(bus: Path) -> int:
    """Byte offset just past the last COMPLETE line of messages.jsonl (0 when absent/empty).

    Taken under a shared lock, so a writer mid-append (it holds the exclusive lock) has
    finished first; a partial tail left by a crashed writer is still excluded, because
    a baseline past it would make that message undeliverable forever.
    """
    path = bus / "messages.jsonl"
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        fcntl.flock(fh, fcntl.LOCK_SH)
        try:
            data = fh.read()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return data.rfind(b"\n") + 1


def _for_me(msg: dict[str, object], me: str) -> bool:
    return msg.get("from") != me and msg.get("to") in (me, "all")


def _cursor_path(bus: Path, me: str) -> Path:
    return bus / "cursors" / f"{me}.offset"


def _tail_cursor_path(bus: Path, me: str, to: str) -> Path:
    """A durable tail's delivery cursor — separate from the inbox cursor of ``me``."""
    return bus / "cursors" / f"{me}.tail.{to}"


def _tail_lock_path(bus: Path, me: str, to: str) -> Path:
    """Held for a durable tail's whole lifetime: two tails on one cursor would both deliver."""
    p = _tail_cursor_path(bus, me, to)
    return p.with_name(p.name + ".lock")


def _read_cursor(path: Path) -> int | None:
    """The byte offset stored in ``path``, or None when the file does not exist.

    A present but empty/garbled cursor (a crash mid-write, or an old non-atomic writer) is a
    hard error naming the file: reading it as 0 would silently replay everything, and reading
    it as EOF would silently drop what arrived while the reader was down. None has meaning in
    one place only — the durable tail's first-run rule (no cursor yet -> start at EOF).
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if text.isdigit():
        return int(text)
    sys.exit(f"bus: cursor file {path} is empty or garbled (holds {text!r}); delete it to reset")


def _checked_cursor(bus: Path, path: Path) -> int | None:
    """``_read_cursor`` plus a bounds check against ``bus``'s messages.jsonl.

    An offset past EOF (a truncated/rotated file, a hand edit) would silently hide every
    message up to it: refused like a garbled cursor. Exactly EOF is valid. To recover,
    delete the cursor (a durable tail also needs --since-start to replay its backlog).
    """
    value = _read_cursor(path)
    if value is None:
        return None
    messages = bus / "messages.jsonl"
    size = messages.stat().st_size if messages.exists() else 0
    if value > size:
        sys.exit(
            f"bus: cursor file {path} holds {value}, beyond the end of {messages} ({size} bytes); "
            "delete it to reset (a durable tail also needs --since-start to replay its backlog)"
        )
    return value


def _get_cursor(bus: Path, me: str) -> int:
    value = _checked_cursor(bus, _cursor_path(bus, me))
    return 0 if value is None else value


def _write_cursor(path: Path, off: int) -> None:
    """Persist a cursor atomically: a temp file in the same dir, then ``os.replace``.

    A crash mid-write therefore leaves the previous cursor intact, never a torn/empty file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-cursor-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(int(off)))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _set_cursor(bus: Path, me: str, off: int) -> None:
    _write_cursor(_cursor_path(bus, me), off)


def fmt(msg: dict[str, object]) -> str:
    ref = f" re:{msg['ref']}" if msg.get("ref") else ""
    if msg.get("to") == "board":
        job = f" @{msg['job']}" if msg.get("job") else ""
        sup = f" supersedes:{msg['supersedes']}" if msg.get("supersedes") else ""
        return f"[{msg['id']}] #{msg.get('topic')} {msg['from']}{job} ({msg['kind']}{ref}{sup}): {msg['body']}"
    return f"[{msg['id']}] {msg['from']} -> {msg['to']} ({msg['kind']}{ref}): {msg['body']}"


def _first_chars(body: object, width: int) -> str:
    """A body's first ``width`` characters on one line (newlines collapsed) — for compact reports."""
    text = " ".join(str(body).split())
    return text[:width] + ("…" if len(text) > width else "")


def _one_line(msg: dict[str, object], width: int = 60) -> str:
    """``fmt`` with the body cut to one bounded line: a 3 KB body must not become a 3 KB answer."""
    ref = f" re:{msg['ref']}" if msg.get("ref") else ""
    return f"[{msg['id']}] {msg['from']} -> {msg['to']} ({msg['kind']}{ref}): {_first_chars(msg['body'], width)}"


def parse_kinds(raw: str | None) -> frozenset[str] | None:
    """The kinds a ``tail`` should emit (``question,blocker``), or None for every kind.

    An unknown or empty set is refused: a typo would otherwise silently hide messages.
    """
    if not raw:
        return None
    want = frozenset(k.strip() for k in raw.split(",") if k.strip())
    unknown = want - set(KINDS)
    if unknown:
        _refuse(f"bus: unknown --kinds value(s) {sorted(unknown)}; possible kinds: {KINDS}")
    if not want:
        _refuse(f"bus: --kinds needs at least one kind; possible kinds: {KINDS}")
    return want


def post(
    bus: Path, me: str, to: str, kind: str, body: str, ref: str | None = None, topic: str | None = None,
    mirror: bool = True, extra: dict[str, object] | None = None, client_id: str | None = None,
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
    if client_id:
        msg["client_id"] = client_id
    if extra:
        msg.update(extra)
    if to == "board":
        if not topic:
            sys.exit("bus: board posts need --topic")
        msg["topic"] = topic.lstrip("#").lower()
    stored = _append(bus, msg)
    if mirror and to == "board" and msg["topic"] in PERSISTENT_TOPICS:
        gdir = global_board_dir()
        if gdir.resolve() != bus.resolve():
            _append(gdir, {**stored, "job": bus.resolve().parent.name})
    return stored


def _require_handoff(bus: Path, me: str) -> None:
    """A `done` post needs `<job dir>/handoffs/<me>.md` holding a lint-clean handoff block (exit 2)."""
    path = bus.resolve().parent / "handoffs" / f"{me}.md"
    if not path.is_file():
        _refuse(f"bus: write your final handoff block to {path}, lint it with handoff_lint.py, then post done")
    spec = importlib.util.spec_from_file_location("handoff_lint", Path(__file__).with_name("handoff_lint.py"))
    assert spec and spec.loader
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)
    text = path.read_text()
    problems = lint.check(text, "reviewer" if re.search(r"^\s*verdict\s*:", text, re.IGNORECASE | re.MULTILINE) else None)
    if problems:
        _refuse(f"bus: {path} fails handoff lint:\n" + "\n".join(problems))


def _all(bus: Path) -> list[dict[str, object]]:
    return _read_from(bus, 0)[0]


def _all_with_offsets(bus: Path) -> list[tuple[dict[str, object], int]]:
    """Every deliverable message with the byte offset just past its line (for unread counts)."""
    path = bus / "messages.jsonl"
    if not path.exists():
        return []
    out: list[tuple[dict[str, object], int]] = []
    offset = 0
    for line in path.read_bytes().splitlines(keepends=True):
        offset += len(line)
        if not line.endswith(b"\n"):  # an unfinished record is not deliverable yet
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if _is_message(rec):
            out.append((rec, offset))
    return out


def _locate(bus: Path, ref: str) -> tuple[Path, dict[str, object]] | None:
    """The bus that HOLDS the message ``ref``, as ``(holding_bus, message)``, or None.

    The selected bus is searched first (a reply to something on it behaves exactly as before),
    then the global board, then every job bus (_job_buses) — so any id ``board search --global``
    prints, or that another job's bus holds, can be replied to. No directory is searched twice:
    the selected bus is often the global board or one of the job buses.
    """
    seen: set[str] = set()
    for src in [bus, global_board_dir(), *_job_buses()]:
        key = str(src.resolve())
        if key in seen:
            continue
        seen.add(key)
        hit = next((m for m in _all(src) if m["id"] == ref), None)
        if hit is not None:
            return src, hit
    return None


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


def _path_needles(paths: list[str]) -> set[str]:
    """Strings whose presence in a post body means it talks about one of ``paths``.

    A file matches by its full path or its basename (posts cite ``nt_runner.py:1337``); a
    directory or glob (``vibe_quant/liquidation/**``) by its path or ``<last dir>/``.
    """
    out: set[str] = set()
    for raw in paths:
        p = raw.strip()
        while p.endswith(("*", "/")):
            p = p.rstrip("*").rstrip("/")
        if not p:
            continue
        base = p.rsplit("/", 1)[-1]
        out |= {p, base if "." in base else f"{base}/"}
    return out


def hidden_posts(msgs: list[dict[str, object]]) -> dict[str, str]:
    """Board ids readers skip by default -> why: retired, superseded, a reply under one, a retire event.

    The log stays append-only: a retire is its own record (``retires`` + ``reason``, ref'ing the
    post, so ``thread`` shows it under the post and the author's inbox gets it).
    """
    out: dict[str, str] = {}
    for m in msgs:
        if m.get("retires"):
            out[str(m["retires"])] = f"retired: {m.get('reason') or m['body']}"
            out[str(m["id"])] = "retire event"
        if m.get("supersedes"):
            out.setdefault(str(m["supersedes"]), f"superseded by {m['id']}")
    for m in msgs:  # a reply follows its parent in file order, so one pass covers whole threads
        if m.get("ref") and str(m["ref"]) in out and str(m["id"]) not in out:
            out[str(m["id"])] = f"reply under hidden {m['ref']}"
    return out


def _retirable(ref: str, msg: dict[str, object]) -> None:
    """Refuse (exit 2) to retire/supersede anything but a plain board post."""
    if msg.get("to") != "board":
        _refuse(f"bus: {ref} is not a board post")
    if msg.get("retires"):
        _refuse(f"bus: {ref} is a retire event; post a correction instead")
    if msg.get("topic") == "needs-user":
        _refuse(f"bus: {ref} is a needs-user question: only the user closes it (needs-user answer --user)")


def retire(bus: Path, me: str, ref: str, reason: str) -> dict[str, object] | None:
    """Hide a stale or wrong board post from brief and search, wherever it lives (see ``_locate``).

    None (nothing written) when it is already hidden. The event is posted on the bus that holds
    the post, under the post's topic, so a job post on a persistent topic is retired on the
    global board too (mirroring, as ``post`` does).
    """
    if not reason.strip():
        _refuse("bus: board retire needs a --reason")
    found = _locate(bus, ref)
    if found is None:
        sys.exit(f"bus: no message {ref}")
    holder, parent = found
    _retirable(ref, parent)
    if ref in hidden_posts(_all(holder)):
        return None
    on_global = holder.resolve() == global_board_dir().resolve() and holder.resolve() != bus.resolve()
    extra: dict[str, object] = {"retires": ref, "reason": reason}
    if on_global:
        extra["job"] = bus.resolve().parent.name
    return post(holder, me, "board", "info", f"retired {ref}: {reason}", ref=ref, topic=str(parent.get("topic")), extra=extra)


def supersede(
    bus: Path, me: str, ref: str, topic: str, body: str, kind: str = "info",
    client_id: str | None = None, extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Post a correction that replaces board post ``ref`` (hidden from then on).

    When ``ref`` lives on a bus the new post does not reach (another job, or the global board
    without mirroring), a retire event naming the new post is written there too.
    """
    found = _locate(bus, ref)
    if found is None:
        sys.exit(f"bus: no message {ref}")
    holder, old = found
    _retirable(ref, old)
    new = post(bus, me, "board", kind, body, topic=topic, client_id=client_id, extra={**(extra or {}), "supersedes": ref})
    gdir = global_board_dir().resolve()
    reached = {bus.resolve()} | ({gdir} if new["topic"] in PERSISTENT_TOPICS else set())
    if holder.resolve() not in reached and ref not in hidden_posts(_all(holder)):
        post(
            holder, me, "board", "info", f"retired {ref}: superseded by {new['id']}", ref=ref,
            topic=str(old.get("topic")), mirror=False, extra={"retires": ref, "reason": f"superseded by {new['id']}"},
        )
    return new


def board_search(
    bus: Path,
    topics: list[str] | None = None,
    paths: list[str] | None = None,
    terms: list[str] | None = None,
    limit: int | None = None,
    include_hidden: bool = False,
) -> list[dict[str, object]]:
    """Board posts matching every given filter, oldest first; never moves a read cursor.

    topics: any of them (OR). paths: the body names any of them (see _path_needles).
    terms: the body contains all of them, case-insensitive (AND). limit: keep the newest N.
    Retired/superseded posts (``hidden_posts``) are skipped unless ``include_hidden``; then each
    comes back as a copy carrying ``hidden`` = why.
    """
    want = {t.lstrip("#").lower() for t in topics or []}
    needles = _path_needles(paths or [])
    words = [t.lower() for t in terms or []]
    msgs = _all(bus)
    hidden = hidden_posts(msgs)
    out = []
    for m in msgs:
        if m.get("to") != "board" or (want and m.get("topic") not in want):
            continue
        if str(m["id"]) in hidden:
            if not include_hidden:
                continue
            m = {**m, "hidden": hidden[str(m["id"])]}
        body = str(m["body"])
        if needles and not any(n in body for n in needles):
            continue
        if words and not all(w in body.lower() for w in words):
            continue
        out.append(m)
    return out[-limit:] if limit else out


def topics(bus: Path) -> dict[str, dict[str, object]]:
    seen: dict[str, dict[str, object]] = {}
    for m in _all(bus):
        if m.get("to") != "board":
            continue
        t = seen.setdefault(str(m.get("topic", "")), {"posts": 0})
        t["posts"] = int(t["posts"]) + 1  # type: ignore[call-overload]
        t["last"] = m
    return seen


def reply(
    bus: Path, me: str, ref: str, body: str, kind: str = "info", client_id: str | None = None
) -> dict[str, object]:
    """Thread under a board post, or answer a DM, wherever the parent lives.

    ``ref`` may be any id another bus printed (e.g. by ``board search --global``): the parent is
    looked up with ``_locate`` and the reply is posted on the bus that HOLDS it.
    """
    found = _locate(bus, ref)
    if found is None:
        sys.exit(f"bus: no message {ref}")
    holder, parent = found
    if parent.get("to") == "board":  # inherit the parent's topic; a topic-less parent never blocks a reply
        # a reply to a post that lives on the GLOBAL board is tagged with the REPLIER's job, exactly
        # as post() tags a mirrored post — the board is shared, so the reply stays on it (no second copy)
        on_global = holder.resolve() == global_board_dir().resolve() and holder.resolve() != bus.resolve()
        extra = {"job": bus.resolve().parent.name} if on_global else None
        return post(
            holder, me, "board", kind, body, ref=ref, topic=str(parent.get("topic") or "general"),
            extra=extra, client_id=client_id,
        )
    target = parent["from"] if parent["from"] != me else parent["to"]
    if kind == "info" and parent.get("kind") == "question":
        kind = "answer"
    return post(holder, me, str(target), kind, body, ref=ref, client_id=client_id)


def pending(bus: Path, me: str) -> list[dict[str, object]]:
    """Messages addressed to ``me`` that still expect a reply, oldest first.

    Kind ``question`` (what ``ask`` posts) or ``blocker``, with no message from ``me`` ref'ing
    it yet. Broadcasts (to ``all``) are not listed: only direct messages to ``me``.
    """
    msgs = _all(bus)
    answered = {str(m.get("ref")) for m in msgs if m.get("from") == me and m.get("ref")}
    out = [
        m
        for m in msgs
        if m.get("to") == me and m.get("kind") in ("question", "blocker") and str(m.get("id")) not in answered
    ]
    return sorted(out, key=_ts_key)


def pending_all(me: str) -> list[dict[str, object]]:
    """Pending questions for ``me`` across the global board and every job bus (job-tagged)."""
    out = pending(global_board_dir(), me)
    for jb in _job_buses():
        out.extend({**m, "job": jb.parent.name} for m in pending(jb, me))
    return sorted(out, key=_ts_key)


def unread_from(bus: Path, sender: str) -> dict[str, int]:
    """Unread DMs per recipient for the messages ``sender`` posted on ``bus``.

    A message counts as unread while it sits past the recipient's own inbox cursor
    (an agent that never read anything has no cursor and counts everything).
    """
    counts: dict[str, int] = {}
    for rec, offset in _all_with_offsets(bus):
        to = str(rec.get("to", ""))
        if rec.get("from") != sender or to in ("all", "board", sender):
            continue
        if offset > _get_cursor(bus, to):
            counts[to] = counts.get(to, 0) + 1
    return counts


def _tail_state(bus: Path, me: str) -> str:
    """Whether a durable tail for ``me`` is alive: its lock file is held (see _durable_tail).

    Detection is exactly the tail's own rule — take the lock non-blocking; a held one means a
    live tail. A lock file left behind by a dead tail is therefore reported as ``none``.
    """
    for lock in sorted((bus / "cursors").glob(f"{me}.tail.*.lock")):
        with lock.open("a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return f"alive ({lock.name})"
            fcntl.flock(fh, fcntl.LOCK_UN)
    return "none"


def status_report(bus: Path, me: str, max_items: int = 5) -> list[str]:
    """One screen about ``bus`` from ``me``'s side: at most 20 lines by construction.

    Answers the questions a chief asks on every wake ("did anything arrive, does it reach the
    workers, is anyone waiting on me?") without dumping the bus: counts first, at most
    ``max_items`` items per list, and message bodies cut to their first 60 characters.
    """
    lines = [f"{me} status — {bus}"]
    pend = pending(bus, me)
    lines.append(f"pending to {me}: {len(pend)}")
    lines += [f"  [{m['id']}] {m['from']} {_age_of(m)} — {_first_chars(m['body'], 60)}" for m in pend[:max_items]]
    if len(pend) > max_items:
        lines.append(f"  (+{len(pend) - max_items} more)")
    unread = unread_from(bus, me)
    shown = sorted(unread.items())[:max_items]
    if unread:
        counts = " ".join(f"{agent}={n}" for agent, n in shown)
        extra = f" (+{len(unread) - len(shown)} more agents)" if len(unread) > len(shown) else ""
        lines.append(f"unread from {me}: {len(unread)} agent(s) {counts}{extra}")
    else:
        lines.append(f"unread from {me}: none")
    lines.append(f"{me} tail: {_tail_state(bus, me)}")
    msgs = _all(bus)
    if msgs:
        last = msgs[-1]
        lines.append(
            f"last message: {last.get('ts', '?')} by {last['from']} "
            f"({last['kind']}, {_age_of(last)} old): {_first_chars(last['body'], 60)}"
        )
    else:
        lines.append("last message: none")
    lines.append(f"open needs-user: {len(needs_user_open(bus))}")
    return lines


def _pending_section(msgs: list[dict[str, object]], me: str, max_items: int = 10) -> list[str]:
    """'PENDING QUESTIONS FOR <me>' block for digest; empty when there are none."""
    if not msgs:
        return []
    lines = [f"PENDING QUESTIONS FOR {me} ({len(msgs)}):"]
    return lines + [f"  [PENDING] {fmt(m)} ({_age_of(m)} old)" for m in msgs[:max_items]]


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


def _ts_key(m: dict[str, object]) -> str:
    """Sort key by timestamp; a record without ``ts`` (it is optional) sorts first, in file order."""
    return str(m.get("ts") or "")


def _age_of(m: dict[str, object]) -> str:
    return _age(str(m.get("ts") or ""))


def _age(ts: str) -> str:
    """Compact age of a message timestamp: 30s / 12m / 5h / 3d; ``?`` when it is missing/unparseable."""
    try:
        s = int(time.time() - time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S")))
    except ValueError:
        return "?"
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
    the global board and job buses are searched (``_locate``) and the answer is posted on the
    bus that holds the question. Mirrored to the global board only when the question itself was.
    """
    found = _locate(bus, ref)
    if found is None:
        sys.exit(f"bus: no message {ref}")
    bus, parent = found
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
    return sorted(open_, key=_ts_key)


def needs_user_open_all() -> list[dict[str, object]]:
    """Open needs-user questions everywhere: the global board plus job-only ones, tagged with their job."""
    g = global_board_dir()
    seen = {m["id"] for m in _all(g)}
    out = needs_user_open(g)
    for jb in _job_buses():
        for m in needs_user_open(jb):
            if m["id"] not in seen:  # a mirrored copy on the global board is authoritative
                out.append({**m, "job": jb.parent.name})
    return sorted(out, key=_ts_key)


def _needs_user_section(msgs: list[dict[str, object]], max_items: int = 5) -> list[str]:
    """'OPEN DECISIONS FOR THE USER' block for brief/digest; empty when there are none."""
    if not msgs:
        return []
    lines = [f"OPEN DECISIONS FOR THE USER ({len(msgs)}):"]
    return lines + [f"  [OPEN] {fmt(m)} ({_age_of(m)} old)" for m in msgs[:max_items]]


def render(bus: Path) -> str:
    msgs = [m for m in _all(bus) if m.get("to") == "board"]
    ids = {m["id"] for m in msgs}
    lines = ["# Swarm board", ""]
    for t in sorted({str(m.get("topic", "")) for m in msgs}):
        lines += [f"## #{t}", ""]
        for root in (m for m in msgs if str(m.get("topic", "")) == t and m.get("ref") not in ids):
            for depth, m in thread(bus, str(root["id"])):
                tag = "" if m["kind"] == "info" else f" **{m['kind']}**"
                job = f" @{m['job']}" if m.get("job") else ""
                lines.append(f"{'  ' * depth}- `{m.get('ts', '?')}` **{m['from']}**{job}{tag}: {m['body']}  <sub>{m['id']}</sub>")
        lines.append("")
    return "\n".join(lines)


def _follow(bus: Path, offset: int, timeout: float | None) -> Iterator[dict[str, object]]:
    deadline = None if timeout is None else time.monotonic() + timeout
    while deadline is None or time.monotonic() < deadline:
        msgs, offset = _read_from(bus, offset)
        yield from msgs
        time.sleep(POLL_S)


def ask(bus: Path, me: str, to: str, body: str, timeout: float, client_id: str | None = None) -> dict[str, object] | None:
    start = (bus / "messages.jsonl").stat().st_size if (bus / "messages.jsonl").exists() else 0
    q = post(bus, me, to, "question", body, client_id=client_id)
    # a retried ask (same client id) returns the EXISTING question: its answer may already be
    # on the bus, before ``start``, so look there first; ``start`` still covers the post->follow race
    for m in _all(bus):
        if m.get("ref") == q["id"] and m.get("from") != me:
            return m
    for m in _follow(bus, start, timeout):
        if m.get("ref") == q["id"] and m.get("from") != me:
            return m
    return None


def _durable_tail(
    bus: Path, me: str, to: str, since_start: bool, timeout: float | None,
    kinds: frozenset[str] | None = None, emit: Callable[[str], None] | None = None,
) -> None:
    """A tail that survives a restart: its own cursor, advanced only after printing.

    First run (no cursor file) starts at the current EOF, so old history is not replayed
    (``--since-start`` overrides to 0), and persists that baseline immediately — so messages
    posted while the watcher was dead are replayed on the next start, not skipped. Each
    fetched batch is emitted (and flushed) before the cursor advances: at-least-once, never a
    silent loss. ``kinds`` drops everything the caller cannot act on; the cursor still advances
    past those records, so a filtered message is never replayed either. The baseline is the end
    of the last complete line (an unfinished record is delivered once it completes), and an
    exclusive lock on ``<cursor>.lock`` makes a second tail for the same (as, to) refuse to
    start. ``emit`` is injectable so tests can simulate a crash between those two steps.
    """
    send = emit if emit is not None else (lambda line: print(line, flush=True))
    cpath = _tail_cursor_path(bus, me, to)
    lpath = _tail_lock_path(bus, me, to)
    lpath.parent.mkdir(parents=True, exist_ok=True)
    with lpath.open("a") as lock:  # closed (= released) when the tail ends or is killed
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"bus: another durable tail for {me}/{to or '*'} is already running (lock {lpath})")
        offset = _checked_cursor(bus, cpath)
        if offset is None:
            offset = 0 if since_start else _complete_end(bus)
            _write_cursor(cpath, offset)
        deadline = None if timeout is None else time.monotonic() + timeout
        while deadline is None or time.monotonic() < deadline:
            msgs, new_offset = _read_from(bus, offset)
            for m in msgs:
                if to and m.get("to") not in (to, "all", "board"):
                    continue
                if kinds and m.get("kind") not in kinds:
                    continue
                send(fmt(m))
            if new_offset != offset:
                _write_cursor(cpath, new_offset)
                offset = new_offset
            time.sleep(POLL_S)


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
        "  python3 scripts/agents/bus.py --as <you> board search --global --path <files you will touch> --topic gotchas findings decisions   # catch up",
        "  python3 scripts/agents/bus.py --as <you> board search --global --grep '<keyword>'   # has someone hit this before?",
        "  python3 scripts/agents/bus.py --as <you> board post --topic findings|gotchas|decisions|model-notes|thoughts|chat --body '...'",
        "  python3 scripts/agents/bus.py --as <you> reply --ref <id> --body '...'   |   inbox   |   ask --to chief --body '...'",
        "Post what future agents should know (findings, gotchas, decisions, thoughts); chat freely on #chat.",
    ]
    everything = _all(g)
    hidden = hidden_posts(everything)
    msgs = [m for m in everything if m.get("to") == "board" and str(m["id"]) not in hidden]
    by_topic: dict[str, list[dict[str, object]]] = {}
    for m in msgs:
        by_topic.setdefault(str(m.get("topic", "")), []).append(m)
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
    p.add_argument("--client-id", help="idempotency key: a repeat post with the same (sender, id) is a no-op")
    p.add_argument("--no-handoff-check", action="store_true", help="skip the handoff-file check of --kind done")
    i = sub.add_parser("inbox")
    i.add_argument("--peek", action="store_true")
    pd = sub.add_parser("pending")
    pd.add_argument("--global", dest="glob", action="store_true")
    a = sub.add_parser("ask")
    a.add_argument("--to", default="chief")
    a.add_argument("--body", required=True)
    a.add_argument("--timeout", type=float, default=900)
    a.add_argument("--client-id")
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
    t.add_argument("--durable", action="store_true", help="persist a delivery cursor so a restart replays undelivered messages")
    t.add_argument("--timeout", type=float, default=None, help="stop after SECONDS (default: forever)")
    t.add_argument(
        "--kinds", metavar="K1,K2",
        help=f"emit only these kinds ({'/'.join(KINDS)}); the durable cursor still advances past the rest",
    )
    sub.add_parser("roster")
    sub.add_parser("status")
    dg = sub.add_parser("digest")
    dg.add_argument("--recent", type=int)
    r = sub.add_parser("reply")
    r.add_argument("--ref", required=True)
    r.add_argument("--body", required=True)
    r.add_argument("--kind", default="info", choices=KINDS)
    r.add_argument("--client-id")
    th = sub.add_parser("thread")
    th.add_argument("msg_id")
    th.add_argument("--global", dest="glob", action="store_true")
    bd = sub.add_parser("board")
    bsub = bd.add_subparsers(dest="bcmd", required=True)
    bp = bsub.add_parser("post")
    bp.add_argument("--topic", required=True)
    bp.add_argument("--body", required=True)
    bp.add_argument("--kind", default="info", choices=KINDS)
    bp.add_argument("--client-id")
    bp.add_argument("--supersedes", metavar="MSG_ID", help="this post corrects MSG_ID, which is hidden from then on")
    brt = bsub.add_parser("retire")
    brt.add_argument("--ref", required=True)
    brt.add_argument("--reason", required=True)
    br = bsub.add_parser("read")
    br.add_argument("--topic")
    br.add_argument("--all", action="store_true")
    br.add_argument("--recent", type=int)
    br.add_argument("--global", dest="glob", action="store_true")
    bs = bsub.add_parser("search")
    bs.add_argument("--topic", action="extend", nargs="+", help="one or more; any of them matches")
    bs.add_argument("--path", action="extend", nargs="+", help="one or more files/dirs/globs the post names (a shell-expanded glob works too)")
    bs.add_argument("--grep", action="append", help="repeatable; all must appear (case-insensitive)")
    bs.add_argument("--limit", type=int, default=40, help="newest N hits (default 40)")
    bs.add_argument("--global", dest="glob", action="store_true")
    bs.add_argument("--all", action="store_true", help="include retired/superseded posts (tagged with why)")
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
        if args.kind == "done" and not args.no_handoff_check:
            _require_handoff(bus, _me(args))
        print(fmt(post(bus, _me(args), args.to, args.kind, args.body, args.ref, client_id=args.client_id)))
    elif args.cmd == "inbox":
        msgs = inbox(bus, _me(args), args.peek)
        for m in msgs:
            print(fmt(m))
        if not msgs:
            print("(inbox empty)")
    elif args.cmd == "pending":
        me = _me(args)
        msgs = pending_all(me) if args.glob else pending(bus, me)  # --global = board + every job bus
        for m in msgs:
            job = f" @{m['job']}" if m.get("job") else ""
            print(f"{_one_line(m)}{job} ({_age_of(m)} old)")  # one bounded line: never a body dump
        if not msgs:
            print(f"(no pending questions for {me})")
    elif args.cmd == "ask":
        ans = ask(bus, _me(args), args.to, args.body, args.timeout, client_id=args.client_id)
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
                print(f"{fmt(m)} ({_age_of(m)} old)")
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
        kinds = parse_kinds(args.kinds)
        path = bus / "messages.jsonl"
        if args.durable:
            _durable_tail(bus, _me(args), str(args.to or ""), args.since_start, args.timeout, kinds)
        else:
            start = 0 if args.since_start or not path.exists() else path.stat().st_size
            for m in _follow(bus, start, args.timeout):
                if args.to and m.get("to") not in (args.to, "all", "board"):
                    continue
                if kinds and m.get("kind") not in kinds:
                    continue
                print(fmt(m), flush=True)
    elif args.cmd == "status":
        for line in status_report(bus, _me(args, "chief")):
            print(line)
    elif args.cmd == "reply":
        print(fmt(reply(bus, _me(args), args.ref, args.body, args.kind, client_id=args.client_id)))
    elif args.cmd == "thread":
        for depth, m in thread(bus, args.msg_id):
            print("  " * depth + fmt(m))
    elif args.cmd == "board":
        if args.bcmd == "post":
            if args.supersedes:
                print(fmt(supersede(bus, _me(args), args.supersedes, args.topic, args.body, args.kind, args.client_id)))
            else:
                print(fmt(post(bus, _me(args), "board", args.kind, args.body, topic=args.topic, client_id=args.client_id)))
        elif args.bcmd == "retire":
            event = retire(bus, _me(args), args.ref, args.reason)
            print(fmt(event) if event else f"(already hidden: {args.ref})")
        elif args.bcmd == "read":
            msgs = board_read(bus, _me(args), args.topic, args.all, args.recent)
            for m in msgs:
                print(fmt(m))
            if not msgs:
                print("(no new board posts)")
        elif args.bcmd == "search":
            if not (args.topic or args.path or args.grep):
                _refuse("board search needs at least one --topic, --path or --grep")
            hits = board_search(bus, args.topic, args.path, args.grep, args.limit, include_hidden=args.all)
            for m in hits:
                print(fmt(m) + (f"  [{m['hidden']}]" if m.get("hidden") else ""))
            if not hits:
                print("(no matching board posts)")
        elif args.bcmd == "topics":
            for name, info in sorted(topics(bus).items()):
                last = info["last"]
                assert isinstance(last, dict)
                print(f"#{name:16} posts={info['posts']:<4} last={last.get('ts', '?')} {last['from']}: {str(last['body'])[:60]}")
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
        me = _me(args)
        for line in _pending_section(pending_all(me), me):
            print(line)
        for line in _needs_user_section(needs_user_open_all()):
            print(line)
        groups = digest(me, args.recent)
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
            s[1] = m.get("ts", "?")
        for name, (n, ts) in sorted(seen.items()):
            print(f"{name:24} msgs={n:<4} last={ts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
