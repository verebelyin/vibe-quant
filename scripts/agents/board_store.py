"""bus.py's board store: the local Vercel Board read and written as OLD-shape bus messages.

bus.py keeps all its logic; its store functions (``_all``, ``_append``, ``_job_buses``)
call ``read``, ``append`` and ``job_channels`` here when ``SWARM_STORE=board``, so every
chief tool (brief, search, retire, pending, needs-user, digest) works on the board
unchanged. Design rulings R5/R6/R9/R11/R12/R14 of job 20261120-board.

Layout (the importer, board_io.py, writes old data the same way):
  channels  ``kb/<topic>`` (KB_TOPICS), ``needs-user``, ``chat``, ``lobby`` (a global message
            without one of those topics), ``job/<job slug>`` (everything else of one job);
            ``$SWARM_BOARD_NS/`` is prepended to every channel (tests: one namespace each)
  labels    legacy-id, ts, job, importance, kind, to, topic, from, ref, user-answer,
            relayed-by, retires, supersedes, state (+ reason, superseded-by), agent/bead/role

Reads and writes go over the board's HTTP API through ONE small client (``_http``): a
message costs ~3 ms there against ~180 ms per `vb` process. Reads are cached per process
and revalidated by the board's head cursor (``GET /events?last=1``) on every call, so a
change by anyone (a label too) is seen. Every write is scanned with
``boardctl.looks_like_secret`` first (exit 2, nothing sent), as the `vb` launcher does.
The token is ``$BOARD_TOKEN``, else the one `vb serve` signs for ``$BOARD_URL`` under
``$XDG_CONFIG_HOME/board/local/`` (where `vb` itself finds it). Never the SQLite file.
Stdlib only.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import boardctl  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

Raw = dict[str, object]

KB_TOPICS = ("gotchas", "findings", "decisions", "model-notes", "thoughts", "self-improvement")
TOP_TOPICS = ("needs-user", "chat")
LOBBY = "lobby"
PAGE = 500  # GET /messages page size (the route's maximum)
_PREFIX_KIND = re.compile(r"^\s*\[(question|blocker|done|claim)\]", re.I)
_PROFILE_OK = re.compile(r"[a-z][a-z0-9-]{0,31}")  # the board stores profiles lowercased
_TS_FMT = "%Y-%m-%dT%H:%M:%S"
_HTTP_TIMEOUT_S = 30.0  # one board HTTP request; $SWARM_BOARD_TIMEOUT (seconds) overrides
_WORKERS = 8  # concurrent reads of one board (one request costs ~3 ms, mostly waiting)
_START = "start it with scripts/agents/board-up.sh"
#: A message whose own state label hides it (S1): what ``bus.hidden_posts`` shows as the reason.
HIDDEN_BY_LABEL = "hidden_by_label"


# ------------------------------------------------------------------ naming (pure)


def job_slug(job: str) -> str:
    """A job dir name as a channel segment: lowercased, anything outside [a-z0-9_-] -> '-'."""
    return re.sub(r"[^a-z0-9_-]", "-", job.lower())


def ns() -> str:
    """The channel prefix of this process: ``$SWARM_BOARD_NS/`` or nothing."""
    raw = os.environ.get("SWARM_BOARD_NS", "").strip("/")
    return f"{raw}/" if raw else ""


def _strip_ns(channel: str, prefix: str) -> str:
    return channel[len(prefix):] if prefix and channel.startswith(prefix) else channel


def topic_channel(topic: str) -> str | None:
    """The global channel a topic owns, or None (it then lives in lobby or a job channel)."""
    if topic in KB_TOPICS:
        return f"kb/{topic}"
    return topic if topic in TOP_TOPICS else None


def channel_topic(channel: str) -> str | None:
    """The topic a channel (no namespace) implies: ``kb/<t>`` -> t, needs-user/chat -> itself."""
    if channel.startswith("kb/"):
        return channel[3:]
    return channel if channel in TOP_TOPICS else None


def global_channels() -> list[str]:
    """Every channel (no namespace) the global board dir sees."""
    return [f"kb/{t}" for t in KB_TOPICS] + [*TOP_TOPICS, LOBBY]


def profile_of(name: str) -> str:
    """A bus agent name as a board profile: lowercase letter first, then [a-z0-9-], <= 32.

    The board lowercases a profile itself (``arch-L2`` reads back ``arch-l2``), so any other
    name is written with a ``from`` label that keeps it."""
    if _PROFILE_OK.fullmatch(name):
        return name
    out = re.sub(r"[^a-z0-9-]", "-", name.lower())
    if not out[:1].isalpha():
        out = "a" + out
    return out[:32]


def _one_line(text: object, width: int) -> str:
    return " ".join(str(text).split())[:width]


def _local_ts(created_at: str) -> str:
    """An ISO UTC ``createdAt`` as the bus's local ``%Y-%m-%dT%H:%M:%S``."""
    stamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return time.strftime(_TS_FMT, time.localtime(stamp.timestamp()))


# ------------------------------------------------------------------ READ mapping (pure)


def _labels(m: Raw) -> dict[str, str]:
    raw = m.get("labels")
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _prefix_kind(text: str) -> str | None:
    hit = _PREFIX_KIND.match(text)
    return hit.group(1).lower() if hit else None


def _is_question(m: Raw) -> bool:
    """Whether ``m`` reads back as a question (only its own label or text prefix can make it one)."""
    lab = _labels(m).get("kind")
    return lab == "question" if lab else _prefix_kind(str(m.get("text") or "")) == "question"


def derived_kind(text: str, parent: Raw | None) -> str:
    """The kind a message without a ``kind`` label reads back as."""
    return _prefix_kind(text) or ("answer" if parent is not None and _is_question(parent) else "info")


def _parent_id(m: Raw) -> str | None:
    for key in ("replyTo", "re"):
        if m.get(key):
            return str(m[key])
    return None


def map_messages(
    raw: list[Raw], targets: Mapping[str, Raw] | None = None, prefix: str = ""
) -> list[dict[str, object]]:
    """Board messages (whole threads, board order) -> old-shape bus messages, deleted skipped.

    ``targets`` holds board messages outside ``raw`` that a ``retires``/``supersedes`` label
    or a reply may point at (by board id); ``prefix`` is the namespace to strip from channels.
    """
    by_id: dict[str, Raw] = {**(targets or {}), **{str(m["id"]): m for m in raw}}

    def old_id(board_id: str) -> str:
        hit = by_id.get(board_id)
        return _labels(hit).get("legacy-id", board_id) if hit is not None else board_id

    out: list[dict[str, object]] = []
    for m in raw:
        if m.get("deleted"):
            continue
        lab = _labels(m)
        text = str(m.get("text") or "")
        channel = _strip_ns(str(m.get("channel") or ""), prefix)
        by = m.get("by")
        profile = by.get("profile") if isinstance(by, dict) else None
        parent = by_id.get(_parent_id(m) or "")
        root = by_id.get(str(m.get("re") or m["id"]))
        topic = lab.get("topic") or channel_topic(channel)
        sender = lab.get("from") or (str(profile) if profile else "user")
        if "to" in lab:
            to = lab["to"]
        elif topic:
            to = "board"
        elif sender != "chief":
            to = "chief"
        else:
            to = _labels(root).get("agent", "all") if root is not None else "all"
        msg: dict[str, object] = {
            "id": old_id(str(m["id"])),
            "ts": lab.get("ts") or _local_ts(str(m.get("createdAt"))),
            "from": sender,
            "to": to,
            "kind": lab.get("kind") or derived_kind(text, parent),
            "body": text,
        }
        if topic:
            msg["topic"] = topic
        if _parent_id(m):
            msg["ref"] = old_id(str(_parent_id(m)))
        elif "ref" in lab:
            msg["ref"] = lab["ref"]
        job = lab.get("job") or (channel[4:] if channel.startswith("job/") else None)
        if job:
            msg["job"] = job
        if lab.get("importance", "").isdigit():
            msg["importance"] = int(lab["importance"])
        if "retires" in lab:
            msg["retires"] = old_id(lab["retires"])
            target = by_id.get(lab["retires"])
            msg["reason"] = _labels(target).get("reason", "") if target is not None else ""
        if "supersedes" in lab:
            msg["supersedes"] = old_id(lab["supersedes"])
        if lab.get("user-answer") == "true":
            msg["user_answer"] = True
        if "relayed-by" in lab:
            msg["by"] = lab["relayed-by"]
        if lab.get("state") in ("retired", "superseded"):  # hidden even without a retire/supersede record
            msg[HIDDEN_BY_LABEL] = f"{lab['state']}: {lab.get('reason') or 'state label'}"
        out.append(msg)
    return out


# ------------------------------------------------------------------ WRITE mapping (pure)


class WritePlan(NamedTuple):
    channel: str | None  # a new thread's channel (namespace included); None for a reply
    re: str | None  # the board id replied to (replyTo is kept)
    text: str
    labels: dict[str, str]
    profile: str
    key: str | None  # idempotency key: <from>:<client_id>


def plan_write(
    msg: Mapping[str, object], *, job: str | None, to_global: bool, lookup: Mapping[str, Raw], ns: str = "",
) -> WritePlan:
    """Where and how an old-shape message is written. ``job`` = the job bus it is appended
    to (None = the global board dir); ``to_global`` = a persistent-topic post that bus.py
    mirrors, written ONCE to its global channel; ``lookup`` = old id -> board message."""
    ns = f"{ns.strip('/')}/" if ns.strip("/") else ""
    topic = str(msg["topic"]) if msg.get("topic") else None
    sender = str(msg["from"])
    parent = lookup.get(str(msg["ref"])) if msg.get("ref") else None
    labels: dict[str, str] = {}
    if parent is not None:
        channel = None
        landing = _strip_ns(str(parent.get("channel") or ""), ns)
    else:
        own = topic_channel(topic) if topic and (job is None or to_global) else None
        landing = own or (LOBBY if job is None else f"job/{job_slug(job)}")
        channel = ns + landing
        if landing.startswith("kb/"):
            labels["state"] = "live"
    if topic and channel_topic(landing) != topic:
        labels["topic"] = topic
    tag = msg.get("job") or job
    if tag:
        labels["job"] = str(tag)
    if msg.get("to") != "board":
        labels["to"] = str(msg["to"])
    if msg.get("kind") != derived_kind(str(msg["body"]), parent):
        labels["kind"] = str(msg["kind"])
    if msg.get("importance") is not None:
        labels["importance"] = str(msg["importance"])
    if msg.get("ref") and parent is None:
        labels["ref"] = str(msg["ref"])
    for field in ("retires", "supersedes"):
        if msg.get(field):
            target = lookup.get(str(msg[field]))
            labels[field] = str(target["id"]) if target is not None else str(msg[field])
    if msg.get("user_answer"):
        labels["user-answer"] = "true"
    if msg.get("by"):
        labels["relayed-by"] = _one_line(msg["by"], 200)
    profile = profile_of(sender)
    if profile != sender:
        labels["from"] = _one_line(sender, 200)
    key = f"{sender}:{msg['client_id']}" if msg.get("client_id") else None
    return WritePlan(channel, str(parent["id"]) if parent is not None else None, str(msg["body"]), labels, profile, key)


def side_effects(
    msg: Mapping[str, object], new_id: str, lookup: Mapping[str, Raw]
) -> list[tuple[str, dict[str, str], dict[str, str | None] | None]]:
    """Labels a write sets on OTHER messages: (board id, labels, ``if`` precondition or None).

    A retire event marks its target ``state=retired`` + ``reason``; a supersede marks the old
    post ``state=superseded`` + ``superseded-by``, unless it is already retired/superseded: the
    write holds only while ``state`` is still what this process read (``live`` or unset), so of
    two concurrent supersedes the first wins, as ``bus.hidden_posts`` says.
    """
    out: list[tuple[str, dict[str, str], dict[str, str | None] | None]] = []
    target = lookup.get(str(msg.get("retires"))) if msg.get("retires") else None
    if target is not None:
        out.append((str(target["id"]), {"state": "retired", "reason": _one_line(msg.get("reason") or msg["body"], 200)}, None))
    old = lookup.get(str(msg.get("supersedes"))) if msg.get("supersedes") else None
    state = _labels(old).get("state") if old is not None else None
    if old is not None and state not in ("retired", "superseded"):
        out.append((str(old["id"]), {"state": "superseded", "superseded-by": new_id}, {"state": state}))
    return out


# ------------------------------------------------------------------ the HTTP client


def board_url() -> str:
    return os.environ.get("BOARD_URL") or boardctl.default_url()


def _token(url: str) -> str:
    """$BOARD_TOKEN, else the token `vb serve` signed for ``url`` (where `vb` finds it)."""
    if os.environ.get("BOARD_TOKEN"):
        return os.environ["BOARD_TOKEN"]
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "board" / "local"
    for path in sorted(base.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("url") == url and data.get("token"):
            return str(data["token"])
    sys.exit(f"bus: no token for the board at {url}: {_START} (or set BOARD_TOKEN)")


def _timeout() -> float:
    """Seconds one board HTTP request may take: ``$SWARM_BOARD_TIMEOUT``, else 30 (a bad value exits 2)."""
    raw = os.environ.get("SWARM_BOARD_TIMEOUT", "").strip()
    if not raw:
        return _HTTP_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        print(f"bus: SWARM_BOARD_TIMEOUT must be a positive number of seconds, not {_one_line(raw, 40)!r}", file=sys.stderr)
        raise SystemExit(2)
    return value


def _is_msg(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get("id"), str | int) and not isinstance(value.get("id"), bool)


def _is_msgs(value: object) -> bool:
    return isinstance(value, list) and all(_is_msg(m) for m in value)


#: What a field the caller names in ``_http(expect=...)`` must hold (R7: a malformed page exits).
_SHAPES: dict[str, Callable[[object], bool]] = {
    "messages": _is_msgs,
    "replies": _is_msgs,
    "message": _is_msg,
    "cursor": lambda v: isinstance(v, str),
    "channels": lambda v: isinstance(v, list) and all(isinstance(c, dict) and isinstance(c.get("path"), str) for c in v),
    "ok": lambda v: v is True,
}

_ACCEPTED = {404: ("message_not_found", "post_not_found"), 409: ("duplicate_id", "label_conflict")}


def _error_code(raw: bytes) -> str:
    """The ``error.code`` of a board error body ("" when it has none)."""
    try:
        err = json.loads(raw.decode("utf-8")).get("error")
    except (ValueError, AttributeError):
        return ""
    return str(err.get("code") or "") if isinstance(err, dict) else ""


def _http(
    method: str, path: str, body: Mapping[str, object] | None = None, *, profile: str | None = None,
    query: Mapping[str, str] | None = None, accept: tuple[str, ...] = (), expect: tuple[str, ...] = (),
) -> dict[str, object]:
    """One board API call, JSON in and out: the ONLY place this module talks to the board.

    ``accept`` names board error codes the caller handles: a 404 ``message_not_found`` /
    ``post_not_found`` or a 409 ``duplicate_id`` / ``label_conflict`` it lists comes back as
    ``{"error": code}``. ``expect`` names the fields a success must carry, each of the type
    ``_SHAPES`` gives (a message page its ``messages`` list, a thread its ``message`` and
    ``replies``). Everything else (another status or code, a timeout, a refused connection,
    a missing token, a body that is not a JSON object or lacks an expected field) exits
    non-zero with one line: a board error never reads as "no messages"."""
    url = board_url()
    full = url + path + ("?" + urllib.parse.urlencode(query) if query else "")
    headers = {"authorization": f"Bearer {_token(url)}", "content-type": "application/json"}
    if profile:
        headers["x-board-profile"] = profile
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(full, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_timeout()) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()
        code = _error_code(detail)
        if code and code in accept and code in _ACCEPTED.get(exc.code, ()):
            return {"error": code}
        sys.exit(f"bus: board refused {method} {path}: {exc.code} {_one_line(detail.decode('utf-8', errors='replace'), 300)}")
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"bus: board at {url} unreachable ({_one_line(exc, 200)}); {_START}")
    try:
        out = json.loads(raw.decode("utf-8"))
    except ValueError:
        out = None
    if not isinstance(out, dict):
        sys.exit(f"bus: board at {url} answered {method} {path} with no JSON object ({_one_line(raw[:80], 80)}); is it the board? {_START}")
    bad = [f for f in expect if f not in out or not _SHAPES[f](out[f])]
    if bad:
        sys.exit(f"bus: board at {url} answered {method} {path} without a valid {'/'.join(bad)} "
                 f"({_one_line(raw[:80], 80)}); is it the board? {_START}")
    return out


def _advanced(prev: str | None, new: object, what: str) -> str:
    """The next page's cursor: a non-empty string past ``prev`` (numeric cursors must grow),
    else exit: a page that announces more without moving on would loop or truncate."""
    ok = isinstance(new, str) and new != "" and new != prev
    if ok and prev is not None and prev.isdigit() and str(new).isdigit():
        ok = int(str(new)) > int(prev)
    if not ok:
        sys.exit(f"bus: board at {board_url()} paged {what} without an advancing cursor "
                 f"({_one_line(prev, 40)} -> {_one_line(new, 40)}); is it the board? {_START}")
    return str(new)


def _msgs(page: Mapping[str, object], field: str = "messages") -> list[Raw]:
    """A list field ``_http(expect=...)`` already checked."""
    raw = page[field]
    return list(raw) if isinstance(raw, list) else []


def read_channel(channel: str) -> list[Raw]:
    """Every message of one channel (full name), oldest first, deleted included.

    Paged forward from the channel's start (cursor ``0``: on a local board cursors are log
    positions); a missing channel is an empty list (the board answers it as an empty page).
    """
    out: list[Raw] = []
    after = "0"
    while True:
        page = _http("GET", "/messages", query={"channel": channel, "limit": str(PAGE), "after": after}, expect=("messages",))
        msgs = _msgs(page)
        out += msgs
        if len(msgs) < PAGE:
            return out
        after = _advanced(after, page.get("cursor"), f"channel {channel}")


def read_thread(board_id: str) -> list[Raw]:
    """One whole thread (root first) around any of its messages; [] when there is none.

    Paged forward from the root (``after`` = the last reply read), ``PAGE`` replies a call,
    until a short page: a thread of any length reads whole."""
    path = f"/messages/{urllib.parse.quote(board_id, safe='')}"

    def page(after: str) -> dict[str, object]:
        return _http("GET", path, query={"limit": str(PAGE), "after": after},
                     accept=("message_not_found",), expect=("message", "replies"))

    first = page(board_id)
    if first.get("error"):
        return []
    root = first["message"]
    assert isinstance(root, dict)  # _SHAPES checked it
    rid = str(root["id"])
    replies = _msgs(first if rid == board_id else page(rid), "replies")  # :id a reply: start at the root
    out: list[Raw] = [root, *replies]
    after = rid
    while len(replies) >= PAGE:
        after = _advanced(after, str(replies[-1]["id"]), f"thread {rid}")
        replies = _msgs(page(after), "replies")
        out += replies
    return out


def post_raw(channel: str | None, text: str, *, labels: Mapping[str, str] | None = None, profile: str,
             re: str | None = None, key: str | None = None) -> Raw:
    """Post a new thread in ``channel`` or a reply to ``re``; the board's message back.

    The text and every label value are scanned first, as the `vb` launcher does: a secret is
    refused (exit 2, nothing sent). ``key`` makes the post idempotent: a repeat answers the
    first message, also when the board calls it a ``duplicate_id`` (same key, other channel or
    thread). The channel, profile, ``re`` and key are scanned too: they persist as well."""
    refuse_secrets(text, channel, profile, re, key, *(labels or {}).values())
    body: dict[str, object] = {"text": text, "labels": dict(labels or {})}
    body.update({"re": re} if re else {"channel": channel})
    if key:
        body["id"] = key
    answer = _http("POST", "/messages", body, profile=profile, accept=("duplicate_id",) if key else (), expect=("message",))
    if answer.get("error") == "duplicate_id" and key:
        answer = _http("GET", f"/posts/{urllib.parse.quote(key, safe='')}", profile=profile,
                       accept=("post_not_found",), expect=("message",))
        if answer.get("error"):
            sys.exit(f"bus: the board refused key {key} as used by someone else, and holds no post of {profile} under it")
    msg = answer["message"]
    assert isinstance(msg, dict)  # _SHAPES checked it
    invalidate()
    return msg


def refuse_secrets(*values: object) -> None:
    """Exit 2 (nothing sent) when any value looks like a secret (``boardctl.looks_like_secret``)."""
    if any(boardctl.looks_like_secret(str(v)) for v in values if v is not None):
        print("bus: a text or label looks like a secret (KEY=value credential or env dump); nothing was sent", file=sys.stderr)
        raise SystemExit(2)


def _label(board_id: str, labels: Mapping[str, str], when: Mapping[str, str | None] | None) -> bool:
    """Set ``labels`` on one message; with ``when``, only while each key still has that value
    (None = unset). False = the board refused the precondition (409 label_conflict), nothing set."""
    refuse_secrets(*labels.values())
    body: dict[str, object] = {"set": dict(labels), **({"if": dict(when)} if when is not None else {})}
    answer = _http("POST", f"/messages/{urllib.parse.quote(board_id, safe='')}/labels", body,
                   accept=("label_conflict",) if when is not None else (), expect=("ok",))
    invalidate()
    return answer.get("error") != "label_conflict"


def set_labels(board_id: str, labels: Mapping[str, str], *, when: Mapping[str, str | None] | None = None) -> bool:
    """Set labels on one message; every value is scanned first (a secret: exit 2, nothing sent).

    ``when``: only while each key still has that value (None = unset); False = the board
    refused the precondition and set nothing."""
    return _label(board_id, labels, when)


# ------------------------------------------------------------------ cache + public API


class _Cache:
    """What one process read from one (board, namespace); revalidated by the head cursor."""

    def __init__(self) -> None:
        self.head: str | None = None
        self.checked: int | None = None  # the pin generation whose head this cache verified
        self.channels: dict[str, list[Raw]] = {}  # full channel name -> raw messages (this head)
        self.raw: dict[str, Raw] = {}  # board id -> raw message (this head)
        self.created: dict[str, str] = {}  # old id -> createdAt (this head)
        self.views: dict[str | None, list[dict[str, object]]] = {}  # read(job)'s answer (this head)
        self.by_job: dict[str, list[dict[str, object]]] | None = None  # global view by job slug (this head)
        self.jobs: dict[str, str] | None = None  # job_channels()'s answer (this head)
        self.board_id: dict[str, str] = {}  # old id -> board id (ids never change: kept across heads)
        self.known: set[str] = set()  # every board id seen (appending one of them is a no-op)

    def forget(self) -> None:
        self.head, self.checked, self.channels, self.raw, self.created = None, None, {}, {}, {}
        self.views, self.by_job, self.jobs = {}, None, None

    def remember(self, msgs: list[Raw]) -> None:
        for m in msgs:
            bid = str(m["id"])
            self.raw[bid] = m
            self.known.add(bid)
            old = _labels(m).get("legacy-id", bid)
            self.board_id[old] = bid
            self.created[old] = str(m.get("createdAt"))


_CACHES: dict[tuple[str, str], _Cache] = {}


def _cache() -> _Cache:
    return _CACHES.setdefault((board_url(), ns()), _Cache())


def invalidate() -> None:
    """Forget what was read: the next read reloads."""
    for c in _CACHES.values():
        c.forget()


class _Pin:
    depth = 0
    gen = 0


@contextlib.contextmanager
def pinned() -> Iterator[None]:
    """Read one board head for the whole block: inside it the head cursor is fetched once
    (per cache), not once per read. A write in the block still reloads (``invalidate``).
    bus.py wraps every multi-bus read (pending_all, needs-user, digest, brief, _locate) in it."""
    if _Pin.depth == 0:
        _Pin.gen += 1
    _Pin.depth += 1
    try:
        yield
    finally:
        _Pin.depth -= 1


def _fresh() -> _Cache:
    """This process's cache, revalidated by the board's head cursor (``GET /events?last=1``):
    a change by anyone, a label or a delete too, moves the head and drops what was read."""
    c = _cache()
    if _Pin.depth and c.checked == _Pin.gen and c.head is not None:
        return c
    head = str(_http("GET", "/events", query={"last": "1"}, expect=("cursor",))["cursor"])
    if head != c.head:
        c.forget()
        c.head = head
    c.checked = _Pin.gen if _Pin.depth else None
    return c


def _parallel[T, U](fn: Callable[[T], U], items: list[T]) -> list[U]:
    """``[fn(x) for x in items]``, up to ``_WORKERS`` at once (an exit in one re-raises here)."""
    if len(items) < 2:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(_WORKERS, len(items))) as pool:
        return list(pool.map(fn, items))


def _load(channels: list[str]) -> _Cache:
    c = _fresh()
    missing = list(dict.fromkeys(ch for ch in channels if ch not in c.channels))
    for ch, msgs in zip(missing, _parallel(read_channel, missing), strict=True):
        c.channels[ch] = msgs
        c.remember(msgs)
    return c


def _list_channels(prefix: str) -> list[str]:
    out: list[str] = []
    after: str | None = None
    while True:
        query = {"prefix": prefix, "limit": "500", **({"after": after} if after else {})}
        page = _http("GET", "/channels", query=query, expect=("channels",))
        out += [str(c["path"]) for c in _msgs(page, "channels")]
        if page.get("next") is None:
            return out
        after = _advanced(after, page["next"], f"channels {prefix}")


def list_channels(prefix: str) -> list[str]:
    """Every channel path (namespace included) that starts with ``prefix``."""
    return _list_channels(prefix)


def _fetch(board_id: str) -> Raw | None:
    """One message by board id (its thread is read and remembered), or None."""
    c = _cache()
    if board_id not in c.raw:
        c.remember(read_thread(board_id))
    return c.raw.get(board_id)


def _targets(msgs: list[Raw]) -> dict[str, Raw]:
    """Board messages outside ``msgs`` that their labels point at (retires/supersedes)."""
    have = {str(m["id"]) for m in msgs}
    out: dict[str, Raw] = {}
    for m in msgs:
        for key in ("retires", "supersedes"):
            bid = _labels(m).get(key)
            if bid and bid not in have and bid not in out and (hit := _fetch(bid)) is not None:
                out[bid] = hit
    return out


def read(job: str | None) -> list[dict[str, object]]:
    """The old-shape messages one bus dir sees, oldest first.

    ``job`` None = the global board dir: every global channel. A job = its ``job/<slug>``
    channel plus every global message whose ``job`` label is that job.
    Mapped once per board head and job (the global messages once for every view): a caller
    gets its own copies.
    """
    prefix = ns()
    names = [prefix + ch for ch in global_channels()]
    own = prefix + f"job/{job_slug(job)}" if job is not None else None
    c = _load(names + ([own] if own else []))
    if job not in c.views:
        if None not in c.views:
            raw = [m for ch in names for m in c.channels[ch]]
            c.views[None] = _ordered(c, map_messages(raw, _targets(raw), prefix))
        if job is not None and own is not None:
            if c.by_job is None:
                c.by_job = {}
                for m in c.views[None]:
                    if m.get("job"):
                        c.by_job.setdefault(job_slug(str(m["job"])), []).append(m)
            mine = c.channels[own]
            c.views[job] = _ordered(c, c.by_job.get(job_slug(job), []) + map_messages(mine, _targets(mine), prefix))
    return [dict(m) for m in c.views[job]]


def _ordered(c: _Cache, mapped: list[dict[str, object]]) -> list[dict[str, object]]:
    """``mapped`` by its board ``createdAt`` (stable: board order within a channel)."""
    return sorted(mapped, key=lambda m: c.created.get(str(m["id"]), ""))


def lookup(old_id: str) -> Raw | None:
    """The board message behind an old-shape id (a legacy id or a board id), or None."""
    c = _cache()
    if old_id not in c.board_id:  # not read yet: every global channel, then every job channel
        c = _load([ns() + ch for ch in global_channels()])
    if old_id not in c.board_id:
        c = _load(_list_channels(ns() + "job/"))
    return _fetch(c.board_id.get(old_id, old_id))


def append(job: str | None, msg: dict[str, object], *, to_global: bool = False) -> dict[str, object]:
    """Write one old-shape message; return it as stored (board id, real ts).

    A message already on the board (bus.py's mirroring second append of the message it
    just stored) is a no-op that returns it. ``client_id`` makes the post idempotent per
    (from, client_id): a repeat returns the first message.
    """
    c = _cache()
    if str(msg.get("id")) in c.known or str(msg.get("id")) in c.board_id:
        return msg
    refuse_secrets(msg.get("reason"), msg.get("by"), msg.get("from"))  # before _one_line folds them
    keys = [str(msg[k]) for k in ("ref", "retires", "supersedes") if msg.get(k)]
    found = {k: hit for k in keys if (hit := lookup(k)) is not None}
    plan = plan_write(msg, job=job, to_global=to_global, lookup=found, ns=ns())
    new = post_raw(plan.channel, plan.text, labels=plan.labels, profile=plan.profile, re=plan.re, key=plan.key)
    for board_id, labels, when in side_effects(msg, str(new["id"]), found):
        _label(board_id, labels, when)  # a refused precondition: another write already won
    return stored(str(new["id"]))


def stored(board_id: str) -> dict[str, object]:
    """One message as bus.py sees it, read back with its thread (for its ref and kind)."""
    invalidate()
    thread = read_thread(board_id)
    c = _cache()
    c.remember(thread)
    want = _labels(c.raw[board_id]).get("legacy-id", board_id)
    return next(m for m in map_messages(thread, _targets(thread), ns()) if m["id"] == want)


def thread_of(old_id: str) -> str | None:
    """The board id of the thread root holding ``old_id`` (what `vb read`/`vb reply` take)."""
    hit = lookup(old_id)
    return str(hit.get("re") or hit["id"]) if hit is not None else None


def job_channels() -> dict[str, str]:
    """Job slug -> ts (old shape, local) of the newest message in its ``job/<slug>`` channel.

    Listed once per board head."""
    c = _fresh()
    if c.jobs is not None:
        return dict(c.jobs)
    prefix = ns() + "job/"
    out: dict[str, str] = {}
    chans = [ch for ch in _list_channels(prefix) if "/" not in ch[len(prefix):]]
    newest = _parallel(lambda ch: _msgs(_http("GET", "/messages", query={"channel": ch, "limit": "1"}, expect=("messages",))), chans)
    for ch, last in zip(chans, newest, strict=True):
        if last:
            out[ch[len(prefix):]] = _labels(last[-1]).get("ts") or _local_ts(str(last[-1].get("createdAt")))
    c.jobs = out
    return dict(out)
