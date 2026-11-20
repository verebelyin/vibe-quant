#!/usr/bin/env python3
"""One-off replay of the old swarm bus (JSONL) into the local Vercel Board, and its check.

    board_io.py import-legacy [--persistent FILE] [--jobs-root DIR] [--url URL] --map FILE [--report FILE]
    board_io.py verify        [--persistent FILE] [--jobs-root DIR] [--url URL] --map FILE [--report FILE] [--query W ...]
    board_io.py export        [--url URL] [--out FILE] [--report FILE]

Sources: the persistent board (``<main>/docs/orchestration/board/messages.jsonl``) first, then
every job bus ``<jobs root>/<job>/bus/messages.jsonl`` by job name (``_*`` dirs are not jobs),
each in file order; a row whose parent comes later is posted after its parent. A job-bus row
whose id is already on the persistent board (a mirror) is skipped.

Every row is written the way bus.py writes it in board mode: ``board_store.plan_write`` picks
the channel, parent and labels (chief rulings R5, R6, R9, R15), ``board_store.side_effects`` the
retire/supersede labels on the target, ``board_store.post_raw``/``set_labels`` send them over the
board's HTTP API (secret-scanned). On top: labels ``legacy-id`` and ``ts``, and the idempotency
key ``legacy-<id>``. A persistent row is written as a global post, a job-bus row into its job.

Re-runs are safe: the map file (old id -> board id, rewritten atomically after each post) is read
first; a row missing from it is posted again under its key, and the board answers the first
post (``remapped``). A key that answers another text or legacy id, or that the board refuses as
someone else's, is a reject. Retire/supersede labels are re-applied on every run.
``import-legacy`` prints one JSON report (progress on stderr) and exits 1 when any row was
rejected; rows the secret scan refuses are listed under ``secret_refused`` and are not a failure.
``verify`` is read-only: it reads the board back through ``board_store.read`` (what bus.py reads
after the cutover) and compares every source row field by field (``FIELDS``) with only the
declared differences below, then checks threads, hidden posts and search on the raw messages;
it exits 1 on any mismatch. BOARD_URL (or --url) is required: there is no default board.

``export`` is the mirror image of ``import-legacy`` (R26): every message of the global view
(kb/*, needs-user, chat, lobby; never a ``job/*`` channel), hidden ones included, one JSON
object per line in the old record shape ``import-legacy`` reads as a persistent file
(``EXPORT_KEYS``, that order, empty fields left out), by ``ts``, ties in board read order. ``id`` is the
``legacy-id`` label, else ``b<board id>``; ``ref``/``retires``/``supersedes`` use the same ids.
No actor, principal, token or URL is written. A message whose own ``state`` label disagrees
with what its retire/supersede records give on import (a bare-label retire, a post shown
again) carries ``state`` (+ ``state_reason`` for its reason label), which ``import-legacy``
sets last. An edited post exports its current text. Every body and reason is secret-scanned
(``boardctl.looks_like_secret``): a hit exits 1 naming the ids, and no file is written.
``$SWARM_BOARD_NS`` namespaces every channel (tests).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

_AGENTS = Path(__file__).resolve().parent
if str(_AGENTS) not in sys.path:
    sys.path.insert(0, str(_AGENTS))

import board_store as bs  # noqa: E402
import boardctl  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from types import ModuleType

Raw = dict[str, object]
PERSISTENT = "persistent"  # Row.source of a persistent-board row; job rows carry their dir name
_MSG_KEYS = frozenset(("id", "from", "to", "kind", "body"))  # what bus.py's readers deliver
#: The old message fields verify compares, each one read back through board_store.read.
FIELDS = (
    "id", "ts", "from", "to", "kind", "body", "topic", "ref", "job", "importance",
    "retires", "reason", "supersedes", "user_answer", "by",
)
_HASHED = frozenset(("body", "reason"))  # shown as length + hash in a mismatch, never as text

# ---- declared read-back differences: verify tolerates these and nothing else
#: Not carried: the board's idempotency key is ``legacy-<id>``; an old ``client_id`` is dropped.
NOT_CARRIED = ("client_id",)
#: The board refuses an empty or whitespace-only text: such a body is posted (and reads) as this.
EMPTY_TEXT = "(empty)"
#: R16: a retire reason is stored on the target as one line (whitespace runs -> one space) cut
#: at this many characters (board_store.side_effects); the retire row reads it back from there.
REASON_WIDTH = 200
#: A job-bus row reads back with ``job`` = its job dir: plan_write labels every job-bus write
#: with its job (the old job file left the field out: the file itself was the job).
JOB_ROW_JOB = "job dir"
#: R25: one board message has one state, so a mirror (a job-bus copy of a persistent post) is
#: hidden in its job view whenever the persistent board hides it (retired, superseded, or a
#: reply under such a post): the old job file never saw the global retire. Non-mirror job rows
#: are compared strictly; a mirror the job hid but the persistent board shows still fails.
MIRROR_HIDDEN = "mirror hidden by the persistent board"
#: An old search hit whose thread root is in one of these channels is listed, not failed:
#: kb/ is the knowledge search, needs-user and chat are no knowledge channels (chief ruling).
#: A hit in lobby or a job channel still fails.
SEARCH_TOP_LEVEL = bs.TOP_TOPICS
#: Over HTTP the text is stored exactly (probed: trailing/leading whitespace, CR, 5900 chars):
#: no trailing-newline rule applies, so none is declared.

#: verify's default search words: the 20 file basenames most often named in persistent-board
#: posts (counted 2026-11-20 over to=board bodies).
DEFAULT_QUERIES = (
    "node.py", "reviewer.md", "nt_runner.py", "mutate.py", "swarm_check.py", "pipeline.py",
    "__main__.py", "architect.md", "exactness_239.py", "templates.py", "guard.py", "CLAUDE.md",
    "discovery.py", "schema.py", "SPEC.md", "reviewer-lite.md", "conftest.py", "extraction.py",
    "risk-officer.md", "bus.py",
)


def _load(name: str) -> ModuleType:
    """A sibling script by file path (as bus.py loads handoff_lint)."""
    spec = importlib.util.spec_from_file_location(f"_board_io_{name}", _AGENTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ sources and mapping (pure)


@dataclass(frozen=True)
class Row:
    rec: dict[str, object]
    source: str  # PERSISTENT or the job directory name

    @property
    def id(self) -> str:
        return str(self.rec["id"])

    def get(self, key: str) -> str | None:
        """The field as text, or None when it is missing, null or empty."""
        value = self.rec.get(key)
        return None if value is None or value == "" else str(value)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Every record a bus.py reader would deliver, in file order."""
    out: list[dict[str, object]] = []
    for line in path.read_bytes().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.keys() >= _MSG_KEYS:
            out.append(rec)
    return out


def load_rows(persistent: Path, jobs_root: Path, limit_jobs: int | None = None) -> list[Row]:
    """Persistent rows, then each job bus (sorted by job name; `_*` dirs skipped), file order."""
    rows = [Row(r, PERSISTENT) for r in _read_jsonl(persistent)]
    jobs = sorted(
        d for d in jobs_root.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "bus" / "messages.jsonl").is_file()
    )
    for job in jobs if limit_jobs is None else jobs[:limit_jobs]:
        rows += [Row(r, job.name) for r in _read_jsonl(job / "bus" / "messages.jsonl")]
    return rows


def root_channel(row: Row) -> str:
    """Where the row's thread must live when it starts one (R5/R6, no namespace): verify's
    expectation, written from the rulings rather than taken from plan_write."""
    if row.source != PERSISTENT:
        return f"job/{bs.job_slug(row.source)}"
    topic = row.get("topic")
    return (bs.topic_channel(topic) if topic else None) or bs.LOBBY


def post_text(body: object) -> str:
    """The text the board must hold: the body, or EMPTY_TEXT when it is empty or whitespace."""
    text = str(body) if body is not None else ""
    return text if text.strip() else EMPTY_TEXT


def one_line(text: object, width: int = REASON_WIDTH) -> str:
    """REASON_WIDTH's rule: one line, whitespace runs folded, cut at ``width``."""
    return " ".join(str(text).split())[:width]


def as_msg(row: Row) -> dict[str, object]:
    """The row as the old-shape message plan_write takes (NOT_CARRIED dropped, text as stored)."""
    msg = {k: v for k, v in row.rec.items() if k not in NOT_CARRIED}
    msg["body"] = post_text(row.rec.get("body"))
    return msg


def idem_key(old_id: str) -> str:
    """The row's idempotency key: ``legacy-<id>``; in a namespace ``legacy-<ns>-<id>`` (a key is
    one per board and author, so two namespaces of one board must not share it)."""
    space = bs.ns().strip("/").replace("/", "-")
    return f"legacy-{space}-{old_id}" if space else f"legacy-{old_id}"


def plan(row: Row, lookup: Mapping[str, Raw]) -> bs.WritePlan:
    """How the row is written: board_store.plan_write + labels legacy-id/ts + key legacy-<id>.

    ``lookup``: old id -> board message, for the parent and the retire/supersede targets."""
    job = None if row.source == PERSISTENT else row.source
    p = bs.plan_write(as_msg(row), job=job, to_global=False, lookup=lookup, ns=bs.ns())
    head = {"legacy-id": row.id, **({"ts": ts} if (ts := row.get("ts")) else {})}
    return p._replace(labels={**head, **p.labels}, key=idem_key(row.id))


def import_order(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    """(rows to import, parents and targets first; rows skipped because their id came earlier).

    The first row with an id wins: a job-bus mirror of a persistent post is skipped."""
    index: dict[str, Row] = {}
    skipped: list[Row] = []
    for row in rows:
        if row.id in index:
            skipped.append(row)
        else:
            index[row.id] = row
    order: list[Row] = []
    placed: set[str] = set()
    active: set[str] = set()

    def visit(row: Row) -> None:
        if row.id in placed or row.id in active:  # `active`: a ref cycle stops here
            return
        active.add(row.id)
        for dep in (row.get("ref"), row.get("retires"), row.get("supersedes")):
            if dep is not None and dep in index:
                visit(index[dep])
        active.discard(row.id)
        placed.add(row.id)
        order.append(row)

    for row in index.values():
        visit(row)
    return order, skipped


def would_be_refused(row: Row) -> bool:
    """True when post_raw's secret scan refuses the row's text or a label (nothing is sent)."""
    p = plan(row, {})
    return any(boardctl.looks_like_secret(v) for v in (p.text, *p.labels.values()))


def _labels(m: Mapping[str, object]) -> dict[str, str]:
    raw = m.get("labels")
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


#: The export's record keys, in file order: the old shape, then R26's state override.
EXPORT_KEYS = (*FIELDS, "state", "state_reason")
_HIDING = ("retired", "superseded")


def export_id(m: Mapping[str, object]) -> str:
    """A board message's id in the export: its ``legacy-id``, else ``b<board id>``."""
    return _labels(m).get("legacy-id") or f"b{m['id']}"


def _derived_states(rows: list[dict[str, object]]) -> dict[str, tuple[str, str | None]]:
    """Export id -> (state, reason) that import-legacy's retire/supersede side effects give it:
    a retire is unconditional (reason, else body), a supersede only on a post not yet hidden."""
    out: dict[str, tuple[str, str | None]] = {}
    for row in rows:
        if row.get("retires"):
            out[str(row["retires"])] = ("retired", one_line(row.get("reason") or row["body"]))
    for row in rows:
        if row.get("supersedes"):
            out.setdefault(str(row["supersedes"]), ("superseded", None))
    return out


def export_rows(raw: list[Raw], targets: Mapping[str, Raw], prefix: str = "") -> list[dict[str, object]]:
    """Global-view board messages (deleted skipped) -> export rows, by ``ts``, ties in the
    board's own read order (``createdAt``, as board_store.read orders; chief ruling: board ids
    are opaque, never compared). Board B imported from the file reads back in file order.

    ``targets``: board messages outside ``raw`` that a retires/supersedes label points at."""
    by_id: dict[str, Raw] = {**targets, **{str(m["id"]): m for m in raw}}

    def eid(board_id: str) -> str:
        hit = by_id.get(board_id)
        return export_id(hit) if hit is not None else board_id  # an orphan label keeps its text

    live = sorted((m for m in raw if not m.get("deleted")), key=lambda m: str(m.get("createdAt") or ""))
    rows: list[dict[str, object]] = []
    for m, msg in zip(live, bs.map_messages(live, {**targets, **by_id}, prefix), strict=True):
        lab = _labels(m)
        row = {k: msg[k] for k in FIELDS if msg.get(k) not in (None, "")}
        row["id"] = export_id(m)
        parent = m.get("replyTo") or m.get("re")
        if parent:
            row["ref"] = eid(str(parent))
        for key in ("retires", "supersedes"):
            if key in lab:
                row[key] = eid(lab[key])
        rows.append(row)
    rows.sort(key=lambda r: str(r["ts"]))  # stable: same-second rows keep the read order
    derived = _derived_states(rows)
    own = {export_id(m): m for m in live}
    for row in rows:
        lab = _labels(own[str(row["id"])])
        state, reason = lab.get("state"), lab.get("reason")
        hidden = state if state in _HIDING else None
        want_state, want_reason = derived.get(str(row["id"]), (None, None))
        if hidden != want_state:
            row["state"] = state or "live"
        if hidden and reason and reason != want_reason:
            row["state_reason"] = reason
    return [{k: row[k] for k in EXPORT_KEYS if k in row} for row in rows]


def render(rows: list[dict[str, object]]) -> str:
    """The export file's text: one JSON object per row, one trailing newline."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def secret_ids(rows: list[dict[str, object]]) -> list[str]:
    """Ids of rows whose body or reason the write path's secret scan would refuse."""
    return [
        str(r["id"]) for r in rows
        if any(boardctl.looks_like_secret(str(r[k])) for k in ("body", "reason", "state_reason") if r.get(k))
    ]


# ------------------------------------------------------------------ board access (board_store)


def board_raw() -> dict[str, Raw]:
    """Every message (deleted included) of this namespace's global and job channels, by board id
    (the only channels an import writes; the board lists no channels for an empty prefix)."""
    out: dict[str, Raw] = {}
    for channel in [bs.ns() + c for c in bs.global_channels()] + bs.list_channels(bs.ns() + "job/"):
        out.update((str(m["id"]), m) for m in bs.read_channel(channel))
    return out


def load_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"board_io: {path} is not an old-id -> new-id object")
    return {str(k): str(v) for k, v in raw.items()}


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temp file beside ``path``, then rename it over ``path``."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_map(path: Path, idmap: dict[str, str]) -> None:
    write_atomic(path, json.dumps(idmap, indent=0) + "\n")


# ------------------------------------------------------------------ import


@dataclass
class _Run:
    map_file: Path
    idmap: dict[str, str]
    raw: dict[str, Raw]  # board id -> message as last seen (labels kept current by this run)
    posted: int = 0
    remapped: int = 0
    already: int = 0
    first_wins: int = 0
    sanitised: list[dict[str, str]] = field(default_factory=list)
    rejects: list[dict[str, str]] = field(default_factory=list)
    secret: list[str] = field(default_factory=list)

    def lookup(self, row: Row) -> dict[str, Raw]:
        """Old id -> board message for the row's parent and retire/supersede targets."""
        out: dict[str, Raw] = {}
        for key in ("ref", "retires", "supersedes"):
            old = row.get(key)
            if old is not None and old in self.idmap and self.idmap[old] in self.raw:
                out[old] = self.raw[self.idmap[old]]
        return out

    def reject(self, row: Row, step: str, error: object) -> None:
        self.rejects.append({"id": row.id, "step": step, "error": one_line(error, 300)})

    def post(self, row: Row) -> None:
        p = plan(row, self.lookup(row))
        try:
            msg = bs.post_raw(p.channel, p.text, labels=p.labels, profile=p.profile, re=p.re, key=p.key)
        except SystemExit as exc:  # post_raw's refusals: 2 = secret scan, else the board's reason
            if exc.code == 2:
                self.secret.append(row.id)
                return
            self.reject(row, "post", exc.code)
            return  # a refused key is reported, never re-posted under another key
        new = str(msg["id"])
        where = msg.get("replyTo") or msg.get("re") if p.re else msg.get("channel")
        if msg.get("text") != p.text or _labels(msg).get("legacy-id") != row.id or where != (p.re or p.channel):
            self.reject(row, "key-reused", f"key {p.key} answers board message {new} with another text, legacy-id or place")
        if new in self.raw:
            self.remapped += 1  # the key answered a message already on the board (resume)
        else:
            self.posted += 1
        self.raw[new] = msg
        self.idmap[row.id] = new
        save_map(self.map_file, self.idmap)
        author = str(row.rec["from"])
        if p.profile != author.lower():
            self.sanitised.append({"id": row.id, "from": author, "as": p.profile})

    def apply_state(self, row: Row) -> None:
        """Label the target of a retire/supersede row (repeatable: safe on a re-run)."""
        if row.id not in self.idmap:
            return
        new = self.idmap[row.id]
        lookup = self.lookup(row)
        old = lookup.get(str(row.get("supersedes")))
        if old is not None and _labels(old).get("state") in ("retired", "superseded") and _labels(old).get("superseded-by") != new:
            self.first_wins += 1  # already retired or superseded by another: the earlier one stays
        for board_id, labels, when in bs.side_effects(as_msg(row), new, lookup):
            try:
                ok = bs.set_labels(board_id, labels, when=when)
            except SystemExit as exc:
                self.reject(row, "state-label", "secret scan refused a label" if exc.code == 2 else exc.code)
                continue
            if ok:
                self.raw[board_id] = {**self.raw[board_id], "labels": {**_labels(self.raw[board_id]), **labels}}
            else:
                self.first_wins += 1


def import_legacy(rows: list[Row], map_file: Path) -> dict[str, object]:
    t0 = time.monotonic()
    order, skipped = import_order(rows)
    persistent_ids = {r.id for r in rows if r.source == PERSISTENT}
    raw = board_raw()
    before = len(raw)
    run = _Run(map_file=map_file, idmap=load_map(map_file), raw=raw)
    for n, row in enumerate(order, 1):
        if row.id in run.idmap:
            run.already += 1
        else:
            run.post(row)
        run.apply_state(row)
        if n % 500 == 0:
            print(f"board_io: {n}/{len(order)} rows, {round(time.monotonic() - t0, 1)} s, "
                  f"{len(run.rejects)} rejects", file=sys.stderr, flush=True)
    for row in order:  # an export's state override (EXPORT_KEYS), after every side effect
        override = {k: v for k, v in (("state", row.get("state")), ("reason", row.get("state_reason"))) if v}
        if override and row.id in run.idmap:
            try:
                bs.set_labels(run.idmap[row.id], override)
            except SystemExit as exc:
                run.reject(row, "state-override", "secret scan refused a label" if exc.code == 2 else exc.code)
    orphans = [
        {"id": r.id, "ref": str(r.get("ref"))} for r in order
        if r.id in run.idmap and r.get("ref") is not None and r.get("ref") not in run.idmap
    ]
    after = len(board_raw())
    mirrors = sum(1 for r in skipped if r.id in persistent_ids)
    return {
        "rows_read": len(rows),
        "persistent_rows": len(persistent_ids),
        "job_rows": sum(1 for r in rows if r.source != PERSISTENT),
        "jobs": len({r.source for r in rows if r.source != PERSISTENT}),
        "distinct_ids": len(order),
        "posted": run.posted,
        "remapped": run.remapped,
        "already_mapped": run.already,
        "mirror_skipped": mirrors,
        "duplicate_skipped": len(skipped) - mirrors,
        "sanitised_authors": {"count": len(run.sanitised), "items": run.sanitised},
        "orphan_refs": {"count": len(orphans), "items": orphans},
        "rejects": {"count": len(run.rejects), "items": run.rejects},
        "secret_refused": {"count": len(run.secret), "ids": run.secret},
        "supersede_first_wins": run.first_wins,
        "map_entries": len(run.idmap),
        "board_before": before,
        "board_after": after,
        "board_added": after - before,
        "wall_seconds": round(time.monotonic() - t0, 1),
    }


# ------------------------------------------------------------------ verify


def expected_value(row: Row, name: str) -> object:
    """What ``board_store.read`` must give for one field of ``row`` (None = absent),
    the declared differences applied."""
    value = row.rec.get(name)
    if name == "body":
        return post_text(value)
    if name == "job" and row.source != PERSISTENT and value in (None, ""):
        return row.source  # JOB_ROW_JOB
    if name == "reason" and value not in (None, ""):
        return one_line(value)  # REASON_WIDTH
    if name == "user_answer":
        return True if value else None
    return None if value in (None, "") else value


def _show(name: str, value: object) -> object:
    """A value for the report: a body or reason only as its length and hash."""
    if value is None or name not in _HASHED:
        return value
    text = str(value)
    return {"len": len(text), "sha1": hashlib.sha1(text.encode()).hexdigest()[:12]}


def readback(view: str, rows: Iterable[Row], got: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    """Every field of every row that ``got`` (old id -> read-back message) holds differently."""
    out: list[dict[str, object]] = []
    for row in rows:
        have = got.get(row.id)
        if have is None:
            out.append({"check": "readback-missing", "view": view, "id": row.id})
            continue
        for name in FIELDS:
            want, now = expected_value(row, name), have.get(name)
            if want != (None if now is None or now == "" or now is False else now):
                out.append({"check": "readback", "view": view, "id": row.id, "field": name,
                            "old": _show(name, want), "new": _show(name, now)})
    return out


def _why(old_reason: str | None, row: Row | None) -> str:
    """Why the old search skipped a message the new one matched."""
    if row is None:
        return "not imported from the old data"
    if old_reason is not None:
        if old_reason.startswith("retired"):
            return "hidden: retired"
        if old_reason.startswith("superseded"):
            return "hidden: superseded"
        if old_reason.startswith("reply under hidden"):
            return "hidden: reply under a hidden post"
        return f"hidden: {old_reason}"
    if row.source != PERSISTENT:
        return f"job-bus row ({row.source})"
    if row.get("to") != "board":
        return f"not a board post (to={row.get('to')})"
    return "old text has no match (case or text differs)"


def search_threads(word: str) -> list[str]:
    """Root ids of `vb read <ns>kb/ --label state=live --text W` (the command workers run;
    chief ruling: the one `vb` use left in verify), every page."""
    env = {**os.environ}
    out: list[str] = []
    before: str | None = None
    seen: set[str] = set()
    while True:
        argv = ["read", f"{bs.ns()}kb/", "--label", "state=live", "--text", word, "--limit", "500", "--json"]
        proc = boardctl.run_vb([*argv, *(["--before", before] if before else [])], env=env)
        if proc.returncode != 0 and '"channel_not_found"' in proc.stderr + proc.stdout and not out:
            return []  # no kb channel at all (a board without kb posts)
        if proc.returncode != 0:
            raise SystemExit(f"board_io verify: vb read --text {word!r} failed ({proc.returncode}): {proc.stderr.strip()[:300]}")
        page = json.loads(proc.stdout)
        out += [str(t["message"]["id"]) for t in page["threads"]]
        before = page.get("older")
        if not before or before in seen:
            return out
        seen.add(before)


def verify(rows: list[Row], map_file: Path, queries: list[str]) -> dict[str, object]:
    t0 = time.monotonic()
    bus = _load("bus")
    order, _ = import_order(rows)
    index = {r.id: r for r in order}
    secret = sorted(i for i, r in index.items() if would_be_refused(r))
    expected = set(index) - set(secret)
    idmap = load_map(map_file)
    msgs = {i: m for i, m in board_raw().items() if not m.get("deleted")}
    by_legacy: dict[str, list[str]] = {}
    for m in msgs.values():
        lid = _labels(m).get("legacy-id")
        if lid is not None:
            by_legacy.setdefault(lid, []).append(str(m["id"]))
    legacy_of = {new: old for old, news in by_legacy.items() for new in news}
    mismatches: list[dict[str, object]] = []

    def miss(check: str, ids: list[str], **extra: object) -> None:
        if ids:
            mismatches.append({"check": check, **extra, "ids": ids})

    miss("map-missing", sorted(expected - idmap.keys()))
    miss("map-extra", sorted(idmap.keys() - expected))
    miss("board-missing", sorted(i for i in expected if i not in by_legacy))
    miss("board-duplicate", sorted(i for i, news in by_legacy.items() if len(news) > 1))
    miss("board-unexpected", sorted(set(by_legacy) - expected))
    miss("map-wrong", sorted(i for i in expected & idmap.keys() if i in by_legacy and idmap[i] not in by_legacy[i]))

    # read-back parity: every source row in the view bus.py will read it from (a mirror in both)
    global_view = {str(m["id"]): m for m in bs.read(None)}
    sources: dict[str, list[Row]] = {}
    for row in rows:
        if row.id in expected:
            sources.setdefault(row.source, []).append(row)
    for source, srows in sources.items():
        view = global_view if source == PERSISTENT else {str(m["id"]): m for m in bs.read(source)}
        mismatches += readback(source, srows, view)

    # threads on the raw messages: the channel of each root, the parent of each message
    prefix = bs.ns()
    for old in sorted(expected & by_legacy.keys()):
        m, row = msgs[by_legacy[old][0]], index[old]
        parent_new = str(m.get("replyTo") or m.get("re") or "") or None
        if parent_new is None and m["channel"] != prefix + root_channel(row):
            mismatches.append({"check": "channel", "id": old, "old": root_channel(row), "new": str(m["channel"])[len(prefix):]})
        want = row.get("ref") if row.get("ref") in expected else None
        have = legacy_of.get(parent_new, f"board:{parent_new}") if parent_new is not None else None
        if have != want or (want is None and _labels(m).get("ref") != row.get("ref")):
            mismatches.append({"check": "parent", "id": old, "old": row.get("ref"), "new": have or _labels(m).get("ref")})

    # hidden parity over every source: bus.hidden_posts on the old rows vs labels + parents
    memo: dict[str, bool] = {}

    def hidden(new: str) -> bool:
        if new not in memo:
            memo[new] = False  # a cycle reads as visible
            labels = _labels(msgs[new])
            parent = msgs[new].get("replyTo") or msgs[new].get("re")
            memo[new] = (
                labels.get("state") in ("retired", "superseded") or "retires" in labels
                or (parent is not None and str(parent) in msgs and hidden(str(parent)))
            )
        return memo[new]

    hidden_report: dict[str, dict[str, list[str]]] = {}
    old_reasons: dict[str, str] = bus.hidden_posts([r.rec for r in rows if r.source == PERSISTENT])
    persistent_ids = {r.id for r in sources.get(PERSISTENT, [])}
    mirror_hidden: list[str] = []
    for source, srows in sources.items():
        reasons: dict[str, str] = old_reasons if source == PERSISTENT else bus.hidden_posts([r.rec for r in rows if r.source == source])
        ids = {r.id for r in srows} & by_legacy.keys()
        old_hidden = set(reasons) & ids
        if source != PERSISTENT:  # MIRROR_HIDDEN
            covered = (ids & persistent_ids & old_reasons.keys()) - old_hidden
            mirror_hidden += sorted(covered)
            old_hidden |= covered
        new_hidden = {i for i in ids if hidden(by_legacy[i][0])}
        hidden_report[source] = {"old": sorted(old_hidden), "new": sorted(new_hidden)}
        if old_hidden != new_hidden:
            mismatches.append({
                "check": "hidden-parity", "source": source,
                "only_old": sorted(old_hidden - new_hidden), "only_new": sorted(new_hidden - old_hidden),
            })

    # search parity: old `board search --global --grep W` (bus.board_search's filter over the
    # persistent rows: to=board, not hidden, W in the body) vs the board's kb search
    search: dict[str, object] = {}
    board_rows = [r for r in sources.get(PERSISTENT, []) if r.get("to") == "board" and r.id not in old_reasons]
    for word in queries:
        old_roots: set[str] = set()
        outside: set[str] = set()
        top_level: set[str] = set()
        for row in board_rows:
            news = by_legacy.get(row.id)
            if not news or word.lower() not in str(row.rec.get("body") or "").lower():
                continue  # a missing one is reported under board-missing
            root = str(msgs[news[0]].get("re") or news[0])
            channel = str(msgs[root]["channel"])[len(prefix):]
            bucket = old_roots if channel.startswith("kb/") else top_level if channel in SEARCH_TOP_LEVEL else outside
            bucket.add(root)
        new_roots = set(search_threads(word))
        miss("search-missing", sorted(legacy_of.get(r, r) for r in old_roots - new_roots), word=word)
        miss("search-outside-kb", sorted(legacy_of.get(r, r) for r in outside), word=word)
        extras: list[dict[str, object]] = []
        for root in sorted(new_roots - old_roots, key=lambda r: legacy_of.get(r, r)):
            matched = [
                {"id": legacy_of.get(i, i), "why": _why(old_reasons.get(legacy_of.get(i, "")), index.get(legacy_of.get(i, "")))}
                for i, m in msgs.items()
                if (i == root or m.get("re") == root) and word.lower() in str(m.get("text", "")).lower()
            ]
            extras.append({"root": legacy_of.get(root, root), "matched": sorted(matched, key=lambda x: x["id"])})
        search[word] = {"old": len(old_roots), "new": len(new_roots), "extras": extras,
                        "top_level": sorted(legacy_of.get(r, r) for r in top_level)}  # SEARCH_TOP_LEVEL

    return {
        "ok": not mismatches,
        "counts": {
            "source_rows": len(rows),
            "distinct_ids": len(index),
            "expected_ids": len(expected),
            "secret_refused": len(secret),
            "map_entries": len(idmap),
            "board_legacy_messages": sum(len(v) for v in by_legacy.values()),
            "readback_rows": sum(len(v) for v in sources.values()),
            "mirror_hidden": len(mirror_hidden),
        },
        "mismatches": mismatches,
        "secret_refused_ids": secret,
        "hidden": hidden_report,
        "mirror_hidden_ids": mirror_hidden,
        "search": search,
        "wall_seconds": round(time.monotonic() - t0, 1),
    }


# ------------------------------------------------------------------ export


def export(out: Path) -> tuple[dict[str, object] | None, list[str]]:
    """Read the global view and write it to ``out`` (temp file + rename): (report, []);
    (None, secret ids) and nothing written when the secret scan refuses a row."""
    t0 = time.monotonic()
    bus = _load("bus")
    prefix = bs.ns()
    raw = [m for ch in bs.global_channels() for m in bs.read_channel(prefix + ch)]
    targets = bs._targets(raw)
    rows = export_rows(raw, targets, prefix)
    refused = secret_ids(rows)
    if refused:
        return None, refused
    write_atomic(out, render(rows))
    channel = {export_id(m): str(m["channel"])[len(prefix):] for m in raw}
    counts: dict[str, int] = {}
    for row in rows:
        counts[channel[str(row["id"])]] = counts.get(channel[str(row["id"])], 0) + 1
    mapped = bs.map_messages(raw, targets, prefix)
    hidden = set(bus.hidden_posts(mapped)) & {str(m["id"]) for m in mapped}
    return {
        "rows": len(rows),
        "channels": dict(sorted(counts.items())),
        "hidden": len(hidden),
        "out": str(out),
        "wall_seconds": round(time.monotonic() - t0, 1),
    }, []


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="board_io.py", description=__doc__.split("\n\n")[0] if __doc__ else None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--url", default=None, help="the board; default $BOARD_URL (one of them is required)")
    ex.add_argument("--out", type=Path, default=None, help="default: <main>/docs/orchestration/board/export.jsonl")
    ex.add_argument("--report", type=Path, default=None, help="also write the JSON report here (temp file + rename)")
    for name in ("import-legacy", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--persistent", type=Path, default=None, help="default: <main>/docs/orchestration/board/messages.jsonl")
        p.add_argument("--jobs-root", type=Path, default=None, help="default: <main>/data/swarm")
        p.add_argument("--url", default=None, help="the board; default $BOARD_URL (one of them is required)")
        p.add_argument("--map", type=Path, required=True, help="old id -> new id JSON, written by import-legacy")
        p.add_argument("--report", type=Path, default=None, help="also write the JSON report here (temp file + rename)")
        if name == "import-legacy":
            p.add_argument("--limit-jobs", type=int, default=None, help=(
                "TEST ONLY: import the first N job buses. A later full run leaves a row whose "
                "parent sits in a later job as an orphan root (ref label), and reports it nowhere"))
        else:
            p.add_argument("--query", nargs="*", default=list(DEFAULT_QUERIES), help="search words (none: skip search)")
    args = ap.parse_args(argv)
    url = args.url or os.environ.get("BOARD_URL")
    if not url:
        print("board_io: no board: pass --url or set BOARD_URL (there is no default board)", file=sys.stderr)
        return 2
    os.environ["BOARD_URL"] = url  # board_store and `vb` read it from the environment
    if args.cmd == "export":
        out = args.out or boardctl.main_checkout() / "docs" / "orchestration" / "board" / "export.jsonl"
        got, refused = export(out)
        if got is None:
            print(f"board_io export: {len(refused)} row(s) look like a secret (KEY=value credential or env "
                  f"dump): {' '.join(refused)}; nothing written", file=sys.stderr)
            return 1
        text = json.dumps(got, indent=1)
        if args.report is not None:
            write_atomic(args.report, text + "\n")
        print(text)
        return 0
    persistent = args.persistent or boardctl.main_checkout() / "docs" / "orchestration" / "board" / "messages.jsonl"
    jobs_root = args.jobs_root or boardctl.main_checkout() / "data" / "swarm"
    if args.cmd == "import-legacy":
        report = import_legacy(load_rows(persistent, jobs_root, args.limit_jobs), args.map)
        ok = report["rejects"] == {"count": 0, "items": []}
    else:
        report = verify(load_rows(persistent, jobs_root), args.map, args.query)
        ok = bool(report["ok"])
    text = json.dumps(report, indent=1)
    if args.report is not None:
        write_atomic(args.report, text + "\n")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
