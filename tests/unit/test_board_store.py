"""scripts/agents/board_store.py — bus.py's second store: the local Vercel Board, read and
written as old-shape bus messages, so every chief tool works on it unchanged (bead m0cbl.3).

Pure mapping tests need no server; behaviour tests drive the public bus functions with
``SWARM_STORE=board`` on the session's ``board_server``, each in its own channel namespace
(``SWARM_BOARD_NS``: a local ``vb serve`` serves no named boards).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Iterator
    from types import ModuleType

    from tests.unit.conftest import BoardServer

_AGENTS = Path(__file__).resolve().parents[2] / "scripts" / "agents"


def _load(name: str, file: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _AGENTS / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bs = _load("board_store", "board_store.py")
bus = _load("swarm_bus_board", "bus.py")

_NOW = 1_800_000_000.0


# ------------------------------------------------------------------ pure READ mapping


def _raw(
    mid: str, channel: str, text: str, *, re: str | None = None, reply_to: str | None = None,
    labels: dict[str, str] | None = None, profile: str | None = "w",
    created: str = "2026-11-20T10:00:00.000Z", deleted: bool = False,
) -> dict[str, object]:
    return {
        "id": mid, "channel": channel, "text": text, "re": re, "replyTo": reply_to,
        "labels": labels or {}, "createdAt": created, "deleted": deleted,
        "by": {"actor": f"local:user:me+{profile}", "principal": "local:user:me", "profile": profile},
    }


def _one(raw: list[dict[str, object]], want_id: str, targets: dict[str, dict[str, object]] | None = None) -> dict[str, object]:
    out = {str(m["id"]): m for m in bs.map_messages(raw, targets=targets)}
    return out[want_id]


def test_read_legacy_id_ts_label_importance_job_and_kb_topic() -> None:
    raw = [_raw("5", "kb/gotchas", "rule x", profile="cc-a", labels={
        "legacy-id": "18e0-0001", "ts": "2026-10-01T12:00:00", "state": "live", "importance": "3",
        "job": "20261001-a",
    })]
    assert bs.map_messages(raw) == [{
        "id": "18e0-0001", "ts": "2026-10-01T12:00:00", "from": "cc-a", "to": "board", "kind": "info",
        "body": "rule x", "topic": "gotchas", "job": "20261001-a", "importance": 3,
    }]


def test_read_created_at_becomes_local_ts_and_board_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "Europe/Budapest")
    time.tzset()
    try:
        got = bs.map_messages([_raw("12", "lobby", "hello", labels={"to": "chief"}, created="2026-11-20T10:00:01.250Z")])
    finally:
        monkeypatch.undo()
        time.tzset()
    assert got == [{"id": "12", "ts": "2026-11-20T11:00:01", "from": "w", "to": "chief", "kind": "info", "body": "hello"}]


def test_read_author_null_profile_is_user_and_from_label_wins() -> None:
    raw = [
        _raw("1", "needs-user", "pick A or B?", profile=None, labels={"kind": "question"}),
        _raw("2", "chat", "hi", profile="drev-x0axf-14-1", labels={"from": "drev-x0axf.14-1"}),
    ]
    got = bs.map_messages(raw)
    assert [(m["from"], m["topic"], m["kind"]) for m in got] == [("user", "needs-user", "question"), ("drev-x0axf.14-1", "chat", "info")]


def test_read_job_channel_topic_label_and_job_from_channel() -> None:
    got = _one([_raw("7", "job/20261120-x", "design note", labels={"topic": "design"})], "7")
    assert got == {
        "id": "7", "ts": got["ts"], "from": "w", "to": "board", "kind": "info", "body": "design note",
        "topic": "design", "job": "20261120-x",
    }


def test_read_to_derivation_in_a_job_thread() -> None:
    raw = [
        _raw("10", "job/j", "task brief", profile="chief", labels={"agent": "cc-1", "bead": "b.1", "role": "maker"}),
        _raw("11", "job/j", "on it", re="10", profile="cc-1"),
        _raw("12", "job/j", "use plan B", re="10", reply_to="11", profile="chief"),
        _raw("20", "job/j", "kickoff", profile="chief"),
        _raw("21", "job/j", "everyone stop", re="20", profile="chief"),
        _raw("22", "job/j", "explicit", re="20", profile="cc-2", labels={"to": "cc-9"}),
    ]
    to = {str(m["id"]): m["to"] for m in bs.map_messages(raw)}
    assert to == {"10": "cc-1", "11": "chief", "12": "cc-1", "20": "all", "21": "all", "22": "cc-9"}


def test_read_kind_from_label_then_prefix_then_answer_after_question_else_info() -> None:
    raw = [
        _raw("1", "job/j", "root", profile="chief", labels={"agent": "w"}),
        _raw("2", "job/j", "  [Question] which file?", re="1"),
        _raw("3", "job/j", "this one", re="1", reply_to="2", profile="chief"),
        _raw("4", "job/j", "[blocker] tests red", re="1"),
        _raw("5", "job/j", "[done] see handoff", re="1"),
        _raw("6", "job/j", "[CLAIM] editing bus.py", re="1"),
        _raw("7", "job/j", "[question] but labelled", re="1", labels={"kind": "info"}),
        _raw("8", "job/j", "plain", re="1"),
        _raw("9", "job/j", "after the root, not the question", re="1", profile="chief"),
    ]
    kinds = {str(m["id"]): m["kind"] for m in bs.map_messages(raw)}
    assert kinds == {
        "1": "info", "2": "question", "3": "answer", "4": "blocker", "5": "done", "6": "claim",
        "7": "info", "8": "info", "9": "info",
    }


def test_read_ref_from_reply_to_else_root_and_legacy_ids_of_both() -> None:
    raw = [
        _raw("1", "kb/findings", "root", labels={"legacy-id": "L-root"}),
        _raw("2", "kb/findings", "first reply", re="1", labels={"legacy-id": "L-r1"}),
        _raw("3", "kb/findings", "reply to the reply", re="1", reply_to="2"),
        _raw("4", "lobby", "orphan", labels={"ref": "L-gone", "to": "chief"}),
    ]
    refs = {str(m["id"]): m.get("ref") for m in bs.map_messages(raw)}
    assert refs == {"L-root": None, "L-r1": "L-root", "3": "L-r1", "4": "L-gone"}


def test_read_retire_event_takes_the_target_reason_and_supersede_its_old_id() -> None:
    raw = [
        _raw("1", "kb/gotchas", "old rule", labels={"legacy-id": "L-old", "state": "retired", "reason": "wrong after NT bump"}),
        _raw("2", "kb/gotchas", "retired L-old: wrong after NT bump", re="1", labels={"retires": "1"}),
        _raw("3", "kb/gotchas", "newer rule", labels={"supersedes": "9", "state": "live"}),
    ]
    targets = {"9": _raw("9", "job/j", "job-only rule", labels={"legacy-id": "L-job", "topic": "gotchas"})}
    got = {str(m["id"]): m for m in bs.map_messages(raw, targets=targets)}
    assert (got["2"]["retires"], got["2"]["reason"], got["2"]["ref"]) == ("L-old", "wrong after NT bump", "L-old")
    assert got["3"]["supersedes"] == "L-job"
    assert "reason" not in got["L-old"] and "retires" not in got["L-old"]


def test_read_user_answer_relayed_by_and_deleted_skipped() -> None:
    raw = [
        _raw("1", "needs-user", "A or B?", labels={"kind": "question"}),
        _raw("2", "needs-user", "A", re="1", profile="user", labels={"user-answer": "true", "relayed-by": "chief", "kind": "answer"}),
        _raw("3", "needs-user", "gone", re="1", deleted=True),
    ]
    got = bs.map_messages(raw)
    assert [m["id"] for m in got] == ["1", "2"]
    assert (got[1]["user_answer"], got[1]["by"], got[1]["kind"], got[1]["from"]) == (True, "chief", "answer", "user")


# ------------------------------------------------------------------ pure WRITE mapping


def _msg(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {"id": "18e9-0001", "ts": "2026-11-20T11:00:00", "from": "cc-1", "to": "board", "kind": "info", "body": "b"}
    base.update(kw)
    return base


def _plan(msg: dict[str, object], job: str | None = None, to_global: bool = False,
          lookup: dict[str, dict[str, object]] | None = None, ns: str = "") -> dict[str, object]:
    return dict(bs.plan_write(msg, job=job, to_global=to_global, lookup=lookup or {}, ns=ns)._asdict())


@pytest.mark.parametrize(
    ("msg", "job", "to_global", "channel", "labels"),
    [
        (_msg(topic="gotchas", importance=3), None, False, "kb/gotchas", {"state": "live", "importance": "3"}),
        (_msg(topic="findings"), "20261120-board", True, "kb/findings", {"state": "live", "job": "20261120-board"}),
        (_msg(topic="findings"), "20261120-board", False, "job/20261120-board", {"topic": "findings", "job": "20261120-board"}),
        (_msg(topic="design", kind="claim"), "20261120-Board X", True, "job/20261120-board-x",
         {"topic": "design", "job": "20261120-Board X", "kind": "claim"}),
        (_msg(topic="design"), None, False, "lobby", {"topic": "design"}),
        (_msg(topic="chat"), None, False, "chat", {}),
        (_msg(topic="needs-user", kind="question"), "j1", True, "needs-user", {"kind": "question", "job": "j1"}),
        (_msg(to="chief", kind="question"), None, False, "lobby", {"to": "chief", "kind": "question"}),
        (_msg(to="chief", kind="question"), "j1", False, "job/j1", {"to": "chief", "kind": "question", "job": "j1"}),
        (_msg(to="cc-2", body="[done] not really done"), "j1", False, "job/j1", {"to": "cc-2", "kind": "info", "job": "j1"}),
    ],
)
def test_write_channel_and_labels_of_a_new_thread(
    msg: dict[str, object], job: str | None, to_global: bool, channel: str, labels: dict[str, str]
) -> None:
    plan = _plan(msg, job, to_global)
    assert (plan["channel"], plan["re"], plan["labels"], plan["profile"], plan["key"]) == (channel, None, labels, "cc-1", None)
    assert plan["text"] == msg["body"]


def test_write_reply_targets_the_exact_parent_and_kind_label_only_when_not_derivable() -> None:
    parent = _raw("41", "job/j1", "[question] which?", re="40")
    answer = _plan(_msg(to="cc-2", kind="answer", ref="Q1"), "j1", lookup={"Q1": parent})
    assert (answer["channel"], answer["re"], answer["labels"]) == (None, "41", {"to": "cc-2", "job": "j1"})
    board_reply = _plan(_msg(topic="gotchas", ref="Q1"), None, lookup={"Q1": _raw("41", "kb/gotchas", "[question] q", re="40")})
    assert board_reply["labels"] == {"kind": "info"}  # a plain reply under a question would read back as answer
    orphan = _plan(_msg(to="chief", ref="L-gone"), "j1")
    assert (orphan["channel"], orphan["re"], orphan["labels"]) == ("job/j1", None, {"to": "chief", "job": "j1", "ref": "L-gone"})


def test_write_idempotency_key_profile_sanitising_namespace_and_user_answer() -> None:
    upper = _plan(_msg(**{"from": "arch-L2"}, to="chief"), None)  # the board lowercases profiles itself
    assert (upper["profile"], upper["labels"]) == ("arch-l2", {"to": "chief", "from": "arch-L2"})
    plan = _plan(_msg(**{"from": "drev-x0axf.14-1", "client_id": "ask-1a2b"}, to="chief"), "j1", ns="t0ab")
    assert (plan["channel"], plan["profile"], plan["key"]) == ("t0ab/job/j1", "drev-x0axf-14-1", "drev-x0axf.14-1:ask-1a2b")
    assert plan["labels"] == {"to": "chief", "job": "j1", "from": "drev-x0axf.14-1"}
    ans = _plan(
        _msg(**{"from": "user"}, kind="answer", topic="needs-user", ref="Q", user_answer=True, by="chief"), None,
        lookup={"Q": _raw("3", "needs-user", "A or B?", labels={"kind": "question"})},
    )
    assert (ans["re"], ans["profile"], ans["labels"]) == ("3", "user", {"user-answer": "true", "relayed-by": "chief"})


def test_write_retire_and_supersede_labels_and_side_effects() -> None:
    target = _raw("7", "kb/gotchas", "old", labels={"legacy-id": "L-7", "state": "live"})
    retire = _msg(topic="gotchas", ref="L-7", retires="L-7", reason="wrong\nafter " + "x" * 300, body="retired L-7: wrong")
    plan = _plan(retire, None, lookup={"L-7": target})
    assert (plan["re"], plan["labels"]) == ("7", {"retires": "7"})
    effects = bs.side_effects(retire, "99", {"L-7": target})
    assert effects == [("7", {"state": "retired", "reason": ("wrong after " + "x" * 300)[:200]}, None)]
    sup = _msg(topic="gotchas", supersedes="L-7")
    assert _plan(sup, None, lookup={"L-7": target})["labels"] == {"state": "live", "supersedes": "7"}
    assert bs.side_effects(sup, "100", {"L-7": target}) == [
        ("7", {"state": "superseded", "superseded-by": "100"}, {"state": "live"}),  # only while still live
    ]
    unlabelled = _raw("8", "job/j", "job post", labels={"legacy-id": "L-8", "topic": "gotchas"})
    assert bs.side_effects(_msg(supersedes="L-8"), "101", {"L-8": unlabelled}) == [
        ("8", {"state": "superseded", "superseded-by": "101"}, {"state": None}),  # only while still unset
    ]
    gone = _raw("7", "kb/gotchas", "old", labels={"state": "superseded", "superseded-by": "50"})
    assert bs.side_effects(sup, "100", {"L-7": gone}) == []


# ------------------------------------------------------------------ behaviour on a real board


@pytest.fixture
def board(board_server: BoardServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """SWARM_STORE=board on the fixture board, a fresh channel namespace; yields the jobs root."""
    for key in ("BOARD_URL", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, board_server.env[key])
    monkeypatch.setenv("SWARM_STORE", "board")
    monkeypatch.setenv("SWARM_BOARD_NS", "t" + uuid.uuid4().hex[:12])
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path / "global"))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path / "jobs"))
    monkeypatch.delenv("SWARM_BUS", raising=False)
    monkeypatch.setattr(bus, "POLL_S", 0.05)
    monkeypatch.setattr(bus, "work_context", lambda: ([], set()))
    monkeypatch.setattr(bus, "comparability_breaks", lambda: [])
    (tmp_path / "jobs").mkdir()
    yield tmp_path / "jobs"


def _job(root: Path, name: str) -> Path:
    b = root / name / "bus"
    b.mkdir(parents=True)
    return b


def _ns() -> str:
    import os

    return os.environ["SWARM_BOARD_NS"]


def _channel(name: str) -> list[dict[str, object]]:
    """Every raw board message of one channel (namespace added), oldest first."""
    return bs.read_channel(f"{_ns()}/{name}")


def _gdir() -> Path:
    return bus.global_board_dir()


def test_post_then_search_by_topic_path_and_two_greps(board: Path) -> None:
    g = _gdir()
    bus.post(g, "cc-a", "board", "info", "nt_runner.py:1337 swallows DataUnavailableError; preflight first", topic="gotchas")
    bus.post(g, "rev-b", "board", "info", "vibe_quant/liquidation/scan.py: funding cut at end_ns-1ms", topic="findings")
    bus.post(g, "chief", "board", "info", "Ruling: clearance fee counts as a cost", topic="decisions")
    bus.post(g, "cc-d", "chief", "done", "liquidation scan.py done")  # a DM, never a search hit
    assert [m["from"] for m in bus.board_search(g, topics=["gotchas", "decisions"])] == ["cc-a", "chief"]
    assert [m["from"] for m in bus.board_search(g, paths=["vibe_quant/liquidation/scan.py"])] == ["rev-b"]
    assert [m["from"] for m in bus.board_search(g, terms=["FUNDING", "end_ns"])] == ["rev-b"]
    assert bus.board_search(g, terms=["funding", "preflight"]) == []
    assert [m["to"] for m in _all_mapped_lobby()] == ["chief"]
    assert [m["labels"] for m in _channel("lobby")] == [{"to": "chief", "kind": "done"}]


def _all_mapped_lobby() -> list[dict[str, object]]:
    return [m for m in bus._all(_gdir()) if m.get("to") != "board"]


def test_reply_threads_under_the_post_and_thread_shows_it(board: Path) -> None:
    g = _gdir()
    root = bus.post(g, "cc-a", "board", "question", "does gc touch needs-user?", topic="findings")
    r1 = bus.reply(g, "cc-b", str(root["id"]), "no, it skips needs-user")
    r2 = bus.reply(g, "cc-a", str(r1["id"]), "thanks")
    tree = [(depth, m["body"], m.get("ref")) for depth, m in bus.thread(g, str(r2["id"]))]
    assert tree == [
        (0, "does gc touch needs-user?", None), (1, "no, it skips needs-user", root["id"]), (2, "thanks", r1["id"]),
    ]
    raw = _channel("kb/findings")
    assert [(m["re"], m["replyTo"]) for m in raw] == [(None, None), (root["id"], None), (root["id"], r1["id"])]


def test_retire_hides_from_search_and_brief_labels_the_target_and_all_shows_why(
    board: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    g = _gdir()
    stale = bus.post(g, "w", "board", "info", "nt_runner.py stale finding", topic="findings")
    bus.post(g, "w", "board", "info", "nt_runner.py fresh gotcha", topic="gotchas")
    event = bus.retire(g, "chief", str(stale["id"]), "pre-break metric")
    assert event is not None and event["retires"] == stale["id"] and event["reason"] == "pre-break metric"
    assert [m["body"] for m in bus.board_search(g, paths=["nt_runner.py"])] == ["nt_runner.py fresh gotcha"]
    assert "stale finding" not in bus.brief(context=([], set()), now=time.time())
    every = bus.board_search(g, paths=["nt_runner.py"], include_hidden=True)
    assert [(m["body"], m.get("hidden")) for m in every] == [
        ("nt_runner.py stale finding", "retired: pre-break metric"), ("nt_runner.py fresh gotcha", None),
    ]
    target = next(m for m in _channel("kb/findings") if m["id"] == stale["id"])
    assert target["labels"] == {"state": "retired", "reason": "pre-break metric"}
    assert bus.retire(g, "chief", str(stale["id"]), "again") is None  # already hidden: nothing written
    assert len(_channel("kb/findings")) == 2


def test_supersede_hides_the_old_post_and_a_second_supersede_keeps_the_first_link(board: Path) -> None:
    g = _gdir()
    old = bus.post(g, "w", "board", "info", "use ruff 0.6 flags", topic="decisions")
    new = bus.supersede(g, "chief", str(old["id"]), "decisions", "use ruff 0.7 flags")
    assert new["supersedes"] == old["id"]
    assert [m["body"] for m in bus.board_search(g, topics=["decisions"])] == ["use ruff 0.7 flags"]
    newer = bus.supersede(g, "chief", str(old["id"]), "decisions", "use ruff 0.8 flags")
    labels = {str(m["id"]): m["labels"] for m in _channel("kb/decisions")}
    assert labels[str(old["id"])] == {"state": "superseded", "superseded-by": str(new["id"])}
    assert labels[str(newer["id"])] == {"state": "live", "supersedes": str(old["id"])}


def _import(channel: str, msg: dict[str, object], re: str | None = None) -> str:
    """Post one old-shape message the way the importer does (legacy-id + ts labels)."""
    labels = {"legacy-id": str(msg["id"]), "ts": str(msg["ts"])}
    for key in ("importance", "job", "kind", "topic"):
        if msg.get(key) is not None and not (key == "kind" and msg[key] == "info") and not (key == "topic" and channel.startswith("kb/")):
            labels[key] = str(msg[key])
    if str(msg.get("to")) != "board":
        labels["to"] = str(msg["to"])
    if channel.startswith("kb/") and re is None:
        labels["state"] = "live"
    out = bs.post_raw(f"{_ns()}/{channel}", str(msg["body"]), labels=labels, profile=str(msg["from"]), re=re)
    return str(out["id"])


def _seed_both(gdir_jsonl: Path, rows: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch) -> None:
    """The same old-shape posts in a JSONL board and (imported) on the board's global channels."""
    board_ids: dict[str, str] = {}
    for m in rows:
        monkeypatch.setenv("SWARM_STORE", "jsonl")
        bus._append(gdir_jsonl, m)
        monkeypatch.setenv("SWARM_STORE", "board")
        topic = str(m["topic"])
        channel = f"kb/{topic}" if topic in bs.KB_TOPICS else ("chat" if topic == "chat" else "lobby")
        parent = board_ids.get(str(m.get("ref")))
        board_ids[str(m["id"])] = _import(channel, m, re=parent)


def _old(days_ago: float, topic: str, body: str, importance: int | None = None, job: str | None = None,
         ref: str | None = None, now: float = _NOW) -> dict[str, object]:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - days_ago * 86400))
    m: dict[str, object] = {"id": f"p{days_ago}-{body[:8]}", "ts": ts, "from": "w", "to": "board", "kind": "info", "topic": topic, "body": body}
    for key, value in (("importance", importance), ("job", job), ("ref", ref)):
        if value:
            m[key] = value
    return m


def test_gc_picks_the_same_posts_as_jsonl_and_is_idempotent(
    board: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        _old(20, "chat", "stale chatter nobody ever cited", importance=1),
        _old(20, "chat", "important chatter to keep", importance=2),
        _old(20, "chat", "cited by a reply", importance=1),
        _old(20, "chat", "cited inside another body", importance=1),
        _old(13, "chat", "thirteen day old chatter", importance=1),
        _old(14.1, "gotchas", "just over fourteen days old", importance=1),
    ]
    rows += [_old(0, "chat", "confirmed reply", ref=str(rows[2]["id"])), _old(0, "chat", f"see {rows[3]['id']} for details", importance=1)]
    jsonl = tmp_path / "jsonl-board"
    _seed_both(jsonl, rows, monkeypatch)
    on_board = bus.gc(_gdir(), "chief", days=14, now=_NOW)
    monkeypatch.setenv("SWARM_STORE", "jsonl")
    assert bus.gc(jsonl, "chief", days=14, now=_NOW) == on_board
    monkeypatch.setenv("SWARM_STORE", "board")
    assert sorted(on_board[0]) == sorted([str(rows[0]["id"]), str(rows[5]["id"])])
    assert bus.gc(_gdir(), "chief", days=14, now=_NOW) == ([], 6)


def test_brief_order_equals_jsonl_for_the_same_posts(board: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = []
    for j in range(3):
        for k in range(4):
            rows.append(_old(1 + k * 5 + j * 0.1, "gotchas", f"job{j} finding number {k}", importance=3, job=f"job{j}"))
    rows += [_old(2, "findings", "nt_runner.py names a changed file"), _old(3, "decisions", "sharpe 1.2 ruling", importance=1)]
    jsonl = tmp_path / "jsonl-board"
    _seed_both(jsonl, rows, monkeypatch)
    ctx = (["scripts/agents/nt_runner.py"], {"finding"})
    on_board = bus.brief(context=ctx, now=_NOW)
    monkeypatch.setenv("SWARM_BOARD", str(jsonl))
    monkeypatch.setenv("SWARM_STORE", "jsonl")
    in_jsonl = bus.brief(context=ctx, now=_NOW)
    assert on_board.splitlines()[8:] == in_jsonl.splitlines()[8:]
    assert len(on_board.splitlines()) == 8 + 1 + 11  # header, "Top N" line, 3 per job x 3 + 2 others


def test_a_worker_question_in_a_task_thread_is_pending_until_the_chief_replies(
    board: Path, board_server: BoardServer
) -> None:
    jb = _job(board, "20261120-pend")
    root = bus.post(jb, "chief", "cc-1", "info", "task: build the thing")
    asked = board_server.vb("reply", str(root["id"]), "[question] which file?", profile="cc-1")
    assert asked.returncode == 0, asked.stderr
    pend = bus.pending(jb, "chief")
    assert [(m["from"], m["to"], m["kind"], m["body"]) for m in pend] == [("cc-1", "chief", "question", "[question] which file?")]
    ans = bus.reply(jb, "chief", str(pend[0]["id"]), "bus.py")
    assert (ans["to"], ans["kind"], ans["ref"]) == ("cc-1", "answer", pend[0]["id"])
    assert bus.pending(jb, "chief") == []
    assert [m["id"] for m in bus.pending_all("chief")] == []


def test_needs_user_ask_open_answer_and_the_user_flag_refusal(
    board: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jb = _job(board, "20261120-nu")
    q = bus.ask_user(jb, "chief", "merge A or B? recommend A")
    assert [m["id"] for m in bus.needs_user_open_all()] == [q["id"]]
    assert [m["job"] for m in bus.needs_user_open(_gdir())] == ["20261120-nu"]
    with pytest.raises(SystemExit) as exc:
        bus.main(["--bus", str(jb), "--as", "chief", "needs-user", "answer", "--ref", str(q["id"]), "--body", "A"])
    assert exc.value.code == 2
    assert bus.main(["--bus", str(jb), "--as", "chief", "needs-user", "answer", "--ref", str(q["id"]), "--body", "A", "--user"]) == 0
    assert bus.needs_user_open_all() == []
    answer = bus.thread(_gdir(), str(q["id"]))[1][1]
    assert (answer["from"], answer["by"], answer["user_answer"], answer["kind"]) == ("user", "chief", True, "answer")
    assert [m["text"] for m in _channel("needs-user")] == ["merge A or B? recommend A", "A"]


def test_client_id_returns_the_first_message(board: Path) -> None:
    g = _gdir()
    first = bus.post(g, "cc-a", "chief", "question", "q?", client_id="ask-1")
    again = bus.post(g, "cc-a", "chief", "question", "q? (edited)", client_id="ask-1")
    other = bus.post(g, "cc-b", "chief", "question", "q?", client_id="ask-1")
    assert again == first and other["id"] != first["id"]
    assert [m["text"] for m in _channel("lobby")] == ["q?", "q?"]


def test_a_job_post_on_a_persistent_topic_is_one_board_message_in_both_views(board: Path) -> None:
    jb = _job(board, "20261120-Mirror")
    stored = bus.post(jb, "cc-a", "board", "info", "pandas rolling sum is not bit-exact", topic="gotchas")
    bus.post(jb, "cc-a", "board", "claim", "editing pipeline.py", topic="design")  # job-only topic
    assert [m["labels"] for m in _channel("kb/gotchas")] == [{"state": "live", "job": "20261120-Mirror"}]
    assert [m["labels"] for m in _channel("job/20261120-mirror")] == [
        {"topic": "design", "job": "20261120-Mirror", "kind": "claim"},
    ]
    in_job = [m["id"] for m in bus._all(jb)]
    in_global = [m["id"] for m in bus._all(_gdir())]
    assert stored["id"] in in_job and in_global == [stored["id"]] and len(in_job) == 2
    assert bus.fmt(stored).endswith("@20261120-Mirror (info): pandas rolling sum is not bit-exact")


def test_locate_finds_a_global_post_from_a_job_bus_and_the_reply_lands_in_its_thread(board: Path) -> None:
    g = _gdir()
    other = _job(board, "20261120-other")
    post = bus.post(g, "rev", "board", "info", "global gotcha", topic="gotchas")
    holder, found = bus._locate(other, str(post["id"]))
    assert holder == g and found["body"] == "global gotcha"
    r = bus.reply(other, "cc-b", str(post["id"]), "seen it too")
    assert (r["ref"], r["job"], r["topic"]) == (post["id"], "20261120-other", "gotchas")
    assert [(m["re"], m["labels"]) for m in _channel("kb/gotchas")][1] == (str(post["id"]), {"job": "20261120-other"})


def test_job_buses_lists_only_jobs_with_a_channel_and_honours_fade_days(board: Path, board_server: BoardServer) -> None:
    live = _job(board, "20261120-live")
    _job(board, "20261120-silent")  # a dir with no channel
    old = _job(board, "20261001-old")
    bus.post(live, "w", "chief", "info", "hi")
    eight_days_ago = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 8 * 86400))
    posted = board_server.vb(
        "post", f"{_ns()}/job/20261001-old", "an imported note", "--label", "legacy-id=18e0-0009",
        "--label", f"ts={eight_days_ago}", profile="w",
    )
    assert posted.returncode == 0, posted.stderr
    assert bus._job_buses() == sorted([live, old])
    assert bus._job_buses(fade_days=9) == sorted([live, old])
    assert bus._job_buses(fade_days=7) == [live]


def test_an_imported_message_reads_back_with_its_legacy_id_and_ts(board: Path) -> None:
    _import("kb/findings", {"id": "18dcedd9943f3be8-6375", "ts": "2026-10-09T14:03:11", "from": "w9", "to": "board",
                            "kind": "info", "body": "needs-user queue landed", "topic": "findings", "job": "20261009-swarm-tooling"})
    got = bus._all(_gdir())
    assert got == [{
        "id": "18dcedd9943f3be8-6375", "ts": "2026-10-09T14:03:11", "from": "w9", "to": "board", "kind": "info",
        "body": "needs-user queue landed", "topic": "findings", "job": "20261009-swarm-tooling",
    }]
    r = bus.reply(_gdir(), "chief", "18dcedd9943f3be8-6375", "still true")
    assert r["ref"] == "18dcedd9943f3be8-6375"


def test_a_sanitised_profile_round_trips_and_answers_close_its_question(board: Path) -> None:
    jb = _job(board, "20261120-drev")
    q = bus.post(jb, "drev-x0axf.14-1", "chief", "question", "ok to proceed?")
    assert q["from"] == "drev-x0axf.14-1"
    bus.reply(jb, "chief", str(q["id"]), "yes")
    assert bus.pending(jb, "chief") == []
    assert [m["from"] for m in bus._all(jb)] == ["drev-x0axf.14-1", "chief"]
    first = _channel("job/20261120-drev")[0]
    assert (first["by"]["profile"], first["labels"]["from"]) == ("drev-x0axf-14-1", "drev-x0axf.14-1")  # type: ignore[index]


def test_channels_longer_than_a_page_read_completely(board: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    g = _gdir()
    for k in range(5):
        bus.post(g, "w", "board", "info", f"post {k}", topic="thoughts")
    monkeypatch.setattr(bs, "PAGE", 2)
    bs.invalidate()
    assert [m["body"] for m in bus._all(g)] == [f"post {k}" for k in range(5)]
    assert [m["text"] for m in _channel("kb/thoughts")] == [f"post {k}" for k in range(5)]


def test_http_writes_interoperate_with_the_real_cli(board: Path, board_server: BoardServer) -> None:
    jb = _job(board, "20261120-interop")
    root = bus.post(jb, "chief", "cc-1", "info", "task root")
    shown = board_server.vb("read", str(root["id"]), "--json")
    assert shown.returncode == 0, shown.stderr
    assert json.loads(shown.stdout)["message"]["by"]["profile"] == "chief"
    out: dict[str, subprocess.CompletedProcess[str]] = {}
    waiter = threading.Thread(target=lambda: out.setdefault("w", board_server.vb(
        "wait", str(root["id"]), "--after", str(root["id"]), "--timeout", "20s", "--json")))
    waiter.start()  # --after <root> also catches a reply that lands before the wait subscribes
    reply = bus.post(jb, "cc-1", "chief", "info", "on it", ref=str(root["id"]))
    waiter.join(30)
    assert out["w"].returncode == 0, out["w"].stderr
    assert str(reply["id"]) in out["w"].stdout


def test_status_in_board_mode_drops_unread_and_tail_lines(board: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jb = _job(board, "20261120-status")
    bus.post(jb, "cc-1", "chief", "question", "which file?")
    assert bus.main(["--bus", str(jb), "--as", "chief", "status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "pending to chief: 1" and "which file?" in lines[2]
    assert not any(line.startswith(("unread from", "chief tail:")) for line in lines)
    assert "asks watcher: use wait-asks" in lines


# ------------------------------------------------------------------ wait-asks + refusals


def test_wait_asks_delivers_a_question_posted_before_the_call_once(
    board: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jb = _job(board, "20261120-asks")
    root = bus.post(jb, "chief", "cc-1", "info", "task root")
    q = bus.post(jb, "cc-1", "chief", "question", "which file?", ref=str(root["id"]))
    assert bus.main(["--bus", str(jb), "--as", "chief", "wait-asks", "--timeout", "5"]) == 0
    out = capsys.readouterr().out
    assert f"[{q['id']}] cc-1 -> chief (question re:{root['id']}): which file?" in out
    assert f"thread {root['id']}" in out
    assert (jb / "chief.asks.seen").read_text().split() == [str(q["id"])]
    assert bus.main(["--bus", str(jb), "--as", "chief", "wait-asks", "--timeout", "0.3"]) == 3
    blocker = bus.post(jb, "cc-1", "chief", "blocker", "tests red", ref=str(root["id"]))
    assert bus.main(["--bus", str(jb), "--as", "chief", "wait-asks", "--timeout", "5"]) == 0
    out = capsys.readouterr().out
    assert str(blocker["id"]) in out and f"[{q['id']}]" not in out


def test_wait_asks_is_refused_on_the_jsonl_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path / "g"))
    with pytest.raises(SystemExit) as exc:
        bus.main(["--bus", str(tmp_path / "b"), "--as", "chief", "wait-asks", "--timeout", "1"])
    assert exc.value.code == 2 and "tail" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "replacement"),
    [
        (["inbox"], "vb read"),
        (["wait"], "vb wait"),
        (["ask", "--body", "q?"], "vb reply"),
        (["tail", "--to", "chief"], "wait-asks"),
        (["board", "read"], "board search"),
    ],
)
def test_offset_cursor_commands_are_refused_in_board_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str], replacement: str
) -> None:
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path / "g"))
    monkeypatch.setenv("SWARM_STORE", "board")
    with pytest.raises(SystemExit) as exc:
        bus.main(["--bus", str(tmp_path / "b"), "--as", "w", *argv])
    err = capsys.readouterr().err
    assert exc.value.code == 2 and replacement in err and len(err.strip().splitlines()) == 1


def test_an_unknown_store_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path / "g"))
    monkeypatch.setenv("SWARM_STORE", "bogus")
    with pytest.raises(SystemExit) as exc:
        bus.main(["--bus", str(tmp_path / "b"), "board", "topics"])
    assert exc.value.code == 2 and "SWARM_STORE" in capsys.readouterr().err


def test_board_mode_is_a_noop_on_the_jsonl_files(board: Path) -> None:
    """Probe of the old behaviour: on base, SWARM_STORE was ignored and posts went to messages.jsonl."""
    jb = _job(board, "20261120-probe")
    bus.post(jb, "w", "chief", "info", "hello")
    assert not (jb / "messages.jsonl").exists()
    assert [m["text"] for m in _channel("job/20261120-probe")] == ["hello"]


# ------------------------------------------------------------------ board errors never read as "no messages"


class _Stub:
    """A tiny HTTP server answering every request with one (status, body)."""

    def __init__(self) -> None:
        import http.server

        self.answer: tuple[int, bytes] = (200, b"{}")
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _reply(self) -> None:
                status, body = stub.answer
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = _reply

            def log_message(self, *args: object) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/api/board"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Stub]:
    s = _Stub()
    monkeypatch.setenv("BOARD_URL", s.url)
    monkeypatch.setenv("BOARD_TOKEN", "t")
    yield s
    s.server.shutdown()


def _err(code: str) -> bytes:
    return json.dumps({"error": {"code": code, "message": "m"}}).encode()


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (500, _err("internal")),  # a server error on a read that accepts not-found
        (400, _err("invalid_request")),  # a 400, but not a not-found code
        (404, _err("not_found")),  # no such ROUTE: the board is not what we think it is
        (404, b"<html>nginx</html>"),  # a 404 with no board error body
        (200, b"<html>not json</html>"),  # garbage JSON
        (200, b"[1, 2]"),  # JSON, but not an object
    ],
)
def test_a_board_error_on_a_read_exits_instead_of_reading_as_empty(
    stub: _Stub, status: int, body: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.answer = (status, body)
    with pytest.raises(SystemExit) as exc:
        bs.read_thread("42")
    assert exc.value.code not in (0, None)
    assert len(str(exc.value.code).splitlines()) == 1


def test_only_the_boards_own_not_found_reads_as_empty(stub: _Stub) -> None:
    stub.answer = (404, _err("message_not_found"))
    assert bs.read_thread("42") == []
    stub.answer = (200, json.dumps({"messages": [], "cursor": "0"}).encode())
    assert bs.read_channel("nobody/posted/here") == []  # the board answers a missing channel as an empty page


def test_an_unreachable_board_or_missing_token_names_board_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.unit.conftest import free_port

    monkeypatch.setenv("BOARD_URL", f"http://127.0.0.1:{free_port()}/api/board")
    monkeypatch.setenv("BOARD_TOKEN", "t")
    with pytest.raises(SystemExit) as exc:
        bs.read_channel("kb/gotchas")
    assert "unreachable" in str(exc.value.code) and "scripts/agents/board-up.sh" in str(exc.value.code)
    monkeypatch.delenv("BOARD_TOKEN")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        bs.read_channel("kb/gotchas")
    assert "no token" in str(exc.value.code) and "scripts/agents/board-up.sh" in str(exc.value.code)


# ------------------------------------------------------------------ R14: every label value is scanned


_ENV_DUMP = "AWS_REGION=eu1\nDB_HOST=10.0.0.5\nDB_USER=admin\nDB_PASS=hunter2hunter2\nREDIS_URL=redis://x"


def test_a_secret_in_a_retire_reason_is_refused_and_nothing_is_sent(board: Path) -> None:
    g = _gdir()
    p = bus.post(g, "chief", "board", "info", "some post", topic="gotchas")
    with pytest.raises(SystemExit) as exc:
        bus.retire(g, "chief", str(p["id"]), _ENV_DUMP)  # the event body alone has 4 KEY=value line starts
    assert exc.value.code == 2
    assert [(m["text"], m["labels"]) for m in _channel("kb/gotchas")] == [("some post", {"state": "live"})]


def test_a_secret_in_any_label_value_is_refused_by_post_raw_and_set_labels(board: Path) -> None:
    secret = "API_TOKEN=abcdefghijklmnop"
    with pytest.raises(SystemExit) as exc:
        bs.post_raw(f"{_ns()}/chat", "harmless", labels={"relayed-by": secret}, profile="w")
    assert exc.value.code == 2 and _channel("chat") == []
    with pytest.raises(SystemExit) as exc:
        bus.post(_gdir(), secret, "board", "info", "harmless", topic="chat")  # the sender becomes a `from` label
    assert exc.value.code == 2 and _channel("chat") == []
    p = bs.post_raw(f"{_ns()}/chat", "harmless", profile="w")
    with pytest.raises(SystemExit) as exc:
        bs.set_labels(str(p["id"]), {"note": "fine", "reason": secret})
    assert exc.value.code == 2 and _channel("chat")[0]["labels"] == {}


# ------------------------------------------------------------------ B2: of two supersedes the first wins


def test_a_supersede_that_lost_the_race_leaves_the_first_link(board: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    g = _gdir()
    old = bus.post(g, "w", "board", "info", "use ruff 0.6 flags", topic="decisions")
    stale = bs.lookup(str(old["id"]))  # what the slower writer read: still live
    first = bus.supersede(g, "wa", str(old["id"]), "decisions", "use ruff 0.7 flags")
    with monkeypatch.context() as m:  # the second writer read before the first wrote
        m.setattr(bs, "lookup", lambda _old_id: stale)
        second = bus.supersede(g, "wb", str(old["id"]), "decisions", "use ruff 0.8 flags")
    labels = {str(m["id"]): m["labels"] for m in _channel("kb/decisions")}
    assert labels[str(old["id"])] == {"state": "superseded", "superseded-by": str(first["id"])}
    assert labels[str(second["id"])] == {"state": "live", "supersedes": str(old["id"])}
    assert bus.hidden_posts(bus._all(g))[str(old["id"])] == f"superseded by {first['id']}"


def test_a_supersede_of_an_unlabelled_job_post_is_conditional_on_unset(board: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jb = _job(board, "20261120-sup")
    old = bus.post(jb, "w", "board", "claim", "editing pipeline.py", topic="design")
    stale = bs.lookup(str(old["id"]))
    first = bus.supersede(jb, "wa", str(old["id"]), "design", "editing pipeline.py and bus.py")
    with monkeypatch.context() as m:
        m.setattr(bs, "lookup", lambda _old_id: stale)
        bus.supersede(jb, "wb", str(old["id"]), "design", "editing bus.py only")
    target = next(m for m in _channel("job/20261120-sup") if m["id"] == old["id"])
    assert target["labels"]["superseded-by"] == str(first["id"])  # type: ignore[index]


# ------------------------------------------------------------------ S1: a bare state label hides a post


def test_a_board_native_state_label_hides_the_post_and_live_shows_it_again(board: Path, board_server: BoardServer) -> None:
    g = _gdir()
    p = bus.post(g, "chief", "board", "info", "stale advice nt_runner.py", topic="gotchas")
    q = bus.post(g, "chief", "board", "info", "superseded advice nt_runner.py", topic="gotchas")
    reply = bus.reply(g, "w", str(p["id"]), "agreed")
    bus.post(g, "chief", "board", "info", "fresh advice nt_runner.py", topic="gotchas")
    for mid, labels in ((p["id"], ["state=retired", "reason=wrong after the NT bump"]), (q["id"], ["state=superseded"])):
        done = board_server.vb("label", str(mid), *labels, profile="vq-inspector")
        assert done.returncode == 0, done.stderr
    assert [m["body"] for m in bus.board_search(g, topics=["gotchas"])] == ["fresh advice nt_runner.py"]
    every = {m["body"]: m.get("hidden") for m in bus.board_search(g, paths=["nt_runner.py"], include_hidden=True)}
    assert every == {
        "stale advice nt_runner.py": "retired: wrong after the NT bump", "superseded advice nt_runner.py": "superseded: state label",
        "fresh advice nt_runner.py": None,
    }
    assert bus.hidden_posts(bus._all(g))[str(reply["id"])] == f"reply under hidden {p['id']}"
    assert "stale advice" not in bus.brief(context=([], set()), now=time.time())
    back = board_server.vb("label", str(p["id"]), "state=live", profile="vq-inspector")
    assert back.returncode == 0, back.stderr
    assert [m["body"] for m in bus.board_search(g, topics=["gotchas"])] == ["stale advice nt_runner.py", "agreed", "fresh advice nt_runner.py"]


def test_a_retire_record_wins_over_the_label_reason(board: Path) -> None:
    g = _gdir()
    p = bus.post(g, "chief", "board", "info", "old rule", topic="findings")
    event = bus.retire(g, "chief", str(p["id"]), "pre-break metric")
    assert event is not None
    hidden = bus.hidden_posts(bus._all(g))
    assert (hidden[str(p["id"])], hidden[str(event["id"])]) == ("retired: pre-break metric", "retire event")


# ------------------------------------------------------------------ S3: another process's writes are seen


def test_reads_see_writes_by_another_process_reply_label_and_delete(board: Path, board_server: BoardServer) -> None:
    g = _gdir()
    jb = _job(board, "20261120-other-proc")
    root = bus.post(jb, "chief", "cc-1", "info", "task root")
    post = bus.post(g, "w", "board", "info", "a finding", topic="findings")
    doomed = bus.post(g, "w", "board", "info", "to be deleted", topic="findings")
    assert [m["body"] for m in bus._all(jb)] == ["task root"]
    assert [m["body"] for m in bus._all(g)] == ["a finding", "to be deleted"]
    asked = board_server.vb("reply", str(root["id"]), "[question] which file?", profile="cc-1")
    assert asked.returncode == 0, asked.stderr
    assert [(m["from"], m["kind"], m["body"]) for m in bus._all(jb)][1:] == [("cc-1", "question", "[question] which file?")]
    assert [m["id"] for m in bus.pending(jb, "chief")] != []
    labelled = board_server.vb("label", str(post["id"]), "importance=3", profile="w")
    assert labelled.returncode == 0, labelled.stderr
    assert next(m for m in bus._all(g) if m["id"] == post["id"]).get("importance") == 3
    gone = board_server.vb("delete", str(doomed["id"]), profile="w")
    assert gone.returncode == 0, gone.stderr
    assert [m["body"] for m in bus._all(g)] == ["a finding"]


# ------------------------------------------------------------------ S2: one head, one mapping, per multi-bus read


def test_pending_all_reads_one_head_and_maps_nothing_again_while_it_is_unchanged(
    board: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for j in range(3):
        jb = _job(board, f"20261120-j{j}")
        root = bus.post(jb, "chief", f"cc-{j}", "info", "task")
        bus.post(jb, f"cc-{j}", "chief", "question", f"q{j}?", ref=str(root["id"]))
    bus.post(_gdir(), "w", "board", "info", "global post", topic="gotchas")
    calls: list[tuple[str, str]] = []
    real_http, real_map = bs._http, bs.map_messages
    mapped: list[int] = []

    def spy(method: str, path: str, *a: object, **kw: object) -> dict[str, object]:
        calls.append((method, path))
        return real_http(method, path, *a, **kw)  # type: ignore[arg-type]

    def spy_map(raw: list[dict[str, object]], *a: object, **kw: object) -> list[dict[str, object]]:
        mapped.append(len(raw))
        return real_map(raw, *a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(bs, "_http", spy)
    monkeypatch.setattr(bs, "map_messages", spy_map)
    bs.invalidate()
    assert sorted(m["body"] for m in bus.pending_all("chief")) == ["q0?", "q1?", "q2?"]
    assert calls.count(("GET", "/events")) == 1
    assert mapped.count(1) == 1  # the global view (one post) is mapped once for all three job views
    calls.clear()
    mapped.clear()
    assert len(bus.pending_all("chief")) == 3
    assert (calls, mapped) == ([("GET", "/events")], [])  # unchanged head: one head request, nothing else


# ------------------------------------------------------------------ F2: digest cursors survive a delete


def test_digest_after_a_delete_neither_redelivers_nor_skips(board: Path, board_server: BoardServer) -> None:
    jb = _job(board, "20261120-digest")
    first = bus.post(jb, "w", "chief", "info", "one")
    bus.post(jb, "w", "chief", "info", "two")
    bus.post(jb, "w", "chief", "info", "three")
    assert [m["body"] for m in bus._read_since(jb, "chief.digest")] == ["one", "two", "three"]
    gone = board_server.vb("delete", str(first["id"]), profile="w")
    assert gone.returncode == 0, gone.stderr
    bus.post(jb, "w", "chief", "info", "four")
    assert [m["body"] for m in bus._read_since(jb, "chief.digest")] == ["four"]
    assert bus._read_since(jb, "chief.digest") == []
    assert [body for _, msgs in bus.digest("cc-x") for body in (m["body"] for m in msgs)] == ["two", "three", "four"]  # another reader


# ------------------------------------------------------------------ F3: client_id returns the first message


def test_client_id_returns_the_first_message_even_in_another_thread(board: Path) -> None:
    jb = _job(board, "20261120-cid")
    root = bus.post(jb, "chief", "cc-a", "info", "task")
    first = bus.post(jb, "cc-a", "chief", "question", "q?", client_id="k")
    again = bus.post(jb, "cc-a", "chief", "question", "q? (edited)", client_id="k")  # same channel: the board answers the first
    moved = bus.post(jb, "cc-a", "chief", "question", "q? in the thread", ref=str(root["id"]), client_id="k")  # 409 duplicate_id
    assert again == first and moved == first
    assert [m["text"] for m in _channel("job/20261120-cid")] == ["task", "q?"]


# ------------------------------------------------------------------ public helpers for board_io.py


def test_set_labels_with_a_precondition_sets_only_while_it_holds(board: Path) -> None:
    p = bs.post_raw(f"{_ns()}/chat", "target", profile="w")
    assert bs.set_labels(str(p["id"]), {"state": "superseded", "superseded-by": "x1"}, when={"state": None}) is True
    assert bs.set_labels(str(p["id"]), {"state": "superseded", "superseded-by": "x2"}, when={"state": "live"}) is False
    assert _channel("chat")[0]["labels"] == {"state": "superseded", "superseded-by": "x1"}
    secret = "API_TOKEN=abcdefghijklmnop"
    with pytest.raises(SystemExit) as exc:
        bs.set_labels(str(p["id"]), {"reason": secret}, when={"state": "superseded"})
    assert exc.value.code == 2 and "reason" not in _channel("chat")[0]["labels"]


def test_list_channels_lists_every_channel_under_a_prefix(board: Path) -> None:
    for ch in ("kb/gotchas", "lobby", "job/20261120-a", "job/20261120-b"):
        bs.post_raw(f"{_ns()}/{ch}", "x", profile="w")
    assert sorted(bs.list_channels(f"{_ns()}/")) == [f"{_ns()}/{c}" for c in ("job/20261120-a", "job/20261120-b", "kb/gotchas", "lobby")]
    assert sorted(bs.list_channels(f"{_ns()}/job/")) == [f"{_ns()}/job/20261120-a", f"{_ns()}/job/20261120-b"]
