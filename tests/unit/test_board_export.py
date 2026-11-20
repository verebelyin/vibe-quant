"""board_io.py export: the public knowledge export of the board, and its round trip.

Board A is seeded through the public bus functions (``SWARM_STORE=board``) and raw `vb`
(a bare-label retire, an edit): one message of every kind the global view holds, plus a
``job/*`` thread that must never be exported. ``export`` A -> ``import-legacy`` of that file
into an empty board B (another ``SWARM_BOARD_NS``) -> ``export`` B: the two files are
byte-identical and bus.py sees the same visible posts, hidden posts and brief on both.
Everything runs on the session's throwaway ``board_server``.
"""

from __future__ import annotations

import contextlib
import getpass
import importlib.util
import io
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

    from tests.unit.conftest import BoardServer

_AGENTS = Path(__file__).resolve().parents[2] / "scripts" / "agents"
_UNROUTABLE = "http://127.0.0.1:9/api/board"
_JOB = "20261120-x"
_SECRET = "API_TOKEN=abcdefghijklmnop"


def _load(name: str, file: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _AGENTS / file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


bio = _load("board_io", "board_io.py")
bs: ModuleType = bio.bs  # the store board_io reads through (same module object)
bus = _load("swarm_bus_export", "bus.py")


def _env(mp: pytest.MonkeyPatch, server: BoardServer, ns: str, root: Path) -> None:
    """bus.py on the board store of ``server``, channel namespace ``ns``, dirs under ``root``."""
    for key in ("BOARD_URL", "XDG_CONFIG_HOME"):
        mp.setenv(key, server.env[key])
    mp.setenv("SWARM_STORE", "board")
    mp.setenv("SWARM_BOARD_NS", ns)
    mp.setenv("SWARM_BOARD", str(root / "global"))
    mp.setenv("SWARM_JOBS", str(root / "jobs"))
    mp.delenv("SWARM_BUS", raising=False)
    mp.setattr(bus, "work_context", lambda: ([], set()))
    mp.setattr(bus, "comparability_breaks", lambda: [])


def _main(*args: str) -> tuple[int, dict[str, Any] | None]:
    """``board_io.main`` in this process: (exit code, the JSON report on stdout or None)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = bio.main(list(args))
    return code, json.loads(out.getvalue()) if out.getvalue().strip() else None


def _export(out: Path, *extra: str) -> tuple[int, dict[str, Any] | None]:
    return _main("export", "--out", str(out), *extra)


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def _line(row: dict[str, object]) -> str:
    return json.dumps(row, ensure_ascii=False)


def _vb(server: BoardServer, *args: str) -> None:
    got = server.vb(*args, profile="w")
    assert got.returncode == 0, got.stderr


@dataclass
class Trip:
    server: BoardServer
    root: Path
    ns_a: str
    ns_b: str
    seeded: dict[str, dict[str, Any]]  # name -> the message bus.py stored on A (board ids)
    file_a: Path
    file_b: Path
    report_a: dict[str, Any]
    report_import: dict[str, Any]


def _seed(server: BoardServer, root: Path) -> dict[str, dict[str, Any]]:
    g = bus.global_board_dir()
    s: dict[str, dict[str, Any]] = {}
    s["live"] = bus.post(g, "cc-a", "board", "info", "RULE: widget.py reads gizmo first", topic="gotchas",
                         extra={"importance": 3, "job": _JOB})
    s["reply1"] = bus.reply(g, "cc-b", str(s["live"]["id"]), "confirmed on sprocket")
    s["reply2"] = bus.reply(g, "cc-a", str(s["reply1"]["id"]), "thanks, ünïcode kept")
    s["stale"] = bus.post(g, "w", "board", "info", "stale finding about gamma", topic="findings")
    retired = bus.retire(g, "chief", str(s["stale"]["id"]), "pre-break metric")
    assert retired is not None
    s["retire"] = retired
    s["old"] = bus.post(g, "w", "board", "info", "use ruff 0.6 flags", topic="decisions")
    s["sup1"] = bus.supersede(g, "chief", str(s["old"]["id"]), "decisions", "use ruff 0.7 flags")
    s["sup2"] = bus.supersede(g, "chief", str(s["old"]["id"]), "decisions", "use ruff 0.8 flags")  # lost: sup1 won
    s["bare"] = bus.post(g, "w", "board", "info", "bare label target", topic="thoughts")
    _vb(server, "label", str(s["bare"]["id"]), "state=retired")  # R23: no retire record
    s["edited"] = bus.post(g, "w", "board", "info", "first text", topic="model-notes")
    _vb(server, "edit", str(s["edited"]["id"]), "edited text")
    s["question"] = bus.post(g, "cc-a", "board", "question", "A or B?", topic="needs-user")
    s["answer"] = bus.needs_user_answer(g, "chief", str(s["question"]["id"]), "Take A.")
    s["lobby"] = bus.post(g, "cc-b", "chief", "done", "lobby note for chief")
    s["mixed"] = bus.post(g, "Arch L2!", "board", "info", "mixed case author on chat", topic="chat")
    jb = root / "jobs" / _JOB / "bus"
    jb.mkdir(parents=True)
    s["job_root"] = bus.post(jb, "chief", "cc-a", "info", "JOBONLY root of a task thread")
    s["job_reply"] = bus.reply(jb, "cc-a", str(s["job_root"]["id"]), "JOBONLY progress")
    return s


@pytest.fixture(scope="module")
def trip(board_server: BoardServer, tmp_path_factory: pytest.TempPathFactory) -> Trip:
    root = tmp_path_factory.mktemp("export")
    ns_a, ns_b = "ea" + uuid.uuid4().hex[:10], "eb" + uuid.uuid4().hex[:10]
    (root / "empty-jobs").mkdir()
    with pytest.MonkeyPatch.context() as mp:
        _env(mp, board_server, ns_a, root)
        seeded = _seed(board_server, root)
        code_a, report_a = _export(root / "a.jsonl")
        mp.setenv("SWARM_BOARD_NS", ns_b)
        code_i, report_i = _main("import-legacy", "--persistent", str(root / "a.jsonl"),
                                 "--jobs-root", str(root / "empty-jobs"), "--map", str(root / "b-map.json"))
        code_b, _ = _export(root / "b.jsonl")
    assert (code_a, code_i, code_b) == (0, 0, 0), report_i
    assert report_a is not None and report_i is not None
    return Trip(board_server, root, ns_a, ns_b, seeded, root / "a.jsonl", root / "b.jsonl", report_a, report_i)


@pytest.fixture
def on_a(trip: Trip, monkeypatch: pytest.MonkeyPatch) -> Trip:
    _env(monkeypatch, trip.server, trip.ns_a, trip.root)
    return trip


def _eid(trip: Trip, name: str) -> str:
    return f"b{trip.seeded[name]['id']}"


# ------------------------------------------------------------------ the round trip


def test_round_trip_is_byte_identical(trip: Trip) -> None:
    a, b = trip.file_a.read_bytes(), trip.file_b.read_bytes()
    assert a == b
    assert a.endswith(b"}\n") and not a.endswith(b"\n\n")
    assert (trip.report_import["rejects"], trip.report_import["secret_refused"]) == (
        {"count": 0, "items": []}, {"count": 0, "ids": []})


def test_bus_sees_the_same_posts_hidden_posts_and_brief_on_both_boards(trip: Trip, monkeypatch: pytest.MonkeyPatch) -> None:
    now = time.time()

    def views(ns: str, ids: Any) -> tuple[object, ...]:
        monkeypatch.setenv("SWARM_BOARD_NS", ns)
        g = bus.global_board_dir()
        every = bus.board_search(g, include_hidden=True)
        return (
            [ids(str(m["id"])) for m in bus.board_search(g)],
            [(ids(str(m["id"])), re.sub(r"\b(\d+)\b", lambda x: ids(x.group(1)), str(m.get("hidden")))) for m in every],
            sorted(ids(i) for i in bus.hidden_posts(bus._all(g))),
            re.sub(r"\[(\d+)\]", lambda x: f"[{ids(x.group(1))}]", bus.brief(context=([], set()), now=now)),
        )

    _env(monkeypatch, trip.server, trip.ns_a, trip.root)
    on_a = views(trip.ns_a, lambda i: f"b{i}")
    on_b = views(trip.ns_b, lambda i: i)
    assert on_a == on_b
    visible, _, hidden, brief = on_a
    e = {n: _eid(trip, n) for n in trip.seeded}
    assert visible == [e[n] for n in ("live", "reply1", "reply2", "sup1", "sup2", "edited", "question", "answer", "mixed")]
    assert hidden == sorted([e["stale"], e["retire"], e["old"], e["bare"]])
    assert f"[{e['live']}] #gotchas cc-a" in brief and e["bare"] not in brief


def test_each_kind_exports_exactly_this_row(trip: Trip) -> None:
    rows, s = _rows(trip.file_a), trip.seeded
    e = {n: _eid(trip, n) for n in s}

    def row(name: str, **fields: object) -> str:
        return _line({"id": e[name], "ts": s[name]["ts"], **fields})

    lines = set(trip.file_a.read_text(encoding="utf-8").splitlines())
    want = [
        row("live", **{"from": "cc-a", "to": "board", "kind": "info", "body": "RULE: widget.py reads gizmo first",
                       "topic": "gotchas", "job": _JOB, "importance": 3}),
        row("reply1", **{"from": "cc-b", "to": "board", "kind": "info", "body": "confirmed on sprocket",
                         "topic": "gotchas", "ref": e["live"]}),
        row("reply2", **{"from": "cc-a", "to": "board", "kind": "info", "body": "thanks, ünïcode kept",
                         "topic": "gotchas", "ref": e["reply1"]}),
        row("retire", **{"from": "chief", "to": "board", "kind": "info", "body": f"retired {s['stale']['id']}: pre-break metric",
                         "topic": "findings", "ref": e["stale"], "retires": e["stale"], "reason": "pre-break metric"}),
        row("sup1", **{"from": "chief", "to": "board", "kind": "info", "body": "use ruff 0.7 flags",
                       "topic": "decisions", "supersedes": e["old"]}),
        row("sup2", **{"from": "chief", "to": "board", "kind": "info", "body": "use ruff 0.8 flags",
                       "topic": "decisions", "supersedes": e["old"]}),
        row("bare", **{"from": "w", "to": "board", "kind": "info", "body": "bare label target", "topic": "thoughts",
                       "state": "retired"}),
        row("edited", **{"from": "w", "to": "board", "kind": "info", "body": "edited text", "topic": "model-notes"}),
        row("question", **{"from": "cc-a", "to": "board", "kind": "question", "body": "A or B?", "topic": "needs-user"}),
        row("answer", **{"from": "user", "to": "board", "kind": "answer", "body": "Take A.", "topic": "needs-user",
                         "ref": e["question"], "user_answer": True, "by": "chief"}),
        row("lobby", **{"from": "cc-b", "to": "chief", "kind": "done", "body": "lobby note for chief"}),
        row("mixed", **{"from": "Arch L2!", "to": "board", "kind": "info", "body": "mixed case author on chat", "topic": "chat"}),
    ]
    assert [w for w in want if w not in lines] == []
    assert rows[e["stale"]] == {"id": e["stale"], "ts": s["stale"]["ts"], "from": "w", "to": "board", "kind": "info",
                                "body": "stale finding about gamma", "topic": "findings"}
    assert len(rows) == 14  # the 14 global messages: nothing of the job thread


def test_rows_are_in_posting_order_and_the_report_counts_them(trip: Trip) -> None:
    rows = list(map(json.loads, trip.file_a.read_text(encoding="utf-8").splitlines()))
    posted = [n for n in trip.seeded if not n.startswith("job_")]
    assert [r["id"] for r in rows] == [_eid(trip, n) for n in posted]  # same-second posts keep the board's order
    report = dict(trip.report_a)
    assert isinstance(report.pop("wall_seconds"), float)
    assert report == {
        "rows": 14, "hidden": 4, "out": str(trip.file_a),
        "channels": {"chat": 1, "kb/decisions": 3, "kb/findings": 2, "kb/gotchas": 3, "kb/model-notes": 1,
                     "kb/thoughts": 1, "lobby": 1, "needs-user": 2},
    }


def test_no_job_channel_no_actor_no_os_user_no_url(trip: Trip) -> None:
    text = trip.file_a.read_text(encoding="utf-8")
    assert "JOBONLY" not in text
    for absent in (getpass.getuser(), "local:", "actor", "principal", trip.server.url, str(trip.server.port), trip.ns_a):
        assert absent not in text, absent


def test_a_second_export_of_an_unchanged_board_is_identical(on_a: Trip) -> None:
    again = on_a.root / "a-again.jsonl"
    assert _export(again)[0] == 0
    assert again.read_bytes() == on_a.file_a.read_bytes()


# ------------------------------------------------------------------ changes, refusals, CLI


@pytest.fixture
def small(board_server: BoardServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An empty namespace of its own with two posts; yields its first export."""
    _env(monkeypatch, board_server, "es" + uuid.uuid4().hex[:10], tmp_path)
    g = bus.global_board_dir()
    bus.post(g, "w", "board", "info", "post one", topic="gotchas")
    bus.post(g, "w", "board", "info", "post two", topic="findings")
    out = tmp_path / "export.jsonl"
    assert _export(out)[0] == 0
    return out


def test_an_edit_or_retire_after_an_export_changes_only_its_own_line(
    small: Path, board_server: BoardServer, tmp_path: Path
) -> None:
    before = small.read_text(encoding="utf-8").splitlines()
    one, two = (json.loads(line) for line in before)
    _vb(board_server, "edit", one["id"][1:], "post one, edited")
    _vb(board_server, "label", two["id"][1:], "state=retired", "reason=wrong")
    assert _export(small)[0] == 0
    assert small.read_text(encoding="utf-8").splitlines() == [
        _line({**one, "body": "post one, edited"}),
        _line({**two, "state": "retired", "state_reason": "wrong"}),
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["export.jsonl", "global"]


def test_a_secret_on_the_board_refuses_the_export_and_keeps_the_old_file(
    small: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = small.read_bytes()
    leaked = bs._http("POST", "/messages", {"channel": f"{bs.ns()}chat", "text": f"oops {_SECRET}", "labels": {}},
                      profile="w", expect=("message",))["message"]  # past the write-path scan, as a raw API client could
    code, report = _export(small)
    err = capsys.readouterr().err
    assert (code, report) == (1, None)
    assert f"b{leaked['id']}" in err and _SECRET not in err and "oops" not in err
    assert small.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["export.jsonl", "global"]


def test_no_board_url_exits_2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("BOARD_URL", raising=False)
    assert _export(tmp_path / "x.jsonl") == (2, None)
    assert "no board" in capsys.readouterr().err and not (tmp_path / "x.jsonl").exists()


def test_board_down_exits_naming_board_up_and_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOARD_URL", _UNROUTABLE)
    monkeypatch.setenv("BOARD_TOKEN", "t")
    with pytest.raises(SystemExit) as exc:
        _export(tmp_path / "x.jsonl")
    assert exc.value.code not in (0, None) and "board-up.sh" in str(exc.value.code)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ pure row building (no board)


def _raw(mid: str, channel: str, text: str, labels: dict[str, str] | None = None, re_: str | None = None,
         created: str = "2026-11-20T10:00:00.000Z", deleted: bool = False) -> dict[str, object]:
    return {"id": mid, "channel": channel, "text": text, "re": re_, "replyTo": None, "labels": labels or {},
            "createdAt": created, "deleted": deleted,
            "by": {"actor": "local:user:someone+w", "principal": "local:user:someone", "profile": "w"}}


def test_rows_keep_legacy_ids_and_derive_state_only_where_the_records_do_not() -> None:
    raw = [
        _raw("1", "kb/gotchas", "legacy post", {"legacy-id": "18e0-0001", "ts": "2026-10-01T08:00:00", "state": "retired", "reason": "x"}),
        _raw("2", "kb/gotchas", "retired 18e0-0001: x", {"ts": "2026-10-01T09:00:00", "retires": "1"}, re_="1"),
        _raw("3", "kb/gotchas", "superseded, then shown again", {"ts": "2026-10-01T10:00:00", "state": "live"}),
        _raw("4", "kb/gotchas", "the correction", {"ts": "2026-10-01T11:00:00", "supersedes": "3", "state": "live"}),
        _raw("5", "chat", "gone", {"ts": "2026-10-01T12:00:00"}, deleted=True),
        _raw("6", "kb/gotchas", "orphan retire", {"ts": "2026-10-01T13:00:00", "retires": "zz-missing"}),
    ]
    got = bio.export_rows(raw, {})
    base = {"from": "w", "to": "board", "kind": "info", "topic": "gotchas"}
    assert got == [
        {"id": "18e0-0001", "ts": "2026-10-01T08:00:00", **base, "body": "legacy post"},
        {"id": "b2", "ts": "2026-10-01T09:00:00", **base, "body": "retired 18e0-0001: x", "ref": "18e0-0001",
         "retires": "18e0-0001", "reason": "x"},
        {"id": "b3", "ts": "2026-10-01T10:00:00", **base, "body": "superseded, then shown again", "state": "live"},
        {"id": "b4", "ts": "2026-10-01T11:00:00", **base, "body": "the correction", "supersedes": "b3"},
        {"id": "b6", "ts": "2026-10-01T13:00:00", **base, "body": "orphan retire", "retires": "zz-missing"},
    ]
    text = bio.render(got)
    assert text.splitlines()[2] == ('{"id": "b3", "ts": "2026-10-01T10:00:00", "from": "w", "to": "board", "kind": "info", '
                                    '"body": "superseded, then shown again", "topic": "gotchas", "state": "live"}')
    assert "someone" not in text and text.endswith("}\n")


def test_same_second_rows_keep_the_board_read_order_never_the_id_text() -> None:
    """Three native posts in one second, a two-digit board id after a one-digit one (chief
    ruling): ``b10`` < ``b9`` as text, so an id sort would put them out of posting order."""
    raw = [
        _raw("11", "kb/gotchas", "third", created="2026-11-20T10:00:00.900Z"),
        _raw("9", "chat", "first", created="2026-11-20T10:00:00.100Z"),
        _raw("10", "kb/gotchas", "second", created="2026-11-20T10:00:00.500Z"),
        _raw("8", "lobby", "a second earlier", {"ts": "2026-11-20T09:59:59", "to": "chief"},
             created="2026-11-20T10:00:00.950Z"),  # a ts label decides before the read order
    ]
    assert [(r["id"], r["body"]) for r in bio.export_rows(raw, {})] == [
        ("b8", "a second earlier"), ("b9", "first"), ("b10", "second"), ("b11", "third"),
    ]
