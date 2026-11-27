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
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
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


# ------------------------------------------------------------------ pure READ mapping


def _raw(
    mid: str,
    channel: str,
    text: str,
    *,
    re: str | None = None,
    reply_to: str | None = None,
    labels: dict[str, str] | None = None,
    profile: str | None = "w",
    created: str = "2026-11-20T10:00:00.000Z",
    deleted: bool = False,
) -> dict[str, object]:
    return {
        "id": mid,
        "channel": channel,
        "text": text,
        "re": re,
        "replyTo": reply_to,
        "labels": labels or {},
        "createdAt": created,
        "deleted": deleted,
        "by": {
            "actor": f"local:user:me+{profile}",
            "principal": "local:user:me",
            "profile": profile,
        },
    }


# ------------------------------------------------------------------ pure WRITE mapping


def _msg(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "18e9-0001",
        "ts": "2026-11-20T11:00:00",
        "from": "cc-1",
        "to": "board",
        "kind": "info",
        "body": "b",
    }
    base.update(kw)
    return base


def _plan(
    msg: dict[str, object],
    job: str | None = None,
    to_global: bool = False,
    lookup: dict[str, dict[str, object]] | None = None,
    ns: str = "",
) -> dict[str, object]:
    return dict(
        bs.plan_write(msg, job=job, to_global=to_global, lookup=lookup or {}, ns=ns)._asdict()
    )


def test_write_retire_and_supersede_labels_and_side_effects() -> None:
    target = _raw("7", "kb/gotchas", "old", labels={"legacy-id": "L-7", "state": "live"})
    retire = _msg(
        topic="gotchas",
        ref="L-7",
        retires="L-7",
        reason="wrong\nafter " + "x" * 300,
        body="retired L-7: wrong",
    )
    plan = _plan(retire, None, lookup={"L-7": target})
    assert (plan["re"], plan["labels"]) == ("7", {"retires": "7"})
    effects = bs.side_effects(retire, "99", {"L-7": target})
    assert effects == [
        ("7", {"state": "retired", "reason": ("wrong after " + "x" * 300)[:200]}, None)
    ]
    sup = _msg(topic="gotchas", supersedes="L-7")
    assert _plan(sup, None, lookup={"L-7": target})["labels"] == {
        "state": "live",
        "supersedes": "7",
    }
    assert bs.side_effects(sup, "100", {"L-7": target}) == [
        (
            "7",
            {"state": "superseded", "superseded-by": "100"},
            {"state": "live"},
        ),  # only while still live
    ]
    unlabelled = _raw("8", "job/j", "job post", labels={"legacy-id": "L-8", "topic": "gotchas"})
    assert bs.side_effects(_msg(supersedes="L-8"), "101", {"L-8": unlabelled}) == [
        (
            "8",
            {"state": "superseded", "superseded-by": "101"},
            {"state": None},
        ),  # only while still unset
    ]
    gone = _raw("7", "kb/gotchas", "old", labels={"state": "superseded", "superseded-by": "50"})
    assert bs.side_effects(sup, "100", {"L-7": gone}) == []


# ------------------------------------------------------------------ behaviour on a real board


@pytest.fixture
def board(
    board_server: BoardServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Path]:
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
    bus.post(
        g,
        "cc-a",
        "board",
        "info",
        "nt_runner.py:1337 swallows DataUnavailableError; preflight first",
        topic="gotchas",
    )
    bus.post(
        g,
        "rev-b",
        "board",
        "info",
        "vibe_quant/liquidation/scan.py: funding cut at end_ns-1ms",
        topic="findings",
    )
    bus.post(
        g, "chief", "board", "info", "Ruling: clearance fee counts as a cost", topic="decisions"
    )
    bus.post(g, "cc-d", "chief", "done", "liquidation scan.py done")  # a DM, never a search hit
    assert [m["from"] for m in bus.board_search(g, topics=["gotchas", "decisions"])] == [
        "cc-a",
        "chief",
    ]
    assert [m["from"] for m in bus.board_search(g, paths=["vibe_quant/liquidation/scan.py"])] == [
        "rev-b"
    ]
    assert [m["from"] for m in bus.board_search(g, terms=["FUNDING", "end_ns"])] == ["rev-b"]
    assert bus.board_search(g, terms=["funding", "preflight"]) == []
    assert [m["to"] for m in _all_mapped_lobby()] == ["chief"]
    assert [m["labels"] for m in _channel("lobby")] == [{"to": "chief", "kind": "done"}]


def _all_mapped_lobby() -> list[dict[str, object]]:
    return [m for m in bus._all(_gdir()) if m.get("to") != "board"]


def test_channels_longer_than_a_page_read_completely(
    board: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    g = _gdir()
    for k in range(5):
        bus.post(g, "w", "board", "info", f"post {k}", topic="thoughts")
    monkeypatch.setattr(bs, "PAGE", 2)
    bs.invalidate()
    assert [m["body"] for m in bus._all(g)] == [f"post {k}" for k in range(5)]
    assert [m["text"] for m in _channel("kb/thoughts")] == [f"post {k}" for k in range(5)]


def test_the_unit_suite_runs_on_the_jsonl_store() -> None:
    """tests/unit/conftest.py pins SWARM_STORE=jsonl: no test reaches a board by default."""
    assert bus.store_kind() == "jsonl"


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


_FULL = 2  # PAGE in the malformed-page tests: a page of 2 announces the next one


@pytest.mark.parametrize(
    ("call", "body"),
    [
        (
            "channel",
            {"error": {"code": "stub_unavailable"}},
        ),  # 200 with an error where a page belongs
        ("channel", {"messages": "none"}),
        ("channel", {"messages": [{"text": "no id"}]}),
        ("channel", {"messages": [{"id": "1"}, {"id": "2"}]}),  # full page, no cursor
        (
            "channel",
            {"messages": [{"id": "1"}, {"id": "2"}], "cursor": "0"},
        ),  # full page, cursor stuck at the start
        ("thread", {"error": {"code": "stub_unavailable"}}),
        ("thread", {"message": {"id": "7"}}),  # no replies list
        (
            "thread",
            {"message": {"id": "7"}, "replies": [{"id": "8"}, {"id": "7"}]},
        ),  # full page, last reply not past
        ("channels", {"error": {"code": "stub_unavailable"}}),
        ("channels", {"channels": [{"path": "a"}], "next": ""}),
        ("pending", {"error": {"code": "stub_unavailable"}}),  # /events without its cursor
    ],
)
def test_a_malformed_success_exits_instead_of_reading_as_empty(
    stub: _Stub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, call: str, body: dict[str, object]
) -> None:
    stub.answer = (200, json.dumps(body).encode())
    monkeypatch.setattr(bs, "PAGE", _FULL)
    monkeypatch.setenv("SWARM_STORE", "board")
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path / "global"))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path))
    bs.invalidate()
    run = {
        "channel": lambda: bs.read_channel("kb/gotchas"),
        "thread": lambda: bs.read_thread("7"),
        "channels": lambda: bs.list_channels("job/"),
        "pending": lambda: bus.pending_all("chief"),
    }[call]
    with pytest.raises(SystemExit) as exc:
        run()
    assert exc.value.code not in (0, None)
    assert len(str(exc.value.code).splitlines()) == 1 and "board-up.sh" in str(exc.value.code)


# ------------------------------------------------------------------ R14: every label value is scanned


_ENV_DUMP = (
    "AWS_REGION=eu1\nDB_HOST=10.0.0.5\nDB_USER=admin\nDB_PASS=hunter2hunter2\nREDIS_URL=redis://x"
)


def test_a_secret_in_a_retire_reason_is_refused_and_nothing_is_sent(board: Path) -> None:
    g = _gdir()
    p = bus.post(g, "chief", "board", "info", "some post", topic="gotchas")
    with pytest.raises(SystemExit) as exc:
        bus.retire(
            g, "chief", str(p["id"]), _ENV_DUMP
        )  # the event body alone has 4 KEY=value line starts
    assert exc.value.code == 2
    assert [(m["text"], m["labels"]) for m in _channel("kb/gotchas")] == [
        ("some post", {"state": "live"})
    ]


def test_a_secret_in_any_label_value_is_refused_by_post_raw_and_set_labels(board: Path) -> None:
    secret = "API_TOKEN=abcdefghijklmnop"
    with pytest.raises(SystemExit) as exc:
        bs.post_raw(f"{_ns()}/chat", "harmless", labels={"relayed-by": secret}, profile="w")
    assert exc.value.code == 2 and _channel("chat") == []
    with pytest.raises(SystemExit) as exc:
        bus.post(
            _gdir(), secret, "board", "info", "harmless", topic="chat"
        )  # the sender becomes a `from` label
    assert exc.value.code == 2 and _channel("chat") == []
    p = bs.post_raw(f"{_ns()}/chat", "harmless", profile="w")
    with pytest.raises(SystemExit) as exc:
        bs.set_labels(str(p["id"]), {"note": "fine", "reason": secret})
    assert exc.value.code == 2 and _channel("chat")[0]["labels"] == {}


@pytest.mark.slow(reason="board_server start/stop ~2 s when run alone; the test itself ~0.1-0.3 s")
@pytest.mark.parametrize("where", ["client_id", "channel", "profile", "re"])
def test_a_secret_in_a_key_channel_profile_or_re_is_refused_and_never_echoed(
    board: Path, where: str, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "FAKE_API_KEY=" + "synthetic" * 3
    with pytest.raises(SystemExit) as exc:
        if where == "client_id":  # the public entry: bus post --client-id
            bus.post(_gdir(), "chief", "board", "info", "harmless", topic="chat", client_id=secret)
        else:
            kw: dict[str, str] = {"channel": f"{_ns()}/chat", "profile": "w", "re": ""}
            kw[where] = secret if where != "channel" else f"{_ns()}/{secret}"
            bs.post_raw(
                kw["channel"] if where != "re" else None,
                "harmless",
                profile=kw["profile"],
                re=kw["re"] or None,
            )
    assert exc.value.code == 2 and _channel("chat") == []
    seen = capsys.readouterr()
    assert secret not in seen.out + seen.err + str(exc.value.code)
    assert bs.list_channels(f"{_ns()}/") == []


# ------------------------------------------------------------------ B2: of two supersedes the first wins


def test_a_supersede_that_lost_the_race_leaves_the_first_link(
    board: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_a_supersede_of_an_unlabelled_job_post_is_conditional_on_unset(
    board: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jb = _job(board, "20261120-sup")
    old = bus.post(jb, "w", "board", "claim", "editing pipeline.py", topic="design")
    stale = bs.lookup(str(old["id"]))
    first = bus.supersede(jb, "wa", str(old["id"]), "design", "editing pipeline.py and bus.py")
    with monkeypatch.context() as m:
        m.setattr(bs, "lookup", lambda _old_id: stale)
        bus.supersede(jb, "wb", str(old["id"]), "design", "editing bus.py only")
    target = next(m for m in _channel("job/20261120-sup") if m["id"] == old["id"])
    assert target["labels"]["superseded-by"] == str(first["id"])  # type: ignore[index]


# ------------------------------------------------------------------ public helpers for board_io.py


def test_set_labels_with_a_precondition_sets_only_while_it_holds(board: Path) -> None:
    p = bs.post_raw(f"{_ns()}/chat", "target", profile="w")
    assert (
        bs.set_labels(
            str(p["id"]), {"state": "superseded", "superseded-by": "x1"}, when={"state": None}
        )
        is True
    )
    assert (
        bs.set_labels(
            str(p["id"]), {"state": "superseded", "superseded-by": "x2"}, when={"state": "live"}
        )
        is False
    )
    assert _channel("chat")[0]["labels"] == {"state": "superseded", "superseded-by": "x1"}
    secret = "API_TOKEN=abcdefghijklmnop"
    with pytest.raises(SystemExit) as exc:
        bs.set_labels(str(p["id"]), {"reason": secret}, when={"state": "superseded"})
    assert exc.value.code == 2 and "reason" not in _channel("chat")[0]["labels"]
