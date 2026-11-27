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
import sys
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
    s["live"] = bus.post(
        g,
        "cc-a",
        "board",
        "info",
        "RULE: widget.py reads gizmo first",
        topic="gotchas",
        extra={"importance": 3, "job": _JOB},
    )
    s["reply1"] = bus.reply(g, "cc-b", str(s["live"]["id"]), "confirmed on sprocket")
    s["reply2"] = bus.reply(g, "cc-a", str(s["reply1"]["id"]), "thanks, ünïcode kept")
    s["stale"] = bus.post(g, "w", "board", "info", "stale finding about gamma", topic="findings")
    retired = bus.retire(g, "chief", str(s["stale"]["id"]), "pre-break metric")
    assert retired is not None
    s["retire"] = retired
    s["old"] = bus.post(g, "w", "board", "info", "use ruff 0.6 flags", topic="decisions")
    s["sup1"] = bus.supersede(g, "chief", str(s["old"]["id"]), "decisions", "use ruff 0.7 flags")
    s["sup2"] = bus.supersede(
        g, "chief", str(s["old"]["id"]), "decisions", "use ruff 0.8 flags"
    )  # lost: sup1 won
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
        code_i, report_i = _main(
            "import-legacy",
            "--persistent",
            str(root / "a.jsonl"),
            "--jobs-root",
            str(root / "empty-jobs"),
            "--map",
            str(root / "b-map.json"),
        )
        code_b, _ = _export(root / "b.jsonl")
    assert (code_a, code_i, code_b) == (0, 0, 0), report_i
    assert report_a is not None and report_i is not None
    return Trip(
        board_server,
        root,
        ns_a,
        ns_b,
        seeded,
        root / "a.jsonl",
        root / "b.jsonl",
        report_a,
        report_i,
    )


# ------------------------------------------------------------------ the round trip


def test_round_trip_is_byte_identical(trip: Trip) -> None:
    a, b = trip.file_a.read_bytes(), trip.file_b.read_bytes()
    assert a == b
    assert a.endswith(b"}\n") and not a.endswith(b"\n\n")
    assert (trip.report_import["rejects"], trip.report_import["secret_refused"]) == (
        {"count": 0, "items": []},
        {"count": 0, "ids": []},
    )


def test_no_job_channel_no_actor_no_os_user_no_url(trip: Trip) -> None:
    text = trip.file_a.read_text(encoding="utf-8")
    assert "JOBONLY" not in text
    for absent in (
        getpass.getuser(),
        "local:",
        "actor",
        "principal",
        trip.server.url,
        str(trip.server.port),
        trip.ns_a,
    ):
        assert absent not in text, absent


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


def test_a_secret_on_the_board_refuses_the_export_and_keeps_the_old_file(
    small: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = small.read_bytes()
    leaked = bs._http(
        "POST",
        "/messages",
        {"channel": f"{bs.ns()}chat", "text": f"oops {_SECRET}", "labels": {}},
        profile="w",
        expect=("message",),
    )["message"]  # past the write-path scan, as a raw API client could
    code, report = _export(small)
    err = capsys.readouterr().err
    assert (code, report) == (1, None)
    assert f"b{leaked['id']}" in err and _SECRET not in err and "oops" not in err
    assert small.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["export.jsonl", "global"]


def test_board_down_exits_naming_board_up_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BOARD_URL", _UNROUTABLE)
    monkeypatch.setenv("BOARD_TOKEN", "t")
    with pytest.raises(SystemExit) as exc:
        _export(tmp_path / "x.jsonl")
    assert exc.value.code not in (0, None) and "board-up.sh" in str(exc.value.code)
    assert list(tmp_path.iterdir()) == []
