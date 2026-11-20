"""t3_history.py: extract runs on a synthetic statev2 fixture, never the real DB."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/t3_history.py"
_spec = importlib.util.spec_from_file_location("t3_history", _PATH)
assert _spec and _spec.loader
t3 = importlib.util.module_from_spec(_spec)
sys.modules["t3_history"] = t3
_spec.loader.exec_module(t3)

SCHEMA = """
CREATE TABLE orchestration_v2_projection_threads (
    thread_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    title TEXT NOT NULL,
    default_provider TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE orchestration_v2_projection_subagents (
    subagent_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    child_thread_id TEXT,
    payload_json TEXT NOT NULL
);
CREATE TABLE orchestration_v2_projection_turn_items (
    turn_item_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""


def _thread(con: sqlite3.Connection, thread_id: str, created: str, model: str = "claude-opus-5-5") -> None:
    con.execute(
        "insert into orchestration_v2_projection_threads values (?,?,?,?,?,?,?)",
        (thread_id, "p1", f"title {thread_id}", "claudeAgent", created, created,
         json.dumps({"modelSelection": {"model": model}})),
    )


def _subagent(
    con: sqlite3.Connection, sub_id: str, parent: str, child: str, provider: str, model: str,
    status: str, started: str, usage: dict[str, object] | None,
) -> None:
    payload: dict[str, object] = {"model": model, "title": f"child {child}"}
    if usage is not None:
        payload["usage"] = usage
    con.execute(
        "insert into orchestration_v2_projection_subagents values (?,?,?,?,?,?,?,?)",
        (sub_id, parent, provider, status, started, started, child, json.dumps(payload)),
    )


def _item(
    con: sqlite3.Connection, item_id: str, thread: str, ordinal: int, kind: str, status: str,
    payload: dict[str, object], updated: str = "2026-10-21T00:00:00.000Z",
) -> None:
    con.execute(
        "insert into orchestration_v2_projection_turn_items values (?,?,?,?,?,?,?)",
        (item_id, thread, ordinal, kind, status, updated, json.dumps(payload)),
    )


def _cmd(output: str | None, command: str) -> dict[str, object]:
    payload: dict[str, object] = {"input": command}
    if output is not None:
        payload["output"] = output
    return payload


def _statev2(path: Path) -> Path:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    # in-window chief, one out-of-window chief, and one child thread row of its own
    _thread(con, "t-parent", "2026-10-20T00:00:00.000Z")
    _thread(con, "t-old", "2026-10-01T00:00:00.000Z")
    _thread(con, "c1", "2026-10-21T00:00:00.000Z", model="claude-sonnet-5-5")
    # s2 started BEFORE the window and is still a child of t-parent
    _subagent(con, "s1", "t-parent", "c1", "claudeAgent", "claude-sonnet-5-5", "completed",
              "2026-10-21T00:00:00.000Z", {"inputTokens": 1000, "outputTokens": 340})
    _subagent(con, "s2", "t-parent", "c2", "codex", "gpt-5.1", "failed",
              "2026-09-01T00:00:00.000Z", {"totalTokens": 999, "approximate": True})
    _subagent(con, "s3", "t-old", "c3", "claudeAgent", "claude-opus-5-5", "completed",
              "2026-10-01T00:00:00.000Z", None)
    sed_err = 'Error: Exit code 1\nsed: 1: "x.py": invalid command code v\n'
    _item(con, "i1", "t-parent", 1, "command_execution", "failed", _cmd(sed_err, "sed -n 1,2p x.py"))
    _item(con, "i2", "t-parent", 2, "user_message", "completed",
          {"text": "no, use the other file", "createdBy": "user"})
    _item(con, "i3", "c1", 1, "command_execution", "failed",
          _cmd("Error: Exit code 2\nzsh: no matches found: bar/*\n", "rg -n foo bar/*"))
    _item(con, "i4", "c1", 2, "command_execution", "failed",
          _cmd("Error: Blocked: sleep 5 followed by: cat x\n", "sleep 5; cat x"))
    _item(con, "i5", "c2", 1, "command_execution", "failed", _cmd(sed_err, "sed -n 1,2p x.py"))
    _item(con, "i6", "c2", 2, "user_message", "completed",
          {"text": "the wrong branch, try main", "createdBy": "agent"})
    _item(con, "i7", "c3", 1, "command_execution", "failed",
          _cmd("Error: Exit code 1\ncat: nope.py: No such file or directory\n", "cat nope.py"))
    # a successful call keeps its row (tool-call volume) but no captured output (scratch size)
    _item(con, "i8", "t-parent", 3, "command_execution", "completed",
          _cmd("Error: Exit code 1\nsed: 1: \"x.py\": invalid command code v\n", "sed -n 1,2p x.py"))
    con.commit()
    con.close()
    return path


def _extracted(tmp_path: Path, since: str = "2026-10-20", until: str = "2026-10-25") -> tuple[Path, Path]:
    src = _statev2(tmp_path / "statev2.sqlite")
    out = tmp_path / "scratch.sqlite"
    t3.extract(since=since, until=until, out=out, source=src)
    return src, out


def _rows(db: Path, sql: str) -> list[tuple[object, ...]]:
    con = sqlite3.connect(db)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_extract_copies_window_threads_all_children_and_their_items(tmp_path: Path) -> None:
    _, db = _extracted(tmp_path)
    assert _rows(db, "select thread_id from threads order by 1") == [("t-parent",)]
    assert _rows(db, "select child_thread_id from children order by 1") == [("c1",), ("c2",)]
    # i7 belongs to c3, whose parent t-old is outside the window
    assert [r[0] for r in _rows(db, "select item_id from commands order by 1")] == [
        "i1", "i3", "i4", "i5", "i8",
    ]
    assert [r[0] for r in _rows(db, "select item_id from messages order by 1")] == ["i2", "i6"]
    assert _rows(db, "select since, until from meta") == [("2026-10-20", "2026-10-25")]


def test_extract_opens_the_source_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _statev2(tmp_path / "statev2.sqlite")
    seen: list[str] = []
    real = sqlite3.connect

    def spy(database: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        seen.append(str(database))
        return real(database, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(t3.sqlite3, "connect", spy)
    t3.extract(since="2026-10-20", until="2026-10-25", out=tmp_path / "s.sqlite", source=src)
    assert f"file:{src}?mode=ro" in seen


def _t3_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / ".t3" / "userdata").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.mark.parametrize("relative", ["userdata/other.sqlite", "scratch.sqlite", "new/deeper/x.sqlite"])
def test_extract_never_writes_under_the_t3_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    home = _t3_home(tmp_path, monkeypatch)
    src = _statev2(tmp_path / "statev2.sqlite")
    dest = home / ".t3" / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("DO NOT OVERWRITE")
    with pytest.raises(ValueError, match=r"\.t3"):
        t3.extract(since="2026-10-20", until="2026-10-25", out=dest, source=src)
    assert dest.read_text() == "DO NOT OVERWRITE"


def test_extract_refuses_a_symlink_into_the_t3_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _t3_home(tmp_path, monkeypatch)
    src = _statev2(tmp_path / "statev2.sqlite")
    victim = home / ".t3" / "userdata" / "other.sqlite"
    victim.write_text("DO NOT OVERWRITE")
    link = tmp_path / "innocent.sqlite"
    link.symlink_to(victim)
    with pytest.raises(ValueError, match=r"\.t3"):
        t3.extract(since="2026-10-20", until="2026-10-25", out=link, source=src)
    assert victim.read_text() == "DO NOT OVERWRITE"


def test_extract_refuses_the_source_as_output(tmp_path: Path) -> None:
    src = _statev2(tmp_path / "statev2.sqlite")
    with pytest.raises(ValueError, match="source"):
        t3.extract(since="2026-10-20", until=None, out=src, source=src)
    assert _rows(src, "select count(*) from orchestration_v2_projection_threads") == [(3,)]
