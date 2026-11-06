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


@pytest.mark.parametrize(
    ("since", "until", "threads"),
    [
        ("2026-10-20T00:00:00.000Z", "2026-10-25", ("t-parent",)),  # created == since is in
        ("2026-10-19", "2026-10-20T00:00:00.000Z", ()),  # created == until is out
        ("2026-10-01", "2026-10-02", ("t-old",)),
    ],
)
def test_extract_window_edges(tmp_path: Path, since: str, until: str, threads: tuple[str, ...]) -> None:
    src = _statev2(tmp_path / "statev2.sqlite")
    out = tmp_path / "scratch.sqlite"
    t3.extract(since=since, until=until, out=out, source=src)
    assert _rows(out, "select thread_id from threads order by 1") == [(t,) for t in threads]


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


def test_extract_records_first_token_exit_code_and_raw_text(tmp_path: Path) -> None:
    _, db = _extracted(tmp_path)
    assert _rows(db, "select first_token, exit_code from commands where item_id='i1'") == [("sed", 1)]
    assert _rows(db, "select first_token, exit_code from commands where item_id='i3'") == [("rg", 2)]
    assert _rows(db, "select exit_code from commands where item_id='i4'") == [(None,)]
    assert _rows(db, "select output from commands where item_id='i1'")[0][0].startswith("Error: Exit code 1")


def test_extract_keeps_completed_calls_but_not_their_output(tmp_path: Path) -> None:
    _, db = _extracted(tmp_path)
    assert _rows(db, "select status, output from commands where item_id='i8'") == [("completed", None)]
    report = t3.signatures(db)
    assert report.failed == 4  # i8 is a completed call: it feeds neither count nor family
    found = {f.signature: f.occurrences for f in report.families}
    assert found["sed:invalid command code"] == 2


def test_extract_an_empty_window_still_writes_a_readable_scratch(tmp_path: Path) -> None:
    src = _statev2(tmp_path / "statev2.sqlite")
    out = tmp_path / "scratch.sqlite"
    counts = t3.extract(since="2030-01-01", until=None, out=out, source=src)
    assert counts == {"threads": 0, "children": 0, "commands": 0, "messages": 0}
    report = t3.signatures(out)
    assert report.history.until is None and report.families == []
    assert "window 2030-01-01..now" in t3.render(report)


@pytest.mark.parametrize(
    ("output", "command", "expected"),
    [
        ('Error: Exit code 1\nsed: 1: "x.py": invalid command code v\n', "sed -n 1p x.py",
         "sed:invalid command code"),
        ("Error: Exit code 1\nsed: 2: \"t.py\": undefined label 'ests/t.py'\n", "sed -i '' 's/a/b/' t.py",
         "sed:undefined label"),
        ("Error: Exit code 2\nzsh: no matches found: bar/*\n", "rg -n foo bar/*", "zsh:no matches found"),
        ("Error: Blocked: sleep 5 followed by: cat x\n", "sleep 5; cat x", "hook:blocked:foreground sleep"),
        ("Error: Exit code 1\nrg: a.py: No such file or directory (os error 2)\n", "rg -n x a.py",
         "rg:No such file or directory"),
        ("Error: Exit code 1\nFAILED tests/unit/test_a.py::test_b - assert 1\n", "pytest -q",
         "pytest:failed"),
        ("Error: Exit code 1\nTraceback (most recent call last):\n  File \"<stdin>\"\n", "python3 -",
         "python:traceback"),
        ("Error: Exit code 127\n", "cargo build", "exit:127"),
        (None, "true", "no-output"),
        # a guard name without a [script] bracket (real rm_guard / shell_guard shape)
        ("Error: PreToolUse:Bash hook error: rm_guard: rm inside a substitution/eval/xargs: "
         "target unknown\n", "rm -f $X", "hook:rm_guard:unknown rm target"),
        # sed truncates the script at the quote: still a sed error, grouped as such
        ('Error: Exit code 1\nsed: 1: "/id="overlay"/,/+8p\n', "sed -n 1p f", "sed:malformed script"),
        ("Error: Exit code 2\nbus: /x/y.md fails handoff lint:\n", "python3 bus.py --x",
         "bus:handoff lint"),
        ("Error: Claude V2 adapter has no active turn for this tool request\n", "ls",
         "error:no active turn"),
        ("Error: Exit code 1\nSURVIVED floor-always-passes (2.0s)\nMUTANTS killed=0 survived=1 "
         "errors=0\n", "python3 scripts/agents/mutate.py --auto-diff main", "mutate:survived"),
        ("Error: Exit code 1\ntests/unit/test_a.py:467: error: Cannot infer type  [misc]\n",
         "mypy", "mypy:error"),
        # prose that looks like `key: value` in a file the command printed is not a diagnostic
        ("Error: Exit code 1\nstatus: done | blocked | failed\n", "cat handoff.md", "exit:1"),
        ("Error: Exit code 1\nbus: /Users/x/handoffs/a.md\n", "cat handoff.md", "exit:1"),
    ],
)
def test_signature_families(output: str | None, command: str, expected: str) -> None:
    assert t3.signature(command, output) == expected


def test_families_of_keeps_every_diagnostic_a_command_shows() -> None:
    output = ('Error: Exit code 1\nsed: 2: "t.py": undefined label \'x\'\n'
              "FAILED tests/unit/test_a.py::test_b - assert 1\n")
    assert t3.families_of("sed -i '' 's/a/b/' t.py; pytest -q", output) == [
        "pytest:failed",
        "sed:undefined label",
    ]
    assert t3.families_of("true", None) == ["no-output"]
    assert t3.families_of("true", "Error: Exit code 3\n") == ["exit:3"]


def test_signatures_counts_occurrences_and_distinct_threads(tmp_path: Path) -> None:
    _, db = _extracted(tmp_path)
    report = t3.signatures(db)
    found = {f.signature: (f.occurrences, f.threads) for f in report.families}
    assert found["sed:invalid command code"] == (2, 2)
    assert found["zsh:no matches found"] == (1, 1)
    assert found["hook:blocked:foreground sleep"] == (1, 1)
    assert report.families[0].signature == "sed:invalid command code"  # most frequent first


def test_signatures_lists_human_corrections_only(tmp_path: Path) -> None:
    _, db = _extracted(tmp_path)
    report = t3.signatures(db)
    assert report.corrections == [("t-parent", "no, use the other file")]
    text = t3.render(report)
    assert "no, use the other file" in text
    assert "the wrong branch" not in text  # agent-authored briefs are not corrections


def test_signatures_groups_models_and_sums_stored_tokens(tmp_path: Path) -> None:
    _, db = _extracted(tmp_path)
    text = t3.render(t3.signatures(db))
    assert "  Opus 1 run(s) tokens unknown" in text
    assert "  Sonnet 1 run(s) tokens=1,340" in text
    assert "  GPT 1 run(s) tokens=999 (approximate)" in text


@pytest.mark.parametrize(
    ("since", "until", "threads"),
    [
        # +02:00 offsets name the same instants as the fixture's 2026-10-20T00:00:00.000Z
        ("2026-10-20T02:00:00+02:00", "2026-10-25", ("t-parent",)),  # created == since is in
        ("2026-10-19", "2026-10-20T02:00:00+02:00", ()),  # created == until is out
        ("2026-10-20T02:00:00.001+02:00", "2026-10-25", ()),  # 1 ms after created: out
        ("2026-10-19", "2026-10-20T02:00:00.001+02:00", ("t-parent",)),  # 1 ms after: in
        ("2026-10-19T22:00:00-02:00", "2026-10-25", ("t-parent",)),  # west of UTC, same instant
        ("2026-10-20T00:00:00Z", "2026-10-20T00:00:00.0005Z", ("t-parent",)),  # sub-ms until
        ("2026-10-20T00:00:00.0005Z", "2026-10-25", ()),  # sub-ms since: row at .000 is before
    ],
)
def test_extract_window_compares_instants_not_clock_strings(
    tmp_path: Path, since: str, until: str, threads: tuple[str, ...]
) -> None:
    src = _statev2(tmp_path / "statev2.sqlite")
    out = tmp_path / "scratch.sqlite"
    counts = t3.extract(since=since, until=until, out=out, source=src)
    assert _rows(out, "select thread_id from threads order by 1") == [(t,) for t in threads]
    assert counts["threads"] == len(threads)
    assert _rows(out, "select since, until from meta") == [(since, until)]  # the caller's words


def test_extract_rejects_an_unparseable_window_before_touching_out(tmp_path: Path) -> None:
    src = _statev2(tmp_path / "statev2.sqlite")
    out = tmp_path / "scratch.sqlite"
    out.write_text("keep")
    with pytest.raises(ValueError, match="not-a-date"):
        t3.extract(since="not-a-date", until=None, out=out, source=src)
    assert out.read_text() == "keep"


def test_extract_follows_nested_children_to_every_depth(tmp_path: Path) -> None:
    src = _statev2(tmp_path / "statev2.sqlite")
    con = sqlite3.connect(src)
    _thread(con, "grandchild", "2026-10-22T00:00:00.000Z")
    _thread(con, "great", "2026-10-23T00:00:00.000Z")
    _subagent(con, "s4", "c1", "grandchild", "codex", "gpt-5.1", "completed",
              "2026-10-22T00:00:00.000Z", {"totalTokens": 50})
    _subagent(con, "s5", "grandchild", "great", "claudeAgent", "claude-haiku-5-5", "completed",
              "2026-10-23T00:00:00.000Z", None)
    _subagent(con, "s6", "great", "grandchild", "claudeAgent", "claude-opus-5-5", "completed",
              "2026-10-24T00:00:00.000Z", None)  # a cycle among descendants must terminate
    _item(con, "g-cmd", "grandchild", 1, "command_execution", "failed",
          _cmd("Error: Exit code 1\nsed: 1: \"x\": invalid command code x\n", "sed -i s/a/b/ f"))
    _item(con, "gg-cmd", "great", 1, "command_execution", "failed",
          _cmd("Error: Exit code 1\ncat: nope: No such file or directory\n", "cat nope"))
    con.commit()
    con.close()
    out = tmp_path / "nested.sqlite"
    counts = t3.extract(since="2026-10-20", until="2026-10-25", out=out, source=src)
    assert _rows(out, "select subagent_id from children order by 1") == [
        ("s1",), ("s2",), ("s4",), ("s5",), ("s6",),
    ]
    assert counts["children"] == 5
    assert _rows(out, "select item_id from commands where item_id like '%-cmd' order by 1") == [
        ("g-cmd",), ("gg-cmd",),
    ]
    report = t3.signatures(out)
    assert {c.child_thread_id for c in report.history.children} >= {"grandchild", "great"}
    assert report.failed == 6  # 4 from the fixture's window + the two nested failures


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


def test_cli_refuses_a_protected_output_with_the_original_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _t3_home(tmp_path, monkeypatch)
    src = _statev2(tmp_path / "statev2.sqlite")
    dest = home / ".t3" / "userdata" / "other.sqlite"
    dest.write_text("DO NOT OVERWRITE")
    with pytest.raises(SystemExit) as exit_info:
        t3.main(["extract", "--since", "2026-10-20", "--db", str(src), "--out", str(dest)])
    assert exit_info.value.code == 2
    assert ".t3" in capsys.readouterr().err
    assert dest.read_text() == "DO NOT OVERWRITE"


@pytest.mark.parametrize(
    "text",
    ["no,use the other file", "No, wrong file", "foo. no,try again", "no,"],
)
def test_correction_marker_needs_no_space_after_no_comma(text: str) -> None:
    assert t3.CORRECTION_MARKERS.search(text)


@pytest.mark.parametrize("text", ["no way", "piano, then", "know, that", "number 5"])
def test_correction_marker_ignores_words_that_only_contain_no(text: str) -> None:
    assert t3.CORRECTION_MARKERS.search(text) is None
