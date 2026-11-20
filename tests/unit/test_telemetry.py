"""scripts/agents/telemetry.py — cross-job swarm task ledger (append-only JSONL)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_TELEMETRY_PATH = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "telemetry.py"

MODEL_A = "xiaomi/mimo-v2.6-pro"
MODEL_B = "deepseek/deepseek-v4.1-flash"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("swarm_telemetry", _TELEMETRY_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swarm_telemetry"] = mod
    spec.loader.exec_module(mod)
    return mod


tel = _load()


@pytest.fixture(autouse=True)
def _isolated_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point SWARM_TELEMETRY into tmp_path — a test must never write the real ledger."""
    ledger = tmp_path / "tasks.jsonl"
    monkeypatch.setenv("SWARM_TELEMETRY", str(ledger))
    return ledger


def _rec(**kw: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "ts": "2026-10-09T10:00:00",
        "job": "j1",
        "task": "t1",
        "agent": "w1",
        "runtime": "cmd",
        "model": MODEL_A,
        "round": 1,
        "outcome": None,
        "review": None,
        "tokens_in": None,
        "tokens_out": None,
        "wall_s": None,
        "bead": None,
        "note": None,
    }
    rec.update(kw)
    return rec


def _seed(ledger: Path, records: list[dict[str, Any]]) -> None:
    ledger.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )


def _ledger_rows(ledger: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]


# hand-built report fixture: 2 models, 2 rounds, review updates, one superseded record
LEDGER = [
    _rec(
        ts="2026-10-09T10:00:00Z",
        task="t1",
        agent="w1",
        model=MODEL_A,
        round=1,
        outcome="done",
        review="PASS",
        tokens_in=100,
        tokens_out=10,
        wall_s=10.0,
    ),
    _rec(
        ts="2026-10-09T11:00:00",
        task="t2",
        agent="w1",
        model=MODEL_A,
        round=1,
        outcome="error",
        review="SHOULD-FIX",
        tokens_in=200,
        tokens_out=20,
        wall_s=20.0,
    ),
    _rec(
        ts="2026-10-09T12:00:00",
        task="t2",
        agent="w2",
        runtime="claude",
        model=MODEL_B,
        round=2,
        outcome="done",
        review="PASS",
        tokens_in=300,
        tokens_out=30,
        wall_s=30.0,
    ),
    _rec(
        ts="2026-10-08T09:00:00",
        task="t3",
        agent="w2",
        model=MODEL_B,
        round=1,
        outcome="done",
        review="BLOCK",
        tokens_in=400,
        tokens_out=40,
        wall_s=40.0,
    ),
    # review update: copy of the previous record with review set (supersedes it)
    _rec(
        ts="2026-10-09T13:00:00",
        task="t3",
        agent="w2",
        model=MODEL_B,
        round=1,
        outcome="done",
        review="PASS",
        tokens_in=400,
        tokens_out=40,
        wall_s=40.0,
    ),
    _rec(
        ts="2026-10-08T10:00:00",
        task="t4",
        agent="w1",
        model=MODEL_A,
        round=1,
        outcome="stalled",
        tokens_in=500,
        tokens_out=50,
        wall_s=50.0,
    ),
]


def test_report_output_tables(_isolated_ledger: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _seed(_isolated_ledger, LEDGER)
    tel.main(["report", "--by", "model"])
    plain = capsys.readouterr().out
    assert "group" in plain.splitlines()[0]
    assert "1/2 (0.50)" in plain
    assert MODEL_A in plain

    tel.main(["report", "--by", "model", "--markdown"])
    md = capsys.readouterr().out
    assert md.splitlines()[0].startswith("| group |")
    assert "| --- |" in md
    assert f"| {MODEL_A} | 3 | 1/2 (0.50) | 1.50 | 200 | 20 | 2 |" in md
    # the attempts column is printed for every --by, not just model
    assert md.splitlines()[0].endswith("| attempts |")
    row_a = next(line for line in md.splitlines() if line.startswith(f"| {MODEL_A} "))
    assert row_a.endswith("| 3 |")  # MODEL_A has 3 attempts in LEDGER


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True,
    )  # fmt: skip


def _copy_script(dst_root: Path) -> ModuleType:
    d = dst_root / "scripts" / "agents"
    d.mkdir(parents=True, exist_ok=True)
    (d / "telemetry.py").write_text(_TELEMETRY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("swarm_telemetry_copy", d / "telemetry.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ledger_path_resolves_the_main_checkout_and_refuses_non_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SWARM_TELEMETRY", raising=False)
    main = (tmp_path / "main").resolve()
    main.mkdir()
    _git(main, "init", "-q")
    _copy_script(main)
    _git(main, "add", "-A")
    _git(main, "commit", "-qm", "init")
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", str(wt))
    mod = _copy_script(wt)  # the worktree's own checkout of the script
    expected = main / "docs" / "orchestration" / "telemetry" / "tasks.jsonl"
    assert mod.ledger_path().resolve() == expected.resolve()
    # outside any git repo there is nowhere to put a shared ledger: a clean error naming the env var
    nogit = _copy_script(tmp_path / "nogit")
    with pytest.raises(SystemExit) as exc:
        nogit.ledger_path()
    assert "SWARM_TELEMETRY" in str(exc.value)


# --- ingest-runs: T3 delegated tasks, Command Code sessions, Claude subagent transcripts -------

JOB = "20261014-demo"
HANDOFF_COMMS = (
    "status: done\ncomms: asked=1 dms=2 replies=0; board posts relied on: "
    "18de7ab5245b2778-1430, 18de7ad6d75e3ec0-0531 (fee gotcha)\nimprove: none"
)


def _t3_db(path: Path, rows: list[dict[str, Any]]) -> Path:
    import sqlite3

    con = sqlite3.connect(path)
    con.execute(
        "create table orchestration_v2_projection_subagents (subagent_id text primary key, "
        "provider text, origin text, status text, started_at text, completed_at text, "
        "payload_json text, child_thread_id text)"
    )
    con.execute(
        "create table orchestration_v2_projection_turn_items (thread_id text, updated_at text)"
    )
    for i, r in enumerate(rows):
        cid = r["cid"]
        child = r.get("child") or f"th-{cid}"
        payload = {
            "id": f"node:delegated-task:command%3Amcp%3Aabc%3Adelegate-task%3A{cid}",
            "prompt": r["prompt"],
            "model": r["model"],
            "result": r.get("result"),
            "usage": r.get("usage"),
        }
        con.execute(
            "insert into orchestration_v2_projection_subagents values (?,?,?,?,?,?,?,?)",
            (f"s{i}", r["provider"], "app_owned", r["status"], r["start"], r.get("end"), json.dumps(payload), child),
        )
        for stamp in r.get("turn_items", []):
            con.execute(
                "insert into orchestration_v2_projection_turn_items values (?,?)", (child, stamp)
            )
    con.commit()
    con.close()
    return path


def _cmd_session(d: Path, start: str, prompt: str, usages: list[tuple[int, int]]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        {"type": "session", "id": "x", "timestamp": start, "cwd": "/repo"},
        {"type": "message", "id": "u1", "timestamp": start,
         "message": {"role": "user", "content": [{"type": "text", "text": f"<t3_code_instructions>mode</t3_code_instructions>\n{prompt}"}]}},
    ]
    for k, (i, o) in enumerate(usages):
        lines.append({"type": "message", "id": f"a{k}", "timestamp": start,
                      "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                      "usage": {"inputTokens": i, "outputTokens": o, "cacheReadTokens": i - 10}})
    (d / f"{start[-6:-1]}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")


def _claude_subagent(d: Path, agent_id: str, name: str, prompt: str, final: str, error: str | None = None) -> None:
    sub = d / "sess1" / "subagents"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"agent-{agent_id}.meta.json").write_text(json.dumps({"agentType": "implementer", "name": name}), encoding="utf-8")
    msgs: list[dict[str, Any]] = [
        {"type": "user", "timestamp": "2026-10-14T10:00:00.000Z", "message": {"role": "user", "content": prompt}},
        # the same API message is logged once per content block: usage must be counted once
        {"type": "assistant", "timestamp": "2026-10-14T10:01:00.000Z", "message": {"id": "m1", "model": "claude-sonnet-5-5",
         "content": [{"type": "thinking", "thinking": "x"}], "usage": {"input_tokens": 3, "cache_creation_input_tokens": 100,
         "cache_read_input_tokens": 1000, "output_tokens": 50}}},
        {"type": "assistant", "timestamp": "2026-10-14T10:01:00.000Z", "message": {"id": "m1", "model": "claude-sonnet-5-5",
         "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}], "usage": {"input_tokens": 3,
         "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000, "output_tokens": 50}}},
        {"type": "assistant", "timestamp": "2026-10-14T10:05:30.000Z", "message": {"id": "m2", "model": "claude-sonnet-5-5",
         "content": [{"type": "text", "text": final}], "usage": {"input_tokens": 2, "cache_creation_input_tokens": 0,
         "cache_read_input_tokens": 2000, "output_tokens": 70}}},
    ]
    if error:
        msgs.append({"type": "assistant", "timestamp": "2026-10-14T10:06:00.000Z", "isApiErrorMessage": True,
                     "message": {"id": "e", "model": "<synthetic>", "content": [{"type": "text", "text": error}]}})
    (sub / f"agent-{agent_id}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in msgs), encoding="utf-8")
    old = time.time() - 3600
    os.utime(sub / f"agent-{agent_id}.jsonl", (old, old))


def _sources(tmp_path: Path) -> dict[str, Path]:
    db = _t3_db(tmp_path / "t3.sqlite", [
        {"cid": "cc-liq-e3-2", "provider": "commandcode_command", "status": "completed",
         "start": "2026-10-14T09:00:00.000Z", "end": "2026-10-14T09:07:30.000Z", "model": "deepseek/deepseek-v4.1-flash",
         "prompt": f"ROLE: Command Code worker (agent id cc-liq-e3), job {JOB}. Fix the docs.", "result": HANDOFF_COMMS,
         "usage": {"totalTokens": 73591, "approximate": True}},
        {"cid": "rev-liq-e3-r1-20261014", "provider": "codex", "status": "failed",
         "start": "2026-10-14T09:10:00.000Z", "end": "2026-10-14T09:10:04.000Z", "model": "gpt-6.1-sol",
         "prompt": f"You are the reviewer (agent id rev-liq-e3-r1), job {JOB}.",
         "result": "You've hit your usage limit. Try again at 1:07 AM.", "usage": {"inputTokens": 900, "outputTokens": 9}},
        {"cid": "cc-other-1", "provider": "commandcode_command", "status": "completed",
         "start": "2026-10-14T09:00:00.000Z", "end": "2026-10-14T09:01:00.000Z", "model": "deepseek/deepseek-v4.1-flash",
         "prompt": "job 20261001-unrelated", "result": "done"},
        {"cid": "cc-running-1", "provider": "commandcode_command", "status": "running",
         "start": "2026-10-14T09:00:00.000Z", "model": "deepseek/deepseek-v4.1-flash", "prompt": f"job {JOB}"},
    ])
    cmd = tmp_path / "cmd"
    _cmd_session(cmd / "proj", "2026-10-14T09:00:01.500Z",
                 f"ROLE: Command Code worker (agent id cc-liq-e3), job {JOB}. Fix the docs.", [(1000, 20), (2000, 30)])
    claude = tmp_path / "claude"
    _claude_subagent(claude, "a1", "impl-liq-t2a-opus5",
                     f"TELEMETRY: task=liq-t2a round=5\nROLE: implementer, job {JOB}", HANDOFF_COMMS)
    _claude_subagent(claude, "a2", "rev-liq-t3-r2", f"ROLE: reviewer, job {JOB}", "verdict BLOCK\ncomms: asked=0 dms=1 replies=0; board posts relied on: none",
                     error="API Error: You've hit your session limit")
    return {"t3_db": db, "cmd_dir": cmd, "claude_dir": claude}


def test_ingest_runs_reads_t3_command_code_and_claude(_isolated_ledger: Path, tmp_path: Path) -> None:
    src = _sources(tmp_path)
    added = tel.ingest_runs(JOB, **src)
    rows = {r["source"]: r for r in _ledger_rows(_isolated_ledger)}
    assert len(added) == 4 and set(rows) == {"t3:cc-liq-e3-2", "t3:rev-liq-e3-r1-20261014", "claude:a1", "claude:a2"}
    cc = rows["t3:cc-liq-e3-2"]
    assert (cc["job"], cc["task"], cc["agent"], cc["round"], cc["runtime"]) == (JOB, "liq-e3", "cc-liq-e3", 2, "cmd")
    # T3's Command Code usage is a last-context approximation: tokens come from the session file
    assert (cc["tokens_in"], cc["tokens_out"], cc["wall_s"], cc["outcome"]) == (3000, 50, 450.0, "done")
    assert (cc["comms_asked"], cc["comms_dms"], cc["comms_replies"], cc["board_relied"]) == (1, 2, 0, 2)
    cdx = rows["t3:rev-liq-e3-r1-20261014"]
    assert (cdx["runtime"], cdx["task"], cdx["outcome"], cdx["tokens_in"], cdx["tokens_out"]) == ("codex", "review:liq-e3", "error", 900, 9)
    assert "usage limit" in cdx["note"]
    a1 = rows["claude:a1"]  # TELEMETRY line beats the name heuristic
    assert (a1["task"], a1["round"], a1["agent"], a1["model"]) == ("liq-t2a", 5, "impl-liq-t2a-opus5", "claude-sonnet-5-5")
    assert (a1["tokens_in"], a1["tokens_out"], a1["wall_s"]) == (1103 + 2002, 120, 330.0)  # m1 counted once
    assert a1["board_relied"] == 2
    a2 = rows["claude:a2"]
    assert (a2["outcome"], a2["task"], a2["round"], a2["board_relied"]) == ("error", "review:liq-t3", 2, 0)
    assert "session limit" in a2["note"]


def test_ingest_runs_is_idempotent_and_skips_manual_records(_isolated_ledger: Path, tmp_path: Path) -> None:
    src = _sources(tmp_path)
    tel.main(["record", "--job", JOB, "--task", "liq-e3", "--agent", "cc-liq-e3", "--runtime", "cmd", "--round", "2"])
    first = tel.ingest_runs(JOB, **src)
    assert {r["source"] for r in first} == {"t3:rev-liq-e3-r1-20261014", "claude:a1", "claude:a2"}
    assert tel.ingest_runs(JOB, **src) == []  # nothing new on a re-run
    assert len(_ledger_rows(_isolated_ledger)) == 4
