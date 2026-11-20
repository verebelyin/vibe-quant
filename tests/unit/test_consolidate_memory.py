"""scripts/agents/consolidate_memory.py — per-job memory consolidation, fail-closed."""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

    from tests.unit.conftest import BoardServer

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "consolidate_memory.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("consolidate_memory", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["consolidate_memory"] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load()
JOB = "20261019-mem"


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(global board dir, job dir) in tmp, with one job post mirrored to the board and an unrelated global post."""
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_STORE", "jsonl")  # these tests read the JSONL bytes by hand
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setattr(cm.bus, "comparability_breaks", lambda: [])
    jdir = tmp_path / "jobs" / JOB
    (jdir / "bus").mkdir(parents=True)
    (jdir / "handoffs").mkdir()
    (jdir / "handoffs" / "w1.md").write_text("improve: guard nt_runner.py fill path | evidence: run 900\n")
    cm.bus.post(jdir / "bus", "w1", "board", "info", "RULE: nt_runner.py fills on the next 1m bar", topic="gotchas")
    cm.bus.post(gdir, "old", "board", "info", "unrelated sizing note", topic="findings")
    return gdir, jdir


def _ids(gdir: Path) -> dict[str, str]:
    return {str(m["body"]): str(m["id"]) for m in cm.bus._all(gdir)}


def test_validate_rejects_ids_the_model_was_not_shown_and_bad_fields() -> None:
    shown = {"a", "b"}
    assert cm.validate([{"op": "retire", "id": "a", "reason": "dup"}], shown) == []
    assert cm.validate([{"op": "add", "topic": "gotchas", "importance": 3, "body": "RULE: x"}], shown) == []
    assert cm.validate([{"op": "retire", "id": "zzz", "reason": "dup"}], shown) == ["op 0: retire id 'zzz' was not shown to the model"]
    errors = cm.validate(
        [
            {"op": "supersede", "id": "a", "topic": "chat", "importance": 4, "body": " "},
            {"op": "retire", "id": "a", "reason": "again"},
            {"op": "delete", "id": "b"},
            {"op": "retire", "id": "b", "reason": ""},
            {"op": "add", "topic": "gotchas", "importance": True, "body": "x" * 1201},
            "not a dict",
        ],
        shown,
    )
    assert errors == [
        "op 0: topic must be one of ('gotchas', 'findings', 'decisions'), got 'chat'",
        "op 0: importance must be 1-3, got 4",
        "op 0: empty body",
        "op 1: id a changed twice",
        "op 2: op must be add|supersede|retire, got 'delete'",
        "op 3: retire needs a reason",
        "op 4: importance must be 1-3, got True",
        "op 4: body has 1201 chars > 1200",
        "op 5: not an object",
    ]
    assert cm.validate({"ops": []}, shown) == ["ops is not a list"]


def test_propose_shows_the_job_posts_and_writes_nothing_on_a_bad_op(
    board: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    gdir, jdir = board
    before = (gdir / "messages.jsonl").read_bytes()
    prompts: list[str] = []

    def bad_model(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"ops": [{"op": "retire", "id": "not-shown", "reason": "stale"}]})

    assert cm.propose(JOB, 40, None, ask=bad_model) == 1
    assert "was not shown to the model" in capsys.readouterr().err
    assert not (jdir / "memory-ops.json").exists() and (gdir / "messages.jsonl").read_bytes() == before
    ids = _ids(gdir)
    assert f"[{ids['RULE: nt_runner.py fills on the next 1m bar']}] #gotchas i2" in prompts[0]  # the job's own post
    assert "improve: guard nt_runner.py fill path" in prompts[0] and "--- handoff w1.md" in prompts[0]


@pytest.mark.parametrize("reply", ["not json", '{"no_ops": []}', "[1, 2]"])
def test_propose_rejects_a_malformed_reply(board: tuple[Path, Path], reply: str) -> None:
    gdir, jdir = board
    before = (gdir / "messages.jsonl").read_bytes()
    assert cm.propose(JOB, 40, None, ask=lambda _p: reply) == 1
    assert not (jdir / "memory-ops.json").exists() and (gdir / "messages.jsonl").read_bytes() == before


def test_propose_then_apply_changes_the_board_once(board: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    gdir, jdir = board
    ids = _ids(gdir)
    job_rule, unrelated = ids["RULE: nt_runner.py fills on the next 1m bar"], ids["unrelated sizing note"]
    reply = {"ops": [
        {"op": "supersede", "id": job_rule, "topic": "gotchas", "importance": 3, "body": "RULE: fills on the NEXT 1m bar open"},
        {"op": "retire", "id": unrelated, "reason": "duplicate of sizing doc"},
        {"op": "add", "topic": "decisions", "importance": 2, "body": "RULE: fade job buses after 7 days"},
    ]}
    fenced = "```json\n" + json.dumps(reply) + "\n```"
    before = (gdir / "messages.jsonl").read_bytes()
    assert cm.propose(JOB, 40, None, ask=lambda _p: fenced) == 0
    assert (gdir / "messages.jsonl").read_bytes() == before  # propose is a dry run
    doc = json.loads((jdir / "memory-ops.json").read_text())
    assert doc["ops"] == reply["ops"] and {job_rule, unrelated} <= set(doc["shown"])
    capsys.readouterr()

    assert cm.apply(jdir / "memory-ops.json", "chief") == 0
    visible = [(m["body"], m.get("importance"), m.get("job")) for m in cm.bus.board_search(gdir, topics=list(cm.TOPICS))]
    assert visible == [
        ("RULE: fills on the NEXT 1m bar open", 3, JOB),
        ("RULE: fade job buses after 7 days", 2, JOB),
    ]
    lines = (gdir / "messages.jsonl").read_bytes()
    assert cm.apply(jdir / "memory-ops.json", "chief") == 0  # a re-run (e.g. after a crash) adds nothing
    assert (gdir / "messages.jsonl").read_bytes() == lines


def test_a_second_apply_for_the_same_job_posts_its_new_adds(board: tuple[Path, Path]) -> None:
    """A job landing in phases runs propose/apply once per phase; a phase-2 add at op index 0
    once collided with phase 1's idempotency key and was silently dropped."""
    gdir, jdir = board
    ops_file = jdir / "memory-ops.json"
    for body in ("RULE: phase one rule", "RULE: phase two rule"):
        ops_file.write_text(json.dumps({"job": JOB, "shown": [], "ops": [
            {"op": "add", "topic": "gotchas", "importance": 1, "body": body}]}))  # fmt: skip
        assert cm.apply(ops_file, "chief") == 0
    bodies = [m["body"] for m in cm.bus.board_search(gdir, topics=list(cm.TOPICS))]
    assert "RULE: phase one rule" in bodies and "RULE: phase two rule" in bodies


def test_apply_revalidates_a_hand_edited_ops_file(board: tuple[Path, Path]) -> None:
    gdir, jdir = board
    ops_file = jdir / "memory-ops.json"
    ops_file.write_text(json.dumps({"job": JOB, "shown": [], "ops": [{"op": "retire", "id": "x", "reason": "r"}]}))
    before = (gdir / "messages.jsonl").read_bytes()
    assert cm.apply(ops_file, "chief") == 1
    ops_file.write_text(json.dumps({"ops": []}))
    assert cm.apply(ops_file, "chief") == 1
    assert (gdir / "messages.jsonl").read_bytes() == before


def test_memory_route_resolves_to_haiku_at_high_effort() -> None:
    assert cm.route() == ("haiku", "high")


def test_board_store_propose_reads_the_job_posts_and_apply_retires_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, board_server: BoardServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """SWARM_STORE=board: a job has no messages.jsonl; its messages decide whether it has a bus."""
    for key in ("BOARD_URL", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, board_server.env[key])
    monkeypatch.setenv("SWARM_STORE", "board")
    monkeypatch.setenv("SWARM_BOARD_NS", "m" + uuid.uuid4().hex[:12])
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path / "global"))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setattr(cm.bus, "comparability_breaks", lambda: [])
    jdir = tmp_path / "jobs" / JOB
    (jdir / "bus").mkdir(parents=True)
    (tmp_path / "jobs" / "20261019-empty" / "bus").mkdir(parents=True)
    gdir = cm.bus.global_board_dir()
    posts = [
        cm.bus.post(jdir / "bus", "w1", "board", "info", "RULE: nt_runner.py fills on the next 1m bar", topic="gotchas"),
        cm.bus.post(jdir / "bus", "w2", "board", "info", "RULE: funding cut at end_ns-1ms", topic="findings"),
        cm.bus.post(jdir / "bus", "chief", "board", "info", "RULE: fees count as a cost", topic="decisions"),
    ]
    assert not (jdir / "bus" / "messages.jsonl").exists()
    assert cm.propose("20261019-empty", 40, None, ask=lambda _p: "unused") == 2
    assert "no job bus at" in capsys.readouterr().err
    shown = cm.shown_posts(JOB, cm.job_material(jdir), 40, 1_800_000_000.0)
    assert sorted(str(m["id"]) for m in shown) == sorted(str(p["id"]) for p in posts)
    retire = {"ops": [{"op": "retire", "id": str(posts[1]["id"]), "reason": "duplicate of the funding doc"}]}
    assert cm.propose(JOB, 40, None, ask=lambda _p: json.dumps(retire)) == 0
    assert cm.apply(jdir / "memory-ops.json", "chief") == 0
    visible = [m["body"] for m in cm.bus.board_search(gdir, topics=list(cm.TOPICS))]
    assert visible == ["RULE: nt_runner.py fills on the next 1m bar", "RULE: fees count as a cost"]


@pytest.mark.slow(reason="board_server start/stop ~2 s when run alone; the test itself ~0.1-0.3 s")
def test_a_worker_kb_post_as_the_board_block_prescribes_is_a_job_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, board_server: BoardServer
) -> None:
    """The board block's raw `vb post kb/...` (run as written, {JOB_NAME} filled the way mkbrief
    fills it) shows in the job's bus view and among consolidation's own job posts, as the old
    `bus board post` from a job bus did."""
    import re
    import shlex

    for key in ("BOARD_URL", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, board_server.env[key])
    ns = "k" + uuid.uuid4().hex[:12]
    monkeypatch.setenv("SWARM_STORE", "board")
    monkeypatch.setenv("SWARM_BOARD_NS", ns)
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path / "global"))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path / "jobs"))
    monkeypatch.setattr(cm.bus, "comparability_breaks", lambda: [])
    jdir = tmp_path / "jobs" / JOB
    (jdir / "bus").mkdir(parents=True)
    block = (SCRIPT.parents[2] / "docs" / "orchestration" / "prompts" / "board-protocol.md").read_text()
    command = next(c for c in re.findall(r"`([^`]+)`", block) if c.startswith("vb post kb/"))
    argv = shlex.split(command.replace("{JOB_NAME}", JOB).replace("'RULE: ... — WHEN: ... — EVIDENCE: ...'", "'RULE: x — WHEN: y — EVIDENCE: z'"))
    argv[2] = f"{ns}/{argv[2]}"  # the test's channel namespace
    proc = board_server.vb(*argv[1:], profile="cc-mem-1")
    assert proc.returncode == 0, proc.stderr
    body = "RULE: x — WHEN: y — EVIDENCE: z"
    assert [m["job"] for m in cm.bus._all(jdir / "bus") if m["body"] == body] == [JOB]
    shown = cm.shown_posts(JOB, "", 0, 1_800_000_000.0)
    assert [(m["body"], m["topic"]) for m in shown] == [(body, "gotchas")]
