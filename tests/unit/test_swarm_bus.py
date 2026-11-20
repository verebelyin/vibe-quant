"""scripts/agents/bus.py — swarm message bus (agents <-> agents <-> chief)."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_BUS_PATH = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "bus.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("swarm_bus", _BUS_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swarm_bus"] = mod
    spec.loader.exec_module(mod)
    return mod


bus = _load()
_REAL_WORK_CONTEXT = bus.work_context
_REAL_BREAKS = bus.comparability_breaks


@pytest.fixture(autouse=True)
def _jsonl_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test here exercises the JSONL store (test_board_store.py covers the board)."""
    monkeypatch.setenv("SWARM_STORE", "jsonl")


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    monkeypatch.setattr(bus, "POLL_S", 0.05)
    # `bd ready --json` costs ~0.5 s and every `status` screen asks it: a `bd` on PATH that
    # answers [] keeps `_ready_count` itself running (chief rule, board 18e53d5a). Nothing here
    # is about bd — test_status_lists_running_agents_and_startable_beads sets the count it wants.
    bd_bin = tmp_path_factory.mktemp("bd-bin")
    (bd_bin / "bd").write_text('#!/bin/sh\nprintf "[]\\n"\n')
    (bd_bin / "bd").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bd_bin}{os.pathsep}{os.environ['PATH']}")
    # board brief ranks by the caller's git branch and in-progress beads: none in tests
    monkeypatch.setattr(bus, "work_context", lambda: ([], set()))
    monkeypatch.setattr(bus, "comparability_breaks", lambda: [])  # tests that demote pass their own
    # never let a test write the real committed board
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path_factory.mktemp("global_board")))
    # ...or discover real job buses (an empty jobs root) / inherit a real identity from the caller's env
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path_factory.mktemp("jobs")))
    monkeypatch.delenv("SWARM_BUS", raising=False)
    monkeypatch.delenv("SWARM_AGENT", raising=False)


def _writer(path: str, name: str, n: int) -> None:
    mod = _load()
    for i in range(n):
        mod.post(Path(path), name, "all", "info", f"{name}-{i}-" + "x" * 500)


def test_concurrent_writers_never_corrupt_lines(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_writer, args=(str(tmp_path), f"w{k}", 50)) for k in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    lines = (tmp_path / "messages.jsonl").read_text().splitlines()
    assert len(lines) == 200
    assert all(json.loads(line)["body"].endswith("x" * 500) for line in lines)


def test_unknown_kind_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        bus.post(tmp_path, "sa", "chief", "gossip", "nope")
    with pytest.raises(SystemExit):  # a board post needs an explicit topic
        bus.post(tmp_path, "sa", "board", "info", "no topic")
    assert not (tmp_path / "messages.jsonl").exists()


# --- message board ---------------------------------------------------------


def test_board_posts_normalise_their_topic_and_topics_counts_them(tmp_path: Path) -> None:
    a = bus.post(tmp_path, "sa", "board", "claim", "touching pipeline.py:_run", topic="#Design")
    bus.post(tmp_path, "sb", "board", "info", "KAMA port exact on 400 bars", topic="findings")
    assert a["topic"] == "design"  # normalised
    assert [m["body"] for m in bus.board_search(tmp_path, topics=["findings"])] == ["KAMA port exact on 400 bars"]
    # topics, on a board of its own so the counts stay exact
    own = tmp_path / "own"
    own.mkdir()
    root = bus.post(own, "sa", "board", "info", "plan: preflight bars", topic="design")
    bus.reply(own, "sb", str(root["id"]), "fine by me")
    bus.post(own, "sc", "board", "blocker", "orval regen changed 180 files", topic="blockers")
    assert {k: v["posts"] for k, v in bus.topics(own).items()} == {"design": 2, "blockers": 1}


def test_reply_threads_on_board(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = bus.post(tmp_path, "sa", "board", "question", "scipy filter exact for min?", topic="design")
    r1 = bus.reply(tmp_path, "se", str(root["id"]), "yes, used in trend_ensemble")
    r2 = bus.reply(tmp_path, "chief", str(r1["id"]), "confirmed, keep the NaN comment")
    assert r1["to"] == "board" and r1["topic"] == "design" and r1["ref"] == root["id"]
    tree = [(d, m["from"]) for d, m in bus.thread(tmp_path, str(r2["id"]))]
    assert tree == [(0, "sa"), (1, "se"), (2, "chief")]
    # a board question posted WITHOUT a topic still takes a reply (topic defaults to "general")
    g = bus.global_board_dir()
    (g / "messages.jsonl").write_text(
        json.dumps(_NO_TS[2] | {"id": "bq", "kind": "question"}) + "\n", encoding="utf-8"
    )
    ans = bus.reply(g, "chief", "bq", "answered")  # must not exit "board posts need --topic"
    assert ans["ref"] == "bq" and ans["to"] == "board" and ans["topic"] == "general"
    assert [(d, m["body"]) for d, m in bus.thread(g, "bq")] == [
        (0, "board post w/o topic"),
        (1, "answered"),
    ]
    assert bus.main(["--bus", str(g), "--as", "chief", "reply", "--ref", "bq", "--body", "again"]) == 0
    assert "answered" in "".join(m["body"] for _, m in bus.thread(g, "bq"))
    capsys.readouterr()
    # a board question WITH a topic keeps it
    bus.post(g, "sa", "board", "question", "q2", topic="design")
    q2 = next(m for m in bus._all(g) if m["body"] == "q2")
    assert bus.reply(g, "chief", str(q2["id"]), "a2")["topic"] == "design"


def test_reply_to_a_dm_question_answers_the_asker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    q = bus.post(tmp_path, "sd", "chief", "question", "which lookback?")
    a = bus.reply(tmp_path, "chief", str(q["id"]), "2*period")
    assert a["to"] == "sd" and a["kind"] == "answer" and a["ref"] == q["id"]
    # The same for a DM that lives on ANOTHER job bus — answered there, the asker reads it there
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    jobs = tmp_path / "jobs"
    monkeypatch.setenv("SWARM_JOBS", str(jobs))
    job_a = _make_job_bus(jobs, "20261015-job-a")
    job_b = _make_job_bus(jobs, "20261015-job-b")
    q2 = bus.post(job_b, "w2", "x", "question", "which lookback?")
    a2 = bus.reply(job_a, "x", str(q2["id"]), "2*period")
    assert a2["to"] == "w2" and a2["kind"] == "answer" and a2["ref"] == q2["id"]
    assert "job" not in a2  # the job tag is for replies landing on the global board only
    assert [m["body"] for m in bus._all(job_b)] == ["which lookback?", "2*period"]  # on the holding bus
    assert bus._all(job_a) == []  # not on the replier's bus
    assert bus._all(gdir) == []  # a DM is not a board post: no mirror


# --- persistent (cross-job) board --------------------------------------------


def test_persistent_topics_mirror_to_global_board_with_job_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    job1 = tmp_path / "20261009-job-a" / "bus"
    job1.mkdir(parents=True)
    bus.post(job1, "sd", "board", "info", "pandas rolling sum is not bit-exact vs numpy cumsum", topic="gotchas")
    bus.post(job1, "sa", "board", "claim", "editing pipeline.py", topic="design")  # job-only topic
    g = _board_posts(gdir)
    assert [(m["topic"], m["job"]) for m in g] == [("gotchas", "20261009-job-a")]
    assert "@20261009-job-a" in bus.fmt(g[0])
    # a brand-new agent in a later job reads the last N cross-job posts
    for k in range(5):
        jb = tmp_path / f"job{k}" / "bus"
        jb.mkdir(parents=True)
        bus.post(jb, f"w{k}", "board", "info", f"finding {k}", topic="findings")
    recent = bus.board_search(gdir, topics=["findings"], limit=3)
    assert [m["body"] for m in recent] == ["finding 2", "finding 3", "finding 4"]


def _seed_search_board(b: Path) -> None:
    bus.post(b, "cc-a", "board", "info", "nt_runner.py:1337 swallows DataUnavailableError; preflight first", topic="gotchas")
    bus.post(b, "rev-b", "board", "info", "vibe_quant/liquidation/scan.py: funding cut at end_ns-1ms", topic="findings")
    bus.post(b, "impl-c", "board", "claim", "editing vibe_quant/liquidation/tiers.py", topic="design")
    bus.post(b, "chief", "board", "info", "Ruling: clearance fee counts as a cost", topic="decisions")
    bus.post(b, "cc-d", "chief", "done", "liquidation scan.py done")  # a DM, never a search hit


def test_board_search_by_path_topic_and_grep(tmp_path: Path) -> None:
    _seed_search_board(tmp_path)
    by_file = bus.board_search(tmp_path, paths=["vibe_quant/screening/nt_runner.py"])  # basename match
    assert [m["from"] for m in by_file] == ["cc-a"]
    by_dir = bus.board_search(tmp_path, paths=["vibe_quant/liquidation/**"])  # glob = directory prefix
    assert [m["from"] for m in by_dir] == ["rev-b", "impl-c"]
    # several --topic are OR, several --grep are AND (case-insensitive), filters combine with AND
    assert [m["from"] for m in bus.board_search(tmp_path, topics=["#Findings", "decisions"])] == ["rev-b", "chief"]
    assert [m["from"] for m in bus.board_search(tmp_path, terms=["CLEARANCE", "cost"])] == ["chief"]
    assert bus.board_search(tmp_path, terms=["clearance", "funding"]) == []
    assert [m["from"] for m in bus.board_search(tmp_path, paths=["vibe_quant/liquidation"], topics=["design"])] == ["impl-c"]
    assert [m["from"] for m in bus.board_search(tmp_path, topics=["findings", "gotchas", "decisions"], limit=2)] == ["rev-b", "chief"]


def test_board_search_cli_requires_a_filter_prints_hits_and_cuts_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    _seed_search_board(gdir)
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        bus.main(["--as", "x", "board", "search", "--global"])
    assert exc.value.code == 2
    assert bus.main(["--as", "x", "board", "search", "--global", "--path", "tiers.py", "nt_runner.py"]) == 0
    out = capsys.readouterr().out
    assert "#gotchas cc-a" in out and "#design impl-c" in out and "rev-b" not in out
    assert bus.main(["--as", "x", "board", "search", "--global", "--grep", "nothing-matches-this"]) == 0
    assert "(no matching board posts)" in capsys.readouterr().out
    # a worker's first search printed 12-27k chars of mostly stale posts (10-17..10-19): newest 15,
    # bodies cut at 400 chars, and a trailer that says how to see more
    big = tmp_path / "big"
    big.mkdir()
    for i in range(20):
        bus.post(big, "w", "board", "info", f"RULE {i}: tests/unit/test_x.py " + "y" * 900, topic="gotchas")
    assert bus.main(["--bus", str(big), "--as", "r", "board", "search", "--topic", "gotchas"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 16 and "RULE 5:" in lines[0] and "RULE 19:" in lines[14]
    assert all(len(ln) < 600 for ln in lines[:15])
    assert lines[-1].startswith("(showing newest 15 of 20, bodies cut at 400 chars")
    assert bus.main(["--bus", str(big), "--as", "r", "board", "search", "--topic", "gotchas", "--full",
                     "--limit", "1"]) == 0  # fmt: skip
    full = capsys.readouterr().out.splitlines()
    assert "y" * 900 in full[0]


# --- lobby default, digest, brief ---------------------------------------------


def test_no_job_bus_means_the_persistent_lobby(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.delenv("SWARM_BUS", raising=False)
    monkeypatch.delenv("SWARM_AGENT", raising=False)
    bus.main(["--as", "architect", "board", "post", "--topic", "thoughts", "--body", "consider caching per symbol"])
    bus.main(["board", "search", "--topic", "thoughts"])  # anonymous reader, no job
    out = capsys.readouterr().out
    assert "#thoughts architect" in out and (gdir / "messages.jsonl").exists()
    # --global reads the same board from a job bus, tagged with the posting job
    jb = tmp_path / "jobx" / "bus"
    jb.mkdir(parents=True)
    bus.main(["--bus", str(jb), "--as", "sa", "board", "post", "--topic", "decisions", "--body", "MiMo default implementer"])
    capsys.readouterr()
    bus.main(["--as", "later", "board", "search", "--global", "--topic", "decisions"])
    assert "#decisions sa @jobx" in capsys.readouterr().out


def test_digest_covers_persistent_board_and_every_job_bus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    gdir = repo / "docs" / "orchestration" / "board"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.setenv("SWARM_JOBS", str(repo / "data" / "swarm"))
    j1 = repo / "data" / "swarm" / "20261009-a" / "bus"
    j2 = repo / "data" / "swarm" / "20261009-b" / "bus"
    j1.mkdir(parents=True)
    j2.mkdir(parents=True)
    bus.post(j1, "sa", "sb", "info", "dm between workers")  # chief sees DMs not addressed to it
    bus.post(j2, "sc", "board", "info", "chat on job b", topic="chat")
    bus.post(j2, "sc", "board", "info", "a gotcha", topic="gotchas")  # mirrored to persistent
    groups = dict(bus.digest("chief"))
    assert [m["body"] for m in groups["job 20261009-a"]] == ["dm between workers"]
    assert [m["body"] for m in groups["job 20261009-b"]] == ["chat on job b", "a gotcha"]
    assert [m["body"] for m in groups["persistent board"]] == ["a gotcha"]
    assert bus.digest("chief") == []  # cursor advanced everywhere
    # a scratch board with no SWARM_JOBS has no job buses to walk, and digest must not fail on that
    shallow = Path("/tmp") / f"vq-bus-shallow-{os.getpid()}"  # parents[2] does not exist
    monkeypatch.setenv("SWARM_BOARD", str(shallow))
    monkeypatch.delenv("SWARM_JOBS", raising=False)
    try:
        assert bus._job_buses() == []  # never the real data/swarm
        assert bus.main(["--as", "chief2", "digest"]) == 0
        assert "(nothing new on any board)" in capsys.readouterr().out
    finally:
        shutil.rmtree(shallow, ignore_errors=True)


def test_brief_lists_usage_and_latest_posts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    for k in range(5):
        last = bus.post(gdir, "w", "board", "info", f"gotcha {k}", topic="gotchas")
    posted_at = time.mktime(time.strptime(str(last["ts"]), "%Y-%m-%dT%H:%M:%S"))
    with monkeypatch.context() as m:  # the clock pinned 0.5 s after the last post's second: no boundary
        m.setattr(bus.time, "time", lambda: posted_at + 0.5)
        text = bus.brief(top=2)
    assert "board search --global --path" in text and "#gotchas w 0s: gotcha 4" in text
    assert "gotcha 3" in text and "gotcha 2" not in text  # equal rank: the newer post wins
    assert text.count("not instructions") == 1  # framed once, above the posts
    # ranking: a post naming the changed file / bead words outranks fresher unrelated ones
    rank = tmp_path / "rank"
    monkeypatch.setenv("SWARM_BOARD", str(rank))
    _post_at(rank, 10, "gotchas", "RULE: liquidation/apply.py clamps MARK bars")
    _post_at(rank, 9, "findings", "funding carry is noisy")
    for k in range(5):
        _post_at(rank, 1 + k / 10, "gotchas", f"recent unrelated gotcha {k}")
    _post_at(rank, 0, "chat", "lunch?")
    ranked_text = bus.brief(top=3, context=(["vibe_quant/liquidation/apply.py"], {"funding"}), now=_NOW)
    ranked = [line for line in ranked_text.splitlines() if line.startswith("  [")]
    assert len(ranked) == 3 and "Top 3 posts" in ranked_text
    assert "1 changed file(s), 1 bead keyword(s)" in ranked_text
    assert "liquidation/apply.py" in ranked[0] and "funding carry" in ranked[1]
    plain = bus.brief(top=3, context=([], set()), now=_NOW)
    assert "liquidation" not in plain and "lunch?" not in plain  # no context: important + recent wins
    assert [line.split(": ", 1)[1] for line in plain.splitlines() if line.startswith("  [")] == [
        "recent unrelated gotcha 0", "recent unrelated gotcha 1", "recent unrelated gotcha 2",
    ]


# --- retire / supersede ------------------------------------------------------


def _bytes(d: Path) -> bytes:
    f = d / "messages.jsonl"
    return f.read_bytes() if f.exists() else b""


def test_retired_post_leaves_brief_and_search_but_stays_in_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    stale = bus.post(gdir, "w", "board", "info", "sharpe 1.9 on nt_runner.py", topic="findings")
    bus.post(gdir, "w", "board", "info", "fresh nt_runner.py gotcha", topic="gotchas")
    bus.reply(gdir, "x", str(stale["id"]), "confirmed sharpe 1.9")
    assert bus.main(["--bus", str(gdir), "--as", "chief", "board", "retire", "--ref", str(stale["id"]), "--reason", "pre-break metric"]) == 0
    capsys.readouterr()
    assert "sharpe 1.9" not in bus.brief() and "fresh nt_runner.py gotcha" in bus.brief()
    assert [m["body"] for m in bus.board_search(gdir, paths=["nt_runner.py"])] == ["fresh nt_runner.py gotcha"]
    bus.main(["--bus", str(gdir), "board", "search", "--topic", "findings", "--all"])
    out = capsys.readouterr().out
    assert f"[{stale['id']}]" in out and "[retired: pre-break metric]" in out
    assert "confirmed sharpe 1.9" in out and f"[reply under hidden {stale['id']}]" in out  # its thread goes too
    assert [m["body"] for m in _board_posts(gdir)][0] == "sharpe 1.9 on nt_runner.py"
    assert bus.thread(gdir, str(stale["id"]))[-1][1]["retires"] == stale["id"]  # the event sits under the post
    # a correction SUPERSEDES a post: hidden from search and brief, visible only with --all
    old = bus.post(gdir, "w", "board", "info", "RULE: use 1.6449", topic="decisions")
    new = bus.supersede(gdir, "chief", str(old["id"]), "decisions", "RULE: use the run's q95 spread")
    assert new["supersedes"] == old["id"] and f"supersedes:{old['id']}" in bus.fmt(new)
    assert [m["body"] for m in bus.board_search(gdir, topics=["decisions"])] == ["RULE: use the run's q95 spread"]
    assert "1.6449" not in bus.brief()
    hits = bus.board_search(gdir, topics=["decisions"], include_hidden=True)
    assert [(m["body"], m.get("hidden")) for m in hits] == [
        ("RULE: use 1.6449", f"superseded by {new['id']}"),
        ("RULE: use the run's q95 spread", None),
    ]


def test_retire_refuses_unknown_ids_and_is_a_noop_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    p = bus.post(gdir, "w", "board", "info", "a gotcha", topic="gotchas")
    before = _bytes(gdir)
    with pytest.raises(SystemExit) as e:
        bus.main(["--bus", str(gdir), "board", "retire", "--ref", "nope", "--reason", "x"])
    assert e.value.code not in (0, None) and _bytes(gdir) == before
    with pytest.raises(SystemExit) as e:
        bus.retire(gdir, "chief", str(p["id"]), "  ")
    assert e.value.code == 2 and _bytes(gdir) == before
    assert bus.retire(gdir, "chief", str(p["id"]), "wrong") is not None
    after_first = _bytes(gdir)
    assert bus.retire(gdir, "chief", str(p["id"]), "again") is None and _bytes(gdir) == after_first
    event = _board_posts(gdir)[-1]
    with pytest.raises(SystemExit) as e:
        bus.retire(gdir, "chief", str(event["id"]), "retire the retire")
    assert e.value.code == 2
    # a needs-user question is the USER's to close: neither retire nor supersede may touch it
    q = bus.ask_user(gdir, "w", "ship or wait?")
    before_needs = _bytes(gdir)
    for call in (
        lambda: bus.retire(gdir, "chief", str(q["id"]), "stale"),
        lambda: bus.supersede(gdir, "chief", str(q["id"]), "decisions", "shipped"),
    ):
        with pytest.raises(SystemExit) as e:
            call()
        assert e.value.code == 2
    assert _bytes(gdir) == before_needs and [m["id"] for m in bus.needs_user_open(gdir)] == [q["id"]]


def test_retire_and_supersede_reach_the_bus_that_holds_the_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path / "jobs"))
    job = tmp_path / "jobs" / "20261019-job" / "bus"
    job.mkdir(parents=True)
    mirrored = bus.post(job, "w", "board", "info", "job gotcha", topic="gotchas")  # copied to global
    bus.retire(job, "chief", str(mirrored["id"]), "wrong")
    assert bus.board_search(job, topics=["gotchas"]) == [] and bus.board_search(gdir, topics=["gotchas"]) == []
    chat = bus.post(job, "w", "board", "info", "job-only design note", topic="design")  # not mirrored
    new = bus.supersede(gdir, "chief", str(chat["id"]), "decisions", "global decision replaces it")
    assert bus.board_search(job, topics=["design"]) == []  # retire event written on the job bus
    assert bus.hidden_posts(bus._all(job))[str(chat["id"])] == f"retired: superseded by {new['id']}"
    assert [m["body"] for m in bus.board_search(gdir, topics=["decisions"])] == ["global decision replaces it"]
    on_global = bus.post(gdir, "w", "board", "info", "global gotcha", topic="gotchas")
    bus.supersede(job, "w2", str(on_global["id"]), "design", "job-only correction")  # not mirrored back
    assert bus.board_search(gdir, topics=["gotchas"]) == []  # so the global board got a retire event


# --- ranked brief -------------------------------------------------------------

_NOW = 1_800_000_000.0  # a fixed clock: ages are exact


def _post_at(
    d: Path, days_ago: float, topic: str, body: str, importance: int | None = None,
    job: str | None = None, now: float | None = None, ref: str | None = None,
) -> dict[str, object]:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime((now if now is not None else _NOW) - days_ago * 86400))
    m: dict[str, object] = {"id": f"p{days_ago}-{body[:8]}", "ts": ts, "from": "w", "to": "board", "kind": "info", "topic": topic, "body": body}
    if importance:
        m["importance"] = importance
    if job:
        m["job"] = job
    if ref:
        m["ref"] = ref
    return bus._append(d, m)


def test_score_is_importance_times_half_life_decay_times_relevance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    old_important = {"id": "a", "ts": "", "body": "x", "topic": "chat", "importance": 3}
    old_important["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_NOW - 90 * 86400))
    fresh_minor = {**old_important, "importance": 1, "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_NOW - 86400))}
    assert bus.score(old_important, _NOW) == pytest.approx(3 * 0.5 ** 1.5)
    assert bus.score(fresh_minor, _NOW) == pytest.approx(0.5 ** (1 / 60))
    assert bus.score(old_important, _NOW) > bus.score(fresh_minor, _NOW)
    assert bus.score({**old_important, "importance": 2}, _NOW) < bus.score(fresh_minor, _NOW)
    assert bus.score({**fresh_minor, "ts": "garbage"}, _NOW) == pytest.approx(0.5 ** 4)  # 4 half-lives
    assert bus.importance_of({"topic": "gotchas"}) == 2 and bus.importance_of({"topic": "chat"}) == 1
    assert bus.importance_of({"topic": "chat", "importance": 9}) == 1  # out of range -> topic default
    assert bus.relevance("see nt_runner.py:12 and the funding gate", {"nt_runner.py"}, {"funding", "gate", "sizing"}) == 6
    # --importance is stored on the post, and an out-of-range value is refused without writing
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    bus.main(["--bus", str(gdir), "--as", "w", "board", "post", "--topic", "chat", "--body", "big", "--importance", "3"])
    bus.main(["--bus", str(gdir), "--as", "w", "board", "post", "--topic", "chat", "--body", "plain"])
    capsys.readouterr()
    assert [(m["body"], m.get("importance")) for m in bus._all(gdir)] == [("big", 3), ("plain", None)]
    with pytest.raises(SystemExit) as e:
        bus.main(["--bus", str(gdir), "board", "post", "--topic", "chat", "--body", "x", "--importance", "4"])
    assert e.value.code == 2 and len(bus._all(gdir)) == 2


def test_work_context_reads_branch_files_and_bead_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = {
        ("git", "merge-base"): "abc123\n",
        ("git", "diff"): "scripts/agents/bus.py\ndocs/orchestration/board/messages.jsonl\n",
        ("git", "ls-files"): "scripts/agents/new_tool.py\n",
        ("bd", "list"): json.dumps([{"title": "Ranked board brief: decay"}, {"title": "Funding carry sizing"}]),
    }
    monkeypatch.setattr(bus, "_run", lambda cmd: outputs.get((cmd[0], cmd[1]), ""))
    files, words = _REAL_WORK_CONTEXT()
    assert files == ["scripts/agents/bus.py", "scripts/agents/new_tool.py"]  # the board's own files never count
    assert words == {"ranked", "brief", "decay", "funding", "carry", "sizing"}  # 'board' is a stopword
    monkeypatch.setattr(bus, "_run", lambda cmd: "")  # git/bd missing or failing
    assert _REAL_WORK_CONTEXT() == ([], set())


_DISCOVERY_MD = """# Discovery
## Comparability breaks

Results from before each date are NOT comparable with later ones:
- **2026-10-19 DSR q95 critical value** (bead): stricter.
- **2026-10-16 NOT a break:** margin limit is opt-in.
- **2026-10-10:** per-instrument fill release.
  - **2026-09-01** an indented sub-bullet is not a break
- **2026-07-09**: screening loads only the strategy timeframe.

## Next section
- **2026-01-01** not in the breaks section
"""


def test_comparability_breaks_are_read_from_the_discovery_doc(tmp_path: Path) -> None:
    doc = tmp_path / "discovery.md"
    doc.write_text(_DISCOVERY_MD)
    assert _REAL_BREAKS(doc) == ["2026-07-09", "2026-10-10", "2026-10-19"]
    assert _REAL_BREAKS(tmp_path / "missing.md") == []
    doc.write_text("# no breaks section\n- **2026-10-19** x\n")
    assert _REAL_BREAKS(doc) == []


def test_metric_posts_older_than_a_break_sink_and_are_tagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    breaks = ["2026-10-10", "2026-10-19"]
    monkeypatch.setattr(bus, "comparability_breaks", lambda: breaks)

    def at(day: str, body: str) -> dict[str, object]:
        return bus._append(gdir, {"id": f"{day}-{body[:6]}", "ts": f"{day}T12:00:00", "from": "w", "to": "board", "kind": "info", "topic": "findings", "body": body})

    old_metric = at("2026-10-09", "strategy 239 sharpe 0.7974 after the fix")
    at("2026-10-09", "the runner.py fill path, a non-metric fact")
    same_day = at("2026-10-19", "run 900 Sharpe from 1.2 to 0.4")
    assert bus.breaks_after(old_metric, breaks) == ["2026-10-10", "2026-10-19"]
    assert bus.breaks_after(same_day, breaks) == []  # posted on the break's day: knew about it
    assert bus.breaks_after({**old_metric, "body": "see dsr.py:46 and fitness.py:168"}, breaks) == []  # paths, not values
    now = time.mktime(time.strptime("2026-10-20T12:00:00", "%Y-%m-%dT%H:%M:%S"))
    assert bus.score(old_metric, now, breaks=breaks) == pytest.approx(bus.score(old_metric, now) * 0.8**2)
    text = bus.brief(top=3, context=([], set()), now=now)
    ranked = [line for line in text.splitlines() if line.startswith("  [")]
    assert "Sharpe from 1.2" in ranked[0] and "non-metric" in ranked[1]  # same age, the metric post sinks
    assert "sharpe 0.7974" in ranked[2] and "(pre-break 2026-10-10, 2 break(s))" in ranked[2]
    bus.main(["--bus", str(gdir), "board", "search", "--grep", "sharpe"])
    out = capsys.readouterr().out
    assert "sharpe 0.7974 after the fix  [pre-break 2026-10-10, 2 break(s)]" in out
    assert "Sharpe from 1.2 to 0.4\n" in out  # untagged


def test_idle_job_buses_fade_from_digest_and_pending_but_not_from_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path / "jobs"))
    live, idle = (tmp_path / "jobs" / name / "bus" for name in ("20261019-live", "20261001-idle"))
    for b in (live, idle):
        b.mkdir(parents=True)
        bus.post(b, "w", "chief", "question", f"question on {b.parent.name}?")
    note = bus.post(idle, "w", "board", "info", "idle job design note", topic="design")
    decision = bus.ask_user(idle, "w", "idle job decision?", job_only=True)
    eight_days_ago = time.time() - 8 * 86400
    os.utime(idle / "messages.jsonl", (eight_days_ago, eight_days_ago))
    assert [m["job"] for m in bus.pending_all("chief")] == ["20261019-live"]
    assert sorted(m["job"] for m in bus.pending_all("chief", fade_days=0)) == ["20261001-idle", "20261019-live"]
    assert [label for label, _ in bus.digest("chief", recent=5)] == ["job 20261019-live"]
    assert [label for label, _ in bus.digest("chief", recent=5, fade_days=10)] == ["job 20261001-idle", "job 20261019-live"]
    assert [m["id"] for m in bus.needs_user_open_all()] == [decision["id"]]  # the user's queue never fades
    assert bus.thread(idle, str(note["id"])) and bus.reply(gdir, "chief", str(note["id"]), "still reachable")["ref"] == note["id"]
    bus.main(["--as", "chief", "pending", "--global", "--fade-days", "0"])
    assert "@20261001-idle" in capsys.readouterr().out


# --- needs-user decision queue ------------------------------------------------


def test_ask_user_persists_to_global_with_job_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    job = tmp_path / "20261009-jobq" / "bus"
    job.mkdir(parents=True)
    assert bus.main(["--bus", str(job), "--as", "w9", "ask-user", "--body", "ship v2 or wait? recommend wait"]) == 0
    qid = capsys.readouterr().out.strip()  # ask-user prints the message id
    g = _board_posts(gdir)
    assert [(m["id"], m["topic"], m["kind"], m["job"]) for m in g] == [(qid, "needs-user", "question", "20261009-jobq")]
    bus.ask_user(job, "w9", "internal question", job_only=True)  # stays on the job bus
    assert [m["body"] for m in _board_posts(gdir)] == ["ship v2 or wait? recommend wait"]


def test_needs_user_answer_closes_with_user_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.delenv("SWARM_AGENT", raising=False)
    monkeypatch.delenv("SWARM_JOBS", raising=False)
    # agent replies of any kind are discussion; only the user closes a question
    q0 = bus.ask_user(gdir, "w0", "ship or wait?")
    bus.reply(gdir, "chief", str(q0["id"]), "recommend ship", kind="answer")
    bus.reply(gdir, "chief", str(q0["id"]), "done: shipped", kind="done")
    assert [m["id"] for m in bus.needs_user_open(gdir)] == [q0["id"]]
    bus.main(["--bus", str(gdir), "needs-user", "open"])
    out = capsys.readouterr().out
    assert "ship or wait?" in out and "old)" in out  # listing shows age
    q1 = bus.ask_user(gdir, "w1", "rename module?")
    bus.main(["--bus", str(gdir), "--as", "chief", "needs-user", "answer", "--ref", str(q1["id"]), "--body", "wait for holdout", "--user"])
    capsys.readouterr()
    assert [m["id"] for m in bus.needs_user_open(gdir)] == [q0["id"]]
    ans = bus.thread(gdir, str(q1["id"]))[-1][1]
    assert ans["from"] == "user" and ans["kind"] == "answer"  # from is always the user
    assert ans["user_answer"] is True and ans["by"] == "chief"  # audit trail: who relayed it
    bus.main(["--bus", str(gdir), "needs-user", "answer", "--ref", str(q0["id"]), "--body", "ship it", "--user"])
    capsys.readouterr()
    q2 = bus.ask_user(gdir, "w2", "drop worst mode?")
    bus.main(["--bus", str(gdir), "needs-user", "answer", "--ref", str(q2["id"]), "--body", "rename it", "--user"])
    capsys.readouterr()
    assert bus.thread(gdir, str(q2["id"]))[-1][1]["by"] == "user"  # no --as, no $SWARM_AGENT
    monkeypatch.setenv("SWARM_AGENT", "w9")
    q3 = bus.ask_user(gdir, "w3", "keep or drop?")
    bus.main(["--bus", str(gdir), "needs-user", "answer", "--ref", str(q3["id"]), "--body", "keep it", "--user"])
    capsys.readouterr()
    ans3 = bus.thread(gdir, str(q3["id"]))[-1][1]
    assert ans3["from"] == "user" and ans3["by"] == "w9"
    assert bus.needs_user_open(gdir) == []


def test_brief_shows_open_decisions_first_and_omits_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    for k in range(3):
        bus.post(gdir, "w", "board", "info", f"gotcha {k}", topic="gotchas")
    qs = [bus.ask_user(gdir, "w1", f"decision {k}?") for k in range(6)]
    text = bus.brief(top=1)
    assert "OPEN DECISIONS FOR THE USER (6):" in text
    assert text.index("OPEN DECISIONS FOR THE USER") < text.index("Top 1 posts")  # before the ranked posts
    section = text[text.index("OPEN DECISIONS FOR THE USER") : text.index("Top 1 posts")]
    assert section.count("[OPEN]") == 5  # max 5, oldest first
    assert "decision 0?" in section and "decision 5?" not in section
    for q in qs:
        bus.needs_user_answer(gdir, "user", str(q["id"]), "decided")
    text = bus.brief(top=1)
    assert "OPEN DECISIONS" not in text and "[OPEN]" not in text  # omitted when none
    # ordering is by ts (oldest first), not by file order
    ordered = tmp_path / "ordered"
    monkeypatch.setenv("SWARM_BOARD", str(ordered))
    monkeypatch.delenv("SWARM_JOBS", raising=False)
    for k, (ts, body) in enumerate([  # appended out of ts order on purpose
        ("2026-10-09T12:00:03", "newest question?"),
        ("2026-10-09T12:00:01", "oldest question?"),
        ("2026-10-09T12:00:02", "middle question?"),
    ]):
        bus._append(ordered, {"id": f"q{k}", "ts": ts, "from": "w", "to": "board", "kind": "question", "topic": "needs-user", "body": body})
    order = ["oldest question?", "middle question?", "newest question?"]
    assert [m["body"] for m in bus.needs_user_open(ordered)] == order
    section = bus.brief(top=3)
    section = section[section.index("OPEN DECISIONS FOR THE USER") :]
    assert [section.index(b) for b in order] == sorted(section.index(b) for b in order)  # brief agrees


def test_digest_shows_open_needs_user_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path))
    job = tmp_path / "20261009-jobq" / "bus"
    job.mkdir(parents=True)
    bus.post(gdir, "w", "chief", "question", "still open?")
    bus.ask_user(gdir, "w1", "global question?")
    bus.ask_user(job, "w2", "job-only question?", job_only=True)
    bus.ask_user(job, "w3", "mirrored question?")  # on both buses, must count once
    bus.main(["--as", "chief", "digest"])
    out = capsys.readouterr().out
    assert "PENDING QUESTIONS FOR chief (1):" in out and "still open?" in out
    assert "OPEN DECISIONS FOR THE USER (3):" in out
    # pending questions, then open decisions, both above the message dump
    assert out.index("PENDING QUESTIONS FOR chief") < out.index("OPEN DECISIONS FOR THE USER")
    assert out.index("OPEN DECISIONS FOR THE USER") < out.index("== persistent board")
    assert "[OPEN]" in out
    assert "global question?" in out and "job-only question?" in out and "mirrored question?" in out
    assert "@20261009-jobq" in out  # job-only items carry their job


def test_needs_user_answer_requires_user_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    q = bus.ask_user(gdir, "w1", "ship or wait?")
    with pytest.raises(SystemExit) as e:
        bus.main(["--bus", str(gdir), "--as", "chief", "needs-user", "answer", "--ref", str(q["id"]), "--body", "ship it"])
    assert e.value.code == 2  # only the user closes decisions
    assert "needs-user answer requires --user: only the user closes decisions; relay the user's verbatim words" in capsys.readouterr().err
    assert [m["id"] for m in bus.needs_user_open(gdir)] == [q["id"]]  # refused: stays open
    assert bus.main(["--bus", str(gdir), "--as", "chief", "needs-user", "answer", "--ref", str(q["id"]), "--body", "ship it", "--user"]) == 0
    assert bus.needs_user_open(gdir) == []  # with --user it closes


def test_needs_user_answer_finds_the_question_on_another_bus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path))
    job = tmp_path / "20261009-jobq" / "bus"
    job.mkdir(parents=True)
    q_job = bus.ask_user(job, "w2", "job-only question?", job_only=True)
    assert bus.main(["--bus", str(gdir), "--as", "chief", "needs-user", "answer", "--ref", str(q_job["id"]), "--body", "wait", "--user"]) == 0
    assert bus.needs_user_open(job) == []  # answered on the job bus that holds it
    assert bus._all(gdir) == []  # job-only question stays job-only (mirror rules unchanged)
    q_glob = bus.ask_user(gdir, "w1", "global question?")
    assert bus.main(["--bus", str(job), "--as", "chief", "needs-user", "answer", "--ref", str(q_glob["id"]), "--body", "ship", "--user"]) == 0
    assert bus.needs_user_open(gdir) == []  # found on the global board from a job bus
    with pytest.raises(SystemExit) as e:  # unknown ids still fail as before
        bus.main(["--bus", str(gdir), "--as", "chief", "needs-user", "answer", "--ref", "nosuchid", "--body", "?", "--user"])
    assert "no message nosuchid" in str(e.value)


def test_needs_user_answer_refuses_non_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    q = bus.ask_user(gdir, "w1", "ship or wait?")
    answered = bus.needs_user_answer(gdir, "user", str(q["id"]), "wait")
    off_topic = bus.post(gdir, "w2", "board", "question", "design question?", topic="design")
    not_a_question = bus.post(gdir, "w3", "board", "info", "needs-user note", topic="needs-user")
    for bad in (answered, off_topic, not_a_question):
        with pytest.raises(SystemExit) as e:
            bus.needs_user_answer(gdir, "chief", str(bad["id"]), "x")
        assert e.value.code == 2
        assert f"{bad['id']} is not a needs-user question" in capsys.readouterr().err


# --- durable tail (never miss a message across a watcher restart) -------------


def test_a_done_post_on_the_lobby_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    # A Claude subagent loses SWARM_BUS between Bash calls; its done post then hit the lobby
    # and the handoff check asked for docs/orchestration/handoffs/<me>.md in the main checkout.
    with pytest.raises(SystemExit) as exc:
        bus.main(["--as", "ver-x-1", "post", "--to", "chief", "--kind", "done", "--body", "ok"])
    assert exc.value.code == 2
    assert "job bus" in capsys.readouterr().err
    assert not (bus.global_board_dir().parent / "handoffs").exists()


# --- atomic cursors -----------------------------------------------------------


def test_set_cursor_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bus._set_cursor(tmp_path, "sa", 5)
    p = bus._cursor_path(tmp_path, "sa")
    assert p.read_text() == "5"

    def boom(src: object, dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(bus.os, "replace", boom)
    with pytest.raises(OSError):
        bus._set_cursor(tmp_path, "sa", 9)
    assert p.read_text() == "5"  # old value intact (no torn write)
    assert [f.name for f in p.parent.iterdir()] == [p.name]  # no temp leftovers


def test_garbled_cursor_is_a_clear_error(tmp_path: Path) -> None:
    bus._set_cursor(tmp_path, "sa", 7)
    p = bus._cursor_path(tmp_path, "sa")
    p.write_text("")  # a crash mid-write leaves an empty cursor file
    with pytest.raises(SystemExit) as exc:
        bus._get_cursor(tmp_path, "sa")
    assert str(p) in str(exc.value)


# --- pending questions --------------------------------------------------------


def test_pending_lists_questions_without_a_reply(tmp_path: Path) -> None:
    q = bus.post(tmp_path, "chief", "sa", "question", "which lookback?")
    bus.post(tmp_path, "chief", "sa", "blocker", "need the catalog")
    bus.post(tmp_path, "chief", "all", "question", "broadcast: not a DM")  # not addressed to me
    assert [m["body"] for m in bus.pending(tmp_path, "sa")] == ["which lookback?", "need the catalog"]
    bus.reply(tmp_path, "sa", str(q["id"]), "2*period")
    assert [m["body"] for m in bus.pending(tmp_path, "sa")] == ["need the catalog"]
    # a reply from someone OTHER than the addressee leaves the question pending
    q2 = bus.post(tmp_path, "chief", "sa", "question", "which lookback, again?")
    bus.post(tmp_path, "other", "chief", "answer", "I think 20", ref=str(q2["id"]))
    assert [m["body"] for m in bus.pending(tmp_path, "sa")] == ["need the catalog", "which lookback, again?"]


# --- idempotent posts (--client-id) -------------------------------------------


def test_client_id_makes_posts_idempotent(tmp_path: Path) -> None:
    a = bus.post(tmp_path, "sa", "chief", "info", "hello", client_id="c1")
    b = bus.post(tmp_path, "sa", "chief", "info", "hello", client_id="c1")
    assert a["id"] == b["id"]  # the existing message is returned, not a duplicate
    bus.post(tmp_path, "sa", "chief", "info", "hello", client_id="c2")  # new id -> new line
    bus.post(tmp_path, "sb", "chief", "info", "hello", client_id="c1")  # other sender -> new line
    lines = (tmp_path / "messages.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["client_id"] == "c1"
    # the CLI passes --client-id through
    for _ in range(2):
        assert bus.main(
            ["--bus", str(tmp_path), "--as", "sa", "post", "--to", "chief", "--body", "x", "--client-id", "z"]
        ) == 0
    assert len((tmp_path / "messages.jsonl").read_text().splitlines()) == 4


# --- regression: findings from the harden review ------------------------------


def _same_key_writer(path: str, barrier: object, rounds: int) -> None:
    for r in range(rounds):
        barrier.wait()  # type: ignore[attr-defined]  # every process posts the SAME key at once
        bus.post(Path(path), "sa", "chief", "info", "retry", client_id=f"k{r}")


def test_concurrent_same_client_id_retries_store_one_line_each(tmp_path: Path) -> None:
    rounds, procs = 25, 6
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(procs)
    workers = [ctx.Process(target=_same_key_writer, args=(str(tmp_path), barrier, rounds)) for _ in range(procs)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=60)
        assert w.exitcode == 0
    msgs = [json.loads(ln) for ln in (tmp_path / "messages.jsonl").read_text().splitlines()]
    assert sorted(m["client_id"] for m in msgs) == sorted(f"k{r}" for r in range(rounds))


def test_pending_global_cli_spans_the_board_and_every_job_bus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs = tmp_path / "jobs"
    monkeypatch.setenv("SWARM_JOBS", str(jobs))
    other_job = jobs / "20261012-other" / "bus"
    other_job.mkdir(parents=True)
    mine = jobs / "20261012-mine" / "bus"
    mine.mkdir(parents=True)
    bus.post(bus.global_board_dir(), "w", "worker", "question", "global-only-unanswered-question")
    bus.post(other_job, "w", "worker", "question", "other-job-question")
    bus.post(mine, "w", "worker", "question", "my-job-question")
    base = ["--bus", str(mine), "--as", "worker", "pending"]
    assert bus.main(base) == 0
    plain = capsys.readouterr().out
    assert "my-job-question" in plain and "global-only" not in plain and "other-job" not in plain
    assert bus.main([*base, "--global"]) == 0
    wide = capsys.readouterr().out
    assert all(b in wide for b in ("global-only-unanswered-question", "other-job-question", "my-job-question"))
    assert "@20261012-other" in wide  # job-tagged


def _delivered(tmp_path: Path) -> list[str]:
    """Bodies the file store's cursor reader (digest's ``_read_since``) delivers, cursor advanced."""
    return [str(m["body"]) for m in bus._read_since(tmp_path, "chief.digest")]


def _board_posts(d: Path) -> list[dict[str, object]]:
    return [m for m in bus._all(d) if m.get("to") == "board"]


def test_append_after_a_crashed_writer_fragment_is_still_delivered(tmp_path: Path) -> None:
    path = tmp_path / "messages.jsonl"
    old = json.dumps({"id": "a", "from": "w", "to": "chief", "kind": "info", "body": "before"}, ensure_ascii=False)
    # a writer died mid-record: a torn tail with no newline, cut inside a multibyte character
    path.write_bytes(old.encode() + b"\n" + b'{"id": "torn", "body": "\xe2\x82')
    stored = bus.post(tmp_path, "sa", "chief", "info", "after", client_id="c1")
    assert _delivered(tmp_path) == ["before", "after"]  # the new record did not get glued to the fragment
    assert bus.post(tmp_path, "sa", "chief", "info", "after", client_id="c1")["id"] == stored["id"]
    assert _delivered(tmp_path) == []  # the retry stored nothing new: the cursor has nothing to deliver
    assert len([ln for ln in path.read_bytes().split(b"\n") if ln]) == 3  # old, fragment, new
    # a FULLY written record that merely lost its newline is deduped and delivered exactly once
    loose = tmp_path / "loose"
    loose.mkdir()
    rec = {"id": "orig", "from": "sa", "to": "chief", "kind": "info", "body": "once", "client_id": "c1"}
    (loose / "messages.jsonl").write_text(json.dumps(rec), encoding="utf-8")
    assert bus.post(loose, "sa", "chief", "info", "once", client_id="c1")["id"] == "orig"
    assert _delivered(loose) == ["once"]  # the acknowledged message is deliverable, exactly once
    assert (loose / "messages.jsonl").read_bytes().endswith(b"\n")


@pytest.mark.parametrize("junk", [b"123", b'{"id": "x", "from": "w"}'])
def test_read_from_keeps_only_complete_message_records(tmp_path: Path, junk: bytes) -> None:
    good = json.dumps({"id": "g", "from": "w", "to": "chief", "kind": "info", "body": "ok"}).encode()
    (tmp_path / "messages.jsonl").write_bytes(junk + b"\n" + good + b"\n" + junk + b"\n")
    msgs, off = bus._read_from(tmp_path, 0)
    assert [m["id"] for m in msgs] == ["g"] and off == (tmp_path / "messages.jsonl").stat().st_size
    # an unfinished last line is left for later: the offset stops at the last complete line
    path = tmp_path / "messages.jsonl"
    path.write_bytes(good + b"\n" + b'{"id": "b"')
    msgs, off = bus._read_from(tmp_path, 0)
    assert [m["id"] for m in msgs] == ["g"] and off == path.read_bytes().index(b"\n") + 1
    assert bus._read_from(tmp_path, off)[1] == off  # exactly that offset is fine
    size = path.stat().st_size
    with pytest.raises(SystemExit) as exc:  # past EOF is refused, never reset
        bus._read_from(tmp_path, size + 1)
    assert str(size + 1) in str(exc.value)
    with pytest.raises(SystemExit):  # a vanished file is shorter than any positive offset too
        bus._read_from(tmp_path / "nowhere", 5)


@pytest.mark.parametrize("value", ["999999", "-5"])
def test_out_of_bounds_cursor_is_a_clear_error_for_the_digest_cursor(tmp_path: Path, value: str) -> None:
    bus.post(tmp_path, "chief", "sa", "info", "must-read")
    size = (tmp_path / "messages.jsonl").stat().st_size
    assert int(value) != size
    cpath = bus._cursor_path(tmp_path, "sa.digest")
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(value, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        bus._read_since(tmp_path, "sa.digest")
    assert str(cpath) in str(exc.value) and value in str(exc.value)
    assert cpath.read_text() == value  # never silently repaired or advanced
    # exactly EOF, on the other hand, is valid
    bus._set_cursor(tmp_path, "sa.digest", size)
    assert bus._read_since(tmp_path, "sa.digest") == []


def test_client_id_retry_over_an_incomplete_record_appends_a_deliverable_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "messages.jsonl"
    path.write_text('{"id":"bad","from":"sa","client_id":"retry"}\n', encoding="utf-8")  # no to/kind/body
    stored = bus.post(tmp_path, "sa", "chief", "info", "real", client_id="retry")
    assert stored["id"] != "bad"  # the junk record is not "already posted": no reader would deliver it
    assert _delivered(tmp_path) == ["real"]
    assert _delivered(tmp_path) == []  # exactly once
    assert bus.main(["--bus", str(tmp_path), "--as", "chief", "thread", str(stored["id"])]) == 0
    assert "real" in capsys.readouterr().out  # the CLI renders it (no KeyError)
    again = bus.post(tmp_path, "sa", "chief", "info", "real", client_id="retry")
    assert again["id"] == stored["id"]  # a real record still dedupes
    # a cursor sitting INSIDE a torn fragment does not wedge the reader either
    torn = tmp_path / "torn"
    torn.mkdir()
    (torn / "messages.jsonl").write_bytes(b'{"id":"torn","body":"123')  # "123" is valid JSON by itself
    bus._set_cursor(torn, "chief.digest", 21)  # the cursor sits INSIDE the fragment (at the digits)
    bus.post(torn, "sa", "chief", "info", "acknowledged")  # repairs the tail, then appends
    assert _delivered(torn) == ["acknowledged"]
    assert bus._get_cursor(torn, "chief.digest") == (torn / "messages.jsonl").stat().st_size  # advanced past the junk
    assert _delivered(torn) == []  # a restart neither crashes nor repeats


_NO_TS = [
    # passes _is_message (id/from/to/kind/body) but carries none of the optional fields (ts, topic, ref, ...)
    {"id": "bad", "from": "sa", "to": "chief", "kind": "question", "body": "old", "client_id": "retry"},
    {"id": "nu", "from": "sa", "to": "board", "kind": "question", "body": "needs the user", "topic": "needs-user"},
    {"id": "bp", "from": "sa", "to": "board", "kind": "info", "body": "board post w/o topic"},
]


@pytest.mark.parametrize(
    ("argv", "expect"),
    [
        (["--as", "chief", "pending"], "old"),
        (["--as", "chief", "needs-user", "open"], "needs the user"),
        (["--as", "chief", "digest"], "needs the user"),
        (["--as", "chief", "digest", "--recent", "9"], "old"),
        (["--as", "chief", "board", "topics"], "needs-user"),
        (["--as", "chief", "board", "brief"], "needs the user"),
        (["--as", "chief", "board", "search", "--grep", "w/o topic", "--all"], "board post w/o topic"),
        (["--as", "chief", "roster"], "sa"),
        (["--as", "chief", "thread", "nu"], "needs the user"),
    ],
)
def test_records_without_optional_fields_work_in_every_consumer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str], expect: str
) -> None:
    g = bus.global_board_dir()  # SWARM_BOARD (autouse) -> a tmp board
    (g / "messages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _NO_TS), encoding="utf-8")
    assert bus.main(["--bus", str(g), *argv]) == 0
    out = capsys.readouterr().out
    assert expect in out


def test_records_without_ts_keep_order_and_retry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    g = bus.global_board_dir()
    recs = [{"id": f"q{i}", "from": "sa", "to": "chief", "kind": "question", "body": f"b{i}"} for i in range(3)]
    (g / "messages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    assert [m["id"] for m in bus.pending(g, "chief")] == ["q0", "q1", "q2"]
    # a client-id retry over such a record returns it (no duplicate) and it stays listable
    (g / "messages.jsonl").write_text(json.dumps(_NO_TS[0]) + "\n", encoding="utf-8")
    assert bus.post(g, "sa", "chief", "question", "old", client_id="retry")["id"] == "bad"
    assert [m["id"] for m in bus.pending(g, "chief")] == ["bad"]
    assert bus.main(["--bus", str(g), "--as", "chief", "pending"]) == 0
    assert "(? old)" in capsys.readouterr().out  # an unknown age renders as ?


# --- reply: the bus that HOLDS the parent (cross-job ids) ---------------------


def _make_job_bus(root: Path, name: str) -> Path:
    b = root / name / "bus"
    b.mkdir(parents=True)
    return b


def test_reply_finds_a_mirrored_post_on_the_global_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An id `board search --global` printed replies on the global board, job-tagged."""
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    jobs = tmp_path / "jobs"
    monkeypatch.setenv("SWARM_JOBS", str(jobs))
    job_a = _make_job_bus(jobs, "20261015-job-a")  # the replier's own bus
    job_b = _make_job_bus(jobs, "20261015-job-b")  # where the mirrored post came from
    root = bus.post(job_b, "w2", "board", "info", "NaN reject rule", topic="findings")
    # the id the board hands out for that post is the mirrored copy's
    assert [m["id"] for m in bus.board_search(bus.global_board_dir(), topics=["findings"])] == [str(root["id"])]
    r = bus.reply(job_a, "x", str(root["id"]), "confirmed")
    assert r["to"] == "board" and r["topic"] == "findings" and r["ref"] == root["id"]
    assert r["job"] == "20261015-job-a"  # tagged with the REPLIER's job, like a mirrored post
    assert [m["id"] for m in bus._all(gdir)] == [root["id"], r["id"]]  # only the global board grew
    assert bus._all(job_a) == []  # never the replier's job bus
    assert [m["id"] for m in bus._all(job_b)] == [root["id"]]  # never the origin job bus
    assert [(d, m["body"]) for d, m in bus.thread(bus.global_board_dir(), str(root["id"]))] == [
        (0, "NaN reject rule"),
        (1, "confirmed"),
    ]
    # the reported bug, through the CLI from a job bus: `bus reply --ref <id from board search --global>`
    argv = ["--bus", str(job_a), "--as", "x", "reply", "--ref", str(root["id"]), "--body", "confirmed again"]
    assert bus.main(argv) == 0
    out = capsys.readouterr().out
    assert "#findings" in out and "confirmed again" in out and "@20261015-job-a" in out
    assert [m["ref"] for m in bus._all(gdir) if m.get("ref")] == [root["id"], root["id"]]


def test_reply_unknown_id_fails_with_the_same_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An id nowhere (even with job buses present) still exits with the unchanged text."""
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path / "global"))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path / "jobs"))
    _make_job_bus(tmp_path / "jobs", "20261015-job-a")
    with pytest.raises(SystemExit) as e:
        bus.reply(tmp_path, "x", "nosuchid", "?")
    assert str(e.value) == "bus: no message nosuchid"


def test_reply_is_idempotent_on_the_holding_bus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeat reply with the same (sender, client_id) is stored once, on the holding bus."""
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    jobs = tmp_path / "jobs"
    monkeypatch.setenv("SWARM_JOBS", str(jobs))
    job_a = _make_job_bus(jobs, "20261015-job-a")
    job_b = _make_job_bus(jobs, "20261015-job-b")
    root = bus.post(job_b, "w2", "board", "info", "a gotcha", topic="gotchas")
    first = bus.reply(job_a, "x", str(root["id"]), "same", client_id="k1")
    again = bus.reply(job_a, "x", str(root["id"]), "same", client_id="k1")
    assert first["id"] == again["id"]
    assert [m["body"] for m in bus._all(gdir)] == ["a gotcha", "same"]  # exactly one reply line


def test_locate_prefers_the_selected_bus_then_global_then_jobs_and_searches_none_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First hit wins in [selected, global, job buses...] and no directory is searched twice."""
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    jobs = tmp_path / "jobs"
    monkeypatch.setenv("SWARM_JOBS", str(jobs))
    job_a = _make_job_bus(jobs, "20261015-job-a")
    job_b = _make_job_bus(jobs, "20261015-job-b")
    gdir.mkdir(parents=True, exist_ok=True)
    for src, body in ((gdir, "on the global board"), (job_a, "on the selected bus"), (job_b, "on another job bus")):
        bus._append(src, {"id": "dup", "from": "w", "to": "board", "kind": "info", "topic": "findings", "body": body})
    found = bus._locate(job_a, "dup")
    assert found is not None and found[0] == job_a and found[1]["body"] == "on the selected bus"
    found = bus._locate(tmp_path / "elsewhere", "dup")  # nothing on the selected bus -> global first
    assert found is not None and found[0] == gdir
    assert bus._locate(job_a, "nowhere") is None
    searched: list[Path] = []
    real_all = bus._all

    def spy(src: Path) -> list[dict[str, object]]:
        searched.append(src)
        return real_all(src)

    monkeypatch.setattr(bus, "_all", spy)
    assert bus._locate(gdir, "nowhere") is None
    assert searched == [gdir, job_a, job_b]  # the global board is asked once, not twice


# --- tail --kinds (a chief Monitor wakes on the kinds it can act on) ----------


# --- status (one screen for the chief) ----------------------------------------


def test_status_reports_counts_and_stays_within_20_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answered = bus.post(tmp_path, "w1", "chief", "question", "already answered?")
    bus.post(tmp_path, "chief", "w1", "answer", "yes", ref=str(answered["id"]))
    for i in range(7):  # 7 unanswered, each with a long body
        bus.post(tmp_path, f"w{i}", "chief", "blocker" if i == 3 else "question", f"open question {i} " + "x" * 200)
    bus.post(tmp_path, "chief", "w2", "info", "dm 3")
    bus.ask_user(tmp_path, "w4", "ship or wait?", job_only=True)
    assert bus.main(["--bus", str(tmp_path), "--as", "chief", "status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) <= 20
    assert "pending to chief: 7" in lines
    items = [ln for ln in lines if ln.startswith("  [")]
    assert len(items) == 5  # at most 5 listed, oldest first (the answered one is not pending)
    assert not any("already answered" in ln for ln in items)
    assert any(ln.startswith("  [") and "200" not in ln.split(" — ")[1][:60] for ln in items)  # body cut to 60
    assert any("(+2 more)" in ln for ln in lines)
    assert any(ln.startswith("last message: ") and "w4" in ln and "ship or wait?" in ln for ln in lines)
    assert "open needs-user: 1" in lines
    # R27: the file store has no worker inbox and no durable tail, so status says nothing about them
    assert not any(ln.startswith(("unread from", "chief tail", "asks watcher")) for ln in lines)


def test_status_says_none_for_an_idle_bus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert bus.main(["--bus", str(tmp_path), "--as", "chief", "status"]) == 0
    out = capsys.readouterr().out
    assert "pending to chief: 0" in out
    assert "last message: none" in out
    assert "open needs-user: 0" in out
    assert "unread from" not in out and "tail" not in out
    assert not (tmp_path / "cursors").exists()  # a read-only screen creates no cursor dir


# --- pending: one bounded line per item ---------------------------------------


def test_pending_cli_prints_one_short_line_per_pending_item(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bus.post(tmp_path, "w1", "chief", "question", "which lookback?\n\nUse 2*period if unsure.\n" + "z" * 3000)
    bus.post(tmp_path, "w2", "chief", "blocker", "need the catalog")
    assert bus.main(["--bus", str(tmp_path), "--as", "chief", "pending"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2  # one line per pending item, whatever the body contains
    assert len(lines[0]) < 200  # the 3k body is cut, not dumped
    assert lines[0].startswith("[") and "which lookback?" in lines[0] and "old)" in lines[0]
    assert "need the catalog" in lines[1]
    assert [m["body"] for m in bus.pending(tmp_path, "chief")]  # pending never advances a cursor


def test_pending_cli_prints_only_the_fallback_line_when_nothing_is_open(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An answered question, an info record and a broadcast question leave nothing pending."""
    bus.post(tmp_path, "w", "chief", "info", "old chatter")
    q = bus.post(tmp_path, "w", "chief", "question", "answered question")
    bus.post(tmp_path, "chief", "w", "answer", "done", ref=str(q["id"]))
    bus.post(tmp_path, "w", "all", "question", "a broadcast is not a DM")
    assert bus.main(["--bus", str(tmp_path), "--as", "chief", "pending"]) == 0
    assert capsys.readouterr().out == "(no pending questions for chief)\n"
    assert bus.main(["--bus", str(tmp_path), "--as", "chief", "pending", "--global"]) == 0
    assert capsys.readouterr().out == "(no pending questions for chief)\n"


# --- scripts/agents/bus (the sh shim) -----------------------------------------


_BUS_SHIM = _BUS_PATH.with_name("bus")


def test_bus_shim_invokes_bus_py(tmp_path: Path) -> None:
    assert _BUS_SHIM.exists(), f"{_BUS_SHIM} is missing"
    assert os.access(_BUS_SHIM, os.X_OK)  # committed executable: `bus ...` works without a python prefix
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    help_run = subprocess.run([str(_BUS_SHIM), "--help"], capture_output=True, text=True, env=env)
    assert help_run.returncode == 0
    assert "status" in help_run.stdout and "needs-user answer" in help_run.stdout
    posted = subprocess.run(
        [str(_BUS_SHIM), "--bus", str(tmp_path), "--as", "sa", "post", "--to", "chief", "--body", "via the shim"],
        capture_output=True, text=True, env=env,
    )  # fmt: skip
    assert posted.returncode == 0 and "via the shim" in posted.stdout


_GOOD_HANDOFF = "## Handoff\nstatus: done\nrole: implementer\nevidence:\n- red then green\n- `GATE PASS` (LOG data/swarm/_gate-logs/x.log)\nimprove: none\n"


def _gated_handoff(root: Path) -> str:
    """_GOOD_HANDOFF citing a real commit and a real PASS LOG of its tree (handoff_lint opens it)."""
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}  # fmt: skip

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                              text=True, env=env).stdout.strip()  # fmt: skip

    root.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    (root / "a.py").write_text("x = 1\n")
    git("add", "a.py")
    git("commit", "-q", "-m", "feat: x")
    sha, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    log = root / "data" / "swarm" / "_gate-logs" / f"wt-{sha[:12]}-x.log"
    log.parent.mkdir(parents=True)
    log.write_text(f"$ ruff check\n[exit 0]\nHEAD {sha} TREE {tree}\nGATE PASS\n")
    return _GOOD_HANDOFF.replace("data/swarm/_gate-logs/x.log", f"{log}, commit {sha[:8]}")


def _done(job: Path, *extra: str) -> int:
    return bus.main(["--bus", str(job / "bus"), "--as", "w1", "post", "--to", "chief", "--kind", "done", "--body", "ok", *extra])


def test_done_refused_without_handoff_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "bus").mkdir()
    with pytest.raises(SystemExit) as exc:
        _done(tmp_path)
    assert exc.value.code == 2
    assert str(tmp_path / "handoffs" / "w1.md") in capsys.readouterr().err
    assert _dms(tmp_path / "bus", "chief") == []


def test_done_refused_when_handoff_fails_lint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "bus").mkdir()
    (tmp_path / "handoffs").mkdir()
    (tmp_path / "handoffs" / "w1.md").write_text("## Handoff\nstatus: done\n")
    with pytest.raises(SystemExit) as exc:
        _done(tmp_path)
    assert exc.value.code == 2
    assert "MISSING evidence" in capsys.readouterr().err
    assert _dms(tmp_path / "bus", "chief") == []
    # a checker block with a bad verdict is refused the same way
    (tmp_path / "handoffs" / "w1.md").write_text(_GOOD_HANDOFF + "verdict: MAYBE\n")
    with pytest.raises(SystemExit) as exc:
        _done(tmp_path)
    assert exc.value.code == 2
    assert "BAD verdict" in capsys.readouterr().err


def test_done_posts_when_the_brief_merely_names_compare_rev(tmp_path: Path) -> None:
    for name in ("bus", "handoffs", "prompts"):
        (tmp_path / name).mkdir()
    (tmp_path / "handoffs" / "w1.md").write_text(_gated_handoff(tmp_path / "repo"))
    (tmp_path / "prompts" / "w1.md").write_text("run mutate.py --compare-rev BASE on every guard\n")
    assert _done(tmp_path) == 0
    assert _dms(tmp_path / "bus", "chief") == ["ok"]


def test_done_posts_with_valid_handoff(tmp_path: Path) -> None:
    (tmp_path / "bus").mkdir()
    (tmp_path / "handoffs").mkdir()
    (tmp_path / "handoffs" / "w1.md").write_text(_GOOD_HANDOFF)
    with pytest.raises(SystemExit) as exc:  # the cited LOG does not exist
        _done(tmp_path)
    assert exc.value.code == 2
    (tmp_path / "handoffs" / "w1.md").write_text(_gated_handoff(tmp_path / "repo"))
    assert _done(tmp_path) == 0
    assert _dms(tmp_path / "bus", "chief") == ["ok"]
    # the escape hatch, and kinds other than done, skip the check entirely
    other = tmp_path / "other"
    (other / "bus").mkdir(parents=True)
    assert _done(other, "--no-handoff-check") == 0
    assert bus.main(["--bus", str(other / "bus"), "--as", "w1", "post", "--to", "chief", "--kind", "info", "--body", "hi"]) == 0


@pytest.mark.parametrize(
    ("body", "posted"),
    [
        # a credential assignment, then a whole env dump: both refused, nothing written
        ("debug: CLAUDE_CODE_MESSAGING_TOKEN=abcd1234efgh5678ijkl", False),
        ("BINANCE_API_KEY: 'k3y-abcdefghijklmnop'", False),
        ("HOME=/x\nPATH=/bin\nUSER=me\nSHELL=/bin/zsh\nLANG=C\n", False),
        # a near miss: the word TOKEN but no assignment
        ("the TOKEN env var must be unset", True),
        ("use SWARM_BUS=/x and SWARM_AGENT=y", True),
    ],
)
def test_cli_refuses_a_body_that_looks_like_a_secret(tmp_path: Path, body: str, posted: bool) -> None:
    argv = ["--bus", str(tmp_path), "--as", "sa", "post", "--to", "chief", "--body", body]
    if posted:
        assert bus.main(argv) == 0
        return
    with pytest.raises(SystemExit) as exc:
        bus.main(argv)
    assert exc.value.code == 2
    assert not (tmp_path / "messages.jsonl").exists()


def test_status_lists_running_agents_and_startable_beads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user asked 'anything left / waiting?' three times in one session: status answers it."""
    job = tmp_path / "20261020-j"
    busdir = job / "bus"
    busdir.mkdir(parents=True)
    (job / "ledger.md").write_text(
        "# ledger 20261020-j\n| agent | runtime | model | task | bead | worktree | status |\n"
        "|---|---|---|---|---|---|---|\n"
        "| cc-a-1 | t3 | flash | a | - | - | 10-20 08:00Z dispatched |\n"
        "| cc-b-1 | t3 | flash | b | - | - | 10-20 08:00Z dispatched |\n"
        "| cc-c-1 | t3 | flash | c | - | - | 10-20 08:00Z dispatched |\n"
        "| cc-c-1 | t3 | flash | c | - | - | 10-20 09:00Z GATE PASS, merged |\n"
    )
    bus.post(busdir, "cc-b-1", "chief", "done", "finished")
    monkeypatch.setattr(bus, "_ready_count", lambda: 4)
    assert bus.main(["--bus", str(busdir), "--as", "chief", "status"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "running: 1 (cc-a-1)" in out  # b posted done; c's latest row is not running
    assert "startable: 4 ready beads (bd ready)" in out
    # no ledger and no bd: zero running, unknown startable
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.setattr(bus, "_ready_count", lambda: None)
    assert bus.main(["--bus", str(bare), "--as", "chief", "status"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "running: 0" in out and "startable: unknown ready beads (bd ready)" in out



def _dms(b: Path, me: str) -> list[object]:
    """Bodies addressed to ``me`` (or all) by someone else, in file order."""
    return [m["body"] for m in bus._all(b) if m.get("from") != me and m.get("to") in (me, "all")]


def _post_cli(b: Path, me: str, to: str, *extra: str) -> int:
    return bus.main(["--bus", str(b), "--as", me, "post", "--to", to, "--kind", "info", "--body", "late finding", *extra])


def test_dm_to_an_agent_that_posted_done_is_refused_naming_its_done_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A finished agent reads nothing more: a DM to it is dead (921 such gate posts and reviewer
    DMs to finished makers in two weeks). The sender is told where to post instead."""
    b = tmp_path / "bus"
    b.mkdir()
    done = bus.post(b, "cc-x-1", "chief", "done", "finished")
    with pytest.raises(SystemExit) as exc:
        _post_cli(b, "rev-x-1", "cc-x-1")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"cc-x-1 posted done at {done['ts']}" in err and "post to chief" in err
    assert _dms(b, "cc-x-1") == []
    assert _post_cli(b, "rev-x-1", "cc-x-1", "--force") == 0
    assert _dms(b, "cc-x-1") == ["late finding"]


def test_dm_to_an_agent_still_working_or_to_the_chief_posts(tmp_path: Path) -> None:
    b = tmp_path / "bus"
    b.mkdir()
    bus.post(b, "cc-x-1", "chief", "done", "round 1 finished")
    bus.post(b, "cc-x-1", "chief", "info", "round 2 started")  # posting again = alive again
    bus.post(b, "cc-y-1", "chief", "question", "which design?")
    for hub in ("chief", "all"):  # never refused, even after the hub itself posted done
        bus.post(b, hub, "cc-y-1", "done", "hub done")
        assert _post_cli(b, "cc-y-1", hub) == 0
    assert _post_cli(b, "rev-x-1", "cc-x-1") == 0
    assert _post_cli(b, "chief", "cc-y-1") == 0
    assert _post_cli(b, "cc-x-1", "chief") == 0
    assert _post_cli(b, "rev-z-1", "cc-never-posted") == 0
    assert _dms(b, "cc-y-1") == ["hub done", "hub done", "late finding"]


# --- brief caps + board gc + supersede hint ----------------------------------


def _listed_ids(text: str) -> list[str]:
    return [line.split("] ")[0].strip("[ ") for line in text.splitlines() if line.startswith("  [")]


def test_brief_caps_three_per_job_and_twelve_total_without_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    owner: dict[str, str] = {}
    for j in range(5):
        for k in range(4):  # interleaved ages: k=0 is the newest post of every job
            m = _post_at(gdir, 1 + k * 5 + j * 0.1, "gotchas", f"job{j} finding number {k}", importance=3, job=f"job{j}")
            owner[str(m["id"])] = f"job{j}"
    text = bus.brief(context=([], set()), now=_NOW)
    ids = _listed_ids(text)
    counts = Counter(owner[i] for i in ids)
    assert len(ids) == 12, "no changed file: 12 posts total"
    assert max(counts.values()) == 3 and set(counts) == set(owner.values()), "3 per job, every job present"
    assert "Top 12 posts" in text


def test_brief_top_flag_still_overrides_the_computed_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    for j in range(5):
        for k in range(4):
            _post_at(gdir, 1 + k * 5 + j * 0.1, "gotchas", f"job{j} finding number {k}", importance=3, job=f"job{j}")
    assert bus.main(["--bus", str(gdir), "board", "brief", "--top", "6"]) == 0
    assert len(_listed_ids(capsys.readouterr().out)) == 6


def test_brief_caps_one_crowding_job_when_a_file_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    for k in range(10):
        _post_at(gdir, 1 + k, "gotchas", f"crowded finding number {k}", importance=3, job="20261010-edge-engine")
    text = bus.brief(context=(["vibe_quant/liquidation/apply.py"], set()), now=_NOW)
    assert len(_listed_ids(text)) == 3, "a changed file keeps the 30 cap but not 10 posts of one job"
    assert "Top 3 posts" in text


def test_brief_counts_posts_without_a_job_as_one_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    for k in range(6):
        _post_at(gdir, 1 + k, "gotchas", f"unowned finding number {k}", importance=3)  # no job key
    text = bus.brief(context=(["vibe_quant/liquidation/apply.py"], set()), now=_NOW)
    assert len(_listed_ids(text)) == 3, "job-less posts share the '-' bucket"


def test_board_gc_hides_only_old_uncited_importance_one_posts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    stale = _post_at(gdir, 20, "chat", "stale chatter nobody ever cited", importance=1)
    important = _post_at(gdir, 20, "chat", "important chatter to keep", importance=2)
    cited_ref = _post_at(gdir, 20, "chat", "cited by a reply", importance=1)
    cited_body = _post_at(gdir, 20, "chat", "cited inside another body", importance=1)
    young = _post_at(gdir, 13, "chat", "thirteen day old chatter", importance=1)
    edge_kept = _post_at(gdir, 14, "chat", "exactly fourteen days old", importance=1)
    edge_hidden = _post_at(gdir, 14.1, "chat", "just over fourteen days old", importance=1)
    _post_at(gdir, 0, "chat", "confirmed reply", ref=str(cited_ref["id"]))
    _post_at(gdir, 0, "chat", f"see {cited_body['id']} for details", importance=1)
    hidden_ids, visible = bus.gc(gdir, "chief", days=14, now=_NOW)
    assert sorted(hidden_ids) == sorted([str(stale["id"]), str(edge_hidden["id"])])
    hidden = bus.hidden_posts(bus._all(gdir))
    assert hidden[str(stale["id"])] == "retired: gc: older than 14d, importance 1, never cited"
    for kept in (important, cited_ref, cited_body, young, edge_kept):
        assert str(kept["id"]) not in hidden
    # gc's own retire events are not visible posts, and the log keeps the post for --all
    assert visible == 9  # 7 candidates + the reply + the post citing cited_body
    hits = bus.board_search(gdir, topics=["chat"], include_hidden=True)
    assert any(m["id"] == stale["id"] for m in hits)


def test_board_gc_cli_reports_dry_run_and_a_second_run_hides_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    now = time.time()
    stale = _post_at(gdir, 20, "chat", "stale chatter nobody cited", importance=1, now=now)
    _post_at(gdir, 20, "chat", "important chatter to keep", importance=2, now=now)
    before = _bytes(gdir)
    assert bus.main(["--bus", str(gdir), "board", "gc", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "gc: would hide 1 of 2"
    assert out.splitlines()[1] == str(stale["id"]) and _bytes(gdir) == before
    assert bus.main(["--bus", str(gdir), "board", "gc"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "gc: hid 1 of 2 visible posts"
    assert bus.main(["--bus", str(gdir), "board", "gc"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "gc: hid 0 of 1 visible posts"


def test_board_post_hints_at_a_near_duplicate_and_stays_silent_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    old = bus.post(gdir, "w", "board", "info", "flush the liquidation cache before backtesting runs", topic="gotchas")
    capsys.readouterr()
    assert bus.main([
        "--bus", str(gdir), "--as", "w2", "board", "post", "--topic", "gotchas",
        "--body", "flush the liquidation cache before backtesting after edits",
    ]) == 0
    err = capsys.readouterr().err
    assert f"hint: similar to [{old['id']}]" in err and f"--supersedes {old['id']}" in err
    assert "flush the liquidation cache" in err
    # an unrelated body on the same topic: no hint
    assert bus.main([
        "--bus", str(gdir), "--as", "w2", "board", "post", "--topic", "gotchas",
        "--body", "deploy dashboard widgets on friday",
    ]) == 0
    assert capsys.readouterr().err == ""
    # --supersedes already says it: no hint, and at most 3 hints otherwise
    assert bus.main([
        "--bus", str(gdir), "--as", "w2", "board", "post", "--topic", "gotchas", "--supersedes", str(old["id"]),
        "--body", "flush the liquidation cache before backtesting after edits",
    ]) == 0
    assert capsys.readouterr().err == ""
    for k in range(4):
        bus.post(gdir, "w", "board", "info", f"flush liquidation cache before backtesting variant {k}", topic="gotchas")
    capsys.readouterr()
    assert bus.main([
        "--bus", str(gdir), "--as", "w2", "board", "post", "--topic", "gotchas",
        "--body", "flush the liquidation cache before backtesting after edits",
    ]) == 0
    assert capsys.readouterr().err.count("hint: similar to") == 3


def test_board_post_hints_at_a_global_post_from_a_fresh_job_bus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The persistent board is where duplicates pile up: a new job's bus has none of it."""
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    old = bus.post(gdir, "w", "board", "info", "flush liquidation cache before backtesting runs", topic="gotchas")
    capsys.readouterr()
    job_bus = tmp_path / "fresh-job" / "bus"
    assert bus.main([
        "--bus", str(job_bus), "--as", "w2", "board", "post", "--topic", "gotchas",
        "--body", "flush liquidation cache before backtesting edits",
    ]) == 0
    err = capsys.readouterr().err
    assert err.count("hint: similar to") == 1  # the new post's own global mirror is excluded
    assert f"hint: similar to [{old['id']}]" in err


def test_board_brief_top_zero_lists_no_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    bus.post(gdir, "w", "board", "info", "flush liquidation cache before backtesting runs", topic="gotchas")
    capsys.readouterr()
    assert bus.main(["--bus", str(gdir), "board", "brief", "--top", "0"]) == 0
    assert not [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("  [")]


def test_board_retire_names_the_replies_it_hides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reply can hold the still-valid rule; retiring its parent hides it too (job 20261029-swarm-retro)."""
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    parent = bus.post(gdir, "w", "board", "info", "RULE: old tool behaviour", topic="gotchas")
    reply = bus.post(gdir, "w2", "board", "info", "RULE: the new behaviour", ref=str(parent["id"]), topic="gotchas")
    capsys.readouterr()
    assert bus.main(["--bus", str(gdir), "--as", "chief", "board", "retire", "--ref", str(parent["id"]), "--reason", "stale"]) == 0
    err = capsys.readouterr().err
    assert f"also hides 1 reply: [{reply['id']}]" in err and "repost" in err
    lone = bus.post(gdir, "w", "board", "info", "RULE: alone", topic="gotchas")
    capsys.readouterr()
    assert bus.main(["--bus", str(gdir), "--as", "chief", "board", "retire", "--ref", str(lone["id"]), "--reason", "stale"]) == 0
    assert "also hides" not in capsys.readouterr().err


def test_board_search_takes_query_and_bare_words_as_grep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # makers typed `board search --query X` and `board search X Y` (nbar-t2-2, costs-t4-1)
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    _seed_search_board(gdir)
    capsys.readouterr()
    assert bus.main(["--as", "x", "board", "search", "--global", "--grep", "tiers"]) == 0
    want = capsys.readouterr().out
    assert "(no matching board posts)" not in want
    assert bus.main(["--as", "x", "board", "search", "--global", "--query", "tiers"]) == 0
    assert capsys.readouterr().out == want
    assert bus.main(["--as", "x", "board", "search", "--global", "tiers"]) == 0
    assert capsys.readouterr().out == want


# ------------------------------------------------------- the main checkout root: one git spawn


def _bus_copy_in_repo(tmp_path: Path, name: str) -> Path:
    """A throwaway git repo holding `scripts/agents/bus.py`; returns that path."""
    repo = tmp_path / name
    script = repo / "scripts" / "agents" / "bus.py"
    script.parent.mkdir(parents=True)
    script.write_text("# stand-in\n")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, env=env)
    return script


def test_the_main_checkout_root_costs_one_git_spawn_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bus command asks `global_board_dir()` from several places; each used to be a
    `git rev-parse --git-common-dir` spawn (25-40 ms on a loaded host)."""
    script = _bus_copy_in_repo(tmp_path, "repo-a")
    spy_bin = tmp_path / "spy-bin"
    spy_bin.mkdir()
    log = tmp_path / "git-calls.log"
    real = shutil.which("git")
    assert real is not None
    (spy_bin / "git").write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{log}"\nexec {real} "$@"\n')
    (spy_bin / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{spy_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("SWARM_BOARD")  # the autouse fixture pins it to a tmp board
    monkeypatch.setattr(bus, "__file__", str(script))

    boards = {bus.global_board_dir() for _ in range(3)}

    assert boards == {(script.parents[2] / "docs" / "orchestration" / "board").resolve()}
    assert len(log.read_text().splitlines()) == 1


def test_the_main_checkout_root_follows_the_script_it_runs_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _bus_copy_in_repo(tmp_path, "repo-a")
    second = _bus_copy_in_repo(tmp_path, "repo-b")
    monkeypatch.setattr(bus, "__file__", str(first))
    assert bus._repo_root() == first.parents[2].resolve()
    monkeypatch.setattr(bus, "__file__", str(second))
    assert bus._repo_root() == second.parents[2].resolve()
    monkeypatch.setattr(bus, "__file__", str(first))
    assert bus._repo_root() == first.parents[2].resolve()


def test_a_failed_root_lookup_is_asked_again_not_remembered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    not_a_repo = tmp_path / "plain" / "scripts" / "agents"
    not_a_repo.mkdir(parents=True)
    monkeypatch.setattr(bus, "__file__", str(not_a_repo / "bus.py"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    with pytest.raises(subprocess.CalledProcessError):
        bus._repo_root()
    subprocess.run(["git", "-C", str(tmp_path / "plain"), "init", "-q"], check=True)
    assert bus._repo_root() == (tmp_path / "plain").resolve()
