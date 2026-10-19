"""scripts/agents/bus.py — swarm message bus (agents <-> agents <-> chief)."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
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


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    monkeypatch.setattr(bus, "POLL_S", 0.05)
    # never let a test write the real committed board
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path_factory.mktemp("global_board")))
    # ...or discover real job buses / inherit a real identity from the caller's env
    monkeypatch.delenv("SWARM_JOBS", raising=False)
    monkeypatch.delenv("SWARM_BUS", raising=False)
    monkeypatch.delenv("SWARM_AGENT", raising=False)


def test_direct_and_broadcast_routing(tmp_path: Path) -> None:
    bus.post(tmp_path, "sa", "sb", "info", "hi sb")
    bus.post(tmp_path, "sa", "all", "claim", "editing pipeline.py")
    bus.post(tmp_path, "sa", "chief", "question", "for chief only")
    sb = [m["body"] for m in bus.inbox(tmp_path, "sb")]
    chief = [m["body"] for m in bus.inbox(tmp_path, "chief")]
    sa = [m["body"] for m in bus.inbox(tmp_path, "sa")]
    assert sb == ["hi sb", "editing pipeline.py"]
    assert chief == ["editing pipeline.py", "for chief only"]
    assert sa == []  # never your own messages, even broadcasts


def test_cursor_advances_and_peek_does_not(tmp_path: Path) -> None:
    bus.post(tmp_path, "chief", "sa", "info", "one")
    assert len(bus.inbox(tmp_path, "sa", peek=True)) == 1
    assert len(bus.inbox(tmp_path, "sa")) == 1
    assert bus.inbox(tmp_path, "sa") == []
    bus.post(tmp_path, "chief", "sa", "info", "two")
    assert [m["body"] for m in bus.inbox(tmp_path, "sa")] == ["two"]


def test_ask_returns_the_answer_that_refs_the_question(tmp_path: Path) -> None:
    def chief() -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            for m in bus.inbox(tmp_path, "chief"):
                if m["kind"] == "question":
                    bus.post(tmp_path, "chief", "all", "info", "unrelated noise")
                    bus.post(tmp_path, "chief", str(m["from"]), "answer", "use 2*period", ref=str(m["id"]))
                    return
            time.sleep(0.02)

    t = threading.Thread(target=chief)
    t.start()
    ans = bus.ask(tmp_path, "sd", "chief", "which lookback?", timeout=5)
    t.join()
    assert ans is not None and ans["body"] == "use 2*period" and ans["kind"] == "answer"


def test_ask_times_out_with_none(tmp_path: Path) -> None:
    assert bus.ask(tmp_path, "sd", "chief", "anyone?", timeout=0.2) is None


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


def test_cli_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert bus.main(["--bus", str(tmp_path), "--as", "sa", "post", "--to", "chief", "--kind", "blocker", "--body", "need data"]) == 0
    assert bus.main(["--bus", str(tmp_path), "--as", "chief", "inbox"]) == 0
    out = capsys.readouterr().out
    assert "sa -> chief (blocker): need data" in out


# --- message board ---------------------------------------------------------


def test_board_posts_are_visible_to_everyone_with_own_cursors(tmp_path: Path) -> None:
    a = bus.post(tmp_path, "sa", "board", "claim", "touching pipeline.py:_run", topic="#Design")
    bus.post(tmp_path, "sb", "board", "info", "KAMA port exact on 400 bars", topic="findings")
    assert a["topic"] == "design"  # normalised
    assert [m["body"] for m in bus.board_read(tmp_path, "sc")] == [
        "touching pipeline.py:_run",
        "KAMA port exact on 400 bars",
    ]
    assert bus.board_read(tmp_path, "sc") == []  # cursor advanced
    assert [m["body"] for m in bus.board_read(tmp_path, "sd", topic="findings")] == ["KAMA port exact on 400 bars"]
    assert len(bus.board_read(tmp_path, "sc", everything=True)) == 2  # full history on demand
    # board posts are not DMs: nobody's inbox fills with them
    assert bus.inbox(tmp_path, "sc") == []


def test_board_post_requires_topic(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        bus.post(tmp_path, "sa", "board", "info", "no topic")


def test_reply_threads_on_board_and_notifies_the_poster(tmp_path: Path) -> None:
    root = bus.post(tmp_path, "sa", "board", "question", "scipy filter exact for min?", topic="design")
    r1 = bus.reply(tmp_path, "se", str(root["id"]), "yes, used in trend_ensemble")
    r2 = bus.reply(tmp_path, "chief", str(r1["id"]), "confirmed, keep the NaN comment")
    assert r1["to"] == "board" and r1["topic"] == "design" and r1["ref"] == root["id"]
    tree = [(d, m["from"]) for d, m in bus.thread(tmp_path, str(r2["id"]))]
    assert tree == [(0, "sa"), (1, "se"), (2, "chief")]
    # the original poster hears about replies to their post in their inbox
    assert [m["from"] for m in bus.inbox(tmp_path, "sa")] == ["se"]


def test_reply_to_a_dm_question_answers_the_asker(tmp_path: Path) -> None:
    q = bus.post(tmp_path, "sd", "chief", "question", "which lookback?")
    a = bus.reply(tmp_path, "chief", str(q["id"]), "2*period")
    assert a["to"] == "sd" and a["kind"] == "answer" and a["ref"] == q["id"]


def test_topics_and_render(tmp_path: Path) -> None:
    root = bus.post(tmp_path, "sa", "board", "info", "plan: preflight bars", topic="design")
    bus.reply(tmp_path, "sb", str(root["id"]), "fine by me")
    bus.post(tmp_path, "sc", "board", "blocker", "orval regen changed 180 files", topic="blockers")
    t = bus.topics(tmp_path)
    assert {k: v["posts"] for k, v in t.items()} == {"design": 2, "blockers": 1}
    md = bus.render(tmp_path)
    assert "## #design" in md and "## #blockers" in md
    assert "- `" in md and "  - `" in md  # reply indented under its root
    assert "**blocker**" in md


# --- persistent (cross-job) board --------------------------------------------


def test_persistent_topics_mirror_to_global_board_with_job_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    job1 = tmp_path / "20261009-job-a" / "bus"
    job1.mkdir(parents=True)
    bus.post(job1, "sd", "board", "info", "pandas rolling sum is not bit-exact vs numpy cumsum", topic="gotchas")
    bus.post(job1, "sa", "board", "claim", "editing pipeline.py", topic="design")  # job-only topic
    g = bus.board_read(gdir, "anyone", everything=True)
    assert [(m["topic"], m["job"]) for m in g] == [("gotchas", "20261009-job-a")]
    assert "@20261009-job-a" in bus.fmt(g[0])


def test_future_job_sees_previous_jobs_knowledge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    for k in range(5):
        jb = tmp_path / f"job{k}" / "bus"
        jb.mkdir(parents=True)
        bus.post(jb, f"w{k}", "board", "info", f"finding {k}", topic="findings")
    # a brand-new agent in a later job reads the last N cross-job posts
    recent = bus.board_read(gdir, "newcomer", topic="findings", recent=3)
    assert [m["body"] for m in recent] == ["finding 2", "finding 3", "finding 4"]
    assert "## #findings" in bus.render(gdir)


def _seed_search_board(b: Path) -> None:
    bus.post(b, "cc-a", "board", "info", "nt_runner.py:1337 swallows DataUnavailableError; preflight first", topic="gotchas")
    bus.post(b, "rev-b", "board", "info", "vibe_quant/liquidation/scan.py: funding cut at end_ns-1ms", topic="findings")
    bus.post(b, "impl-c", "board", "claim", "editing vibe_quant/liquidation/tiers.py", topic="design")
    bus.post(b, "chief", "board", "info", "Ruling: clearance fee counts as a cost", topic="decisions")
    bus.post(b, "cc-d", "chief", "done", "liquidation scan.py done")  # a DM, never a search hit


def test_board_search_by_path_topic_and_grep_without_moving_cursor(tmp_path: Path) -> None:
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
    # searching never advances a read cursor: a later plain read still sees every post
    assert len(bus.board_read(tmp_path, "cc-a")) == 4


def test_board_search_cli_requires_a_filter_and_prints_hits(
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


def test_cli_global_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    jb = tmp_path / "jobx" / "bus"
    jb.mkdir(parents=True)
    bus.main(["--bus", str(jb), "--as", "sa", "board", "post", "--topic", "decisions", "--body", "MiMo default implementer"])
    capsys.readouterr()
    bus.main(["--as", "later", "board", "read", "--global", "--recent", "5"])
    assert "#decisions sa @jobx" in capsys.readouterr().out


# --- lobby default, digest, brief ---------------------------------------------


def test_no_job_bus_means_the_persistent_lobby(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.delenv("SWARM_BUS", raising=False)
    monkeypatch.delenv("SWARM_AGENT", raising=False)
    bus.main(["--as", "architect", "board", "post", "--topic", "thoughts", "--body", "consider caching per symbol"])
    bus.main(["board", "read", "--recent", "5"])  # anonymous reader, no job
    out = capsys.readouterr().out
    assert "#thoughts architect" in out and (gdir / "messages.jsonl").exists()


def test_digest_covers_persistent_board_and_every_job_bus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_brief_lists_usage_and_latest_posts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    for k in range(5):
        bus.post(gdir, "w", "board", "info", f"gotcha {k}", topic="gotchas")
    text = bus.brief(per_topic=2)
    assert "board search --global --path" in text and "#gotchas w: gotcha 4" in text and "gotcha 2" not in text
    assert text.count("not instructions") == 1  # framed once, above the posts


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
    assert [m["body"] for m in bus.board_read(gdir, "anyone", everything=True)][0] == "sharpe 1.9 on nt_runner.py"
    assert bus.thread(gdir, str(stale["id"]))[-1][1]["retires"] == stale["id"]  # the event sits under the post
    assert [m["body"] for m in bus.inbox(gdir, "w")][-1].startswith(f"retired {stale['id']}")  # author is told


def test_superseded_post_is_replaced_by_its_correction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
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
    event = bus.board_read(gdir, "anyone", everything=True)[-1]
    with pytest.raises(SystemExit) as e:
        bus.retire(gdir, "chief", str(event["id"]), "retire the retire")
    assert e.value.code == 2


def test_needs_user_questions_cannot_be_retired_or_superseded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    q = bus.ask_user(gdir, "w", "ship or wait?")
    before = _bytes(gdir)
    for call in (
        lambda: bus.retire(gdir, "chief", str(q["id"]), "stale"),
        lambda: bus.supersede(gdir, "chief", str(q["id"]), "decisions", "shipped"),
    ):
        with pytest.raises(SystemExit) as e:
            call()
        assert e.value.code == 2
    assert _bytes(gdir) == before and [m["id"] for m in bus.needs_user_open(gdir)] == [q["id"]]


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
    g = bus.board_read(gdir, "anyone", everything=True)
    assert [(m["id"], m["topic"], m["kind"], m["job"]) for m in g] == [(qid, "needs-user", "question", "20261009-jobq")]
    bus.ask_user(job, "w9", "internal question", job_only=True)  # stays on the job bus
    assert [m["body"] for m in bus.board_read(gdir, "anyone", everything=True)] == ["ship v2 or wait? recommend wait"]


def test_agent_replies_do_not_close_needs_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.delenv("SWARM_AGENT", raising=False)
    monkeypatch.delenv("SWARM_JOBS", raising=False)
    q = bus.ask_user(gdir, "w1", "ship or wait?")
    bus.reply(gdir, "chief", str(q["id"]), "recommend ship", kind="answer")
    bus.reply(gdir, "chief", str(q["id"]), "done: shipped", kind="done")
    assert [m["id"] for m in bus.needs_user_open(gdir)] == [q["id"]]  # only the user decides
    bus.main(["--bus", str(gdir), "needs-user", "open"])
    out = capsys.readouterr().out
    assert "ship or wait?" in out and "old)" in out  # listing shows age


def test_needs_user_answer_closes_with_user_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.delenv("SWARM_AGENT", raising=False)
    monkeypatch.delenv("SWARM_JOBS", raising=False)
    q1 = bus.ask_user(gdir, "w1", "ship or wait?")
    q2 = bus.ask_user(gdir, "w2", "rename module?")
    bus.main(["--bus", str(gdir), "--as", "chief", "needs-user", "answer", "--ref", str(q1["id"]), "--body", "wait for holdout", "--user"])
    capsys.readouterr()
    assert [m["id"] for m in bus.needs_user_open(gdir)] == [q2["id"]]
    ans = bus.thread(gdir, str(q1["id"]))[-1][1]
    assert ans["from"] == "user" and ans["kind"] == "answer"  # from is always the user
    assert ans["user_answer"] is True and ans["by"] == "chief"  # audit trail: who relayed it
    bus.main(["--bus", str(gdir), "needs-user", "answer", "--ref", str(q2["id"]), "--body", "rename it", "--user"])
    capsys.readouterr()
    assert bus.thread(gdir, str(q2["id"]))[-1][1]["by"] == "user"  # no --as, no $SWARM_AGENT
    monkeypatch.setenv("SWARM_AGENT", "w9")
    q3 = bus.ask_user(gdir, "w3", "drop worst mode?")
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
    text = bus.brief(per_topic=1)
    assert "OPEN DECISIONS FOR THE USER (6):" in text
    assert text.index("OPEN DECISIONS FOR THE USER") < text.index("Latest posts")  # before Latest posts
    section = text[text.index("OPEN DECISIONS FOR THE USER") : text.index("Latest posts")]
    assert section.count("[OPEN]") == 5  # max 5, oldest first
    assert "decision 0?" in section and "decision 5?" not in section
    for q in qs:
        bus.needs_user_answer(gdir, "user", str(q["id"]), "decided")
    text = bus.brief(per_topic=1)
    assert "OPEN DECISIONS" not in text and "[OPEN]" not in text  # omitted when none


def test_needs_user_ordering_is_oldest_first_despite_file_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.delenv("SWARM_JOBS", raising=False)
    for k, (ts, body) in enumerate([  # appended out of ts order on purpose
        ("2026-10-09T12:00:03", "newest question?"),
        ("2026-10-09T12:00:01", "oldest question?"),
        ("2026-10-09T12:00:02", "middle question?"),
    ]):
        bus._append(gdir, {"id": f"q{k}", "ts": ts, "from": "w", "to": "board", "kind": "question", "topic": "needs-user", "body": body})
    order = ["oldest question?", "middle question?", "newest question?"]
    assert [m["body"] for m in bus.needs_user_open(gdir)] == order
    section = bus.brief(per_topic=3)
    section = section[section.index("OPEN DECISIONS FOR THE USER") :]
    assert [section.index(b) for b in order] == sorted(section.index(b) for b in order)  # brief agrees


def test_digest_shows_open_needs_user_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    gdir = repo / "docs" / "orchestration" / "board"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    bus.ask_user(gdir, "w1", "ship or wait?")
    bus.main(["--as", "chief", "digest"])
    out = capsys.readouterr().out
    assert "OPEN DECISIONS FOR THE USER (1):" in out
    assert out.index("OPEN DECISIONS FOR THE USER") < out.index("== persistent board")  # at the top
    assert "[OPEN]" in out and "ship or wait?" in out


def test_digest_lists_job_only_needs_user_items_labelled_with_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path))
    job = tmp_path / "20261009-jobq" / "bus"
    job.mkdir(parents=True)
    bus.ask_user(gdir, "w1", "global question?")
    bus.ask_user(job, "w2", "job-only question?", job_only=True)
    bus.ask_user(job, "w3", "mirrored question?")  # on both buses, must count once
    bus.main(["--as", "chief", "digest"])
    out = capsys.readouterr().out
    assert "OPEN DECISIONS FOR THE USER (3):" in out
    assert "global question?" in out and "job-only question?" in out and "mirrored question?" in out
    assert "@20261009-jobq" in out  # job-only items carry their job


def test_digest_does_not_crash_on_shallow_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    shallow = Path("/tmp") / f"vq-bus-shallow-{os.getpid()}"  # parents[2] does not exist
    monkeypatch.setenv("SWARM_BOARD", str(shallow))
    monkeypatch.delenv("SWARM_JOBS", raising=False)
    try:
        # the skip rule itself: scratch board + no SWARM_JOBS -> no job buses, never real data/swarm
        assert bus._job_buses() == []
        assert bus.main(["--as", "chief", "digest"]) == 0
        assert "(nothing new on any board)" in capsys.readouterr().out
    finally:
        shutil.rmtree(shallow, ignore_errors=True)


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


def _tail_proc(bus_dir: Path, timeout: float, durable: bool = True) -> subprocess.Popen[str]:
    args = [
        sys.executable, str(_BUS_PATH), "--bus", str(bus_dir), "--as", "chief",
        "tail", "--to", "chief", "--timeout", str(timeout),
    ]
    if durable:
        args.append("--durable")
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _wait_for(path: Path, deadline_s: float = 15.0) -> None:
    deadline = time.monotonic() + deadline_s
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists(), f"{path} never appeared"


def test_durable_tail_skips_history_and_replays_across_a_kill(tmp_path: Path) -> None:
    bus.post(tmp_path, "w", "chief", "info", "old history")  # before the watcher: must not replay
    first = _tail_proc(tmp_path, 30)
    try:
        _wait_for(bus._tail_cursor_path(tmp_path, "chief", "chief"))  # baseline persisted at start
    finally:
        first.kill()
        first.wait(timeout=10)
    for i in range(3):
        bus.post(tmp_path, "w", "chief", "info", f"m{i}")  # arrive while the watcher is dead
    second = _tail_proc(tmp_path, 1.0)
    out, _ = second.communicate(timeout=30)
    bodies = [ln.split("): ", 1)[1] for ln in out.splitlines() if "): " in ln]
    assert bodies == ["m0", "m1", "m2"]  # each undelivered message exactly once
    assert "old history" not in out
    third = _tail_proc(tmp_path, 1.0)
    out3, _ = third.communicate(timeout=30)
    assert out3.strip() == ""  # nothing new


def test_durable_tail_cursor_not_advanced_before_printing(tmp_path: Path) -> None:
    for i in range(3):
        bus.post(tmp_path, "w", "chief", "info", f"m{i}")
    seen: list[str] = []

    def emit(line: str) -> None:
        seen.append(line)
        raise RuntimeError("crash after the first delivered line")

    with pytest.raises(RuntimeError):
        bus._durable_tail(tmp_path, "chief", "chief", True, 1.0, emit=emit)
    assert len(seen) == 1
    # the cursor still points at the start: the two unprinted messages must replay
    assert bus._read_cursor(bus._tail_cursor_path(tmp_path, "chief", "chief")) == 0


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


def test_pending_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bus.post(tmp_path, "chief", "sa", "question", "which lookback?")
    assert bus.main(["--bus", str(tmp_path), "--as", "sa", "pending"]) == 0
    out = capsys.readouterr().out
    assert "which lookback?" in out and "old)" in out
    assert [m["body"] for m in bus.pending(tmp_path, "sa")]  # still unread (pending never advances)


def test_digest_shows_pending_questions_first(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    g = bus.global_board_dir()  # the autouse fixture points SWARM_BOARD at a tmp board
    bus.post(g, "w", "chief", "question", "still open?")
    assert bus.main(["--as", "chief", "digest"]) == 0
    out = capsys.readouterr().out
    assert "PENDING QUESTIONS FOR chief (1):" in out and "still open?" in out
    assert out.index("PENDING QUESTIONS FOR chief") < out.index("== persistent board")


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


def test_post_cli_accepts_client_id(tmp_path: Path) -> None:
    for _ in range(2):
        assert bus.main(
            ["--bus", str(tmp_path), "--as", "sa", "post", "--to", "chief", "--body", "x", "--client-id", "z"]
        ) == 0
    assert len((tmp_path / "messages.jsonl").read_text().splitlines()) == 1


# --- regression: findings from the harden review ------------------------------


def test_durable_baseline_is_the_end_of_the_last_complete_line(tmp_path: Path) -> None:
    """A writer mid-append (partial last line) must not push the first-run baseline past it."""
    bus.post(tmp_path, "w", "chief", "info", "old history")
    path = tmp_path / "messages.jsonl"
    complete = path.stat().st_size
    full = json.dumps({"id": "p1", "from": "w", "to": "chief", "kind": "info", "body": "half-written"}) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(full[:20])  # the writer is mid-append: no newline yet
    sent: list[str] = []
    bus._durable_tail(tmp_path, "chief", "chief", False, 0, emit=sent.append)  # timeout 0: baseline only
    assert bus._read_cursor(bus._tail_cursor_path(tmp_path, "chief", "chief")) == complete
    with path.open("a", encoding="utf-8") as fh:
        fh.write(full[20:])  # the append finishes
    bus._durable_tail(tmp_path, "chief", "chief", False, 0.3, emit=sent.append)
    assert [ln.split("): ", 1)[1] for ln in sent] == ["half-written"]  # delivered, not lost


def test_overlapping_durable_tails_do_not_both_deliver(tmp_path: Path) -> None:
    import fcntl

    bus.post(tmp_path, "w", "chief", "info", "m0")
    first = bus._tail_lock_path(tmp_path, "chief", "chief")
    first.parent.mkdir(parents=True, exist_ok=True)
    with first.open("a") as held:  # a live tail for the same (as, to) holds this lock
        fcntl.flock(held, fcntl.LOCK_EX)
        sent: list[str] = []
        with pytest.raises(SystemExit) as exc:
            bus._durable_tail(tmp_path, "chief", "chief", True, 0.3, emit=sent.append)
    assert "already running" in str(exc.value) and str(first) in str(exc.value)
    assert sent == []  # the second tail delivered nothing
    bus._durable_tail(tmp_path, "chief", "chief", True, 0.3, emit=sent.append)  # lock released: runs
    assert len(sent) == 1


def test_overlapping_tail_for_another_cursor_is_not_blocked(tmp_path: Path) -> None:
    import fcntl

    other = bus._tail_lock_path(tmp_path, "chief", "sa")
    other.parent.mkdir(parents=True, exist_ok=True)
    with other.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        bus._durable_tail(tmp_path, "chief", "chief", True, 0, emit=lambda line: None)


def test_ask_retry_with_same_client_id_returns_the_existing_answer(tmp_path: Path) -> None:
    q = bus.post(tmp_path, "sa", "chief", "question", "which lookback?", client_id="ask-1")
    bus.reply(tmp_path, "chief", str(q["id"]), "2*period")  # answered before the retry
    ans = bus.ask(tmp_path, "sa", "chief", "which lookback?", timeout=0.3, client_id="ask-1")
    assert ans is not None and ans["body"] == "2*period"
    assert len((tmp_path / "messages.jsonl").read_text().splitlines()) == 2  # no duplicate question


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


def test_durable_tail_delivers_only_what_is_addressed_to_it(tmp_path: Path) -> None:
    bus.post(tmp_path, "w", "chief", "info", "for chief")
    bus.post(tmp_path, "w", "sa", "info", "for sa")
    bus.post(tmp_path, "w", "all", "info", "for all")
    bus.post(tmp_path, "w", "board", "info", "board post", topic="chat")
    sent: list[str] = []
    bus._durable_tail(tmp_path, "chief", "chief", True, 0.3, emit=sent.append)
    assert [ln.split("): ", 1)[1] for ln in sent] == ["for chief", "for all", "board post"]


def test_foreign_reply_does_not_close_a_pending_question(tmp_path: Path) -> None:
    q = bus.post(tmp_path, "chief", "sa", "question", "which lookback?")
    bus.post(tmp_path, "other", "chief", "answer", "I think 20", ref=str(q["id"]))  # not sa's reply
    assert [m["body"] for m in bus.pending(tmp_path, "sa")] == ["which lookback?"]


def test_read_from_leaves_an_unfinished_line_for_later(tmp_path: Path) -> None:
    path = tmp_path / "messages.jsonl"
    path.write_text('{"id": "a", "from": "w", "to": "chief", "kind": "info", "body": "x"}\n{"id": "b"', encoding="utf-8")
    msgs, off = bus._read_from(tmp_path, 0)
    assert [m["id"] for m in msgs] == ["a"] and off == path.read_text().index("\n") + 1


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
    sent: list[str] = []
    bus._durable_tail(tmp_path, "chief", "chief", True, 0.3, emit=sent.append)
    return [ln.split("): ", 1)[1] for ln in sent]


def test_append_after_a_crashed_writer_fragment_is_still_delivered(tmp_path: Path) -> None:
    path = tmp_path / "messages.jsonl"
    old = json.dumps({"id": "a", "from": "w", "to": "chief", "kind": "info", "body": "before"}, ensure_ascii=False)
    # a writer died mid-record: a torn tail with no newline, cut inside a multibyte character
    path.write_bytes(old.encode() + b"\n" + b'{"id": "torn", "body": "\xe2\x82')
    stored = bus.post(tmp_path, "sa", "chief", "info", "after", client_id="c1")
    assert _delivered(tmp_path) == ["before", "after"]  # the new record did not get glued to the fragment
    assert bus.post(tmp_path, "sa", "chief", "info", "after", client_id="c1")["id"] == stored["id"]
    assert _delivered(tmp_path) == []  # the retry stored nothing new: the durable cursor has nothing to deliver
    assert len([ln for ln in path.read_bytes().split(b"\n") if ln]) == 3  # old, fragment, new


def test_client_id_retry_over_an_unterminated_complete_record_delivers_once(tmp_path: Path) -> None:
    path = tmp_path / "messages.jsonl"
    rec = {"id": "orig", "from": "sa", "to": "chief", "kind": "info", "body": "once", "client_id": "c1"}
    path.write_text(json.dumps(rec), encoding="utf-8")  # fully written, newline lost to a crash
    assert bus.post(tmp_path, "sa", "chief", "info", "once", client_id="c1")["id"] == "orig"
    assert _delivered(tmp_path) == ["once"]  # the acknowledged message is deliverable, exactly once
    assert path.read_bytes().endswith(b"\n")


def test_cursor_inside_a_torn_fragment_does_not_wedge_the_durable_tail(tmp_path: Path) -> None:
    path = tmp_path / "messages.jsonl"
    path.write_bytes(b'{"id":"torn","body":"123')  # a crashed writer's fragment; "123" is valid JSON by itself
    cpath = bus._tail_cursor_path(tmp_path, "chief", "chief")
    bus._write_cursor(cpath, 21)  # the cursor sits INSIDE the fragment (at the digits)
    bus.post(tmp_path, "sa", "chief", "info", "acknowledged")  # repairs the tail, then appends
    assert _delivered_resume(tmp_path) == ["acknowledged"]
    assert bus._read_cursor(cpath) == path.stat().st_size  # advanced past the junk
    assert _delivered_resume(tmp_path) == []  # a restart neither crashes nor repeats


def _delivered_resume(tmp_path: Path) -> list[str]:
    sent: list[str] = []
    bus._durable_tail(tmp_path, "chief", "chief", False, 0.3, emit=sent.append)
    return [ln.split("): ", 1)[1] for ln in sent]


@pytest.mark.parametrize("junk", [b"123", b"null", b'"str"', b"[1, 2]", b'{"id": "x"}', b'{"id": "x", "from": "w"}'])
def test_read_from_keeps_only_complete_message_records(tmp_path: Path, junk: bytes) -> None:
    good = json.dumps({"id": "g", "from": "w", "to": "chief", "kind": "info", "body": "ok"}).encode()
    (tmp_path / "messages.jsonl").write_bytes(junk + b"\n" + good + b"\n" + junk + b"\n")
    msgs, off = bus._read_from(tmp_path, 0)
    assert [m["id"] for m in msgs] == ["g"] and off == (tmp_path / "messages.jsonl").stat().st_size


@pytest.mark.parametrize("value", ["999999", "-5"])
def test_out_of_bounds_cursor_is_a_clear_error_on_every_reader(tmp_path: Path, value: str) -> None:
    bus.post(tmp_path, "chief", "sa", "info", "must-read")
    size = (tmp_path / "messages.jsonl").stat().st_size
    assert int(value) != size
    cases: dict[str, tuple[Path, Callable[[], object]]] = {
        "inbox": (bus._cursor_path(tmp_path, "sa"), lambda: bus.inbox(tmp_path, "sa")),
        "board": (bus._cursor_path(tmp_path, "sa.board._all"), lambda: bus.board_read(tmp_path, "sa")),
        "tail": (
            bus._tail_cursor_path(tmp_path, "sa", "sa"),
            lambda: bus._durable_tail(tmp_path, "sa", "sa", False, 0.2, emit=lambda line: None),
        ),
    }
    for name, (cpath, call) in cases.items():
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(value, encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            call()
        assert str(cpath) in str(exc.value) and value in str(exc.value), name
        assert cpath.read_text() == value  # never silently repaired or advanced


def test_cursor_exactly_at_eof_is_valid(tmp_path: Path) -> None:
    bus.post(tmp_path, "chief", "sa", "info", "seen")
    size = (tmp_path / "messages.jsonl").stat().st_size
    bus._set_cursor(tmp_path, "sa", size)
    assert bus.inbox(tmp_path, "sa") == []
    bus._write_cursor(bus._tail_cursor_path(tmp_path, "sa", "sa"), size)
    bus._durable_tail(tmp_path, "sa", "sa", False, 0.2, emit=lambda line: None)


def test_client_id_retry_over_an_incomplete_record_appends_a_deliverable_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "messages.jsonl"
    path.write_text('{"id":"bad","from":"sa","client_id":"retry"}\n', encoding="utf-8")  # no to/kind/body
    stored = bus.post(tmp_path, "sa", "chief", "info", "real", client_id="retry")
    assert stored["id"] != "bad"  # the junk record is not "already posted": no reader would deliver it
    assert [m["body"] for m in bus.inbox(tmp_path, "chief")] == ["real"]
    assert bus.inbox(tmp_path, "chief") == []  # exactly once
    assert bus.main(["--bus", str(tmp_path), "--as", "chief2", "tail", "--to", "chief", "--since-start", "--timeout", "0.2"]) == 0
    assert "real" in capsys.readouterr().out  # the CLI renders it (no KeyError)
    again = bus.post(tmp_path, "sa", "chief", "info", "real", client_id="retry")
    assert again["id"] == stored["id"]  # a real record still dedupes


def test_file_shrinking_under_a_running_durable_tail_is_an_explicit_error(tmp_path: Path) -> None:
    bus.post(tmp_path, "w", "chief", "info", "a long first message " * 5)
    bus.post(tmp_path, "w", "chief", "info", "a long second message " * 5)
    path = tmp_path / "messages.jsonl"
    old_size = path.stat().st_size
    sent: list[str] = []

    def emit(line: str) -> None:
        sent.append(line)
        if len(sent) == 2:  # a checkout/rebase shrinks the file while the tail follows
            path.write_text("", encoding="utf-8")
            bus.post(tmp_path, "w", "chief", "info", "short")  # acknowledged, but behind the held offset

    with pytest.raises(SystemExit) as exc:
        bus._durable_tail(tmp_path, "chief", "chief", True, 5.0, emit=emit)
    assert str(path) in str(exc.value) and str(old_size) in str(exc.value)
    assert not any("short" in ln for ln in sent)  # never silently skipped past, never reset


def test_read_from_refuses_an_offset_beyond_eof(tmp_path: Path) -> None:
    bus.post(tmp_path, "w", "chief", "info", "x")
    size = (tmp_path / "messages.jsonl").stat().st_size
    assert bus._read_from(tmp_path, size)[1] == size  # exactly EOF is fine
    with pytest.raises(SystemExit) as exc:
        bus._read_from(tmp_path, size + 1)
    assert str(size + 1) in str(exc.value)
    with pytest.raises(SystemExit):  # a vanished file is shorter than any positive offset too
        bus._read_from(tmp_path / "nowhere", 5)


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
        (["--as", "chief", "pending", "--global"], "old"),
        (["--as", "chief", "needs-user", "open"], "needs the user"),
        (["--as", "chief", "digest"], "needs the user"),
        (["--as", "chief", "digest", "--recent", "5"], "board post w/o topic"),
        (["--as", "chief", "tail", "--to", "chief", "--since-start", "--timeout", "0.2"], "old"),
        (["--as", "chief", "board", "render"], "board post w/o topic"),
        (["--as", "chief", "board", "topics"], "needs-user"),
        (["--as", "chief", "board", "brief"], "needs the user"),
        (["--as", "chief", "board", "read", "--all"], "board post w/o topic"),
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


def test_retry_over_a_record_without_ts_returns_it_and_it_stays_listable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    g = bus.global_board_dir()
    (g / "messages.jsonl").write_text(json.dumps(_NO_TS[0]) + "\n", encoding="utf-8")
    assert bus.post(g, "sa", "chief", "question", "old", client_id="retry")["id"] == "bad"
    assert [m["id"] for m in bus.pending(g, "chief")] == ["bad"]
    assert bus.main(["--bus", str(g), "--as", "chief", "pending"]) == 0
    assert "(? old)" in capsys.readouterr().out  # an unknown age renders as ?


def test_missing_ts_keeps_file_order_among_themselves(tmp_path: Path) -> None:
    g = bus.global_board_dir()
    recs = [{"id": f"q{i}", "from": "sa", "to": "chief", "kind": "question", "body": f"b{i}"} for i in range(3)]
    (g / "messages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    assert [m["id"] for m in bus.pending(g, "chief")] == ["q0", "q1", "q2"]


def test_reply_to_a_board_question_without_a_topic_is_always_stored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    g = bus.global_board_dir()
    (g / "messages.jsonl").write_text(json.dumps(_NO_TS[2] | {"id": "bq", "kind": "question"}) + "\n", encoding="utf-8")
    ans = bus.reply(g, "chief", "bq", "answered")  # must not exit "board posts need --topic"
    assert ans["ref"] == "bq" and ans["to"] == "board" and ans["topic"] == "general"
    assert [(d, m["body"]) for d, m in bus.thread(g, "bq")] == [(0, "board post w/o topic"), (1, "answered")]
    assert bus.main(["--bus", str(g), "--as", "chief", "reply", "--ref", "bq", "--body", "again"]) == 0
    assert "answered" in "".join(m["body"] for _, m in bus.thread(g, "bq"))
    capsys.readouterr()
    # a board question WITH a topic keeps it
    bus.post(g, "sa", "board", "question", "q2", topic="design")
    q2 = next(m for m in bus._all(g) if m["body"] == "q2")
    assert bus.reply(g, "chief", str(q2["id"]), "a2")["topic"] == "design"
    # a NEW board post still requires an explicit topic
    with pytest.raises(SystemExit):
        bus.post(g, "sa", "board", "info", "no topic")


# --- reply: the bus that HOLDS the parent (cross-job ids) ---------------------


def _make_job_bus(root: Path, name: str) -> Path:
    b = root / name / "bus"
    b.mkdir(parents=True)
    return b


def test_reply_finds_a_mirrored_post_on_the_global_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3/R8: an id `board search --global` printed replies on the global board, job-tagged."""
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


def test_reply_finds_a_dm_on_another_job_bus_and_answers_the_asker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4: an id that lives on another job bus (a DM) is answered there; the asker's inbox gets it."""
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    jobs = tmp_path / "jobs"
    monkeypatch.setenv("SWARM_JOBS", str(jobs))
    job_a = _make_job_bus(jobs, "20261015-job-a")
    job_b = _make_job_bus(jobs, "20261015-job-b")
    q = bus.post(job_b, "w2", "x", "question", "which lookback?")
    a = bus.reply(job_a, "x", str(q["id"]), "2*period")
    assert a["to"] == "w2" and a["kind"] == "answer" and a["ref"] == q["id"]
    assert "job" not in a  # the job tag is for replies landing on the global board only
    assert [m["body"] for m in bus._all(job_b)] == ["which lookback?", "2*period"]  # on the holding bus
    assert bus._all(job_a) == []  # not on the replier's bus
    assert bus._all(gdir) == []  # a DM is not a board post: no mirror
    assert [m["body"] for m in bus.inbox(job_b, "w2")] == ["2*period"]  # the asker reads it there


def test_reply_unknown_id_fails_with_the_same_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5: an id nowhere (even with job buses present) still exits with the unchanged text."""
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path / "global"))
    monkeypatch.setenv("SWARM_JOBS", str(tmp_path / "jobs"))
    _make_job_bus(tmp_path / "jobs", "20261015-job-a")
    with pytest.raises(SystemExit) as e:
        bus.reply(tmp_path, "x", "nosuchid", "?")
    assert str(e.value) == "bus: no message nosuchid"


def test_reply_is_idempotent_on_the_holding_bus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R6: a repeat reply with the same (sender, client_id) is stored once, on the holding bus."""
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


def test_reply_cli_from_a_job_bus_reaches_a_mirrored_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reported bug: `bus reply --ref <id from board search --global>` from a job bus."""
    gdir = tmp_path / "global"
    monkeypatch.setenv("SWARM_BOARD", str(gdir))
    jobs = tmp_path / "jobs"
    monkeypatch.setenv("SWARM_JOBS", str(jobs))
    job_a = _make_job_bus(jobs, "20261015-job-a")
    job_b = _make_job_bus(jobs, "20261015-job-b")
    root = bus.post(job_b, "w2", "board", "info", "a gotcha", topic="gotchas")
    argv = ["--bus", str(job_a), "--as", "x", "reply", "--ref", str(root["id"]), "--body", "confirmed"]
    assert bus.main(argv) == 0
    out = capsys.readouterr().out
    assert "#gotchas" in out and "confirmed" in out and "@20261015-job-a" in out
    assert [m["ref"] for m in bus._all(gdir) if m.get("ref")] == [root["id"]]


def test_locate_prefers_the_selected_bus_then_global_then_jobs_and_searches_none_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2: first hit wins in [selected, global, job buses...] and no directory is searched twice."""
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


def test_tail_kinds_emits_only_the_named_kinds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bus.post(tmp_path, "w", "chief", "info", "chatter")
    bus.post(tmp_path, "w", "chief", "question", "which lookback?")
    bus.post(tmp_path, "w", "chief", "done", "finished")
    bus.post(tmp_path, "w", "chief", "blocker", "need the catalog")
    argv = [
        "--bus", str(tmp_path), "--as", "chief", "tail", "--to", "chief",
        "--since-start", "--timeout", "0.2", "--kinds", "question,blocker",
    ]  # fmt: skip
    assert bus.main(argv) == 0
    out = capsys.readouterr().out
    assert "which lookback?" in out and "need the catalog" in out
    assert "chatter" not in out and "finished" not in out


def test_tail_without_kinds_emits_every_kind(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bus.post(tmp_path, "w", "chief", "info", "chatter")
    bus.post(tmp_path, "w", "chief", "done", "finished")
    argv = ["--bus", str(tmp_path), "--as", "chief", "tail", "--to", "chief", "--since-start", "--timeout", "0.2"]
    assert bus.main(argv) == 0
    out = capsys.readouterr().out
    assert "chatter" in out and "finished" in out


def test_tail_kinds_refuses_an_unknown_kind(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (["tail", "--kinds", "question,gossip"], ["tail", "--kinds", ","]):
        with pytest.raises(SystemExit) as exc:
            bus.main(["--bus", str(tmp_path), *argv])
        assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "gossip" in err and "--kinds needs at least one kind" in err


def test_durable_tail_kinds_filter_still_advances_the_cursor(tmp_path: Path) -> None:
    bus.post(tmp_path, "w", "chief", "info", "noise before")
    bus.post(tmp_path, "w", "chief", "question", "the one question")
    bus.post(tmp_path, "w", "chief", "info", "noise after")
    sent: list[str] = []
    bus._durable_tail(tmp_path, "chief", "chief", True, 0.3, kinds=frozenset({"question"}), emit=sent.append)
    assert [ln.split("): ", 1)[1] for ln in sent] == ["the one question"]
    # the cursor moved past the filtered messages: a restart does not replay them
    assert bus._read_cursor(bus._tail_cursor_path(tmp_path, "chief", "chief")) == (tmp_path / "messages.jsonl").stat().st_size
    bus._durable_tail(tmp_path, "chief", "chief", False, 0.3, kinds=frozenset({"question"}), emit=sent.append)
    assert len(sent) == 1


# --- status (one screen for the chief) ----------------------------------------


def test_status_reports_counts_and_stays_within_20_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import fcntl

    answered = bus.post(tmp_path, "w1", "chief", "question", "already answered?")
    bus.post(tmp_path, "chief", "w1", "answer", "yes", ref=str(answered["id"]))
    for i in range(7):  # 7 unanswered, each with a long body
        bus.post(tmp_path, f"w{i}", "chief", "blocker" if i == 3 else "question", f"open question {i} " + "x" * 200)
    bus.post(tmp_path, "chief", "w1", "info", "dm 1")
    bus.post(tmp_path, "chief", "w1", "info", "dm 2")
    bus.post(tmp_path, "chief", "w2", "info", "dm 3")
    bus.post(tmp_path, "chief", "w3", "blocker", "need the data")
    assert len(bus.inbox(tmp_path, "w1")) == 3  # w1 read everything chief sent: nothing unread
    bus.ask_user(tmp_path, "w4", "ship or wait?", job_only=True)
    lock = bus._tail_lock_path(tmp_path, "chief", "chief")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # a live durable chief tail
        assert bus.main(["--bus", str(tmp_path), "--as", "chief", "status"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) <= 20
    assert "pending to chief: 7" in lines
    items = [ln for ln in lines if ln.startswith("  [")]
    assert len(items) == 5  # at most 5 listed, oldest first (the answered one is not pending)
    assert not any("already answered" in ln for ln in items)
    assert any(ln.startswith("  [") and "200" not in ln.split(" — ")[1][:60] for ln in items)  # body cut to 60
    assert any("(+2 more)" in ln for ln in lines)
    assert any(ln.startswith("unread from chief: 2 agent(s)") and "w2=1" in ln and "w3=1" in ln for ln in lines)
    assert any(ln.startswith("chief tail: alive") for ln in lines)
    assert any(ln.startswith("last message: ") and "w4" in ln and "ship or wait?" in ln for ln in lines)
    assert "open needs-user: 1" in lines


def test_status_says_none_for_an_idle_bus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert bus.main(["--bus", str(tmp_path), "--as", "chief", "status"]) == 0
    out = capsys.readouterr().out
    assert "pending to chief: 0" in out
    assert "unread from chief: none" in out
    assert "chief tail: none" in out
    assert "last message: none" in out
    assert "open needs-user: 0" in out


def test_status_reports_no_tail_when_the_lock_is_free(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bus.post(tmp_path, "w", "chief", "info", "hello")
    lock = bus._tail_lock_path(tmp_path, "chief", "chief")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()  # an abandoned lock file from a dead tail: not held, so the tail is not alive
    assert bus.main(["--bus", str(tmp_path), "--as", "chief", "status"]) == 0
    assert "chief tail: none" in capsys.readouterr().out


# --- pending: one bounded line per item ---------------------------------------


def test_pending_cli_prints_only_the_fallback_line_for_old_inbox_traffic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bus full of old inbox traffic and no open question prints exactly the fallback line."""
    for i in range(200):
        bus.post(tmp_path, "w", "chief", "info", f"old chatter {i} " + "y" * 200)
        q = bus.post(tmp_path, "w", "chief", "question", f"answered question {i} " + "y" * 200)
        bus.post(tmp_path, "chief", "w", "answer", "done", ref=str(q["id"]))
    bus.post(tmp_path, "w", "all", "question", "a broadcast is not a DM")
    assert bus.main(["--bus", str(tmp_path), "--as", "chief", "pending"]) == 0
    out = capsys.readouterr().out
    assert out == "(no pending questions for chief)\n"
    assert bus.main(["--bus", str(tmp_path), "--as", "chief", "pending", "--global"]) == 0
    assert capsys.readouterr().out == "(no pending questions for chief)\n"


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


# --- scripts/agents/bus (the sh shim) -----------------------------------------


_BUS_SHIM = _BUS_PATH.with_name("bus")


def test_bus_shim_invokes_bus_py(tmp_path: Path) -> None:
    assert _BUS_SHIM.exists(), f"{_BUS_SHIM} is missing"
    assert os.access(_BUS_SHIM, os.X_OK)  # committed executable: `bus ...` works without a python prefix
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    help_run = subprocess.run([str(_BUS_SHIM), "--help"], capture_output=True, text=True, env=env)
    assert help_run.returncode == 0
    assert "status" in help_run.stdout and "--kinds" in help_run.stdout
    posted = subprocess.run(
        [str(_BUS_SHIM), "--bus", str(tmp_path), "--as", "sa", "post", "--to", "chief", "--body", "via the shim"],
        capture_output=True, text=True, env=env,
    )  # fmt: skip
    assert posted.returncode == 0 and "via the shim" in posted.stdout


_GOOD_HANDOFF = "## Handoff\nstatus: done\nrole: implementer\nevidence:\n- red then green\nimprove: none\n"


def _done(job: Path, *extra: str) -> int:
    return bus.main(["--bus", str(job / "bus"), "--as", "w1", "post", "--to", "chief", "--kind", "done", "--body", "ok", *extra])


def test_done_refused_without_handoff_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "bus").mkdir()
    with pytest.raises(SystemExit) as exc:
        _done(tmp_path)
    assert exc.value.code == 2
    assert str(tmp_path / "handoffs" / "w1.md") in capsys.readouterr().err
    assert bus.inbox(tmp_path / "bus", "chief") == []


def test_done_refused_when_handoff_fails_lint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "bus").mkdir()
    (tmp_path / "handoffs").mkdir()
    (tmp_path / "handoffs" / "w1.md").write_text("## Handoff\nstatus: done\n")
    with pytest.raises(SystemExit) as exc:
        _done(tmp_path)
    assert exc.value.code == 2
    assert "MISSING evidence" in capsys.readouterr().err
    assert bus.inbox(tmp_path / "bus", "chief") == []


def test_done_posts_with_valid_handoff(tmp_path: Path) -> None:
    (tmp_path / "bus").mkdir()
    (tmp_path / "handoffs").mkdir()
    (tmp_path / "handoffs" / "w1.md").write_text(_GOOD_HANDOFF)
    assert _done(tmp_path) == 0
    assert [m["body"] for m in bus.inbox(tmp_path / "bus", "chief")] == ["ok"]


def test_done_reviewer_handoff_needs_verdict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "bus").mkdir()
    (tmp_path / "handoffs").mkdir()
    (tmp_path / "handoffs" / "w1.md").write_text(_GOOD_HANDOFF + "verdict: MAYBE\n")
    with pytest.raises(SystemExit) as exc:
        _done(tmp_path)
    assert exc.value.code == 2
    assert "BAD verdict" in capsys.readouterr().err


def test_done_escape_hatch_and_other_kinds_unchecked(tmp_path: Path) -> None:
    (tmp_path / "bus").mkdir()
    assert _done(tmp_path, "--no-handoff-check") == 0
    assert bus.main(["--bus", str(tmp_path / "bus"), "--as", "w1", "post", "--to", "chief", "--kind", "info", "--body", "hi"]) == 0
