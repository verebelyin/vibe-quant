"""scripts/agents/bus.py — swarm message bus (agents <-> agents <-> chief)."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import sys
import threading
import time
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


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    monkeypatch.setattr(bus, "POLL_S", 0.05)
    # never let a test write the real committed board
    monkeypatch.setenv("SWARM_BOARD", str(tmp_path_factory.mktemp("global_board")))


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
    assert "board read --global" in text and "#gotchas w: gotcha 4" in text and "gotcha 2" not in text
