"""scripts/agents/bus.py — swarm message bus (agents <-> agents <-> chief)."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
import sys
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


# --- regression: findings from the harden review ------------------------------


def _delivered(tmp_path: Path) -> list[str]:
    """Bodies the file store's cursor reader (digest's ``_read_since``) delivers, cursor advanced."""
    return [str(m["body"]) for m in bus._read_since(tmp_path, "chief.digest")]


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


def _done(job: Path, *extra: str) -> int:
    return bus.main(["--bus", str(job / "bus"), "--as", "w1", "post", "--to", "chief", "--kind", "done", "--body", "ok", *extra])


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


def _dms(b: Path, me: str) -> list[object]:
    """Bodies addressed to ``me`` (or all) by someone else, in file order."""
    return [m["body"] for m in bus._all(b) if m.get("from") != me and m.get("to") in (me, "all")]
