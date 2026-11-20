"""chief.py: one command that ledgers a row, lints a done handoff, and prints bus status + ETA.

Chiefs ran ledger.py 1,349 times in Oct 14-28 plus handoff_lint and `bus.py status` by hand,
and the user asked "status? ETA?" 10+ times.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/chief.py"
_spec = importlib.util.spec_from_file_location("chief_under_test", _PATH)
assert _spec and _spec.loader
chief = importlib.util.module_from_spec(_spec)
sys.modules["chief_under_test"] = chief
_spec.loader.exec_module(chief)

HEADER = (
    "# ledger 20261029-x\n"
    "| agent | runtime | model | task | bead | worktree | status |\n"
    "|---|---|---|---|---|---|---|\n"
)


@pytest.fixture(autouse=True)
def _telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic: the default ledger lives in a git checkout, and mutate.py's scratch copy is not one."""
    monkeypatch.setenv("SWARM_TELEMETRY", str(tmp_path / "telemetry.jsonl"))


@pytest.fixture(autouse=True)
def _fast_bd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`bd ready --json` costs ~0.5 s and every status block asks it (`chief` teaches no bd
    semantics). A `bd` on PATH that answers $VQ_TEST_BD_READY keeps `bus._ready_count` ITSELF
    running — argv, returncode, json.loads — rather than replacing the production function
    (chief rule, board 18e53d5a). 14 status blocks paid the real `bd` before this."""
    bin_dir = tmp_path / "bd-bin"
    bin_dir.mkdir()
    shim = bin_dir / "bd"
    shim.write_text(
        '#!/bin/sh\n'
        'if [ "$1" = "show" ]; then printf "%s\\n" "$VQ_TEST_BD_SHOW";'
        ' else printf "%s\\n" "$VQ_TEST_BD_READY"; fi\n'
    )
    shim.chmod(0o755)
    monkeypatch.setenv("VQ_TEST_BD_READY", "[]")
    monkeypatch.setenv("VQ_TEST_BD_SHOW", "[]")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def _ledger(job: Path, *rows: str) -> None:
    job.mkdir(parents=True, exist_ok=True)
    (job / "ledger.md").write_text(HEADER + "".join(row + "\n" for row in rows))


CLEAN_HANDOFF = "## Handoff\nstatus: blocked\nimprove: none\n- nothing to route\n"
DIRTY_HANDOFF = "## Handoff\nstatus: blocked\n"


# --- step: done rows are linted into a stop -------------------------------------------------


def test_step_refuses_a_dirty_done_handoff_and_appends_no_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "20261029-x"
    handoff = tmp_path / "h.md"
    handoff.write_text(DIRTY_HANDOFF)
    assert chief.main(["step", "--job", str(job), "--id", "cc-a-1", "--status", "done GATE PASS", "--handoff", str(handoff)]) == 1
    out = capsys.readouterr().out
    assert "MISSING improve" in out
    assert "step refused: handoff not lint-clean" in out
    assert not (job / "ledger.md").exists()


def test_step_appends_the_row_for_a_clean_done_handoff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "20261029-x"
    handoff = tmp_path / "h.md"
    handoff.write_text(CLEAN_HANDOFF)
    rc = chief.main(
        ["step", "--job", str(job), "--id", "cc-a-1", "--runtime", "t3", "--model", "deepseek-flash",
         "--task", "P1-A", "--status", "done 6aabd780", "--handoff", str(handoff)]
    )  # fmt: skip
    assert rc == 0
    lines = (job / "ledger.md").read_text().splitlines()
    assert lines[-1].startswith("| cc-a-1 | t3 | deepseek-flash | P1-A | - | - | ")
    assert lines[-1].endswith(" done 6aabd780 |")
    assert "refused" not in capsys.readouterr().out


# --- status: running agents with an ETA from telemetry medians ------------------------------


def test_status_never_writes_the_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z rendered |")
    before = (job / "ledger.md").read_bytes()
    monkeypatch.setenv("SWARM_TELEMETRY", str(tmp_path / "none.jsonl"))
    assert chief.main(["status", "--job", str(job)]) == 0
    assert (job / "ledger.md").read_bytes() == before


# --- handoff / resume: fixed sections in a fixed order ------------------------------------

HANDOFF_SECTIONS = ("Main", "Running children", "Ledger", "Open user questions", "Next 3 steps")


def test_handoff_writes_five_sections_and_resume_prints_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "20261029-x"
    _ledger(
        job,
        "| cc-a-1 | t3 | deepseek-flash | P1-A taskId node:x | b1 | wt | 10-29 07:51Z rendered |",
        "| cc-b-1 | t3 | deepseek-flash | P1-B | b2 | wt | 10-29 08:10Z done abc1234 |",
    )
    rc = chief.main(
        ["handoff", "--job", str(job), "--next", "land P1", "--next", "review P2",
         "--next", "tell the user"]
    )  # fmt: skip
    assert rc == 0
    path = job / "HANDOFF.md"
    text = path.read_text()
    assert [line for line in text.splitlines() if line.startswith("## ")] == [
        f"## {section}" for section in HANDOFF_SECTIONS
    ]
    assert "| clientRequestId cc-a-1 | taskId node:x" in text
    assert "- cc-b-1 — " in text and "done abc1234" in text
    assert "- land P1" in text and "- tell the user" in text
    capsys.readouterr()
    assert chief.main(["resume", "--job", str(job)]) == 0
    out = capsys.readouterr().out
    for section in HANDOFF_SECTIONS:
        assert f"## {section}" in out
    assert "| taskId node:x" in out


# --- init: a job dir mkbrief accepts (bus/, handoffs/, prompts/, brief.md, ledger.md) -------

_LEDGER_PATH = Path(__file__).resolve().parents[2] / "scripts/agents/ledger.py"
_ledger_spec = importlib.util.spec_from_file_location("chief_ledger_under_test", _LEDGER_PATH)
assert _ledger_spec and _ledger_spec.loader
ledger = importlib.util.module_from_spec(_ledger_spec)
sys.modules["chief_ledger_under_test"] = ledger
_ledger_spec.loader.exec_module(ledger)

_INIT_HEADINGS = (
    "## Goal",
    "## Constraints",
    "## Profile",
    "## Finish condition",
    "## Tasks",
    "## Answers",
)


def test_init_creates_the_dirs_brief_and_a_ledger_mkbrief_accepts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "20261106-x"
    assert chief.main(["init", "--job", str(job)]) == 0
    for sub in ("bus", "handoffs", "prompts"):
        assert (job / sub).is_dir(), sub
    brief = (job / "brief.md").read_text()
    assert brief.startswith("# Job 20261106-x\n")
    assert [line for line in brief.splitlines() if line.startswith("## ")] == list(_INIT_HEADINGS)
    assert "## Profile\nlean\n" in brief
    assert (job / "ledger.md").read_text() == ledger.header_text("20261106-x")
    assert capsys.readouterr().out.splitlines()[-1] == f"export SWARM_BUS={job / 'bus'}"


def test_init_refuses_an_existing_brief_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "j"
    job.mkdir()
    brief = job / "brief.md"
    brief.write_text("# Job j\n\n## Goal\nhand-written\n")
    before = brief.read_bytes()
    assert chief.main(["init", "--job", str(job)]) == 2
    assert brief.read_bytes() == before
    assert sorted(path.name for path in job.iterdir()) == ["brief.md"]
    assert "brief.md" in capsys.readouterr().err
