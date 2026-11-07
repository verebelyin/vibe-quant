"""chief.py: one command that ledgers a row, lints a done handoff, and prints bus status + ETA.

Chiefs ran ledger.py 1,349 times in Oct 14-28 plus handoff_lint and `bus.py status` by hand,
and the user asked "status? ETA?" 10+ times.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from datetime import UTC, datetime
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
MAP = {
    "deepseek-flash": "deepseek/deepseek-v4.1-flash",
    "gpt-sol": "gpt-6.1-sol",
}


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


def _rec(model: str, wall: float | None, outcome: str = "done") -> dict[str, object]:
    return {"model": model, "wall_s": wall, "outcome": outcome}


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


def test_step_lints_the_default_handoff_path_only_for_done(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "20261029-x"
    (job / "handoffs").mkdir(parents=True)
    (job / "handoffs" / "cc-a-1.md").write_text(DIRTY_HANDOFF)
    assert chief.main(["step", "--job", str(job), "--id", "cc-a-1", "--status", "in-progress"]) == 0
    assert "MISSING" not in capsys.readouterr().out
    assert (job / "ledger.md").read_text().splitlines()[-1].endswith(" in-progress |")


# --- status: running agents with an ETA from telemetry medians ------------------------------


def test_running_line_uses_the_median_of_three_done_runs(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z rendered |")
    recs = [
        _rec("deepseek/deepseek-v4.1-flash", 2400),
        _rec("deepseek/deepseek-v4.1-flash", 3000),
        _rec("deepseek/deepseek-v4.1-flash", 9000),  # asymmetric: a mean would read 80m
        _rec("deepseek/deepseek-v4.1-flash", float("nan")),  # malformed hand record: ignored
        _rec("deepseek/deepseek-v4.1-flash", 9999, "failed"),
        _rec("deepseek/deepseek-v4.1-flash", None),
    ]
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 3, tzinfo=UTC), recs, MAP)
    assert (
        "  P1-A — started 10-29 07:51Z, elapsed 12m, ETA ~38m "
        "(median 50m of 3 runs, deepseek/deepseek-v4.1-flash) [cc-a-1 deepseek-flash]"
    ) in lines


def test_running_line_says_overdue_past_one_and_a_half_medians(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z dispatched |")
    recs = [_rec("deepseek/deepseek-v4.1-flash", 1000), _rec("deepseek/deepseek-v4.1-flash", 3000), _rec("deepseek/deepseek-v4.1-flash", 2000)]
    lines = chief.status_block(job, datetime(2026, 10, 29, 9, 30, tzinfo=UTC), recs, MAP)
    assert (
        "  P1-A — started 10-29 07:51Z, elapsed 99m, ETA overdue "
        "(median 33m of 3 runs, deepseek/deepseek-v4.1-flash) [cc-a-1 deepseek-flash]"
    ) in lines


def test_running_line_is_unknown_below_three_runs(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 08:00Z in-progress |")
    recs = [_rec("deepseek/deepseek-v4.1-flash", 600), _rec("deepseek/deepseek-v4.1-flash", 1800)]
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 5, tzinfo=UTC), recs, MAP)
    assert "  P1-A — started 10-29 08:00Z, elapsed 5m, ETA unknown (2 runs) [cc-a-1 deepseek-flash]" in lines


def test_finished_agent_is_not_listed_and_two_rows_use_the_latest(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    _ledger(
        job,
        "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:00Z rendered |",
        "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 08:00Z done 6aabd780 |",
        "| cc-b-1 | t3 | deepseek-flash | P1-B | b2 | - | 10-29 07:10Z rendered |",
        "| cc-b-1 | t3 | deepseek-flash | P1-B | b2 | - | 10-29 07:50Z in-progress |",
    )
    recs = [_rec("deepseek/deepseek-v4.1-flash", w) for w in (600, 1200, 1800)]
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 10, tzinfo=UTC), recs, MAP)
    body = "\n".join(lines)
    assert "cc-a-1" not in body
    assert "  P1-B — started 10-29 07:50Z, elapsed 20m" in body


def test_ledger_timestamp_without_a_year_rolls_back_over_new_year(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 12-31 23:00Z rendered |")
    lines = chief.status_block(job, datetime(2026, 1, 1, 0, 10, tzinfo=UTC), [], MAP)
    assert "  P1-A — started 12-31 23:00Z, elapsed 70m, ETA unknown (0 runs) [cc-a-1 deepseek-flash]" in lines


def test_a_ledger_cell_with_a_stray_byte_does_not_break_status(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    job.mkdir()
    (job / "ledger.md").write_bytes(
        HEADER.encode()
        + b"| cc-a-1 | t3 | deepseek-flash | P1-\xff | b1 | - | 10-29 07:51Z rendered |\n"
    )
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 0, tzinfo=UTC), [], MAP)
    assert lines[-1] == "  P1-\ufffd — started 10-29 07:51Z, elapsed 9m, ETA unknown (0 runs) [cc-a-1 deepseek-flash]"


def test_model_key_maps_through_the_real_models_json(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    _ledger(
        job,
        "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z rendered |",
        "| cc-b-1 | t3 | gpt-sol | P1-B | b2 | - | 10-29 07:51Z rendered |",
    )
    recs = [_rec("gpt-6.1-sol", 600), _rec("gpt-6.1-sol", 600), _rec("gpt-6.1-sol", 600)]
    recs += [_rec("deepseek/deepseek-v4.1-flash", 300), _rec("deepseek/deepseek-v4.1-flash", 300), _rec("deepseek/deepseek-v4.1-flash", 300)]
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 0, tzinfo=UTC), recs)
    body = "\n".join(lines)
    assert "(median 10m of 3 runs, gpt-6.1-sol)" in body
    assert "(median 5m of 3 runs, deepseek/deepseek-v4.1-flash)" in body


def test_status_block_prints_the_bus_screen_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z rendered |")
    monkeypatch.setenv("SWARM_TELEMETRY", str(tmp_path / "none.jsonl"))
    monkeypatch.setenv("VQ_TEST_BD_READY", '[{"id": "vq-1"}, {"id": "vq-2"}]')
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 0, tzinfo=UTC), [], MAP)
    assert lines[0] == f"chief status — {job / 'bus'}"
    assert "pending to chief: 0" in lines
    assert lines.index("Running:") > lines.index("pending to chief: 0")
    assert lines[-1].endswith("[cc-a-1 deepseek-flash]")
    # one bead per row of the JSON bd printed: `_ready_count`'s own parse, not a stub's 0
    assert "startable: 2 ready beads (bd ready)" in lines


def test_status_never_writes_the_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z rendered |")
    before = (job / "ledger.md").read_bytes()
    monkeypatch.setenv("SWARM_TELEMETRY", str(tmp_path / "none.jsonl"))
    assert chief.main(["status", "--job", str(job)]) == 0
    assert (job / "ledger.md").read_bytes() == before


def test_cli_step_prints_the_same_status_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "20261029-x"
    monkeypatch.setenv("SWARM_TELEMETRY", str(tmp_path / "none.jsonl"))
    assert chief.main(["step", "--job", str(job), "--id", "cc-a-1", "--status", "rendered"]) == 0
    out = capsys.readouterr().out
    assert f"chief status — {job / 'bus'}" in out
    assert "Running:" in out
    assert "  unlabelled item — started " in out


def test_cli_step_with_a_bare_job_name_writes_under_the_swarm_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2a: `step --job <bare name>` resolves through job_path, never into the cwd."""
    swarm = tmp_path / "data" / "swarm"
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(chief, "SWARM_ROOT", swarm)
    assert chief.main(["step", "--job", "20261106-x", "--id", "cc-a-1", "--status", "rendered"]) == 0
    assert (swarm / "20261106-x" / "ledger.md").is_file()
    assert list(cwd.iterdir()) == []


def test_cli_status_with_a_bare_job_name_reads_under_the_swarm_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2a: `status --job <bare name>` reads <swarm root>/<name>, not a cwd dir of that name."""
    swarm = tmp_path / "data" / "swarm"
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(chief, "SWARM_ROOT", swarm)
    _ledger(swarm / "20261106-x", "| cc-real-1 | t3 | deepseek-flash | REAL-A | b1 | - | 10-29 07:00Z rendered |")
    _ledger(cwd / "20261106-x", "| cc-cwd-1 | t3 | deepseek-flash | CWD-A | b1 | - | 10-29 07:00Z rendered |")
    assert chief.main(["status", "--job", "20261106-x"]) == 0
    out = capsys.readouterr().out
    assert "REAL-A" in out and "[cc-real-1 deepseek-flash]" in out
    assert "CWD-A" not in out


def test_a_running_row_is_running_in_status_and_hands_off_as_a_child(
    tmp_path: Path,
) -> None:
    """R2b: a `running` ledger row (delegate_task cr=...) is unfinished work, not finished."""
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z running (delegate_task cr=abc) |")
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 3, tzinfo=UTC), [], MAP)
    assert "  P1-A — started 10-29 07:51Z, elapsed 12m, ETA unknown (0 runs) [cc-a-1 deepseek-flash]" in lines
    assert chief.main(["handoff", "--job", str(job)]) == 0
    text = (job / "HANDOFF.md").read_text()
    assert "- cc-a-1 | runtime t3 | model deepseek-flash | clientRequestId cc-a-1 | taskId -" in text
    assert "## Running children\n- cc-a-1" in text


def test_running_the_script_exits_1_on_a_refused_step(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    handoff = tmp_path / "h.md"
    handoff.write_text(DIRTY_HANDOFF)
    env = {**os.environ, "SWARM_TELEMETRY": str(tmp_path / "none.jsonl")}
    proc = subprocess.run(
        [sys.executable, str(_PATH), "step", "--job", str(job), "--id", "cc-a-1",
         "--status", "done x", "--handoff", str(handoff)],
        capture_output=True, text=True, env=env,
    )  # fmt: skip
    assert proc.returncode == 1
    assert "step refused: handoff not lint-clean" in proc.stdout
    assert not (job / "ledger.md").exists()


def test_running_line_drops_the_t3_task_id_from_the_task_cell(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P3 board taskId node:delegated-task:x%3Ay | b1 | - | 10-29 07:51Z dispatched |")
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 3, tzinfo=UTC), [], MAP)
    assert "  P3 board — started 10-29 07:51Z, elapsed 12m, ETA unknown (0 runs) [cc-a-1 deepseek-flash]" in lines


def test_running_line_says_what_the_item_does_from_the_bead_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user asked "explain simply": the screen must say what the item does, not a bare id."""
    job = tmp_path / "20261029-x"
    _ledger(
        job,
        "| cc-a-1 | t3 | deepseek-flash | P1-A | vibe-quant-ntmoh | - | 10-29 07:51Z rendered |",
    )
    monkeypatch.setenv(
        "VQ_TEST_BD_SHOW",
        '{"id": "vibe-quant-ntmoh", "title": "Add a plain-words status footer with ETA"}',
    )
    recs = [_rec("deepseek/deepseek-v4.1-flash", 6000) for _ in range(3)]
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 3, tzinfo=UTC), recs, MAP)
    assert (
        "  Add a plain-words status footer with ETA — started 10-29 07:51Z, elapsed 12m, "
        "ETA ~88m (median 100m of 3 runs, deepseek/deepseek-v4.1-flash) [cc-a-1 deepseek-flash]"
    ) in lines


def test_running_lines_name_the_task_when_one_bead_has_several(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "20261029-x"
    _ledger(
        job,
        "| cc-a-1 | t3 | deepseek-flash | mirrors | vibe-quant-ntmoh | - | 10-29 07:51Z dispatched |",
        "| cc-b-1 | t3 | deepseek-flash | docs | vibe-quant-ntmoh | - | 10-29 07:52Z dispatched |",
    )
    monkeypatch.setenv("VQ_TEST_BD_SHOW", '{"id": "vibe-quant-ntmoh", "title": "Switch indicators"}')
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 3, tzinfo=UTC), [], MAP)
    assert any(line.startswith("  Switch indicators: mirrors — started") for line in lines)
    assert any(line.startswith("  Switch indicators: docs — started") for line in lines)


def test_a_status_only_step_keeps_the_agents_earlier_cells(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | vibe-quant-ntmoh | vq-a | 10-29 07:51Z rendered |")
    assert chief.main(["step", "--job", str(job), "--id", "cc-a-1", "--status", "dispatched"]) == 0
    last = (job / "ledger.md").read_text().splitlines()[-1]
    assert last.startswith("| cc-a-1 | t3 | deepseek-flash | P1-A | vibe-quant-ntmoh | vq-a | ")
    assert last.endswith(" dispatched |")


def test_running_line_uses_the_task_cell_when_bd_has_no_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z rendered |")
    monkeypatch.setenv("VQ_TEST_BD_SHOW", '[{"id": "b1"}]')
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 3, tzinfo=UTC), [], MAP)
    assert "  P1-A — started 10-29 07:51Z, elapsed 12m, ETA unknown (0 runs) [cc-a-1 deepseek-flash]" in lines


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


def test_handoff_lists_open_questions_from_the_bus(tmp_path: Path) -> None:
    job = tmp_path / "j"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z done x |")
    bus = chief._sibling("bus")
    bus.post(job / "bus", "cc-a-1", "board", "question", "Should paper trading start today?",
             topic="needs-user", mirror=False)  # fmt: skip
    bus.post(job / "bus", "cc-a-1", "chief", "question", "Can I touch the catalog?", mirror=False)
    assert chief.main(["handoff", "--job", str(job)]) == 0
    text = (job / "HANDOFF.md").read_text()
    assert "Should paper trading start today?" in text
    assert "Can I touch the catalog?" in text


def test_handoff_carries_each_question_body_in_full(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A >80-character question must reach HANDOFF.md and resume whole (chief R1/SF1)."""
    job = tmp_path / "j"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z done x |")
    body = (
        "You hold BTC and ETH at 14:32. ETH fails to save and the screen shows 0 positions.\n"
        "Should we halt both positions or keep BTC trading?"
    )
    chief._sibling("bus").post(
        job / "bus", "cc-a-1", "board", "question", body, topic="needs-user", mirror=False
    )
    assert chief.main(["handoff", "--job", str(job), "--next", "land P1"]) == 0
    first, second = body.splitlines()
    text = (job / "HANDOFF.md").read_text()
    assert first in text
    assert f"  {second}" in text
    capsys.readouterr()
    assert chief.main(["resume", "--job", str(job)]) == 0
    out = capsys.readouterr().out
    assert first in out and f"  {second}" in out


def test_handoff_does_not_print_the_ledger_header_as_a_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SF2: latest_rows must skip the parsed header; a header-only ledger has no rows."""
    job = tmp_path / "j"
    job.mkdir()
    (job / "ledger.md").write_text(chief._sibling("ledger").header_text("j"))
    assert chief.ledger_lines(chief.latest_rows(job / "ledger.md")) == ["- (no rows)"]
    assert chief.main(["handoff", "--job", str(job)]) == 0
    text = (job / "HANDOFF.md").read_text()
    assert "- agent — status" not in text
    assert "- (no rows)" in text


def test_handoff_keeps_at_most_three_next_steps(tmp_path: Path) -> None:
    job = tmp_path / "j"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z done x |")
    assert chief.main(
        ["handoff", "--job", str(job), "--next", "a", "--next", "b", "--next", "c", "--next", "d"]
    ) == 0
    body = (job / "HANDOFF.md").read_text().split("## Next 3 steps", 1)[1]
    assert "- a" in body and "- b" in body and "- c" in body and "- d" not in body


def test_handoff_without_next_steps_writes_a_placeholder_resume_rejects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "j"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z done x |")
    assert chief.main(["handoff", "--job", str(job)]) == 0
    assert "- TODO:" in (job / "HANDOFF.md").read_text()
    assert chief.main(["resume", "--job", str(job)]) == 1
    assert "EMPTY section: ## Next 3 steps" in capsys.readouterr().out


def test_resume_exits_1_when_a_section_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "j"
    job.mkdir()
    (job / "HANDOFF.md").write_text(
        "# Handoff j\n\n## Main\n- main: abc\n\n## Running children\n- (none)\n\n"
        "## Ledger\n- (none)\n\n## Next 3 steps\n- one\n"
    )
    assert chief.main(["resume", "--job", str(job)]) == 1
    assert "MISSING section: ## Open user questions" in capsys.readouterr().out


def test_resume_exits_1_when_a_section_is_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "j"
    job.mkdir()
    (job / "HANDOFF.md").write_text(
        "# Handoff j\n\n## Main\n- main: abc\n\n## Running children\n- (none)\n\n"
        "## Ledger\n- (none)\n\n## Open user questions\n\n## Next 3 steps\n- one\n"
    )
    assert chief.main(["resume", "--job", str(job)]) == 1
    assert "EMPTY section: ## Open user questions" in capsys.readouterr().out


def test_resume_exits_1_when_the_handoff_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert chief.main(["resume", "--job", str(tmp_path / "j")]) == 1
    assert "HANDOFF.md" in capsys.readouterr().out


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


def test_init_ledger_accepts_the_first_row(tmp_path: Path) -> None:
    """A header-only ledger.md is what ledger.py itself writes, so append_row keeps appending."""
    job = tmp_path / "j"
    assert chief.main(["init", "--job", str(job)]) == 0
    ledger.append_row(job, {"agent": "cc-a-1", "status": "rendered"})
    assert (job / "ledger.md").read_text().splitlines() == ledger.header_text("j").splitlines() + [
        "| cc-a-1 | - | - | - | - | - | rendered |"
    ]


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


def test_a_bare_job_name_resolves_under_the_swarm_root_like_ledger(tmp_path: Path) -> None:
    assert chief.job_path(Path("20261106-x")) == ledger.SWARM_ROOT / "20261106-x"
    assert chief.job_path(tmp_path / "j") == tmp_path / "j"
