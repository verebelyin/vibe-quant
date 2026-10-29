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
        "  cc-a-1 deepseek-flash P1-A started 10-29 07:51Z elapsed 12m "
        "ETA ~38m (median 50m of 3 runs, deepseek/deepseek-v4.1-flash)"
    ) in lines


def test_running_line_says_overdue_past_one_and_a_half_medians(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 07:51Z dispatched |")
    recs = [_rec("deepseek/deepseek-v4.1-flash", 1000), _rec("deepseek/deepseek-v4.1-flash", 3000), _rec("deepseek/deepseek-v4.1-flash", 2000)]
    lines = chief.status_block(job, datetime(2026, 10, 29, 9, 30, tzinfo=UTC), recs, MAP)
    assert (
        "  cc-a-1 deepseek-flash P1-A started 10-29 07:51Z elapsed 99m ETA overdue "
        "(median 33m of 3 runs, deepseek/deepseek-v4.1-flash)"
    ) in lines


def test_running_line_is_unknown_below_three_runs(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 10-29 08:00Z in-progress |")
    recs = [_rec("deepseek/deepseek-v4.1-flash", 600), _rec("deepseek/deepseek-v4.1-flash", 1800)]
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 5, tzinfo=UTC), recs, MAP)
    assert "  cc-a-1 deepseek-flash P1-A started 10-29 08:00Z elapsed 5m ETA unknown (2 runs)" in lines


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
    assert "  cc-b-1 deepseek-flash P1-B started 10-29 07:50Z elapsed 20m" in body


def test_ledger_timestamp_without_a_year_rolls_back_over_new_year(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    _ledger(job, "| cc-a-1 | t3 | deepseek-flash | P1-A | b1 | - | 12-31 23:00Z rendered |")
    lines = chief.status_block(job, datetime(2026, 1, 1, 0, 10, tzinfo=UTC), [], MAP)
    assert "  cc-a-1 deepseek-flash P1-A started 12-31 23:00Z elapsed 70m ETA unknown (0 runs)" in lines


def test_a_ledger_cell_with_a_stray_byte_does_not_break_status(tmp_path: Path) -> None:
    job = tmp_path / "20261029-x"
    job.mkdir()
    (job / "ledger.md").write_bytes(
        HEADER.encode()
        + b"| cc-a-1 | t3 | deepseek-flash | P1-\xff | b1 | - | 10-29 07:51Z rendered |\n"
    )
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 0, tzinfo=UTC), [], MAP)
    assert lines[-1] == "  cc-a-1 deepseek-flash P1-\ufffd started 10-29 07:51Z elapsed 9m ETA unknown (0 runs)"


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
    lines = chief.status_block(job, datetime(2026, 10, 29, 8, 0, tzinfo=UTC), [], MAP)
    assert lines[0] == f"chief status — {job / 'bus'}"
    assert "pending to chief: 0" in lines
    assert lines.index("Running:") > lines.index("pending to chief: 0")
    assert lines[-1].startswith("  cc-a-1 ")


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
    assert "  cc-a-1 - - started " in out


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
    assert "  cc-a-1 deepseek-flash P3 board started 10-29 07:51Z elapsed 12m ETA unknown (0 runs)" in lines
