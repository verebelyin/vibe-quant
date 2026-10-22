"""Characterization goldens for a full mock discovery run (rabmo.7 T3).

Pins what ``vibe_quant.discovery.__main__.main()`` persists and prints for the
four success shapes (0- and 5-champion singles, multi-seed and bankroll grid)
and the four failure classes, so the rabmo.7 move of the run body into
``run.py`` cannot change any of it. This is characterization only: no
production code is touched and every argument set is a small mock run
(population 6, 2 generations) — never a real data run.

Each case is selectable by id, e.g. ``pytest ... -k plain`` or
``-k "multi_seed or bankroll_grid"`` (F11 wants the champion shapes runnable on
their own).

Volatile values are masked/removed before comparison: the elapsed-time fields
in ``GEN``/``DISCOVERY COMPLETE``/``CONVERGED`` log lines, the result row's
``execution_time_seconds`` and ``created_at``, and the notes'
``compiler_version`` hash. Everything else — including the exact notes JSON
byte order (R37) — is compared literally.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from vibe_quant.db.state_manager import StateManager

GOLDEN_PATH = Path(__file__).parent.parent / "fixtures" / "discovery" / "mock_run_golden.json"

# Shared mock-run head. The seed/population/generation limits keep this cheap.
_BASE = [
    "--mock",
    "--seed",
    "7",
    "--population-size",
    "6",
    "--max-generations",
    "2",
    "--max-workers",
    "-1",
    "--start-date",
    "2024-01-01",
    "--end-date",
    "2025-01-01",
    "--timeframe",
    "4h",
    "--symbols",
    "BTCUSDT",
]
_NO_GATES = ["--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate"]

SUCCESS_CASES: dict[str, list[str]] = {
    "plain": _BASE,
    "champions": _BASE + _NO_GATES,
    "multi_seed": _BASE + _NO_GATES + ["--num-seeds", "2"],
    "bankroll_grid": _BASE
    + _NO_GATES
    + [
        "--bankroll-modes",
        "isolated",
        "--bankroll-leverages",
        "5,10",
        "--bankroll-max-margin-pcts",
        "0.5",
    ],
}

FAILURE_CASES: dict[str, list[str]] = {
    "run_missing": _BASE,
    "partial_grid": _BASE + ["--bankroll-modes", "isolated"],
    "no_catalog_data": [arg for arg in _BASE if arg != "--mock"],
    "seed_from_run_empty": _BASE + ["--seed-from-run", "PRIOR"],
}

_COMPILER_VERSION = re.compile(r'"compiler_version": "[^"]*"')
_TIMING_SUBS = (
    (re.compile(r"gen_time=[0-9.]+s total=[0-9.]+s ETA=[0-9.]+s"), "gen_time=<t> total=<t> ETA=<t>"),
    (
        re.compile(r"=== DISCOVERY COMPLETE: [0-9.]+s total \([0-9.]+s/gen avg\)"),
        "=== DISCOVERY COMPLETE: <t> total (<t>/gen avg)",
    ),
    (re.compile(r"after [0-9.]+s ==="), "after <t> ==="),
)


def mask_notes(raw: str) -> str:
    """Mask the compiler hash so the golden is stable across rebuilds."""
    return _COMPILER_VERSION.sub('"compiler_version": "<masked>"', raw)


def mask_log_line(line: str) -> str:
    """Mask elapsed-time fields; they depend on machine speed, not behaviour."""
    for pattern, replacement in _TIMING_SUBS:
        line = pattern.sub(replacement, line)
    return line


def _make_run(db: Path) -> int:
    state = StateManager(db)
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2025-01-01",
        parameters={},
    )
    state.close()
    return run_id


def _read(db: Path, run_id: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    state = StateManager(db)
    run = state.get_backtest_run(run_id)
    result = state.get_backtest_result(run_id)
    state.close()
    return run, result


def capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    *,
    run_id: int | None = None,
    seed_from_prior: bool = False,
    patch_no_data: bool = False,
) -> dict[str, Any]:
    """Run ``main()`` once on a temp DB and record everything the golden pins."""
    import vibe_quant.discovery.__main__ as dm

    db = tmp_path / "discovery.db"
    prior_run_id: int | None = None
    if seed_from_prior:
        # A completed mock run that persisted zero champions is the warm-start
        # source the ValueError row needs.
        prior_run_id = _make_run(db)
        monkeypatch.setattr(
            sys, "argv", ["prog", *_BASE, "--run-id", str(prior_run_id), "--db", str(db)]
        )
        assert dm.main() == 0
    if run_id is None:
        run_id = _make_run(db)
    elif not seed_from_prior:
        # A brand-new DB races the heartbeat thread's first ``PRAGMA
        # journal_mode=WAL`` against main()'s lazily opened connection
        # ("database is locked", unrelated to this row). Create the DB and one
        # real run first; the requested id is still absent.
        _make_run(db)
    if patch_no_data:
        monkeypatch.setattr(dm, "_check_data_available", lambda symbols: False)

    resolved = [str(prior_run_id) if arg == "PRIOR" else arg for arg in args]
    monkeypatch.setattr(sys, "argv", ["prog", *resolved, "--run-id", str(run_id), "--db", str(db)])

    caplog.clear()
    caplog.set_level(logging.INFO)
    capsys.readouterr()  # drop setup output
    code = dm.main()
    captured = capsys.readouterr()

    run_row, result_row = _read(db, run_id)
    record: dict[str, Any] = {
        "exit": code,
        "stdout": captured.out.splitlines(),
        "traceback": "Traceback (most recent call last)" in captured.err,
        "run": (
            None
            if run_row is None
            else {"status": run_row["status"], "error_message": run_row["error_message"]}
        ),
        "result": None,
        "log_lines": [
            mask_log_line(record.getMessage())
            for record in caplog.records
            if record.getMessage().startswith("===")
            or record.getMessage().startswith("Discovery seed:")
        ],
    }
    if result_row is not None:
        row = {key: value for key, value in result_row.items() if key != "execution_time_seconds"}
        row["created_at"] = "<created_at>"
        row["notes"] = mask_notes(row["notes"])
        record["result"] = row
    return record


def load_golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GOLDEN_PATH.read_text())
    return data


@pytest.mark.parametrize("case_id", list(SUCCESS_CASES), ids=list(SUCCESS_CASES))
def test_mock_runs_match_golden(
    case_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every mock success shape persists and prints the golden record."""
    got = capture(tmp_path, monkeypatch, caplog, capsys, SUCCESS_CASES[case_id])
    expected = load_golden()["success"][case_id]

    assert got["exit"] == 0
    assert got["run"] == {"status": "completed", "error_message": None}
    assert got == expected

    notes = json.loads(got["result"]["notes"])
    assert len(notes["top_strategies"]) == (0 if case_id == "plain" else 5)
    assert notes["mock"] is True


@pytest.mark.parametrize("case_id", list(FAILURE_CASES), ids=list(FAILURE_CASES))
def test_failure_rows(
    case_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each failure class keeps its exit code, status, error text and traceback."""
    got = capture(
        tmp_path,
        monkeypatch,
        caplog,
        capsys,
        FAILURE_CASES[case_id],
        run_id=999 if case_id == "run_missing" else None,
        seed_from_prior=case_id == "seed_from_run_empty",
        patch_no_data=case_id == "no_catalog_data",
    )
    expected = load_golden()["failure"][case_id]

    assert got == expected
