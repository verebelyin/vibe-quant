"""Tests for scripts/paper_reconciliation.py (paper log + DB trades)."""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType, create_event
from vibe_quant.logging.writer import EventWriter

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "paper_reconciliation.py"
_spec = importlib.util.spec_from_file_location("paper_reconciliation", _SCRIPT)
assert _spec is not None and _spec.loader is not None
paper_reconciliation = importlib.util.module_from_spec(_spec)
sys.modules["paper_reconciliation"] = paper_reconciliation
_spec.loader.exec_module(paper_reconciliation)

ENTRY = "2026-01-01T00:00:00+00:00"
EXIT = "2026-01-01T00:10:00+00:00"


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway state DB that the script's StateManager() resolves to."""
    db = tmp_path / "state.db"
    monkeypatch.setattr("vibe_quant.db.connection.DEFAULT_DB_PATH", db)
    return db


def _insert_validation_run(db: Path) -> int:
    state = StateManager(db_path=db)
    try:
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1m",
            start_date="2025-01-01",
            end_date="2025-01-02",
            parameters={},
        )
        state.save_trades_batch(
            run_id,
            [
                {
                    "symbol": "BTCUSDT",
                    "direction": "LONG",
                    "leverage": 1,
                    "entry_time": ENTRY,
                    "exit_time": EXIT,
                    "entry_price": 50000.0,
                    "exit_price": 51000.0,
                    "quantity": 0.1,
                    "gross_pnl": 102.0,
                    "net_pnl": 100.0,
                    "exit_reason": "signal",
                }
            ],
        )
        return run_id
    finally:
        state.close()


def _write_paper_pair(logs_dir: Path, trader_id: str) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    with EventWriter(run_id=trader_id, base_path=logs_dir) as writer:
        writer.write(
            create_event(
                EventType.POSITION_OPEN,
                run_id=trader_id,
                strategy_name="s",
                data={
                    "position_id": "p1",
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "quantity": 0.1,
                    "entry_price": 50000.0,
                    "leverage": 1,
                },
                timestamp=datetime.fromisoformat(ENTRY),
            )
        )
        writer.write(
            create_event(
                EventType.POSITION_CLOSE,
                run_id=trader_id,
                strategy_name="s",
                data={
                    "position_id": "p1",
                    "symbol": "BTCUSDT",
                    "exit_price": 51000.0,
                    "gross_pnl": 102.0,
                    "net_pnl": 100.0,
                    "exit_reason": "signal",
                },
                timestamp=datetime.fromisoformat(EXIT),
            )
        )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *argv: str,
) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["paper_reconciliation.py", *argv])
    code = paper_reconciliation.main()
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_script_reads_paper_default_dir_and_db_trades(
    tmp_path: Path,
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Paper trades come from the event log, validation trades from the DB."""
    _ = tmp_db
    val_id = _insert_validation_run(tmp_db)
    logs_dir = tmp_path / "paper_logs"
    _write_paper_pair(logs_dir, "PAPER-042")

    code, out, err = _run(
        monkeypatch,
        capsys,
        "--paper-run",
        "PAPER-042",
        "--validation-run",
        str(val_id),
        "--logs-path",
        str(logs_dir),
    )

    assert code == 0, err
    assert re.search(r"Matched:\s+1\b", out), out
    assert "Parity rate:" in out and "100.0%" in out


def test_script_no_trades_on_either_side_exits_one(
    tmp_path: Path,
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty comparison is a failure, not a TypeError or a 100% parity."""
    val_id = _insert_validation_run(tmp_db)
    state = StateManager(db_path=tmp_db)
    state.conn.execute("DELETE FROM trades WHERE run_id = ?", (val_id,))
    state.conn.commit()
    state.close()
    logs_dir = tmp_path / "paper_logs"
    logs_dir.mkdir()
    (logs_dir / "PAPER-043.jsonl").write_text("")

    code, out, err = _run(
        monkeypatch,
        capsys,
        "--paper-run",
        "PAPER-043",
        "--validation-run",
        str(val_id),
        "--logs-path",
        str(logs_dir),
    )

    assert code == 1
    assert "FAIL: no trades on either side" in out + err
