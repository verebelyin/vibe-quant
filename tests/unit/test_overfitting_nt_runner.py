"""Tests for overfitting NTWFARunner adapter (bd-yfbg).

Covers construction-time DSL+symbols resolution and the
single-combo optimize() collapse. The real NT backtest path is covered
by integration smoke tests run manually — unit tests here focus on the
glue layer.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import pytest

from vibe_quant.overfitting.nt_runner import NTWFARunner

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def stub_db(tmp_path: Path) -> Path:
    """Minimal state DB with a strategy + run for NTWFARunner to resolve."""
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dsl_config TEXT NOT NULL
        );
        CREATE TABLE backtest_runs (
            id INTEGER PRIMARY KEY,
            strategy_id INTEGER,
            symbols TEXT NOT NULL
        );
        """
    )
    dsl = {"name": "stub", "timeframe": "4h", "indicators": {}}
    conn.execute(
        "INSERT INTO strategies (id, name, dsl_config) VALUES (?,?,?)",
        (1, "stub", json.dumps(dsl)),
    )
    conn.execute(
        "INSERT INTO backtest_runs (id, strategy_id, symbols) VALUES (?,?,?)",
        (42, 1, json.dumps(["BTCUSDT", "ETHUSDT"])),
    )
    conn.commit()
    conn.close()
    return db


def test_resolves_dsl_and_symbols(stub_db: Path) -> None:
    runner = NTWFARunner(run_id=42, db_path=stub_db)
    # Private attrs — test is looking at the glue, not the NT call.
    assert runner._dsl_dict["name"] == "stub"  # noqa: SLF001
    assert runner._symbols == ["BTCUSDT", "ETHUSDT"]  # noqa: SLF001


def test_missing_run_raises(stub_db: Path) -> None:
    with pytest.raises(ValueError, match="backtest_runs.id=999"):
        NTWFARunner(run_id=999, db_path=stub_db)


def test_missing_strategy_raises(tmp_path: Path) -> None:
    db = tmp_path / "orphan.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT, dsl_config TEXT);
        CREATE TABLE backtest_runs (id INTEGER PRIMARY KEY, strategy_id INTEGER, symbols TEXT);
        """
    )
    # run points at a non-existent strategy_id.
    conn.execute(
        "INSERT INTO backtest_runs VALUES (?,?,?)",
        (7, 99, json.dumps(["X"])),
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="strategies.id=99"):
        NTWFARunner(run_id=7, db_path=db)


def test_optimize_collapses_single_combo_grid(stub_db: Path, monkeypatch) -> None:
    """optimize() must delegate to backtest() with the single combo the
    overfitting pipeline passes — we don't want any real GA search."""
    import datetime

    runner = NTWFARunner(run_id=42, db_path=stub_db)

    captured: dict[str, object] = {}

    def fake_backtest(
        _strategy_id: str,
        _start: datetime.date,
        _end: datetime.date,
        params: dict[str, object],
    ) -> tuple[float, float]:
        captured["params"] = params
        return 1.5, 0.2

    monkeypatch.setattr(runner, "backtest", fake_backtest)

    best, sharpe, ret = runner.optimize(
        strategy_id="ignored",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 6, 1),
        param_grid={"period": [14]},
    )
    assert best == {"period": 14}
    assert sharpe == 1.5
    assert ret == 0.2
    assert captured["params"] == {"period": 14}
