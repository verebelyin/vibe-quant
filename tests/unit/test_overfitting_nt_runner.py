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

from vibe_quant.fees import FeeSchedule
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
            symbols TEXT NOT NULL,
            parameters JSON NOT NULL
        );
        CREATE TABLE sweep_results (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            starting_balance REAL
        );
        CREATE TABLE backtest_results (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            starting_balance REAL
        );
        """
    )
    dsl = {"name": "stub", "timeframe": "4h", "indicators": {}}
    conn.execute(
        "INSERT INTO strategies (id, name, dsl_config) VALUES (?,?,?)",
        (1, "stub", json.dumps(dsl)),
    )
    conn.execute(
        "INSERT INTO backtest_runs (id, strategy_id, symbols, parameters) VALUES (?,?,?,?)",
        (42, 1, json.dumps(["BTCUSDT", "ETHUSDT"]), "{}"),
    )
    conn.execute(
        "INSERT INTO backtest_results (run_id, starting_balance) VALUES (?,?)", (42, 1000.0)
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
        CREATE TABLE backtest_runs (id INTEGER PRIMARY KEY, strategy_id INTEGER, symbols TEXT,
                                    parameters JSON NOT NULL);
        CREATE TABLE sweep_results (id INTEGER PRIMARY KEY, run_id INTEGER,
                                    starting_balance REAL);
        CREATE TABLE backtest_results (id INTEGER PRIMARY KEY, run_id INTEGER,
                                       starting_balance REAL);
        """
    )
    # run points at a non-existent strategy_id.
    conn.execute(
        "INSERT INTO backtest_runs VALUES (?,?,?,?)",
        (7, 99, json.dumps(["X"]), "{}"),
    )
    conn.execute(
        "INSERT INTO backtest_results (run_id, starting_balance) VALUES (?,?)", (7, 1000.0)
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


# ---------------------------------------------------------------------------
# E2 (vibe-quant-ldbwy + vibe-quant-rawtz): the runner must backtest with the
# PARENT run's own fee schedule, not a silent VIP0 default.
# ---------------------------------------------------------------------------


def _seed_fee_run(tmp_path: Path, *, parameters: dict[str, object]) -> Path:
    """Minimal state DB whose run row carries ``parameters`` (E2 contract)."""
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
            symbols TEXT NOT NULL,
            parameters JSON NOT NULL
        );
        CREATE TABLE sweep_results (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            starting_balance REAL
        );
        CREATE TABLE backtest_results (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            starting_balance REAL
        );
        """
    )
    dsl = {"name": "stub", "timeframe": "4h", "indicators": {}}
    conn.execute(
        "INSERT INTO strategies (id, name, dsl_config) VALUES (?,?,?)",
        (1, "stub", json.dumps(dsl)),
    )
    conn.execute(
        "INSERT INTO backtest_runs (id, strategy_id, symbols, parameters) VALUES (?,?,?,?)",
        (42, 1, json.dumps(["BTCUSDT"]), json.dumps(parameters)),
    )
    conn.execute(
        "INSERT INTO backtest_results (run_id, starting_balance) VALUES (?,?)", (42, 1000.0)
    )
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def fee_runner_kwargs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Capture the kwargs every NTScreeningRunner is built with (E2)."""
    import vibe_quant.screening.nt_runner as ntr
    import vibe_quant.screening.pipeline as sp

    captured: list[dict[str, object]] = []

    class _NoTicks:
        def get(self, _start: str, _end: str) -> None:
            return None

    def _build(**kwargs: object) -> object:
        captured.append(kwargs)

        def _run(_params: dict[str, object]) -> object:
            return type("M", (), {"sharpe_ratio": 1.0, "total_return": 0.1})()

        return _run

    monkeypatch.setattr(ntr, "NTScreeningRunner", _build)
    monkeypatch.setattr(sp, "ParentFillTicks", lambda *a, **k: _NoTicks())
    return captured


def test_wfa_runner_builds_default_fee_schedule(
    tmp_path: Path, fee_runner_kwargs: list[dict[str, object]]
) -> None:
    """E2 row 1: no fee params -> VIP0/no-BNB, so numbers stay unchanged."""
    import datetime

    runner = NTWFARunner(run_id=42, db_path=_seed_fee_run(tmp_path, parameters={}))
    runner.backtest("x", datetime.date(2024, 2, 1), datetime.date(2024, 3, 1), {})

    assert fee_runner_kwargs[0]["launch"].fee_schedule == FeeSchedule()
    assert fee_runner_kwargs[0]["launch"].fee_schedule.is_default is True  # type: ignore[union-attr]


def test_wfa_runner_reads_bnb_discount(
    tmp_path: Path, fee_runner_kwargs: list[dict[str, object]]
) -> None:
    """E2 row 2: the parent run's BNB discount reaches the runner (rawtz).

    VIP0 is the only tier that ships, so the reachable non-default schedule
    is VIP0 + BNB; the runner must forward it, not the plain default.
    """
    import datetime

    runner = NTWFARunner(
        run_id=42, db_path=_seed_fee_run(tmp_path, parameters={"bnb_discount": True})
    )
    runner.backtest("x", datetime.date(2024, 2, 1), datetime.date(2024, 3, 1), {})

    assert fee_runner_kwargs[0]["launch"].fee_schedule == FeeSchedule(tier="VIP0", bnb_discount=True)


def test_wfa_runner_unknown_fee_tier_fails_closed(tmp_path: Path) -> None:
    """E2 row 3: an unknown tier raises at construction, never a silent VIP0."""
    with pytest.raises(ValueError, match="unknown fee tier"):
        NTWFARunner(run_id=42, db_path=_seed_fee_run(tmp_path, parameters={"fee_tier": "VIP1"}))
