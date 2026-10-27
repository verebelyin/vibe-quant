"""WFA / purged-k-fold re-runs bind the typed (string/bool) row params (vibe-quant-ger69).

The overfitting gates must test the strategy the screener actually ran. Since
vibe-quant-na00j a ``sweep_results.parameters`` row also carries the run's fixed
top-level overrides under their compiled config-field names, typed: e.g.
``stop_loss_type="atr"``, ``log_events=False``. The sweep channel is numeric-only
(vibe-quant-fahwt), so the typed values must ride the screening runner's fixed
override channel (``strategy_overrides=``) instead of being silently dropped.

Every assertion goes through the public overfitting entry
(``OverfittingPipeline.run`` with a real ``NTWFARunner`` / ``NTPurgedKFoldRunner``)
and the REAL ``NTScreeningRunner``: only its engine run (``run_engine``) and the
catalog data preflight are replaced. The strategy is instantiated from the runner's
own ``ImportableStrategyConfig`` with NT's ``StrategyFactory``, so the assertions
are on the ACTUAL ``strategy.config`` the engine would have run.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner
from vibe_quant.overfitting.nt_runner import NTWFARunner
from vibe_quant.overfitting.pipeline import OverfittingPipeline
from vibe_quant.overfitting.types import FilterConfig
from vibe_quant.screening.nt_runner import UnknownStrategyParamError

if TYPE_CHECKING:
    from pathlib import Path


# A DSL that really compiles: its config declares stop_loss_type / log_events, so the
# typed values below land on real generated-config fields.
_DSL: dict[str, Any] = {
    "name": "typed_params_probe",
    "timeframe": "1m",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi < 30"]},
    "exit_conditions": {"long": ["rsi > 70"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 3.0},
}

_RUN_ID = 42
_START = "2024-01-01"
_END = "2025-12-31"

# 1000 4h bars: enough for the default 5-fold split of n_samples=1000.
_BAR_TS_NS = [
    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1e9) + i * 4 * 3600 * 10**9
    for i in range(1000)
]

_INSTRUMENT = "BTCUSDT-PERP.BINANCE"


def _seed_db(tmp_path: Path, row_params: dict[str, object]) -> Path:
    """One screening run with a single sweep_results row carrying ``row_params``."""
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
            run_mode TEXT NOT NULL,
            symbols TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            parameters JSON NOT NULL
        );
        CREATE TABLE sweep_results (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            parameters TEXT NOT NULL,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            max_drawdown REAL,
            total_return REAL,
            profit_factor REAL,
            win_rate REAL,
            total_trades INTEGER,
            total_fees REAL,
            total_funding REAL,
            skewness REAL,
            kurtosis REAL,
            mtm_sharpe_daily REAL,
            mtm_skewness REAL,
            mtm_kurtosis REAL,
            mtm_days INTEGER,
            is_pareto_optimal BOOLEAN DEFAULT 0,
            passed_deflated_sharpe BOOLEAN,
            passed_walk_forward BOOLEAN,
            passed_purged_kfold BOOLEAN,
            starting_balance REAL
        );
        CREATE TABLE backtest_results (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            starting_balance REAL
        );
        """
    )
    conn.execute(
        "INSERT INTO strategies (id, name, dsl_config) VALUES (?,?,?)",
        (1, "typed_params_probe", json.dumps(_DSL)),
    )
    conn.execute(
        "INSERT INTO backtest_runs VALUES (?,?,?,?,?,?,?,?)",
        (
            _RUN_ID,
            1,
            "screening",
            json.dumps(["BTCUSDT"]),
            "1m",
            _START,
            _END,
            "{}",
        ),
    )
    conn.execute(
        "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_return, "
        "starting_balance) VALUES (?,?,?,?,?)",
        (_RUN_ID, json.dumps(row_params), 1.5, 0.2, 1000.0),
    )
    conn.execute(
        "INSERT INTO backtest_results (run_id, starting_balance) VALUES (?,?)",
        (_RUN_ID, 1000.0),
    )
    conn.commit()
    conn.close()
    return db


class _Capture:
    """The strategies the runner built for the engine, and their decoded config dicts."""

    def __init__(self) -> None:
        self.configs: list[Any] = []
        self.raw: list[dict[str, Any]] = []


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> _Capture:
    """Replace only the engine run, BELOW the runner's engine-spec construction.

    The real ``NTScreeningRunner`` compiles the DSL, resolves the sweep keys and the
    fixed overrides, and hands its ``ImportableStrategyConfig`` to ``run_engine``;
    this fixture then instantiates that config with NT's ``StrategyFactory`` and keeps
    the actual ``strategy.config``. The runner, the resolver and the spec construction
    are all production code, so a value lost between the constructor and the config
    fails here. The catalog data preflight and the engine execution are the only
    substitutions (the fixtures carry no catalog).
    """
    from nautilus_trader.trading.config import StrategyFactory

    import vibe_quant.screening.nt_runner as ntr

    cap = _Capture()

    def _fake_run_engine(
        spec: Any, consume: Any, *, prepare: Any = None
    ) -> Any:
        for importable in spec.strategy_configs:
            strategy = StrategyFactory.create(importable)
            cap.configs.append(strategy.config)
            cap.raw.append(dict(importable.config))
        if prepare is not None:
            prepare(SimpleNamespace(get_engines=list))
        return consume(SimpleNamespace(engine=None, bt_result=None))

    monkeypatch.setattr(ntr, "run_engine", _fake_run_engine)
    monkeypatch.setattr(ntr, "require_bars_in_window", lambda *a, **k: None)
    monkeypatch.setattr(
        NTPurgedKFoldRunner,
        "_load_bar_timestamps",
        lambda self: setattr(self, "_bar_ts_ns", list(_BAR_TS_NS)),
    )
    return cap


def _built_config(**params: Any) -> dict[str, Any]:
    """The config dict a 1m screening build makes from ``params`` (fill mode: delay 1.0)."""
    return {
        "instrument_id": _INSTRUMENT,
        **params,
        "execution_delay_probability": 1.0,
    }


def _run_wfa(db: Path) -> Any:
    pipeline = OverfittingPipeline(db, wfa_runner=NTWFARunner(run_id=_RUN_ID, db_path=db))
    try:
        return pipeline.run(
            run_id=_RUN_ID,
            config=FilterConfig.wfa_only(),
            data_start=date.fromisoformat(_START),
            data_end=date.fromisoformat(_END),
        )
    finally:
        pipeline.close()


def _run_cv(db: Path) -> Any:
    pipeline = OverfittingPipeline(
        db, cv_runner=NTPurgedKFoldRunner(run_id=_RUN_ID, db_path=db)
    )
    try:
        return pipeline.run(
            run_id=_RUN_ID,
            config=FilterConfig.cv_only(),
            n_samples=len(_BAR_TS_NS),
        )
    finally:
        pipeline.close()


# --- WFA: typed row params reach the config-bound channel, with their type -------


@pytest.mark.parametrize(
    "typed",
    [
        {"stop_loss_type": "atr", "log_events": False},
        {"stop_loss_type": "fixed_pct", "log_events": True},
    ],
)
def test_wfa_rerun_binds_typed_row_params(
    tmp_path: Path, capture: _Capture, typed: dict[str, object]
) -> None:
    """A WFA re-run of a typed row must run the strategy the sweeper ran."""
    db = _seed_db(tmp_path, typed)

    _run_wfa(db)

    assert capture.configs, "WFA never built a strategy"
    for config in capture.configs:
        assert config.stop_loss_type == typed["stop_loss_type"]
        assert config.log_events is typed["log_events"]
    # Nothing rides the numeric sweep channel: the override pair is the whole row.
    assert capture.raw == [_built_config(**typed)] * len(capture.raw)


def test_wfa_rerun_numeric_only_row_is_byte_identical(
    tmp_path: Path, capture: _Capture
) -> None:
    """A numeric-only row binds exactly as before: same config dict, no typed override."""
    db = _seed_db(tmp_path, {"rsi_period": 14, "stop_loss_percent": 2.5})

    _run_wfa(db)

    assert capture.configs
    assert capture.raw == [
        _built_config(rsi_period=14, stop_loss_percent=2.5)
    ] * len(capture.raw)
    for config in capture.configs:
        assert config.rsi_period == 14
        assert config.stop_loss_percent == 2.5
        assert config.stop_loss_type == "fixed_pct"


def test_wfa_rerun_mixed_row_splits_channels(tmp_path: Path, capture: _Capture) -> None:
    """A real na00j row carries both: the grid point stays swept, the override typed."""
    db = _seed_db(tmp_path, {"rsi_period": 9, "stop_loss_type": "atr", "log_events": True})

    _run_wfa(db)

    assert capture.configs
    for config in capture.configs:
        assert config.stop_loss_type == "atr"
        assert config.log_events is True
        assert config.rsi_period == 9
    assert capture.raw == [
        _built_config(rsi_period=9, stop_loss_type="atr", log_events=True)
    ] * len(capture.raw)


# --- purged k-fold: same contract ------------------------------------------------


@pytest.mark.parametrize(
    "typed",
    [
        {"stop_loss_type": "atr", "log_events": False},
        {"stop_loss_type": "fixed_pct", "log_events": True},
    ],
)
def test_cv_rerun_binds_typed_row_params(
    tmp_path: Path, capture: _Capture, typed: dict[str, object]
) -> None:
    db = _seed_db(tmp_path, typed)

    _run_cv(db)

    # Two backtests per fold, every one of them bound to the row's typed values.
    assert len(capture.configs) == 10
    for config in capture.configs:
        assert config.stop_loss_type == typed["stop_loss_type"]
        assert config.log_events is typed["log_events"]
    assert capture.raw == [_built_config(**typed)] * len(capture.raw)


def test_cv_rerun_numeric_only_row_is_byte_identical(
    tmp_path: Path, capture: _Capture
) -> None:
    db = _seed_db(tmp_path, {"rsi_period": 14, "stop_loss_percent": 2.5})

    _run_cv(db)

    assert capture.configs
    assert capture.raw == [
        _built_config(rsi_period=14, stop_loss_percent=2.5)
    ] * len(capture.raw)
    for config in capture.configs:
        assert config.rsi_period == 14
        assert config.stop_loss_percent == 2.5


def test_cv_rerun_mixed_row_splits_channels(tmp_path: Path, capture: _Capture) -> None:
    db = _seed_db(tmp_path, {"rsi_period": 9, "stop_loss_type": "atr", "log_events": True})

    _run_cv(db)

    assert capture.configs
    for config in capture.configs:
        assert config.stop_loss_type == "atr"
        assert config.log_events is True
        assert config.rsi_period == 9
    assert capture.raw == [
        _built_config(rsi_period=9, stop_loss_type="atr", log_events=True)
    ] * len(capture.raw)


# --- fail closed: an unknown typed key is never silently dropped ------------------


def test_wfa_rerun_unknown_typed_key_raises(tmp_path: Path, capture: _Capture) -> None:
    """The runner surfaces the resolver's error naming the key (no silent drop)."""
    db = _seed_db(tmp_path, {"stop_loss_type": "atr", "stop_lossy_typo": "x"})
    runner = NTWFARunner(run_id=_RUN_ID, db_path=db)

    with pytest.raises(UnknownStrategyParamError, match="stop_lossy_typo"):
        runner.backtest("ignored", date(2024, 2, 1), date(2024, 3, 1), {"stop_lossy_typo": "x"})

    assert capture.configs == []


def test_wfa_pipeline_fails_candidate_closed_on_unknown_typed_key(
    tmp_path: Path, capture: _Capture, caplog: pytest.LogCaptureFixture
) -> None:
    """Through the gate: the candidate fails, and the key is named in the log."""
    db = _seed_db(tmp_path, {"stop_lossy_typo": "x"})

    with caplog.at_level(logging.WARNING):
        result = _run_wfa(db)

    assert result.candidates[0].passed_wfa is False
    assert "stop_lossy_typo" in caplog.text
    assert capture.configs == []


def test_cv_rerun_unknown_typed_key_fails_closed(
    tmp_path: Path, capture: _Capture
) -> None:
    db = _seed_db(tmp_path, {"stop_lossy_typo": "x"})

    with pytest.raises(UnknownStrategyParamError, match="stop_lossy_typo"):
        _run_cv(db)

    assert capture.configs == []
