"""Screening + overfitting honour the run's ``SizingSettings`` (bead vibe-quant-2gyk1.2, task S2).

Binding rows: S14 (a sizing key is a RUN-level setting, never a sweep parameter -- at the pure
resolver, at the runner constructor and at run time), S15 (a corrupt STORED value fails the
screening run through ``__main__`` and raises out of both overfitting runner constructors),
S19 (an all-default sizing leaves the strategy config dict byte-identical), S20 (the factory,
the pipeline and the overfitting runners thread the SAME :class:`SizingSettings`) and S24 (the
liquidation scan keeps the run's leverage while the sizing rides along).

The rule itself (S1-S13) is pinned by ``tests/unit/test_sizing_settings.py``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner
from vibe_quant.overfitting.nt_runner import NTWFARunner
from vibe_quant.screening.nt_runner import (
    NTScreeningRunner,
    UnknownStrategyParamError,
    resolve_strategy_params,
)
from vibe_quant.screening.pipeline import create_screening_pipeline
from vibe_quant.sizing_settings import SizingConfigError, SizingSettings

if TYPE_CHECKING:
    from pathlib import Path

_DSL_NAME = "sizing_probe_1m"


def _dsl_dict(*, sweep: dict[str, Any] | None = None) -> dict[str, Any]:
    """A valid 1m DSL: no tick set, no catalog read, EMA only (nothing needs an archive)."""
    dsl: dict[str, Any] = {
        "name": _DSL_NAME,
        "timeframe": "1m",
        "indicators": {"ema": {"type": "EMA", "period": 3}},
        "entry_conditions": {"long": ["ema > 0"]},
        "stop_loss": {"type": "fixed_pct", "percent": 10.0},
        "take_profit": {"type": "fixed_pct", "percent": 90.0},
    }
    if sweep is not None:
        dsl["sweep"] = sweep
    return dsl


def _runner(tmp_path: Path, **kwargs: Any) -> NTScreeningRunner:
    """A real runner on the probe DSL (nothing is compiled or read until it is used)."""
    return NTScreeningRunner(
        _dsl_dict(),
        ["BTCUSDT"],
        "2024-01-01",
        "2024-01-02",
        catalog_path=str(tmp_path),
        funding_archive_path=str(tmp_path / "no_funding.db"),
        **kwargs,
    )


def _strategy_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> dict[str, Any]:
    """The strategy config dict ``_build_run_config`` hands NT (bar coverage stubbed out).

    ``require_bars_in_window`` is a data-availability guard with its own tests; it is
    stubbed so this file checks the config dict without a parquet catalog.
    """
    import vibe_quant.screening.nt_runner as ntr

    monkeypatch.setattr(ntr, "require_bars_in_window", lambda *_a, **_k: None)
    run_config, _tick_dirs = _runner(tmp_path, **kwargs)._build_run_config({})
    assert run_config is not None
    assert run_config.engine is not None
    return dict(run_config.engine.strategies[0].config)


def _seed(tmp_path: Path, name: str, *, parameters: dict[str, object]) -> tuple[Path, int]:
    """A minimal state DB: one strategy + one screening run with stored ``parameters``.

    Carries the ``backtest_results`` row ``launch_settings.parent_effective_balance`` reads
    (R11), so the
    overfitting constructors get past the balance lookup and reach the sizing parse.
    """
    db = tmp_path / f"{name}.db"
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
            timeframe TEXT,
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
    conn.execute(
        "INSERT INTO strategies (id, name, dsl_config) VALUES (?,?,?)",
        (1, "stub", '{"name": "stub", "timeframe": "4h", "indicators": {}}'),
    )
    conn.execute(
        "INSERT INTO backtest_runs (id, strategy_id, symbols, timeframe, parameters) "
        "VALUES (?,?,?,?,?)",
        (42, 1, '["BTCUSDT"]', "4h", json.dumps(parameters)),
    )
    conn.execute(
        "INSERT INTO backtest_results (run_id, starting_balance) VALUES (?,?)", (42, 1000.0)
    )
    conn.commit()
    conn.close()
    return db, 42


class _NoTicks:
    """``ParentFillTicks`` stand-in: no tick set for any window."""

    def get(self, _start: str, _end: str) -> None:
        return None


# ------------------------------------------------------------------ S14: sweep keys


def test_sweep_sizing_key_rejected(tmp_path: Path) -> None:
    """S14: a sizing key is a run-level setting, so a sweep row carrying one fails closed."""
    fields = ("instrument_id", "ema_period", "max_position_pct", "risk_per_trade")
    for key, value in (("max_position_pct", 1.0), ("risk_per_trade", 0.01)):
        with pytest.raises(UnknownStrategyParamError) as excinfo:
            resolve_strategy_params({key: value}, fields)
        assert str(excinfo.value) == (
            f"{key} is a run-level sizing setting (parameters.{key}), not a sweep parameter"
        )
    # The DSL's own sweep: rejected at construction, before any engine could exist.
    with pytest.raises(UnknownStrategyParamError, match="not a sweep parameter"):
        NTScreeningRunner(
            _dsl_dict(sweep={"max_position_pct": [0.5, 1.0]}),
            ["BTCUSDT"],
            "2024-01-01",
            "2024-01-02",
            catalog_path=str(tmp_path),
        )
    # A legacy stored sweep combination reaches the same rejection at run time.
    with pytest.raises(UnknownStrategyParamError, match="not a sweep parameter"):
        _runner(tmp_path)({"max_position_pct": 1.0})


def test_dotted_sizing_alias_rejected_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S14b: a spelling that normalizes to a sizing field is rejected, naming the normalized key.

    Through the resolver, the constructor (DSL sweep) and config construction -- before
    the config-field lookup, which would otherwise match the generated config's own field.
    """
    import vibe_quant.screening.nt_runner as ntr

    monkeypatch.setattr(ntr, "require_bars_in_window", lambda *_a, **_k: None)
    fields = ("instrument_id", "ema_period", "max_position_pct", "risk_per_trade")
    for alias, norm in (
        ("max.position.pct", "max_position_pct"),
        ("risk.per.trade", "risk_per_trade"),
    ):
        expected = f"{norm} is a run-level sizing setting (parameters.{norm}), not a sweep parameter"
        with pytest.raises(UnknownStrategyParamError) as excinfo:
            resolve_strategy_params({alias: 1.0}, fields)
        assert str(excinfo.value) == expected
        with pytest.raises(UnknownStrategyParamError) as excinfo:
            NTScreeningRunner(
                _dsl_dict(sweep={alias: [0.01, 0.02]}),
                ["BTCUSDT"],
                "2024-01-01",
                "2024-01-02",
                catalog_path=str(tmp_path),
            )
        assert str(excinfo.value) == expected
        with pytest.raises(UnknownStrategyParamError) as excinfo:
            _runner(tmp_path)._build_run_config({alias: 1.0})  # noqa: SLF001
        assert str(excinfo.value) == expected


def test_max_margin_not_sweepable() -> None:
    """R42: ``max_margin_pct`` is a run-level sizing setting; ``margin_leverage`` (what the
    runner derives from the run leverage) is runner-owned, so neither can be swept."""
    fields = ("instrument_id", "ema_period", "max_margin_pct", "margin_leverage")
    with pytest.raises(UnknownStrategyParamError) as excinfo:
        resolve_strategy_params({"max_margin_pct": 0.5}, fields)
    assert str(excinfo.value) == (
        "max_margin_pct is a run-level sizing setting (parameters.max_margin_pct), "
        "not a sweep parameter"
    )
    with pytest.raises(UnknownStrategyParamError, match="margin_leverage"):
        resolve_strategy_params({"margin_leverage": 1.0}, fields)


def test_margin_limit_reaches_the_strategy_config_with_run_leverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R36/R37: on, the config gets the limit AND the run leverage; off, neither key exists."""
    off = _strategy_config(tmp_path, monkeypatch, leverage=4)
    assert "max_margin_pct" not in off
    assert "margin_leverage" not in off
    on = _strategy_config(
        tmp_path, monkeypatch, leverage=4, sizing=SizingSettings(max_margin_pct=0.5)
    )
    assert on["max_margin_pct"] == 0.5
    assert on["margin_leverage"] == 4.0


def test_screening_logs_refused_entry_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R40 (screening half): refused entries are summed over the strategies and logged;
    an off run logs nothing about the limit."""
    import nautilus_trader.backtest.node as nt_node

    import vibe_quant.screening.nt_runner as ntr

    strategies = [SimpleNamespace(_margin_refusals=2), SimpleNamespace(_margin_refusals=3)]
    engine = SimpleNamespace(
        get_result=lambda: SimpleNamespace(total_orders=0, total_events=0),
        trader=SimpleNamespace(strategies=lambda: strategies),
        kernel=SimpleNamespace(
            get_log_guard=lambda: None,
            portfolio=SimpleNamespace(
                analyzer=SimpleNamespace(register_statistic=lambda _s: None)
            ),
        ),
        reset=lambda: None,
    )

    class _Node:
        def __init__(self, configs: object) -> None:
            pass

        def build(self) -> None:
            pass

        def run(self) -> None:
            pass

        def get_engines(self) -> list[Any]:
            return [engine]

        def get_engine(self, _id: object) -> Any:
            return engine

        def dispose(self) -> None:
            pass

    monkeypatch.setattr(nt_node, "BacktestNode", _Node)
    monkeypatch.setattr(ntr, "cleanup_epoch_parquet", lambda *_a, **_k: None, raising=False)
    for sizing, expected in (
        (SizingSettings(max_margin_pct=0.5), True),
        (SizingSettings(), False),
    ):
        runner = _runner(tmp_path, sizing=sizing)
        runner._resolved_catalog_path = tmp_path  # noqa: SLF001
        monkeypatch.setattr(
            runner, "_build_run_config", lambda _p: (SimpleNamespace(id="r"), [])
        )
        monkeypatch.setattr(
            runner, "_extract_metrics", lambda params, *_a, **_k: SimpleNamespace(
                sharpe_ratio=0.0, total_trades=0, total_return=0.0, max_drawdown=0.0,
                execution_time_seconds=0.0,
            ),
        )
        caplog.clear()
        with caplog.at_level("INFO", logger="vibe_quant.screening.nt_runner"):
            runner._run_backtest({}, time.time())  # noqa: SLF001
        logged = [r.getMessage() for r in caplog.records if "margin limit" in r.getMessage()]
        assert logged == (["Screening: margin limit refused 5 entries"] if expected else [])


# ------------------------------------------------- S19: the default config is untouched


def test_default_config_dict_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S19: an all-default sizing adds no key -- the config dict is what it was before."""
    plain = _strategy_config(tmp_path, monkeypatch)
    default = _strategy_config(tmp_path, monkeypatch, sizing=SizingSettings())
    assert default == plain
    assert "max_position_pct" not in plain
    assert "risk_per_trade" not in plain


def test_non_default_sizing_reaches_the_strategy_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S19/S20: only the keys the run moved off the default are merged into the config."""
    cfg = _strategy_config(
        tmp_path, monkeypatch, sizing=SizingSettings(max_position_pct=1.0)
    )
    assert cfg["max_position_pct"] == 1.0
    assert "risk_per_trade" not in cfg
    risk_only = _strategy_config(
        tmp_path, monkeypatch, sizing=SizingSettings(risk_per_trade=0.01)
    )
    assert risk_only["risk_per_trade"] == 0.01
    assert "max_position_pct" not in risk_only
    both = _strategy_config(
        tmp_path, monkeypatch, sizing=SizingSettings(max_position_pct=1.0, risk_per_trade=0.01)
    )
    assert both["max_position_pct"] == 1.0
    assert both["risk_per_trade"] == 0.01


# ------------------------------------------------------- S20: every runner gets the sizing


def test_pipeline_and_overfitting_runners_get_sizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S20: the pipeline AND both overfitting runners thread the run's own SizingSettings.

    One assertion per constructor site (a dropped forwarding at any one of them must fail).
    """
    import vibe_quant.screening.nt_runner as ntr
    import vibe_quant.screening.pipeline as sp

    captured: list[dict[str, Any]] = []

    class _Capture:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

        def __call__(self, _params: dict[str, float | int]) -> Any:
            return SimpleNamespace(sharpe_ratio=1.0, total_return=0.1)

    monkeypatch.setattr(ntr, "NTScreeningRunner", _Capture)
    monkeypatch.setattr(sp, "ParentFillTicks", lambda *_a, **_k: _NoTicks())

    create_screening_pipeline(
        validate_strategy_dict(_dsl_dict()),
        symbols=["BTCUSDT"],
        start_date="2024-01-01",
        end_date="2024-02-01",
        catalog_path=str(tmp_path),
        parameters={"max_position_pct": 1.0, "risk_per_trade": 0.01},
    )
    db, run_id = _seed(tmp_path, "overfit_sizing", parameters={"max_position_pct": 2.0})
    NTWFARunner(run_id=run_id, db_path=db).backtest(
        "x", date(2024, 2, 1), date(2024, 3, 1), {}
    )
    monkeypatch.setattr(NTPurgedKFoldRunner, "_load_bar_timestamps", lambda _self: None)
    NTPurgedKFoldRunner(
        run_id=run_id, db_path=db, catalog_path=tmp_path / "catalog"
    )._backtest("2024-02-01", "2024-03-01")  # noqa: SLF001

    assert [kwargs["sizing"] for kwargs in captured] == [
        SizingSettings(max_position_pct=1.0, risk_per_trade=0.01),
        SizingSettings(max_position_pct=2.0),
        SizingSettings(max_position_pct=2.0),
    ]


# ---------------------------------------------------- S15: a corrupt stored value fails


def test_runner_rejects_a_cap_above_its_leverage(tmp_path: Path) -> None:
    """N4 defence: the runner re-checks the cap against the leverage it runs at."""
    with pytest.raises(SizingConfigError) as excinfo:
        _runner(tmp_path, leverage=3, sizing=SizingSettings(max_position_pct=20))
    assert str(excinfo.value) == "max_position_pct (20.0) must not exceed leverage (3.0)"
    # equality is allowed: a cap exactly at the leverage is fully marginable (S13)
    assert (
        _runner(tmp_path, leverage=3, sizing=SizingSettings(max_position_pct=3))
        ._sizing.max_position_pct
        == 3.0
    )


def test_pipeline_rejects_stored_garbage_before_any_engine(tmp_path: Path) -> None:
    """S15/S13: the factory parses the run's sizing before the mock/real split."""
    dsl = validate_strategy_dict(_dsl_dict())
    with pytest.raises(SizingConfigError) as excinfo:
        create_screening_pipeline(
            dsl, symbols=["BTCUSDT"], catalog_path=str(tmp_path), parameters={"max_position_pct": -1}
        )
    assert str(excinfo.value) == "max_position_pct must be a number in (0, 125], got -1"
    with pytest.raises(SizingConfigError) as excinfo:
        create_screening_pipeline(
            dsl, symbols=["BTCUSDT"], catalog_path=str(tmp_path), parameters={"max_position_pct": 20}
        )
    assert str(excinfo.value) == "max_position_pct (20.0) must not exceed leverage (10.0)"


def test_factory_rejects_a_cap_above_leverage_even_with_mock(tmp_path: Path) -> None:
    """S13: the factory's own leverage check holds when no real runner would re-check it."""
    with pytest.raises(SizingConfigError) as excinfo:
        create_screening_pipeline(
            validate_strategy_dict(_dsl_dict()),
            symbols=["BTCUSDT"],
            catalog_path=str(tmp_path),
            use_mock=True,
            parameters={"max_position_pct": 20},
        )
    assert str(excinfo.value) == "max_position_pct (20.0) must not exceed leverage (10.0)"


def test_stored_garbage_fails_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """S15: a stored cap of -1 fails the RUN through ``__main__`` with the class prefix."""
    from vibe_quant.screening.__main__ import cmd_run

    db = tmp_path / "screening_garbage.db"
    state = StateManager(db)
    try:
        state.create_strategy(_DSL_NAME, {"timeframe": "1m", "indicators": {}})
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="1m",
            start_date="2024-01-01",
            end_date="2024-02-01",
            parameters={"dsl_override": _dsl_dict(), "max_position_pct": -1},
        )
        state.update_backtest_run_status(run_id, "running")
    finally:
        state.close()

    assert cmd_run(argparse.Namespace(run_id=run_id, db=str(db))) == 1

    state = StateManager(db)
    try:
        run = state.get_backtest_run(run_id)
    finally:
        state.close()
    assert run is not None
    assert run["status"] == "failed"
    assert run["error_message"] == (
        "SizingConfigError: max_position_pct must be a number in (0, 125], got -1"
    )
    assert "max_position_pct must be a number in (0, 125], got -1" in capsys.readouterr().out


def test_main_prints_the_screening_sizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The run log names the sizing every screening run was launched with."""
    from vibe_quant.screening import pipeline as pipeline_mod
    from vibe_quant.screening.__main__ import cmd_run
    from vibe_quant.screening.types import BacktestMetrics

    # One worker: the stand-in runner is a local class and cannot cross a process pool.
    monkeypatch.setattr(pipeline_mod, "cpu_count", lambda: 1)

    class _OkRunner:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self, params: dict[str, float | int]) -> BacktestMetrics:
            return BacktestMetrics(parameters=params, sharpe_ratio=1.0, total_return=0.1)

    monkeypatch.setattr("vibe_quant.screening.nt_runner.NTScreeningRunner", _OkRunner)

    db = tmp_path / "screening_printed.db"
    state = StateManager(db)
    try:
        state.create_strategy(_DSL_NAME, {"timeframe": "1m", "indicators": {}})
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="1m",
            start_date="2024-01-01",
            end_date="2024-02-01",
            parameters={"dsl_override": _dsl_dict(), "max_position_pct": 1.0},
        )
    finally:
        state.close()

    assert cmd_run(argparse.Namespace(run_id=run_id, db=str(db))) == 0
    assert (
        "Screening sizing: max_position_pct=1 risk_per_trade=0.02" in capsys.readouterr().out
    )


def test_overfitting_runner_constructor_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S15: a corrupt stored sizing raises out of BOTH overfitting constructors (N4)."""
    monkeypatch.setattr(NTPurgedKFoldRunner, "_load_bar_timestamps", lambda _self: None)
    db, run_id = _seed(tmp_path, "garbage_wfa", parameters={"max_position_pct": -1})
    with pytest.raises(SizingConfigError) as excinfo:
        NTWFARunner(run_id=run_id, db_path=db)
    assert str(excinfo.value) == "max_position_pct must be a number in (0, 125], got -1"
    with pytest.raises(SizingConfigError, match="got -1"):
        NTPurgedKFoldRunner(run_id=run_id, db_path=db, catalog_path=tmp_path / "catalog")

    # The CV constructor re-checks the cap against the leverage it will rerun at.
    db3, run_id3 = _seed(
        tmp_path, "cv_above_leverage", parameters={"max_position_pct": 20, "leverage": 3}
    )
    with pytest.raises(SizingConfigError) as excinfo:
        NTPurgedKFoldRunner(run_id=run_id3, db_path=db3, catalog_path=tmp_path / "catalog")
    assert str(excinfo.value) == "max_position_pct (20.0) must not exceed leverage (3.0)"

    # A cap above the run's own leverage is the same fail-closed constructor (never a
    # silently smaller position than the parent screened with).
    db2, run_id2 = _seed(
        tmp_path, "above_leverage", parameters={"max_position_pct": 20, "leverage": 3}
    )
    with pytest.raises(SizingConfigError) as excinfo:
        NTWFARunner(run_id=run_id2, db_path=db2)
    assert str(excinfo.value) == "max_position_pct (20.0) must not exceed leverage (3.0)"


# ------------------------------------------------------ S24: the overlay keeps the venue


def test_overlay_receives_unchanged_leverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S24: a mode-on run with a non-default cap still scans at ``parameters.leverage``.

    The sizing rides along in the strategy config; it never becomes the liquidation
    leverage (a cap of 1.0 at 3x must not liquidate as if it were 1x).
    """
    import vibe_quant.liquidation.apply as apply_module

    captured: list[Any] = []
    real = apply_module.apply_liquidation

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs["leverage"])
        return real(*args, **kwargs)

    monkeypatch.setattr(apply_module, "apply_liquidation", spy)
    runner = NTScreeningRunner(
        _dsl_dict(),
        ["BTCUSDT"],
        "2025-01-01",
        "2025-01-04",
        funding_archive_path=str(tmp_path / "no_funding.db"),
        liquidation_mode="isolated",
        leverage=3,
        starting_balance=1000,
        sizing=SizingSettings(max_position_pct=1.0),
    )
    # What _build_run_config would have set: a 1m run on a catalog with no positions.
    runner._all_timeframes = {"1m"}  # noqa: SLF001
    runner._resolved_catalog_path = tmp_path  # noqa: SLF001
    engine = SimpleNamespace(
        kernel=SimpleNamespace(
            cache=SimpleNamespace(positions=list, position_snapshots=list, bars=list)
        )
    )
    bt_result = SimpleNamespace(
        total_positions=0,
        elapsed_time=0.0,
        stats_pnls={"USDT": {"PnL% (total)": 0.0, "Win Rate": 0.0}},
        stats_returns={},
    )

    runner._extract_metrics({}, bt_result, engine, time.time(), 1000.0)  # noqa: SLF001

    assert captured == [3.0]
    assert runner._leverage == 3.0  # noqa: SLF001
    assert runner._venue_config().default_leverage == Decimal("3")
    assert runner.last_liquidation_notes is not None
