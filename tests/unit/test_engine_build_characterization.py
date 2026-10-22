"""Engine-build characterization (bead vibe-quant-rabmo.3, task S1).

Every cell drives ONE public entry point over the S1 matrix and pins the single
``BacktestRunConfig`` that tier hands to ``nautilus_trader.backtest.node.BacktestNode``.
rabmo.3 (``launch_settings``) and rabmo.4 (``engine_run``) must reproduce the same
configs, so this file is the refactor's safety net: it is green on base and must stay
green through both.

Capture (writes ``tests/unit/goldens/engine_build_matrix.json``)::

    VIBE_QUANT_CAPTURE_GOLDENS=1 pytest tests/unit/test_engine_build_characterization.py

Capture twice and compare the two files: they must be byte-identical.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nautilus_trader.backtest.node as nt_node
import pytest

from vibe_quant.data import catalog as catalog_mod
from vibe_quant.data.catalog import (
    CatalogManager,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.data.fill_ticks import FillTickSet
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.engine_run import loaded_timeframes
from vibe_quant.screening import pipeline as pipeline_mod
from vibe_quant.screening.nt_runner import screening_fill_mode
from vibe_quant.screening.pipeline import create_screening_pipeline
from vibe_quant.validation.extraction import date_to_ns
from vibe_quant.validation.runner import ValidationRunner

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = Path(__file__).parent / "goldens"
DSL_GOLDEN = GOLDENS / "strategy_239_dsl.json"
MATRIX_GOLDEN = GOLDENS / "engine_build_matrix.json"
BASE_COMMIT = "a557a612"

SYMBOLS = ["ETHUSDT", "BTCUSDT"]
START = "2024-01-01"
END = "2024-01-08"
START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 1, 8)
_DAY0_MS = 1_704_067_200_000
_BARS_PER_SYMBOL = 400
RUN_ID = 42

#: The S1 parameter matrix. WFA/CV take their balance from the parent run's
#: stored ``starting_balance`` instead of ``parameters`` (row R28).
PARAMS: dict[str, dict[str, Any]] = {
    "default": {},
    "VIP0+bnb": {"fee_tier": "VIP0", "bnb_discount": True},
    "isolated@20": {"liquidation_mode": "isolated", "leverage": 20},
    "cross@5": {"liquidation_mode": "cross", "leverage": 5},
    "off@3": {"liquidation_mode": "off", "leverage": 3},
    "cap2+risk.05": {"max_position_pct": 2, "risk_per_trade": 0.05},
    "margin .3": {"max_margin_pct": 0.3},
    "initial_balance 25000": {"initial_balance": 25000},
}
_NO_OUTBOX = "1m"

TIERS = ("screening", "validation", "wfa", "cv")


@dataclass(frozen=True)
class Cell:
    tier: str
    label: str
    timeframe: str = "4h"

    @property
    def cell_id(self) -> str:
        return f"{self.tier}-{self.label}"

    def params(self) -> dict[str, Any]:
        return dict(PARAMS.get(self.label, {}))

    @property
    def sweep_balance(self) -> float:
        # WFA/CV read the parent's recorded balance; the label is the matrix row.
        return 25000.0 if self.label == "initial_balance 25000" else 1000.0


CELLS = [
    *(Cell(tier, label) for tier in TIERS for label in PARAMS),
    *(Cell(tier, _NO_OUTBOX, timeframe=_NO_OUTBOX) for tier in TIERS),
]


class _Captured(BaseException):
    """Raised by the recorder: the run stops at the constructor it must not go past."""


#: ``[configs]`` of every BacktestNode construction in the current test.
_CALLS: list[list[Any]] = []


class _Recorder:
    """Stands in for ``BacktestNode``: records ``configs`` and stops the run."""

    def __init__(self, configs: list[Any]) -> None:
        _CALLS.append(list(configs))
        raise _Captured


@dataclass
class Harness:
    tmp: Path
    catalog: Path
    catalogs: dict[str, Path]
    configs: list[list[Any]]
    writes: list[str] = field(default_factory=list)

    def use_interval(self, interval: str) -> None:
        """Point the default catalog at bars of ``interval`` (the 1m matrix row)."""
        self.catalog = self.catalogs[interval]
        catalog_mod.DEFAULT_CATALOG_PATH = self.catalog

    def run_cell(self, cell: Cell) -> None:
        dsl_dict = _dsl(cell.timeframe)
        if cell.tier == "screening":
            self._screening(dsl_dict, cell.params())
        elif cell.tier == "validation":
            self._validation(dsl_dict, cell.params())
        elif cell.tier == "wfa":
            self._wfa(dsl_dict, cell.params(), cell.sweep_balance)
        else:
            self._cv(dsl_dict, cell.params(), cell.sweep_balance)

    def _screening(self, dsl_dict: dict[str, Any], params: dict[str, Any]) -> None:
        pipeline = create_screening_pipeline(
            validate_strategy_dict(dsl_dict),
            use_mock=False,
            symbols=SYMBOLS,
            start_date=START,
            end_date=END,
            catalog_path=str(self.catalog),
            parameters=params,
        )
        # ScreeningPipeline keeps its runner private; one call is one engine build.
        pipeline._runner({})  # noqa: SLF001

    def _validation(self, dsl_dict: dict[str, Any], params: dict[str, Any]) -> None:
        state = StateManager(self.tmp / "state.db")
        try:
            strategy_id = state.create_strategy(name=dsl_dict["name"], dsl_config=dsl_dict)
            run_id = state.create_backtest_run(
                strategy_id=strategy_id,
                run_mode="validation",
                symbols=SYMBOLS,
                timeframe=dsl_dict["timeframe"],
                start_date=START,
                end_date=END,
                parameters=params,
            )
        finally:
            state.close()
        runner = ValidationRunner(db_path=self.tmp / "state.db", logs_path=self.tmp / "logs")
        try:
            runner.run(run_id)
        finally:
            runner.close()

    def _wfa(self, dsl_dict: dict[str, Any], params: dict[str, Any], balance: float) -> None:
        from vibe_quant.overfitting.nt_runner import NTWFARunner

        runner = NTWFARunner(RUN_ID, _raw_db(self.tmp, dsl_dict, params, balance))
        runner.backtest("x", START_DATE, END_DATE, {})

    def _cv(self, dsl_dict: dict[str, Any], params: dict[str, Any], balance: float) -> None:
        from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

        db = _raw_db(self.tmp, dsl_dict, params, balance)
        runner = NTPurgedKFoldRunner(RUN_ID, db, catalog_path=self.catalog)
        runner._backtest(START, END)  # noqa: SLF001


def _dsl(timeframe: str) -> dict[str, Any]:
    dsl: dict[str, Any] = json.loads(DSL_GOLDEN.read_text())["dsl"]
    if timeframe != dsl["timeframe"]:
        dsl["timeframe"] = timeframe
    return dsl


def _bars(symbol: str, interval: str, count: int) -> list[Any]:
    instrument = create_instrument(symbol)
    klines: list[Any] = [
        {
            "open_time": _DAY0_MS + i * 60_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
            "close_time": _DAY0_MS + i * 60_000 + 59_999,
        }
        for i in range(count)
    ]
    return klines_to_bars(
        klines,
        instrument.id,
        get_bar_type(symbol, interval),
        instrument.size_precision,
        instrument.price_precision,
    )


def _make_catalog(root: Path, interval: str) -> Path:
    catalog = CatalogManager(root)
    for symbol in SYMBOLS:
        instrument = create_instrument(symbol)
        catalog.write_instrument(instrument)
        catalog.write_bars(_bars(symbol, interval, _BARS_PER_SYMBOL))
    return root


def _fake_fill_ticks(
    dsl_dict: dict[str, Any],
    symbols: list[str] | tuple[str, ...],
    start: str,
    end: str,
    catalog_path: str | Path | None = None,
) -> FillTickSet | None:
    """Parent-side resolver over tmp dirs: one empty catalog dir per symbol."""
    tick_tf = screening_fill_mode(
        str(dsl_dict["timeframe"]), loaded_timeframes(dsl_dict)
    ).tick_timeframe
    if tick_tf is None:
        return None
    base = Path(catalog_path) if catalog_path else Path(catalog_mod.DEFAULT_CATALOG_PATH)
    ticks_root = base.parent / "ticks"
    paths: dict[str, str] = {}
    for symbol in symbols:
        directory = ticks_root / symbol
        directory.mkdir(parents=True, exist_ok=True)
        paths[symbol] = str(directory)
    return FillTickSet(
        timeframe=tick_tf,
        paths=paths,
        missing_boundaries={},
        start_ns=date_to_ns(start),
        end_ns=date_to_ns(end),
    )


def _raw_db(
    tmp: Path, dsl_dict: dict[str, Any], params: dict[str, Any], balance: float
) -> Path:
    """Minimal raw schema the WFA/CV ``_resolve`` reads."""
    db = tmp / "raw.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(
            """
            CREATE TABLE strategies (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL, dsl_config TEXT NOT NULL
            );
            CREATE TABLE backtest_runs (
                id INTEGER PRIMARY KEY, strategy_id INTEGER, symbols TEXT NOT NULL,
                timeframe TEXT, parameters JSON NOT NULL
            );
            CREATE TABLE sweep_results (
                id INTEGER PRIMARY KEY, run_id INTEGER, starting_balance REAL
            );
            CREATE TABLE backtest_results (
                id INTEGER PRIMARY KEY, run_id INTEGER, starting_balance REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO strategies(id, name, dsl_config) VALUES (239, ?, ?)",
            (dsl_dict["name"], json.dumps(dsl_dict)),
        )
        conn.execute(
            "INSERT INTO backtest_runs(id, strategy_id, symbols, timeframe, parameters) "
            "VALUES (?, 239, ?, ?, ?)",
            (RUN_ID, json.dumps(SYMBOLS), dsl_dict["timeframe"], json.dumps(params)),
        )
        conn.execute(
            "INSERT INTO sweep_results(run_id, starting_balance) VALUES (?, ?)",
            (RUN_ID, balance),
        )
        conn.commit()
    finally:
        conn.close()
    return db


def _normalize(calls: list[list[Any]], tmp: Path) -> dict[str, Any]:
    """Exactly one recorded constructor call, decoded and path-normalized."""
    assert len(calls) == 1, f"expected exactly one BacktestNode call, got {len(calls)}"
    (config,) = calls[0]
    decoded = json.loads(config.json())
    decoded.pop("id", None)
    text = json.dumps(decoded, sort_keys=True)
    for prefix in (str(tmp.resolve()), str(tmp)):
        text = text.replace(prefix, "<TMP>")
    text = text.replace(str(ROOT), "<ROOT>")
    result: dict[str, Any] = json.loads(text)
    return result


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    # Both catalogs are built BEFORE the write recorder, so its log holds only
    # the writes the entry points themselves make (R18).
    catalogs = {
        interval: _make_catalog(tmp_path / f"catalog_{interval}", interval)
        for interval in ("4h", _NO_OUTBOX)
    }
    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", catalogs["4h"])
    monkeypatch.setattr(pipeline_mod, "resolve_parent_fill_ticks", _fake_fill_ticks)
    monkeypatch.setattr(ValidationRunner, "_clamp_run_window", lambda self, rc, ft: (rc, None))

    state = Harness(
        tmp=tmp_path, catalog=catalogs["4h"], catalogs=catalogs, configs=_CALLS
    )
    _CALLS.clear()

    def _record_write(_self: CatalogManager, instrument: Any) -> None:
        state.writes.append(str(instrument.id))

    monkeypatch.setattr(CatalogManager, "write_instrument", _record_write)
    monkeypatch.setattr(nt_node, "BacktestNode", _Recorder)
    yield state


@pytest.mark.parametrize("cell", CELLS, ids=[cell.cell_id for cell in CELLS])
def test_build_matrix_matches_golden(cell: Cell, harness: Harness) -> None:
    if cell.timeframe != "4h":
        harness.use_interval(cell.timeframe)

    with pytest.raises(_Captured):
        harness.run_cell(cell)

    normalized = _normalize(harness.configs, harness.tmp)
    expected_writes = (
        [f"{symbol}-PERP.BINANCE" for symbol in SYMBOLS] if cell.tier == "validation" else []
    )
    assert harness.writes == expected_writes

    if os.environ.get("VIBE_QUANT_CAPTURE_GOLDENS") == "1":
        _store_golden(cell.cell_id, normalized)
        return
    goldens = json.loads(MATRIX_GOLDEN.read_text())["cells"]
    assert cell.cell_id in goldens, f"golden missing {cell.cell_id}; re-capture"
    assert normalized == goldens[cell.cell_id]


def _store_golden(cell_id: str, normalized: dict[str, Any]) -> None:
    document: dict[str, Any] = {"base": BASE_COMMIT, "strategy": "strategy_239_dsl.json", "cells": {}}
    if MATRIX_GOLDEN.exists():
        document = json.loads(MATRIX_GOLDEN.read_text())
    document["cells"][cell_id] = normalized
    MATRIX_GOLDEN.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _bar_types(config: dict[str, Any]) -> list[str]:
    return [
        bar_type
        for data in config["data"]
        for bar_type in data["bar_types"] or []
    ]


def test_screening_sorts_symbols(harness: Harness) -> None:
    """R22: screening builds configs in ``sorted(symbols)`` order."""
    with pytest.raises(_Captured):
        harness.run_cell(Cell("screening", "default"))
    assert _bar_types(_normalize(harness.configs, harness.tmp)) == [
        "BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        "ETHUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
    ]


def test_validation_keeps_the_runs_symbol_order(harness: Harness) -> None:
    """R22: validation keeps the run row's symbol order (ETH first here)."""
    with pytest.raises(_Captured):
        harness.run_cell(Cell("validation", "default"))
    assert _bar_types(_normalize(harness.configs, harness.tmp)) == [
        "ETHUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        "ETHUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL",
        "BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL",
    ]


def test_strategy_239_golden_dsl_validates_and_compiles() -> None:
    """The pinned DSL is strategy 239's stored DSL, and it validates and compiles."""
    from vibe_quant.dsl.compiler import StrategyCompiler

    dsl = validate_strategy_dict(_dsl("4h"))
    assert dsl.name == "genome_47b1365029e0"
    module = StrategyCompiler().compile_to_module(dsl)
    assert hasattr(module, "Genome47b1365029e0Strategy")
