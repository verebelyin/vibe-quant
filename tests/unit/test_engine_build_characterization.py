"""The money values of the ``BacktestRunConfig`` each tier builds (bead vibe-quant-qlx46).

Every cell drives ONE public entry point over the tier/parameter matrix and checks the
venue values that move fills, costs and sizing: ``bar_execution``, the fill model, the
fee model, the absence of a latency model, the account size and the leverage.

This replaces the rabmo.3 golden matrix, which pinned every field of the whole config
dict while the rabmo refactor was in flight. That epic is closed, so the pins are gone
and only these outcomes remain, one row per tier plus the liquidation-leverage row.
The values they do not repeat live in the owning tests: fees in
``test_fee_schedule*.py``, liquidation and the run leverage in
``test_validation_liquidation.py``/``test_screening_liquidation.py``, the fill-tick and
outbox configs in ``test_screening_fill_mode.py``, the venue factory in
``test_validation_venue.py``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
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
from vibe_quant.screening import nt_runner
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

SYMBOLS = ["ETHUSDT", "BTCUSDT"]
START = "2024-01-01"
END = "2024-01-08"
START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 1, 8)
_DAY0_MS = 1_704_067_200_000
_BARS_PER_SYMBOL = 400
RUN_ID = 42

#: Matrix rows whose values reach the venue: the tier venue and the run leverage
#: (``liquidation_mode`` selects it; "isolated" uses ``parameters.leverage``).
PARAMS: dict[str, dict[str, Any]] = {
    "default": {},
    "isolated@20": {"liquidation_mode": "isolated", "leverage": 20},
}
#: The venue's fill model per tier: validation degrades the book, the rest fill bars.
FILL_MODEL = {
    "screening": "nautilus_trader.backtest.models:FillModel",
    "validation": "vibe_quant.validation.fill_model:VolumeSlippageFillModel",
    "wfa": "nautilus_trader.backtest.models:FillModel",
    "cv": "nautilus_trader.backtest.models:FillModel",
}
TIERS = tuple(FILL_MODEL)


@dataclass(frozen=True)
class Cell:
    tier: str
    label: str

    @property
    def cell_id(self) -> str:
        return f"{self.tier}-{self.label}"

    def params(self) -> dict[str, Any]:
        return dict(PARAMS[self.label])


CELLS = [Cell(tier, label) for tier in TIERS for label in PARAMS]


class _Captured(BaseException):
    """Raised by the recorder: the run stops at the constructor it must not go past."""


#: ``[configs]`` of every BacktestNode construction in the current test.
_CALLS: list[list[Any]] = []


class _Recorder:
    """Stands in for ``BacktestNode``: records ``configs`` and stops the run."""

    def __init__(self, configs: list[Any]) -> None:
        _CALLS.append(list(configs))
        raise _Captured


class Harness:
    def __init__(self, tmp: Path, catalog: Path, configs: list[list[Any]]) -> None:
        self.tmp = tmp
        self.catalog = catalog
        self.configs = configs

    def run_cell(self, cell: Cell) -> None:
        dsl_dict = _dsl()
        if cell.tier == "screening":
            self._screening(dsl_dict, cell.params())
        elif cell.tier == "validation":
            self._validation(dsl_dict, cell.params())
        elif cell.tier == "wfa":
            self._wfa(dsl_dict, cell.params(), 1000.0)
        else:
            self._cv(dsl_dict, cell.params(), 1000.0)

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


def _dsl() -> dict[str, Any]:
    return dict(json.loads(DSL_GOLDEN.read_text())["dsl"])


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


def _venue_of(calls: list[list[Any]]) -> dict[str, Any]:
    """The single recorded constructor call, decoded; its one venue config."""
    assert len(calls) == 1, f"expected exactly one BacktestNode call, got {len(calls)}"
    (config,) = calls[0]
    venues = json.loads(config.json())["venues"]
    assert len(venues) == 1, venues
    return dict(venues[0])


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    catalog = _make_catalog(tmp_path / "catalog_4h", "4h")
    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", catalog)
    monkeypatch.setattr(pipeline_mod, "resolve_parent_fill_ticks", _fake_fill_ticks)
    monkeypatch.setattr(ValidationRunner, "_clamp_run_window", lambda self, rc, ft: (rc, None))
    monkeypatch.setattr(nt_runner, "require_bars_in_window", lambda *a, **k: None)
    monkeypatch.setattr(nt_node, "BacktestNode", _Recorder)
    _CALLS.clear()
    yield Harness(tmp=tmp_path, catalog=catalog, configs=_CALLS)


@pytest.mark.parametrize("cell", CELLS, ids=[cell.cell_id for cell in CELLS])
def test_tier_venue_money_values(cell: Cell, harness: Harness) -> None:
    """Each tier hands NT the same money contract: bar fills, its fill model, no latency,
    the run's account size and leverage, and the Binance maker/taker fee model."""
    with pytest.raises(_Captured):
        harness.run_cell(cell)

    venue = _venue_of(harness.configs)
    assert venue["bar_execution"] is True
    assert venue["account_type"] == "MARGIN"
    assert venue["oms_type"] == "NETTING"
    assert venue["fill_model"]["fill_model_path"] == FILL_MODEL[cell.tier]
    assert (
        venue["fee_model"]["fee_model_path"]
        == "nautilus_trader.backtest.models:MakerTakerFeeModel"
    )
    assert venue["latency_model"] is None
    assert venue["starting_balances"] == ["1000 USDT"]
    assert venue["default_leverage"] == (20.0 if cell.label == "isolated@20" else 10.0)
