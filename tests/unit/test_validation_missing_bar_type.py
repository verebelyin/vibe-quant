"""Validation fails closed when a loaded bar type has no bars in the window.

vibe-quant-nxmxd.7: screening already checks every loaded bar type with
``require_bars_in_window`` (vibe-quant-c8va4); validation only logged an
"Unknown timeframe, skipping" warning and could run a silent zero-trade
backtest over a bar type the catalog never ingested. Validation now runs the
same check for every bar type it loads -- each strategy/extra timeframe AND
the detail (sub-bar) bar type -- and reports the missing bar type in the
window.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.catalog import (
    CatalogManager,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError

if TYPE_CHECKING:
    from pathlib import Path

#: Effective [start_date, end_date] the runner clamps to; the fake coverage below
#: spans it exactly, so the bar-coverage checks see the requested dates.
_START = "2024-01-01"
_END = "2024-02-01"
_COVERED = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2026, 3, 17, tzinfo=UTC))
_DAY0 = 1_704_067_200_000  # 2024-01-01T00:00Z in ms


def _dsl(timeframe: str = "4h") -> dict[str, object]:
    return {
        "name": f"missing_bar_probe_{timeframe}",
        "timeframe": timeframe,
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi < 30"]},
        "exit_conditions": {"long": ["rsi > 70"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }


class _NeverBuiltNode:
    """BacktestNode stand-in: building it means the check ran too late."""

    built: bool = False  # noqa: RUF012 - per-test capture, reset by fixture

    def __init__(self, configs: list[Any]) -> None:
        pass

    def build(self) -> None:
        _NeverBuiltNode.built = True
        raise AssertionError("BacktestNode must not be built when a bar type is missing")

    def get_engines(self) -> list[Any]:
        return []

    def dispose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _empty_catalog_with_detail_coverage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Temp catalog, 1m coverage faked (so the clamp and detail resolution pass),
    no bars written; the real ``require_bars_in_window`` runs against it."""
    import nautilus_trader.backtest.node as nt_node

    import vibe_quant.data.catalog as catalog_mod

    def date_range(
        self: CatalogManager, symbol: str, interval: str
    ) -> tuple[datetime, datetime] | None:
        return _COVERED if interval == "1m" else None

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    monkeypatch.setattr(CatalogManager, "get_bar_date_range", date_range)
    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", catalog)
    monkeypatch.setattr(nt_node, "BacktestNode", _NeverBuiltNode)
    _NeverBuiltNode.built = False
    return catalog


def _bars(symbol: str, interval: str, n: int = 1440) -> list[Any]:
    inst = create_instrument(symbol)
    klines = [
        {
            "open_time": _DAY0 + i * 60_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
            "close_time": _DAY0 + i * 60_000 + 59_999,
        }
        for i in range(n)
    ]
    return klines_to_bars(
        klines,
        inst.id,
        get_bar_type(symbol, interval),
        inst.size_precision,
        inst.price_precision,
    )


def _run(tmp_path: Path, timeframe: str, symbols: list[str]) -> None:
    db = tmp_path / "s.db"
    state = StateManager(db)
    sid = state.create_strategy(name=f"missing_bar_probe_{timeframe}", dsl_config=_dsl(timeframe))
    run_id = state.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=symbols,
        timeframe=timeframe,
        start_date=_START,
        end_date=_END,
        parameters={},
    )
    state.close()
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    try:
        with pytest.raises(
            ValidationRunnerError,
            match=r"BTCUSDT-PERP\.BINANCE-4-HOUR-LAST-EXTERNAL",
        ) as ei:
            runner.run(run_id=run_id)
    finally:
        runner.close()
    assert f"{_START}..{_END}" in str(ei.value)
    assert _NeverBuiltNode.built is False


def test_validation_fails_without_strategy_timeframe_bars(tmp_path: Path) -> None:
    """A 4h strategy over a catalog with no 4h bars fails, naming the bar type."""
    _run(tmp_path, "4h", ["BTCUSDT"])


def test_validation_fails_for_second_symbol_missing_bars(
    tmp_path: Path, _empty_catalog_with_detail_coverage: Path
) -> None:
    """BTCUSDT has 4h + 1m bars; ETHUSDT has none -> fails naming ETHUSDT."""
    catalog = _empty_catalog_with_detail_coverage
    cm = CatalogManager(catalog)
    for interval in ("4h", "1m"):
        inst = create_instrument("BTCUSDT")
        cm.write_instrument(inst)
        cm.write_bars(_bars("BTCUSDT", interval))

    db = tmp_path / "s.db"
    state = StateManager(db)
    sid = state.create_strategy(name="missing_bar_probe_4h", dsl_config=_dsl("4h"))
    run_id = state.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframe="4h",
        start_date=_START,
        end_date=_END,
        parameters={},
    )
    state.close()
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    try:
        with pytest.raises(
            ValidationRunnerError,
            match=r"ETHUSDT-PERP\.BINANCE-4-HOUR-LAST-EXTERNAL",
        ):
            runner.run(run_id=run_id)
    finally:
        runner.close()
    assert _NeverBuiltNode.built is False


def test_validation_fails_without_detail_bars(
    tmp_path: Path, _empty_catalog_with_detail_coverage: Path
) -> None:
    """A 4h strategy whose 4h bars exist but whose 1m detail bars are missing fails,
    naming the detail bar type it would load for sub-bar fill resolution."""
    cm = CatalogManager(_empty_catalog_with_detail_coverage)
    inst = create_instrument("BTCUSDT")
    cm.write_instrument(inst)
    cm.write_bars(_bars("BTCUSDT", "4h"))

    db = tmp_path / "s.db"
    state = StateManager(db)
    sid = state.create_strategy(name="missing_bar_probe_4h", dsl_config=_dsl("4h"))
    run_id = state.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date=_START,
        end_date=_END,
        parameters={},
    )
    state.close()
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    try:
        with pytest.raises(
            ValidationRunnerError,
            match=r"BTCUSDT-PERP\.BINANCE-1-MINUTE-LAST-EXTERNAL",
        ):
            runner.run(run_id=run_id)
    finally:
        runner.close()
    assert _NeverBuiltNode.built is False
