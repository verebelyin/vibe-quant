"""Paper trading refuses strategies that set ``max_bars_in_trade`` (D1=A, T6).

The bar count is not restored after a restart, so a paper strategy with N set
would exit late (a BTC long filled Nov 3 08:00 with N=12 resets its count on a
Nov 4 08:00 restart and exits Nov 6 08:00, 18 bars instead of 12). Strategies
without the field are unchanged.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
)
from vibe_quant.paper.node import PaperTradingNode

#: Sentinel for "the DSL has no max_bars_in_trade key at all".
_ABSENT = object()


def _dsl_config(max_bars_in_trade: object = _ABSENT) -> dict[str, Any]:
    """A minimal valid DSL, optionally carrying max_bars_in_trade."""
    dsl: dict[str, Any] = {
        "name": "test_strategy",
        "description": "Test strategy",
        "version": 1,
        "timeframe": "1h",
        "additional_timeframes": [],
        "indicators": {
            "rsi_14": {"type": "RSI", "period": 14, "source": "close"},
            "atr_14": {"type": "ATR", "period": 14},
        },
        "entry_conditions": {"long": ["rsi_14 < 30"], "short": ["rsi_14 > 70"]},
        "exit_conditions": {"long": [], "short": []},
        "time_filters": {},
        "stop_loss": {"type": "atr_fixed", "atr_multiplier": 2.0, "indicator": "atr_14"},
        "take_profit": {"type": "risk_reward", "risk_reward_ratio": 2.0},
        "position_management": {
            "scale_in": {"enabled": False},
            "partial_exit": {"enabled": False},
        },
        "sweep": {},
    }
    if max_bars_in_trade is not _ABSENT:
        dsl["max_bars_in_trade"] = max_bars_in_trade
    return dsl


def _write_strategy_db(db_file: Path, dsl: dict[str, Any]) -> Path:
    """Store one strategy row in a fresh state DB and return its path."""
    conn = sqlite3.connect(db_file)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                dsl_config JSON NOT NULL,
                strategy_type TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                is_active BOOLEAN DEFAULT 1,
                version INTEGER DEFAULT 1
            );
        """)
        conn.execute(
            "INSERT INTO strategies (name, description, dsl_config) VALUES (?, ?, ?)",
            ("test_strategy", "A test strategy", json.dumps(dsl)),
        )
        conn.commit()
    finally:
        conn.close()
    return db_file


def _node(db_file: Path) -> PaperTradingNode:
    return PaperTradingNode(
        PaperTradingConfig(
            trader_id="PAPER-001",
            binance=BinanceTestnetConfig("key", "secret"),
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_file,
        )
    )


@pytest.mark.parametrize(
    ("max_bars_in_trade", "expected"),
    [
        (_ABSENT, None),
        (None, None),
    ],
    ids=["absent", "null"],
)
def test_unset_max_bars_in_trade_still_loads(
    tmp_path: Path, max_bars_in_trade: object, expected: int | None
) -> None:
    """No key and an explicit null both mean "off" and load unchanged."""
    db_file = _write_strategy_db(tmp_path / "test_paper.db", _dsl_config(max_bars_in_trade))
    node = _node(db_file)
    try:
        strategy = node._load_strategy()
        assert strategy.name == "test_strategy"
        assert strategy.max_bars_in_trade is expected
    finally:
        node._state_manager.close()


@pytest.mark.parametrize("n_bars", [1, 12])
def test_loader_refuses_max_bars_in_trade(tmp_path: Path, n_bars: int) -> None:
    """Any set value is refused by the loader, not only the example N=12."""
    db_file = _write_strategy_db(tmp_path / "test_paper.db", _dsl_config(n_bars))
    node = _node(db_file)
    try:
        with pytest.raises(ConfigurationError, match="does not support max_bars_in_trade"):
            node._load_strategy()
    finally:
        node._state_manager.close()


@pytest.mark.asyncio
async def test_startup_refuses_max_bars_in_trade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup fails with ConfigurationError, before any venue/trading-node call."""
    db_file = _write_strategy_db(tmp_path / "test_paper.db", _dsl_config(12))
    node = _node(db_file)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_create_live_trading_node must not run")

    monkeypatch.setattr(PaperTradingNode, "_create_live_trading_node", _fail)
    try:
        with pytest.raises(ConfigurationError, match="does not support max_bars_in_trade"):
            await node._initialize()
    finally:
        await node._shutdown()
