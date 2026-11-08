"""Paper trading refuses opt-in features awaiting a risk review (design-v2 rows R17).

``position_mode: reverse`` and ``signal_level`` stops/targets are research features
whose risk review has not happened yet (the precedent is ``sizing_mode="atr"``). A
paper start fails with ``ConfigurationError`` before adoption or any node build, so
no trading node exists and no orders are structurally possible. Strategies carrying
neither feature, or a directional override of a plain type, run as today.
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

#: The four directional stop/target fields that may carry ``signal_level``.
_SIGNAL_LEVEL_FIELDS = (
    "stop_loss_long",
    "stop_loss_short",
    "take_profit_long",
    "take_profit_short",
)


def _signal_level(field: str) -> dict[str, str]:
    """A valid ``signal_level`` config for one directional field."""
    indicator = "dc.lower" if field.startswith("stop_loss") else "dc.upper"
    return {"type": "signal_level", "indicator": indicator}


def _dsl_config(**over: Any) -> dict[str, Any]:
    """A minimal valid DSL; ``over`` replaces/adds top-level keys."""
    dsl: dict[str, Any] = {
        "name": "test_strategy",
        "description": "Test strategy",
        "version": 1,
        "timeframe": "1h",
        "additional_timeframes": [],
        "indicators": {
            "rsi_14": {"type": "RSI", "period": 14, "source": "close"},
            "dc": {"type": "DONCHIAN", "period": 2},
        },
        "entry_conditions": {"long": ["rsi_14 < 30"], "short": ["rsi_14 > 70"]},
        "exit_conditions": {"long": [], "short": []},
        "time_filters": {},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
        "position_management": {
            "scale_in": {"enabled": False},
            "partial_exit": {"enabled": False},
        },
        "sweep": {},
    }
    dsl.update(over)
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


@pytest.mark.parametrize("field", _SIGNAL_LEVEL_FIELDS)
def test_paper_refuses_directional_signal_level(tmp_path: Path, field: str) -> None:
    """Each of the four directional fields refuses a signal_level config."""
    dsl = _dsl_config(**{field: _signal_level(field)})
    db_file = _write_strategy_db(tmp_path / "test_paper.db", dsl)
    node = _node(db_file)
    try:
        with pytest.raises(ConfigurationError, match="does not support signal_level stops/targets"):
            node._load_strategy()
    finally:
        node._state_manager.close()


def test_paper_refuses_reverse_position_mode(tmp_path: Path) -> None:
    """``position_mode: reverse`` is refused by the loader."""
    db_file = _write_strategy_db(tmp_path / "test_paper.db", _dsl_config(position_mode="reverse"))
    node = _node(db_file)
    try:
        with pytest.raises(ConfigurationError, match="does not support position_mode='reverse'"):
            node._load_strategy()
    finally:
        node._state_manager.close()


def test_paper_refuses_generic_fixed_with_directional_signal_target(tmp_path: Path) -> None:
    """A plain generic stop plus a directional signal_level target is still refused."""
    dsl = _dsl_config(
        stop_loss={"type": "fixed_pct", "percent": 2.0},
        take_profit_short=_signal_level("take_profit_short"),
    )
    db_file = _write_strategy_db(tmp_path / "test_paper.db", dsl)
    node = _node(db_file)
    try:
        with pytest.raises(ConfigurationError, match="does not support signal_level stops/targets"):
            node._load_strategy()
    finally:
        node._state_manager.close()


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({}, id="legacy"),
        pytest.param({"position_mode": None}, id="position-mode-null"),
        pytest.param(
            {"stop_loss_long": {"type": "fixed_pct", "percent": 1.5}},
            id="directional-plain-override",
        ),
    ],
)
def test_paper_runs_as_today_without_the_features(tmp_path: Path, over: dict[str, Any]) -> None:
    """Neither feature, or a plain directional override, loads unchanged."""
    db_file = _write_strategy_db(tmp_path / "test_paper.db", _dsl_config(**over))
    node = _node(db_file)
    try:
        strategy = node._load_strategy()
        assert strategy.name == "test_strategy"
    finally:
        node._state_manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("over", "message"),
    [
        (
            {"position_mode": "reverse"},
            "does not support position_mode='reverse'",
        ),
        (
            {"take_profit_short": {"type": "signal_level", "indicator": "dc.upper"}},
            "does not support signal_level stops/targets",
        ),
    ],
    ids=["reverse", "signal-level"],
)
async def test_startup_refuses_before_adoption_or_node_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    over: dict[str, Any],
    message: str,
) -> None:
    """Startup fails before the DSL is compiled or any trading node is built."""
    db_file = _write_strategy_db(tmp_path / "test_paper.db", _dsl_config(**over))
    node = _node(db_file)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no compile/adoption/node build may run")

    monkeypatch.setattr(PaperTradingNode, "_compile_strategy", _fail)
    monkeypatch.setattr(PaperTradingNode, "_create_live_trading_node", _fail)
    try:
        with pytest.raises(ConfigurationError, match=message):
            await node._initialize()
    finally:
        await node._shutdown()
