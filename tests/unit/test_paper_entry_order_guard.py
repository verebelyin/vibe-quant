"""Paper trading must refuse a validation run with a non-market entry_order.

Post-only maker entries are being added to
validation only and have had no risk review for paper trading, so a paper node
started from such a run must raise ConfigurationError naming the parameter
instead of silently dropping it (entry_order is not a generated-config field,
so today it would be logged as "not applied").

Two independent paths are covered:

* a run that carries ``entry_order`` other than ``"market"`` (or a value that
  is not the string ``"market"`` at all) is refused by ``_resolve_run_config``;
* a strategy launched by ``strategy_id`` alone (no ``validation_run_id``) skips
  that branch and trades the compiled defaults, so the guard also inspects the
  *effective* config before any strategy is instantiated.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_audit_paper_node import (
    fake_tradingnode as _audit_fake_tradingnode,  # noqa: F401  (via getfixturevalue)
)
from tests.unit.test_audit_paper_node import seeded as _audit_seeded  # noqa: F401
from vibe_quant.db.state_manager import StateManager
from vibe_quant.paper.config import ConfigurationError
from vibe_quant.paper.node import PaperTradingNode, ResolvedRunConfig

# The audit suite's `seeded` (strategy + completed validation run) and
# `fake_tradingnode` (no real NT node) fixtures are imported under aliases and
# pulled in by name: importing them under their own names shadows the test
# parameters (ruff F811) and a wrapper fixture parameter would shadow too.

_NON_MARKET_ENTRY_ORDERS: list[object] = [
    None,
    "",
    "MARKET",
    "limit",
    "post_only",
    0,
    False,
    ["market"],
    {"x": 1},
]


def _run_with_entry_order(seeded: dict[str, Any], entry_order: object) -> int:
    sm = StateManager(seeded["db"])
    try:
        rid = sm.create_backtest_run(
            strategy_id=seeded["sid"],
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={
                "rsi_14_period": 21,
                "take_profit_risk_reward": 3.0,
                "leverage": 5,
                "entry_order": entry_order,
            },
        )
        sm.update_backtest_run_status(rid, "completed")
    finally:
        sm.close()
    return rid


def _run_without_entry_order(seeded: dict[str, Any]) -> int:
    sm = StateManager(seeded["db"])
    try:
        rid = sm.create_backtest_run(
            strategy_id=seeded["sid"],
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={"rsi_14_period": 21, "take_profit_risk_reward": 3.0, "leverage": 5},
        )
        sm.update_backtest_run_status(rid, "completed")
    finally:
        sm.close()
    return rid


@pytest.mark.parametrize("entry_order", _NON_MARKET_ENTRY_ORDERS)
async def test_paper_refuses_non_market_entry_order(
    request: pytest.FixtureRequest,
    entry_order: object,
) -> None:
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    fake_tradingnode: type[Any] = request.getfixturevalue("_audit_fake_tradingnode")
    rid = _run_with_entry_order(seeded, entry_order)
    node = PaperTradingNode(_config(seeded, validation_run_id=rid))
    try:
        with pytest.raises(ConfigurationError, match="entry_order") as excinfo:
            await node._initialize()
        assert str(rid) in str(excinfo.value)
    finally:
        await node._shutdown()
    assert fake_tradingnode.instances == []


async def test_paper_accepts_absent_entry_order(
    request: pytest.FixtureRequest,
) -> None:
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    request.getfixturevalue("_audit_fake_tradingnode")
    rid = _run_without_entry_order(seeded)
    node = PaperTradingNode(_config(seeded, validation_run_id=rid))
    await node._initialize()
    try:
        assert node.resolved_config is not None
    finally:
        await node._shutdown()


async def test_paper_accepts_market_entry_order(
    request: pytest.FixtureRequest,
) -> None:
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    request.getfixturevalue("_audit_fake_tradingnode")
    rid = _run_with_entry_order(seeded, "market")
    node = PaperTradingNode(_config(seeded, validation_run_id=rid))
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.symbols == ["BTCUSDT"]
        # A parallel task adds entry_order as a generated config field with
        # default "market"; once it lands the parameter is kept instead of
        # being logged as "not applied". Assert the validated values are
        # present and the effective entry_order is "market" (absent or equal),
        # without pinning the exact param dict.
        params = resolved.strategy_params
        assert params["rsi_14_period"] == 21
        assert params["take_profit_risk_reward"] == 3.0
        assert params.get("entry_order", "market") == "market"
    finally:
        await node._shutdown()


def _fake_compiled_module(dsl: Any, entry_order: object) -> ModuleType:
    """A stand-in compiled module whose config carries ``entry_order``."""
    strategy_name, config_name = PaperTradingNode._class_names(dsl)

    class _Config:
        def __init__(self) -> None:
            self.entry_order = entry_order

    class _ConfigCls:
        @staticmethod
        def parse(payload: str) -> Any:
            return _Config()

    class _Strategy:
        def __init__(self, config: Any) -> None:
            self.config = config

    module = ModuleType("fake_compiled_module")
    setattr(module, strategy_name, _Strategy)
    setattr(module, config_name, _ConfigCls)
    return module


async def test_effective_non_market_config_refused_without_validation_run(
    request: pytest.FixtureRequest,
) -> None:
    """strategy_id-only launch: no validation branch, guard the effective config."""
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    node = PaperTradingNode(
        _config(seeded, validation_run_id=None, symbols=["BTCUSDT"]),
    )
    dsl = node._load_strategy()
    node._resolved = ResolvedRunConfig(
        symbols=["BTCUSDT"],
        strategy_params={},
        leverage=1,
        margin_type="CROSSED",
    )
    with pytest.raises(ConfigurationError, match="entry_order") as excinfo:
        node._instantiate_strategies(_fake_compiled_module(dsl, "post_only"), dsl)
    assert "post_only" in str(excinfo.value)

    strategies = node._instantiate_strategies(_fake_compiled_module(dsl, "market"), dsl)
    assert len(strategies) == 1
