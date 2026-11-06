"""Paper reports a validated non-default fee schedule as an unverified assumption.

Paper trades a real Binance (testnet/live)
account whose commission comes from the exchange, so the validated BNB/VIP fee
schedule is never applied to the node nor forwarded to the strategy. A completed
validation run that used the BNB discount must still surface in the config diff
the node reports (and in the "not applied" log), a default schedule must stay
invisible, and an invalid schedule must fail closed before any node is built.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from tests.unit.test_audit_paper_node import (
    LIQ_OFF_ENTRY,
    _config,
)
from tests.unit.test_audit_paper_node import (
    fake_tradingnode as _audit_fake_tradingnode,  # noqa: F401  (via getfixturevalue)
)
from tests.unit.test_audit_paper_node import seeded as _audit_seeded  # noqa: F401
from vibe_quant.db.state_manager import StateManager
from vibe_quant.paper.config import ConfigurationError
from vibe_quant.paper.node import PaperTradingNode

_FEE_KEYS = ("fee_tier", "bnb_discount")
_BASE_PARAMS: dict[str, Any] = {
    "rsi_14_period": 21,
    "take_profit_risk_reward": 3.0,
    "leverage": 5,
}


def _completed_run(seeded: dict[str, Any], parameters: dict[str, Any]) -> int:
    sm = StateManager(seeded["db"])
    try:
        rid = sm.create_backtest_run(
            strategy_id=seeded["sid"],
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters=parameters,
        )
        sm.update_backtest_run_status(rid, "completed")
    finally:
        sm.close()
    return rid


async def test_paper_reports_unverified_bnb_fee_schedule(
    request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture
) -> None:
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    request.getfixturevalue("_audit_fake_tradingnode")
    rid = _completed_run(seeded, {**_BASE_PARAMS, "bnb_discount": True})
    node = PaperTradingNode(_config(seeded, validation_run_id=rid))
    with caplog.at_level(logging.INFO, logger="vibe_quant.paper.node"):
        await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        fee_entries = [d for d in resolved.diff if d["field"] == "fee_schedule"]
        assert len(fee_entries) == 1
        entry = fee_entries[0]
        assert entry["validated"] == {"fee_tier": "VIP0", "bnb_discount": True}
        assert "exchange" in str(entry["paper"])
        assert "unverified" in str(entry["note"])
        # The venue assumption is never forwarded to the strategy config.
        assert not any(k in resolved.strategy_params for k in _FEE_KEYS)
        # And it is named in the "not applied" log, not silently dropped.
        assert "not applied" in caplog.text
        assert "bnb_discount" in caplog.text
    finally:
        await node._shutdown()


async def test_default_schedule_adds_no_diff_entry(request: pytest.FixtureRequest) -> None:
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    request.getfixturevalue("_audit_fake_tradingnode")
    node = PaperTradingNode(_config(seeded))  # seeded["rid"]: no fee params
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.diff == [LIQ_OFF_ENTRY], "a default fee schedule must not add a diff entry"
        assert resolved.validated is not None
        assert resolved.validated["fee_schedule"] == {"fee_tier": "VIP0", "bnb_discount": False}
    finally:
        await node._shutdown()


@pytest.mark.parametrize(
    "bad",
    [
        {"fee_tier": "VIP9"},
        {"bnb_discount": "true"},
        {"bnb_discount": 1},
        {"bnb_discount": None},
    ],
)
async def test_paper_refuses_invalid_fee_schedule(
    request: pytest.FixtureRequest, bad: dict[str, Any]
) -> None:
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    fake_tradingnode: type[Any] = request.getfixturevalue("_audit_fake_tradingnode")
    rid = _completed_run(seeded, {**_BASE_PARAMS, **bad})
    node = PaperTradingNode(_config(seeded, validation_run_id=rid))
    try:
        with pytest.raises(ConfigurationError, match="fee schedule"):
            await node._initialize()
    finally:
        await node._shutdown()
    assert fake_tradingnode.instances == []


async def test_paper_never_forwards_fee_keys_to_strategy(request: pytest.FixtureRequest) -> None:
    seeded: dict[str, Any] = request.getfixturevalue("_audit_seeded")
    request.getfixturevalue("_audit_fake_tradingnode")
    rid = _completed_run(seeded, {**_BASE_PARAMS, "bnb_discount": True})
    node = PaperTradingNode(_config(seeded, validation_run_id=rid))
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.strategy_params == {
            "rsi_14_period": 21,
            "take_profit_risk_reward": 3.0,
        }
        strat: Any = node._compiled_strategies[0]
        assert getattr(strat.config, "bnb_discount", None) is None
        assert getattr(strat.config, "fee_tier", None) is None
    finally:
        await node._shutdown()
