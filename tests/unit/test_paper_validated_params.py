"""Paper inherits a validation run's overrides under the names validation resolved them to.

Validation maps a stored override (``stop_loss.percent``, ``take_profit.risk_reward_ratio``)
onto its compiled config field through the one resolver (bead vibe-quant-nxmxd.5). Paper
starts from the completed validation row, so it must run the SAME effective values -- not
silently fall back to the compiled defaults -- and fail closed on a key validation would
refuse. No exchange is contacted and no order is placed: only the config resolution runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.unit.test_validation_strategy_params import _dsl
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.paper.config import BinanceTestnetConfig, ConfigurationError, PaperTradingConfig
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path


def _resolve(
    tmp_path: Path, params: dict[str, object], dsl_dict: dict[str, object] | None = None
) -> tuple[type, dict[str, object]]:
    """Seed a completed validation run with ``params``; return (config class, paper params)."""
    source = dsl_dict or _dsl("1m")
    state = StateManager(tmp_path / "state.db")
    sid = state.create_strategy(name=str(source["name"]), dsl_config=source)
    rid = state.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1m",
        start_date="2025-01-01",
        end_date="2025-02-01",
        parameters=params,
    )
    state.update_backtest_run_status(rid, "completed")
    state.close()
    dsl = validate_strategy_dict(source)
    module = StrategyCompiler().compile_to_module(dsl)
    node = PaperTradingNode(
        PaperTradingConfig(
            trader_id="PAPER-007",
            binance=BinanceTestnetConfig("key", "secret", testnet=True),
            strategy_id=sid,
            validation_run_id=rid,
            db_path=tmp_path / "state.db",
            logs_path=tmp_path / "logs",
        )
    )
    try:
        resolved = node._resolve_run_config(module, dsl)
    finally:
        node._state_manager.close()
    return module.StrategyParamsProbeConfig, dict(resolved.strategy_params)


def test_paper_runs_validated_dotted_overrides(tmp_path: Path) -> None:
    config_cls, params = _resolve(tmp_path, {"stop_loss.percent": 3.5, "rsi.period": 10})
    effective = config_cls(instrument_id="BTCUSDT-PERP.BINANCE", **params)
    assert (effective.stop_loss_percent, effective.rsi_period) == (3.5, 10)


def test_paper_runs_validated_suffix_alias(tmp_path: Path) -> None:
    dsl = _dsl("1m")
    dsl["take_profit"] = {"type": "risk_reward", "risk_reward_ratio": 2.0}
    config_cls, params = _resolve(
        tmp_path, {"take_profit.risk_reward_ratio": 3.0}, dsl_dict=dsl
    )
    effective = config_cls(instrument_id="BTCUSDT-PERP.BINANCE", **params)
    assert effective.take_profit_risk_reward == 3.0


def test_paper_still_inherits_underscore_overrides_and_sizing(tmp_path: Path) -> None:
    _, params = _resolve(
        tmp_path, {"rsi_period": 10, "max_position_pct": 1, "risk_per_trade": 0.01}
    )
    assert params["rsi_period"] == 10
    assert params["max_position_pct"] == 1.0
    assert params["risk_per_trade"] == 0.01


def test_paper_refuses_dotted_post_only_entry(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="post_only"):
        _resolve(tmp_path, {"entry.order": "post_only"})


def test_paper_fails_closed_on_unknown_stored_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="rsi_perid"):
        _resolve(tmp_path, {"rsi_perid": 10})


def test_paper_ignores_run_level_keys(tmp_path: Path) -> None:
    _, params = _resolve(
        tmp_path,
        {"leverage": 5, "initial_balance": 1000, "latency_preset": "retail", "rsi.period": 10},
    )
    assert params["rsi_period"] == 10
    assert not {"leverage", "initial_balance", "latency_preset"} & set(params)
