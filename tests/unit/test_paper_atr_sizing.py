"""Paper ATR sizing (vibe-quant-hrhok): the risk-officer's required tests, built while paper
still REFUSES ATR runs until the user approves (``ATR_SIZING_APPROVED``).

No exchange is contacted and no order is placed: the config resolution and the strategy
config the node builds are checked on literal numbers.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from tests.unit.test_paper_restore_trader_match import ctx  # noqa: F401  (fixture)
from tests.unit.test_validation_strategy_params import _dsl
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.paper import node as node_mod
from vibe_quant.paper.cli import load_config_from_json
from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
    SizingModuleConfig,
)
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path

    from tests.unit.test_paper_restore_trader_match import Ctx

_ATR = {"sizing_mode": "atr", "sizing_atr_period": 20, "sizing_atr_multiplier": 1.5}


def _node(
    tmp_path: Path,
    params: dict[str, object],
    timeframe: str = "1m",
    sizing: SizingModuleConfig | None = None,
    source: dict[str, object] | None = None,
) -> tuple[PaperTradingNode, object, object]:
    """A node on a completed validation run stored with ``params``; (node, module, dsl)."""
    source = _dsl(timeframe) if source is None else source
    state = StateManager(tmp_path / "state.db")
    sid = state.create_strategy(name=str(source["name"]), dsl_config=source)
    rid = state.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe=timeframe,
        start_date="2025-01-01",
        end_date="2025-02-01",
        parameters=params,
    )
    state.update_backtest_run_status(rid, "completed")
    state.close()
    dsl = validate_strategy_dict(source)
    module = StrategyCompiler().compile_to_module(dsl)
    cfg = PaperTradingConfig(
        trader_id="PAPER-007",
        binance=BinanceTestnetConfig("key", "secret", testnet=True),
        strategy_id=sid,
        validation_run_id=rid,
        db_path=tmp_path / "state.db",
        logs_path=tmp_path / "logs",
    )
    if sizing is not None:
        cfg.sizing = sizing
    return PaperTradingNode(cfg), module, dsl


def _resolve(
    tmp_path: Path,
    params: dict[str, object],
    timeframe: str = "1m",
    sizing: SizingModuleConfig | None = None,
) -> node_mod.ResolvedRunConfig:
    node, module, dsl = _node(tmp_path, params, timeframe, sizing)
    try:
        return node._resolve_run_config(module, dsl)  # type: ignore[arg-type]
    finally:
        node._state_manager.close()


# ---- R1: the approval switch --------------------------------------------------------


def test_atr_run_is_refused_while_atr_sizing_is_not_approved(tmp_path: Path) -> None:
    assert node_mod.ATR_SIZING_APPROVED is False
    with pytest.raises(ConfigurationError) as excinfo:
        _resolve(tmp_path, dict(_ATR), timeframe="4h")
    assert "sizing_mode='atr'; paper trading supports only 'risk' sizing" in str(excinfo.value)
    assert "needs a risk review and approval first" in str(excinfo.value)


# ---- R2: the warm-up cap (checked before the approval switch) -----------------------


@pytest.mark.parametrize(
    ("timeframe", "period"),
    [("1d", 7), ("4h", 42), ("1m", 10_000), ("1m", 20)],
)
def test_warmup_within_seven_days_reaches_the_approval_switch(
    tmp_path: Path, timeframe: str, period: int
) -> None:
    """7 d exactly (1d x 7, 4h x 42) and 6.94 d (1m x 10000) pass the cap; the switch still
    refuses, so the operator sees the approval text, not a warm-up text."""
    params = {**_ATR, "sizing_atr_period": period}
    with pytest.raises(ConfigurationError, match="needs a risk review and approval first"):
        _resolve(tmp_path, params, timeframe=timeframe)


@pytest.mark.parametrize(
    ("timeframe", "period", "length"),
    [("1d", 8, "8 d"), ("4h", 43, "172 h")],
)
def test_warmup_over_seven_days_is_refused_naming_period_timeframe_and_length(
    tmp_path: Path, timeframe: str, period: int, length: str
) -> None:
    params = {**_ATR, "sizing_atr_period": period}
    with pytest.raises(ConfigurationError) as excinfo:
        _resolve(tmp_path, params, timeframe=timeframe)
    text = str(excinfo.value)
    assert f"sizing_atr_period={period}" in text
    assert f"timeframe {timeframe}" in text
    assert f"warm-up of {length}" in text
    assert "needs a risk review" not in text


def test_warmup_cap_applies_when_atr_sizing_is_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node_mod, "ATR_SIZING_APPROVED", True)
    resolved = _resolve(tmp_path, {**_ATR, "sizing_atr_period": 42}, timeframe="4h")
    assert resolved.strategy_params["sizing_atr_period"] == 42
    with pytest.raises(ConfigurationError, match="sizing_atr_period=43"):
        _resolve(tmp_path / "b", {**_ATR, "sizing_atr_period": 43}, timeframe="4h")


# ---- R7: the dead paper-side atr_multiplier is refused ------------------------------


def _write_config(tmp_path: Path, sizing: dict[str, object]) -> Path:
    path = tmp_path / "paper.json"
    path.write_text(json.dumps({"trader_id": "PAPER-007", "strategy_id": 1, "sizing": sizing}))
    return path


@pytest.mark.parametrize("value", [None, 2.0, "2.0", 2])
def test_default_atr_multiplier_loads(tmp_path: Path, value: object) -> None:
    sizing: dict[str, object] = {} if value is None else {"atr_multiplier": value}
    config = load_config_from_json(_write_config(tmp_path, sizing))
    assert config.sizing.atr_multiplier == 2


@pytest.mark.parametrize("value", [1.5, "3", 0, -2])
def test_non_default_atr_multiplier_is_refused(tmp_path: Path, value: object) -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        load_config_from_json(_write_config(tmp_path, {"atr_multiplier": value}))
    assert str(excinfo.value) == (
        "paper sizing.atr_multiplier is unused; ATR sizing comes from the validated run"
    )


def test_non_numeric_atr_multiplier_keeps_the_decimal_parse_error(tmp_path: Path) -> None:
    from decimal import InvalidOperation

    with pytest.raises(InvalidOperation):
        load_config_from_json(_write_config(tmp_path, {"atr_multiplier": "abc"}))


def test_sizing_module_default_is_the_accepted_value() -> None:
    default: Decimal = SizingModuleConfig().atr_multiplier
    assert default == 2


# ---- R3-R5, R8: with the approval switch on ---------------------------------------------


@pytest.fixture()
def approved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node_mod, "ATR_SIZING_APPROVED", True)


@pytest.mark.parametrize(
    ("timeframe", "period", "mult"),
    [("4h", 20, 1.5), ("1h", 33, 4.0)],
)
@pytest.mark.usefixtures("approved")
def test_the_built_strategy_config_carries_the_validated_atr_params(
    tmp_path: Path, timeframe: str, period: int, mult: float
) -> None:
    params = {"sizing_mode": "atr", "sizing_atr_period": period, "sizing_atr_multiplier": mult}
    node, module, dsl = _node(tmp_path, params, timeframe)
    try:
        strategies = node._instantiate_strategies(module, dsl)  # type: ignore[arg-type]
    finally:
        node._state_manager.close()
    assert len(strategies) == 1
    config = strategies[0].config  # type: ignore[attr-defined]
    assert (config.sizing_mode, config.sizing_atr_period, config.sizing_atr_multiplier) == (
        "atr",
        period,
        mult,
    )


@pytest.mark.usefixtures("approved")
def test_a_paper_risk_override_keeps_atr_mode_and_is_in_the_logged_diff(tmp_path: Path) -> None:
    sizing = SizingModuleConfig(risk_per_trade=Decimal("0.01"))
    resolved = _resolve(tmp_path, {**_ATR, "risk_per_trade": 0.005}, timeframe="4h", sizing=sizing)
    params = resolved.strategy_params
    assert params["risk_per_trade"] == 0.01
    assert (params["sizing_mode"], params["sizing_atr_period"]) == ("atr", 20)
    assert params["sizing_atr_multiplier"] == 1.5
    # (validation ran with liquidation off, so the diff also carries that parity note)
    assert [d for d in resolved.diff if str(d["field"]).startswith("strategy_params.")] == [
        {"field": "strategy_params.risk_per_trade", "validated": 0.005, "paper": 0.01}
    ]


@pytest.mark.usefixtures("approved")
def test_each_atr_strategy_logs_its_warmup_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    params = {"sizing_mode": "atr", "sizing_atr_period": 30, "sizing_atr_multiplier": 2.5}
    node, module, dsl = _node(tmp_path, params, "4h")
    try:
        with caplog.at_level(logging.INFO, logger=node_mod.logger.name):
            node._instantiate_strategies(module, dsl)  # type: ignore[arg-type]
    finally:
        node._state_manager.close()
    lines = [r.getMessage() for r in caplog.records if "sizing ATR" in r.getMessage()]
    assert lines == [
        "BTCUSDT-PERP.BINANCE sizing ATR(30) warming up: entries skipped for the first 30 "
        "bars (~120.0 h)"
    ]


def test_a_risk_mode_strategy_logs_no_atr_warmup(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    node, module, dsl = _node(tmp_path, {}, "4h")
    try:
        with caplog.at_level(logging.INFO, logger=node_mod.logger.name):
            node._instantiate_strategies(module, dsl)  # type: ignore[arg-type]
    finally:
        node._state_manager.close()
    assert "sizing ATR" not in caplog.text


# ---- R5: /api/paper/restore of an ATR trader --------------------------------------------


async def test_restore_of_an_atr_trader_re_resolves_the_validated_atr_params(
    ctx: Ctx,  # noqa: F811
    approved: None,
) -> None:
    run_id = ctx.state.create_backtest_run(
        strategy_id=ctx.strategy_a,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-01-01",
        end_date="2025-02-01",
        parameters={"sizing_mode": "atr", "sizing_atr_period": 25, "sizing_atr_multiplier": 3.5},
    )
    ctx.state.update_backtest_run_status(run_id, "completed")
    prior = {**ctx.config("PAPER-001", ctx.strategy_a, ["BTCUSDT"]), "validation_run_id": run_id}
    ctx.write("paper_1.json", prior, 1_000)

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 201, r.text
    saved = ctx.cfg_dir / f"paper_{r.json()['run_id']}.json"
    node = PaperTradingNode(load_config_from_json(saved))
    try:
        strategy = node._load_strategy()
        resolved = node._resolve_run_config(node._compile_strategy(strategy), strategy)
    finally:
        node._state_manager.close()
    assert resolved.strategy_params["sizing_mode"] == "atr"
    assert resolved.strategy_params["sizing_atr_period"] == 25
    assert resolved.strategy_params["sizing_atr_multiplier"] == 3.5
