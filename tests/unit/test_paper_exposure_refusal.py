"""Paper refuses to start when the exposure cap exceeds leverage (D-S5, task P1).

One shared rule, :func:`vibe_quant.paper.config.check_paper_exposure`, refuses a
paper start whose effective ``max_position_pct`` exceeds paper's effective
leverage. The node calls it in ``_resolve_run_config`` right after the
``max_leverage`` check (before any strategy, checkpoint restore or TradingNode),
and the API calls it before credentials and before any run row.

Contract rows P1-P8 and P11-P13 (design 20261016-sizing, D-S5); P9/P10 live in
``tests/unit/api/test_paper_exposure_api.py``. The cap/leverage comparison is a
money comparison, so the equality edge is pinned on both sides (P2 ok, P3 just
above refuses).
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import DSL
from tests.unit.test_audit_paper_node import _FakeTradingNode as FakeTradingNode
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
    SizingModuleConfig,
    check_paper_exposure,
)
from vibe_quant.paper.journal import FakeJournal
from vibe_quant.paper.node import NodeState, PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path

#: The contract's run id for the pure-rule tests, so the pinned texts match the
#: design rows byte for byte (the node tests use the real, auto-incremented id).
RUN_ID = 870


@pytest.fixture()
def fake_tradingnode(monkeypatch: pytest.MonkeyPatch) -> type[FakeTradingNode]:
    """The audit suite's fake NT TradingNode, re-declared so the name is a fixture."""
    import nautilus_trader.live.node as live_node

    FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", FakeTradingNode)
    return FakeTradingNode


@pytest.fixture()
def seed(tmp_path: Path) -> dict[str, Any]:
    db = tmp_path / "state.db"
    sm = StateManager(db)
    sid = sm.create_strategy("test_strategy", DSL)
    sm.close()
    return {"db": db, "sid": sid, "tmp": tmp_path}


def _run(seed: dict[str, Any], parameters: dict[str, Any]) -> int:
    """A completed validation run with ``parameters`` (the stored run config)."""
    sm = StateManager(seed["db"])
    try:
        rid = sm.create_backtest_run(
            strategy_id=seed["sid"],
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


def _config(seed: dict[str, Any], run_id: int, **kw: Any) -> PaperTradingConfig:
    base: dict[str, Any] = {
        "trader_id": "PAPER-007",
        "binance": BinanceTestnetConfig("key", "secret", testnet=True),
        "strategy_id": seed["sid"],
        "validation_run_id": run_id,
        "db_path": seed["db"],
        "logs_path": seed["tmp"] / "logs",
    }
    base.update(kw)
    return PaperTradingConfig(**base)


def _rule(
    *,
    validation_run_id: int | None = RUN_ID,
    run_parameters: object = None,
    cap: Decimal | float | None = None,
    leverage: int | None = None,
) -> None:
    check_paper_exposure(
        validation_run_id=validation_run_id,
        run_parameters=run_parameters,
        max_position_pct_override=cap,
        leverage_override=leverage,
    )


# ------------------------------------------------------------------- P1


async def test_default_run_starts(seed: dict[str, Any], fake_tradingnode: object) -> None:
    """P1: a default run (cap 0.5, leverage 10) resolves and starts unchanged."""
    rid = _run(seed, {})
    node = PaperTradingNode(_config(seed, rid))
    try:
        await node._initialize()
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.leverage == 10
        assert resolved.strategy_params == {}
        assert len(FakeTradingNode.instances) == 1
    finally:
        await node._shutdown()


# ------------------------------------------------------------ P2 / P3 (edge)


def test_cap_equal_leverage_ok() -> None:
    """P2: cap == leverage is allowed (equality edge, the ``>`` mutant's red side)."""
    _rule(run_parameters={"max_position_pct": 2.0, "leverage": 2})


def test_cap_just_above_leverage_refused() -> None:
    """P3: 2.000001 vs 2 refuses with the exact pinned text and both sources."""
    with pytest.raises(ConfigurationError) as exc:
        _rule(run_parameters={"max_position_pct": 2.000001, "leverage": 2})
    assert str(exc.value) == (
        "paper refuses to start: max_position_pct (2.000001) must not exceed leverage (2.0) "
        "(max_position_pct from validation run 870; leverage from validation run 870)"
    )


# ------------------------------------------------------------ P4 / P5 / P6


def test_leverage_override_below_cap_refused() -> None:
    """P4: a leverage override below the run's cap refuses, naming the override source."""
    with pytest.raises(ConfigurationError) as exc:
        _rule(run_parameters={"max_position_pct": 3.0, "leverage": 5}, leverage=2)
    assert str(exc.value) == (
        "paper refuses to start: max_position_pct (3.0) must not exceed leverage (2.0) "
        "(max_position_pct from validation run 870; leverage from paper leverage override)"
    )


async def test_node_leverage_override_below_cap_refused(
    seed: dict[str, Any], fake_tradingnode: object
) -> None:
    """P4 (node path): ``cfg.leverage`` reaches the rule as the leverage override.

    ``_resolve_run_config`` must pass ``cfg.leverage`` into ``check_paper_exposure``,
    not ``None``: with the run's leverage (5) the cap (3.0) would pass, so only the
    override refusing proves the node wires the override through.
    """
    rid = _run(seed, {"max_position_pct": 3.0, "leverage": 5})
    node = PaperTradingNode(_config(seed, rid, leverage=2))
    try:
        with pytest.raises(ConfigurationError) as exc:
            await node._initialize()
    finally:
        await node._shutdown()
    assert str(exc.value) == (
        "paper refuses to start: max_position_pct (3.0) must not exceed leverage (2.0) "
        f"(max_position_pct from validation run {rid}; leverage from paper leverage override)"
    )
    assert FakeTradingNode.instances == [], "the node must refuse before building a TradingNode"


async def test_node_leverage_override_equal_to_cap_starts(
    seed: dict[str, Any], fake_tradingnode: object
) -> None:
    """P4 equality edge (node path): cap == override leverage starts, no refusal."""
    rid = _run(seed, {"max_position_pct": 2.0, "leverage": 5})
    node = PaperTradingNode(_config(seed, rid, leverage=2))
    try:
        await node._initialize()
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.leverage == 2
        assert len(FakeTradingNode.instances) == 1
    finally:
        await node._shutdown()


def test_cap_override_above_leverage_refused() -> None:
    """P5: a cap override above the run's leverage refuses, naming the override source."""
    with pytest.raises(ConfigurationError) as exc:
        _rule(run_parameters={"leverage": 2}, cap=Decimal("3"))
    assert str(exc.value) == (
        "paper refuses to start: max_position_pct (3.0) must not exceed leverage (2.0) "
        "(max_position_pct from paper sizing override; leverage from validation run 870)"
    )


def test_both_overrides_equal_ok() -> None:
    """P6: equal overrides are allowed (equality edge on the override path)."""
    _rule(validation_run_id=None, cap=Decimal("2"), leverage=2)


# ------------------------------------------------------------------- P7


def test_strategy_only_uses_defaults() -> None:
    """P7: no validation run -> defaults (0.5 vs 10); an override above 10 refuses."""
    _rule(validation_run_id=None)
    with pytest.raises(ConfigurationError) as exc:
        _rule(validation_run_id=None, cap=Decimal("12"))
    assert str(exc.value) == (
        "paper refuses to start: max_position_pct (12.0) must not exceed leverage (10.0) "
        "(max_position_pct from paper sizing override; leverage from default)"
    )


# ------------------------------------------------------------------- P8


def test_garbage_validated_sizing_fails_closed() -> None:
    """P8: a corrupt recorded cap is an error, not a silent default, with the run's id."""
    with pytest.raises(ConfigurationError) as exc:
        _rule(run_parameters={"max_position_pct": -1})
    assert str(exc.value) == (
        "Validation run 870 has invalid sizing settings: "
        "max_position_pct must be a number in (0, 125], got -1"
    )


# ------------------------------------------------------------ P11 precedence


async def test_max_leverage_error_wins(seed: dict[str, Any], fake_tradingnode: object) -> None:
    """P11: the ``max_leverage`` check precedes exposure, so its error is the one raised."""
    rid = _run(seed, {"leverage": 20})
    cfg = _config(
        seed,
        rid,
        sizing=SizingModuleConfig(max_leverage=Decimal("5"), max_position_pct=Decimal("30")),
    )
    node = PaperTradingNode(cfg)
    try:
        with pytest.raises(ConfigurationError) as exc:
            await node._initialize()
    finally:
        await node._shutdown()
    assert "exceeds max_leverage" in str(exc.value)
    assert "paper refuses to start" not in str(exc.value), "exposure must run AFTER max_leverage"
    assert FakeTradingNode.instances == []


# ------------------------------------------------------------------- P12


async def test_node_refusal_on_restore_path(seed: dict[str, Any], fake_tradingnode: object) -> None:
    """P12: a prior config whose cap override now violates is refused by the node."""
    rid = _run(seed, {"leverage": 2})
    cfg = _config(seed, rid, sizing=SizingModuleConfig(max_position_pct=Decimal("3")))
    node = PaperTradingNode(cfg)
    try:
        with pytest.raises(ConfigurationError) as exc:
            await node._initialize()
    finally:
        await node._shutdown()
    assert str(exc.value) == (
        "paper refuses to start: max_position_pct (3.0) must not exceed leverage (2.0) "
        f"(max_position_pct from paper sizing override; leverage from validation run {rid})"
    )
    assert FakeTradingNode.instances == [], "the node must refuse before building a TradingNode"


# ------------------------------------------------------------------- P13


async def test_refusal_before_any_node(
    seed: dict[str, Any], fake_tradingnode: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P13: the raise site is inside ``_resolve_run_config``; start() writes + alerts."""
    rid = _run(seed, {"max_position_pct": 3, "leverage": 2})
    node = PaperTradingNode(_config(seed, rid))
    fake = FakeJournal()
    node._journal = fake
    alerts: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(node, "_setup_signal_handlers", lambda: None)
    monkeypatch.setattr(
        node,
        "_send_telegram_alert",
        lambda kind, message, *, critical=False: alerts.append((kind, message, critical)),
    )

    expected = (
        "paper refuses to start: max_position_pct (3.0) must not exceed leverage (2.0) "
        f"(max_position_pct from validation run {rid}; leverage from validation run {rid})"
    )
    with pytest.raises(ConfigurationError) as exc:
        await node.start()

    assert str(exc.value) == expected
    assert FakeTradingNode.instances == [], "no TradingNode may be built"
    assert node._compiled_strategies == [], "no strategy may be instantiated"
    assert node._resolved is None, "no resolved config may be produced"
    writes = [(call[1], call[2], call[3]) for call in fake.calls if call[0] == "write"]
    assert writes[0] == (
        EventType.RISK_CHECK,
        {"action": "error", "error": expected, "passed": False},
        None,
    )
    assert alerts == [("error", f"Paper node failed: {expected}", True)]
    assert node.status.state is NodeState.ERROR
