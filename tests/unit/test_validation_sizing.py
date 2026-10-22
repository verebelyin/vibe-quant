"""Validation honours a run's sizing settings (vibe-quant-2gyk1.2 S3, rows S15/S21).

``run`` / ``run_walk_forward`` validate ``parameters.max_position_pct`` and
``parameters.risk_per_trade`` through the one shared rule
(:mod:`vibe_quant.sizing_settings`) right after the liquidation settings, so a stored sizing
value the engine could not honour fails the run before it reaches ``running``. The generated
StrategyConfig carries the values (it declares both fields as floats), and paper inherits them
from the same ``_build_strategy_params`` source.

The real ``_run_backtest`` runs here; only the NT node and the result extraction are replaced.
"""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.results import ValidationResult
from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError

if TYPE_CHECKING:
    from pathlib import Path

_COVERED = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2026, 3, 17, tzinfo=UTC))

# The texts the one sizing rule raises (sizing_settings.M1); validation never re-words them.
GARBAGE_CAP = "max_position_pct must be a number in (0, 125], got -1"
GARBAGE_RISK = (
    "risk_per_trade must be more than 0 and at most 0.5 (50% of equity per trade), got 2.0"
)
ABOVE_LEVERAGE = "max_position_pct (20.0) must not exceed leverage (10.0)"


class _CapturingNode:
    """Records the ``BacktestRunConfig`` it is built with and finishes instantly."""

    seen: list[Any] = []  # noqa: RUF012 - per-test capture, reset by the fixture

    def __init__(self, configs: list[Any]) -> None:
        type(self).seen.extend(configs)

    def build(self) -> None:
        pass

    def run(self) -> None:
        pass

    def get_engine(self, _id: object) -> SimpleNamespace:
        result = SimpleNamespace(
            iterations=1, total_events=1, total_orders=0, total_positions=0, elapsed_time=86_400.0
        )
        # The margin-limit note sums the live strategies' refusals; no strategy here.
        trader = SimpleNamespace(strategies=lambda: [])
        return SimpleNamespace(get_result=lambda: result, trader=trader)

    def get_engines(self) -> list[Any]:
        return []

    def dispose(self) -> None:
        pass


class _ResettableEngine:
    """A built engine with the three lifecycle reads teardown/registration make."""

    def __init__(self) -> None:
        self.resets = 0
        self.kernel = SimpleNamespace(
            get_log_guard=lambda: None,
            portfolio=SimpleNamespace(
                analyzer=SimpleNamespace(register_statistic=lambda _stat: None)
            ),
        )

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No catalog data and no engine: the run reaches the strategy-config build only."""
    import nautilus_trader.backtest.node as nt_node

    import vibe_quant.data.catalog as catalog_mod

    def date_range(
        self: CatalogManager, symbol: str, interval: str
    ) -> tuple[datetime, datetime] | None:
        return _COVERED if interval == "1m" else None

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", date_range)
    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", tmp_path / "catalog")
    monkeypatch.setattr(nt_node, "BacktestNode", _CapturingNode)
    _CapturingNode.seen = []

    def fake_extract(self: ValidationRunner, **kw: Any) -> ValidationResult:
        return ValidationResult(run_id=kw["run_id"], strategy_name="p", total_trades=3)

    monkeypatch.setattr(ValidationRunner, "_extract_results", fake_extract)


def _dsl() -> dict[str, object]:
    return {
        "name": "sizing_probe",
        "timeframe": "4h",
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi < 30"]},
        "exit_conditions": {"long": ["rsi > 70"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }


def _launch(tmp_path: Path, parameters: dict[str, object]) -> int:
    state = StateManager(tmp_path / "s.db")
    try:
        sid = state.create_strategy(name="sizing_probe", dsl_config=_dsl())
        return state.create_backtest_run(
            strategy_id=sid, run_mode="validation", symbols=["BTCUSDT"], timeframe="4h",
            start_date="2025-01-01", end_date="2025-02-01", parameters=parameters,
        )
    finally:
        state.close()


def _runner(tmp_path: Path) -> ValidationRunner:
    return ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")


def _stored_run(tmp_path: Path, run_id: int) -> dict[str, Any]:
    state = StateManager(tmp_path / "s.db")
    try:
        row = state.get_backtest_run(run_id)
    finally:
        state.close()
    assert row is not None
    return row


def _strategy_configs() -> list[Any]:
    """The generated strategies of the one captured ``BacktestRunConfig``."""
    assert len(_CapturingNode.seen) == 1, "the run never built an engine config"
    return list(_CapturingNode.seen[0].engine.strategies)


def _decode(strategy_config: Any) -> Any:
    """The generated StrategyConfig, decoded the way NT decodes the config dict."""
    import msgspec

    module_path, class_name = str(strategy_config.config_path).split(":")
    config_cls = getattr(importlib.import_module(module_path), class_name)
    return msgspec.convert(strategy_config.config, type=config_cls)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"max_position_pct": -1}, GARBAGE_CAP),
        ({"risk_per_trade": 2.0}, GARBAGE_RISK),
        ({"max_position_pct": 20}, ABOVE_LEVERAGE),
        ({"max_position_pct": None}, "max_position_pct must be a number in (0, 125], got None"),
    ],
    ids=["cap-below-zero", "risk-above-half", "cap-above-leverage", "explicit-null"],
)
def test_stored_garbage_fails_before_running(
    tmp_path: Path, parameters: dict[str, object], message: str
) -> None:
    """S15: the run fails before ``running``, with the rule's own text and no engine."""
    run_id = _launch(tmp_path, parameters)
    runner = _runner(tmp_path)
    try:
        with pytest.raises(ValidationRunnerError) as excinfo:
            runner.run(run_id=run_id)
    finally:
        runner.close()

    assert str(excinfo.value) == message
    assert _CapturingNode.seen == [], "the engine config was built for an invalid sizing"
    stored = _stored_run(tmp_path, run_id)
    assert stored["status"] == "failed"
    assert stored["error_message"] == message


def test_walk_forward_garbage_fails_before_running(tmp_path: Path) -> None:
    """S15: every walk-forward window is guarded by the same single check."""
    run_id = _launch(tmp_path, {"max_position_pct": -1})
    runner = _runner(tmp_path)
    try:
        with pytest.raises(ValidationRunnerError) as excinfo:
            runner.run_walk_forward(run_id=run_id, train_days=7, test_days=7, step_days=7)
    finally:
        runner.close()

    assert str(excinfo.value) == GARBAGE_CAP
    assert _CapturingNode.seen == []
    assert _stored_run(tmp_path, run_id)["error_message"] == GARBAGE_CAP


def test_cap_exactly_at_leverage_runs(tmp_path: Path) -> None:
    """S13 equality edge: a cap exactly at the run's leverage is marginable."""
    run_id = _launch(tmp_path, {"max_position_pct": 10, "leverage": 10})
    runner = _runner(tmp_path)
    try:
        runner.run(run_id=run_id)
    finally:
        runner.close()
    assert _strategy_configs()


def test_cap_just_above_leverage_fails(tmp_path: Path) -> None:
    """S13: one ulp of exposure above the leverage is refused."""
    run_id = _launch(tmp_path, {"max_position_pct": 10.000001, "leverage": 10})
    runner = _runner(tmp_path)
    try:
        with pytest.raises(ValidationRunnerError) as excinfo:
            runner.run(run_id=run_id)
    finally:
        runner.close()
    assert str(excinfo.value) == "max_position_pct (10.000001) must not exceed leverage (10.0)"
    assert _CapturingNode.seen == []


def test_validation_injects_sizing(tmp_path: Path) -> None:
    """S21: the generated StrategyConfig carries the run's sizing as floats."""
    run_id = _launch(tmp_path, {"max_position_pct": 1, "risk_per_trade": 0.01})
    runner = _runner(tmp_path)
    try:
        runner.run(run_id=run_id)
    finally:
        runner.close()

    (strategy_config,) = _strategy_configs()
    assert strategy_config.config["max_position_pct"] == 1.0
    decoded = _decode(strategy_config)
    assert decoded.max_position_pct == 1.0
    assert isinstance(decoded.max_position_pct, float)
    assert decoded.risk_per_trade == 0.01
    assert isinstance(decoded.risk_per_trade, float)


def test_default_sizing_is_not_injected(tmp_path: Path) -> None:
    """A default launch stores nothing, so its strategy config is today's (S2/S19)."""
    run_id = _launch(tmp_path, {})
    runner = _runner(tmp_path)
    try:
        runner.run(run_id=run_id)
    finally:
        runner.close()

    (strategy_config,) = _strategy_configs()
    assert "max_position_pct" not in strategy_config.config
    assert "risk_per_trade" not in strategy_config.config
    assert "margin_leverage" not in strategy_config.config


def test_run_forwards_the_run_leverage_as_margin_leverage(tmp_path: Path) -> None:
    """The margin limit divides by the RUN leverage, taken from the ONE parsed launch.

    ``run`` passes the same :class:`~vibe_quant.launch_settings.LaunchSettings` to the venue
    and to ``_run_backtest`` (rabmo.3 S5), so a stored ``max_margin_pct`` at leverage 3 still
    reaches the strategy config as ``margin_leverage`` 3.0 — never the 10 default.
    """
    run_id = _launch(tmp_path, {"max_margin_pct": 0.3, "leverage": 3})
    runner = _runner(tmp_path)
    try:
        runner.run(run_id=run_id)
    finally:
        runner.close()

    (strategy_config,) = _strategy_configs()
    assert strategy_config.config["max_margin_pct"] == 0.3
    assert strategy_config.config["margin_leverage"] == 3.0
    decoded = _decode(strategy_config)
    assert decoded.margin_leverage == 3.0


def test_validation_engine_missing_raises_and_tears_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R24: a build that yields no engine fails the run by name, after teardown."""
    import nautilus_trader.backtest.node as nt_node

    import vibe_quant.data.catalog as catalog_mod

    engine = _ResettableEngine()
    disposed: list[bool] = []
    cleaned: list[Path] = []

    class _NoEngineNode(_CapturingNode):
        def get_engine(self, _id: object) -> None:
            return None

        def get_engines(self) -> list[Any]:
            return [engine]

        def dispose(self) -> None:
            disposed.append(True)

    monkeypatch.setattr(nt_node, "BacktestNode", _NoEngineNode)
    monkeypatch.setattr(CatalogManager, "_cleanup_epoch_parquet", lambda self: None)
    monkeypatch.setattr(catalog_mod, "cleanup_epoch_parquet", cleaned.append)

    run_id = _launch(tmp_path, {})
    runner = _runner(tmp_path)
    try:
        with pytest.raises(ValidationRunnerError) as excinfo:
            runner.run(run_id=run_id)
    finally:
        runner.close()

    (run_config,) = _CapturingNode.seen
    assert str(excinfo.value) == (
        f"ValidationRunnerError: Backtest engine not found for run config {run_config.id}"
    )
    assert engine.resets == 1
    assert disposed == [True]
    assert cleaned == [tmp_path / "catalog"]


def test_validation_extraction_failure_still_cleans_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R36: an extraction error still resets, disposes and cleans the catalog."""
    import nautilus_trader.backtest.node as nt_node

    import vibe_quant.data.catalog as catalog_mod

    engine = _ResettableEngine()
    disposed: list[bool] = []
    cleaned: list[Path] = []

    class _EngineNode(_CapturingNode):
        def get_engines(self) -> list[Any]:
            return [engine]

        def dispose(self) -> None:
            disposed.append(True)

    def exploding_extract(self: ValidationRunner, **kw: Any) -> ValidationResult:
        raise RuntimeError("extraction exploded")

    monkeypatch.setattr(nt_node, "BacktestNode", _EngineNode)
    monkeypatch.setattr(CatalogManager, "_cleanup_epoch_parquet", lambda self: None)
    monkeypatch.setattr(catalog_mod, "cleanup_epoch_parquet", cleaned.append)
    monkeypatch.setattr(ValidationRunner, "_extract_results", exploding_extract)

    run_id = _launch(tmp_path, {})
    runner = _runner(tmp_path)
    try:
        with pytest.raises(
            ValidationRunnerError, match="RuntimeError: extraction exploded"
        ):
            runner.run(run_id=run_id)
    finally:
        runner.close()

    assert engine.resets == 1
    assert disposed == [True]
    assert cleaned == [tmp_path / "catalog"]


def test_paper_inherits_stored_sizing(tmp_path: Path) -> None:
    """S21: paper reads the same stored sizing through the same builder."""
    from vibe_quant.paper.node import validated_run_strategy_params

    run_id = _launch(tmp_path, {"max_position_pct": 1, "risk_per_trade": 0.01})
    params = validated_run_strategy_params(_stored_run(tmp_path, run_id))
    assert params["max_position_pct"] == 1.0
    assert params["risk_per_trade"] == 0.01


def _screening_at_sizing(
    tmp_path: Path, strategy_id: int, parameters: dict[str, object], sharpe: float
) -> int:
    """A completed screening run over the validation window/symbols with one sweep row."""
    state = StateManager(tmp_path / "s.db")
    try:
        run_id = state.create_backtest_run(
            strategy_id=strategy_id, run_mode="screening", symbols=["BTCUSDT"], timeframe="4h",
            start_date="2025-01-01", end_date="2025-02-01", parameters=parameters,
        )
        state.update_backtest_run_status(run_id, "completed")
        state.conn.execute(
            "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades)"
            " VALUES (?, '{}', ?, 50)",
            (run_id, sharpe),
        )
        state.conn.commit()
        return run_id
    finally:
        state.close()


def test_consistency_reference_must_share_the_runs_sizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SF2: the runner forwards its sizing to the reference lookup.

    Two same-scope screening runs: an older one at the run's cap (1.0, screening Sharpe 4.0)
    and a newer one at the defaults (Sharpe 0.46). The validation collapse is reported against
    the OLDER matching-cap run; without the forwarding the newer default-cap run is chosen and
    no collapse fires.
    """
    state = StateManager(tmp_path / "s.db")
    try:
        sid = state.create_strategy(name="sizing_probe", dsl_config=_dsl())
        matching = _screening_at_sizing(tmp_path, sid, {"max_position_pct": 1.0}, sharpe=4.0)
        default = _screening_at_sizing(tmp_path, sid, {}, sharpe=0.46)
        assert default > matching, "the newer run must be the default-cap one"
        run_id = state.create_backtest_run(
            strategy_id=sid, run_mode="validation", symbols=["BTCUSDT"], timeframe="4h",
            start_date="2025-01-01", end_date="2025-02-01",
            parameters={"max_position_pct": 1.0},
        )
    finally:
        state.close()

    def fake_extract(self: ValidationRunner, **kw: Any) -> ValidationResult:
        return ValidationResult(
            run_id=kw["run_id"], strategy_name="p", total_trades=50, sharpe_ratio=1.0
        )

    monkeypatch.setattr(ValidationRunner, "_extract_results", fake_extract)

    runner = _runner(tmp_path)
    try:
        runner.run(run_id=run_id)
    finally:
        runner.close()

    state = StateManager(tmp_path / "s.db")
    try:
        notes = json.loads(state.get_result_notes_raw(run_id) or "{}")
    finally:
        state.close()
    report = notes["consistency"]
    assert report["screen_source"] == f"screening_run:{matching}", report
    assert any("validation-collapse" in f for f in report["flags"])
