"""Validation fails closed on unknown strategy params, like screening (vibe-quant-nxmxd.5).

Screening routes a run's parameter overrides through
:func:`vibe_quant.screening.nt_runner.resolve_strategy_params`, which maps dot notation
(``rsi.period`` -> ``rsi_period``) and raises ``UnknownStrategyParamError`` on a key the
compiled config has no field for. Validation used to silently drop every key not in the
compiled ``__struct_fields__`` -- including dotted keys, which never mapped -- so a typo
(``rsi_perid``) ran the compiled defaults instead of failing.

The one whitelist is :data:`vibe_quant.launch_settings.RUN_LEVEL_KEYS`: the run-level knobs
a validation ``parameters`` row legitimately carries (venue, launch, sizing, latency), which
never belong to the strategy config. Everything else must resolve to a compiled config field
or the run fails.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError

if TYPE_CHECKING:
    from pathlib import Path


def _dsl(timeframe: str) -> dict[str, object]:
    return {
        "name": "strategy_params_probe",
        "timeframe": timeframe,
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi < 30"]},
        "exit_conditions": {"long": ["rsi > 70"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }


_COVERED = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2026, 3, 17, tzinfo=UTC))


class _Captured(Exception):  # noqa: N818 - control-flow sentinel, not an error
    pass


class _FakeNode:
    """Records the BacktestRunConfig and stops before any engine work."""

    seen: list[Any] = []  # noqa: RUF012 - per-test capture, reset by fixture

    def __init__(self, configs: list[Any]) -> None:
        _FakeNode.seen.extend(configs)

    def build(self) -> None:
        raise _Captured

    def get_engines(self) -> list[Any]:
        return []

    def dispose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every symbol has 1m (not 5s) coverage; the catalog is a temp dir; the
    BacktestNode only captures its config."""
    import nautilus_trader.backtest.node as nt_node

    import vibe_quant.data.catalog as catalog_mod
    from vibe_quant.screening import nt_runner

    def date_range(
        self: CatalogManager, symbol: str, interval: str
    ) -> tuple[datetime, datetime] | None:
        return _COVERED if interval == "1m" else None

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", date_range)
    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", tmp_path / "catalog")
    monkeypatch.setattr(nt_node, "BacktestNode", _FakeNode)
    monkeypatch.setattr(nt_runner, "require_bars_in_window", lambda *a, **k: None)
    _FakeNode.seen = []


def _create_run(
    db_path: Path, timeframe: str, parameters: dict[str, object] | None = None
) -> int:
    state = StateManager(db_path)
    sid = state.create_strategy(name="strategy_params_probe", dsl_config=_dsl(timeframe))
    run_id = state.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe=timeframe,
        start_date="2025-01-01",
        end_date="2025-02-01",
        parameters=parameters or {},
    )
    state.close()
    return run_id


def _run_status(db_path: Path, run_id: int) -> str | None:
    state = StateManager(db_path)
    status = state.get_backtest_run_status(run_id)
    state.close()
    return status


def _captured_config(
    tmp_path: Path, timeframe: str, parameters: dict[str, object] | None = None
) -> dict[str, Any]:
    """Launch a validation run and return the strategy config the engine was given."""
    db = tmp_path / "s.db"
    run_id = _create_run(db, timeframe, parameters)
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    with pytest.raises(ValidationRunnerError, match="_Captured"):
        runner.run(run_id=run_id)
    runner.close()
    assert len(_FakeNode.seen) == 1
    (strategy,) = _FakeNode.seen[0].engine.strategies
    return dict(strategy.config)


def test_validation_unknown_strategy_param_fails_run(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    run_id = _create_run(db, "1m", {"rsi_perid": 14})
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    with pytest.raises(ValidationRunnerError, match="rsi_perid"):
        runner.run(run_id=run_id)
    runner.close()
    assert _FakeNode.seen == []
    assert _run_status(db, run_id) == "failed"


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        ({"rsi.period": 10}, {"rsi_period": 10}),
        ({"stop_loss.percent": 3.5}, {"stop_loss_percent": 3.5}),
    ],
)
def test_validation_dotted_param_reaches_config(
    tmp_path: Path, parameters: dict[str, object], expected: dict[str, object]
) -> None:
    """Dot notation used to be dropped (never mapped), silently keeping the default."""
    config = _captured_config(tmp_path, "1m", parameters)
    for key, value in expected.items():
        assert config[key] == value


# One key from each RUN_LEVEL_KEYS family that the compiled config does NOT declare.
_RUN_LEVEL_NOT_IN_CONFIG = (
    "sweep",
    "overfitting_filters",
    "fee_tier",
    "bnb_discount",
    "fee_schedule_source",
    "liquidation_mode",
    "leverage",
    "liquidation_leverage",
    "initial_balance",
    "latency_preset",
    "detail_timeframe",
    "promote_source",
)


def test_validation_run_level_keys_never_reach_config(tmp_path: Path) -> None:
    """A run's own settings are consumed by the runner/venue, never the strategy config.

    The sizing settings are the exception: they are merged from the parsed launch settings
    after the resolver (screening order), so they arrive validated -- and only when moved
    off the default.
    """
    config = _captured_config(
        tmp_path,
        "1m",
        {
            "sweep": {"rsi_period": [10, 14]},
            "overfitting_filters": {"min_trades": 10},
            "fee_tier": "VIP0",
            "bnb_discount": True,
            "fee_schedule_source": "run",
            "liquidation_mode": "off",
            "leverage": 5,
            "liquidation_leverage": 5,
            "initial_balance": 5000,
            "max_position_pct": 0.4,
            "risk_per_trade": 0.01,
            "max_margin_pct": 0.25,
            "latency_preset": "cloud",
            "detail_timeframe": "1m",
            "promote_source": "screening",
            # A run-level key that IS a config field: detail data overrides it to 0.0.
            "execution_delay_probability": 0.45,
        },
    )
    for key in _RUN_LEVEL_NOT_IN_CONFIG:
        assert key not in config
    # Detail data releases orders on the strategy's own bar, so the stored delay is
    # overridden -- the run-level key reached the runner, not the compiled defaults.
    assert config["execution_delay_probability"] == 0.0
    # Sizing comes from the parsed launch settings, never from the raw parameters.
    assert config["max_position_pct"] == 0.4
    assert config["risk_per_trade"] == 0.01
    assert config["max_margin_pct"] == 0.25
    assert config["margin_leverage"] == 5.0
    assert config["command_release"] == "bar"


def test_validation_run_level_config_fields_still_reach_config(tmp_path: Path) -> None:
    """Entry/maker knobs are run-level AND compiled config fields: they must arrive."""
    config = _captured_config(
        tmp_path,
        "1m",
        {
            "entry_order": "post_only",
            "maker_offset_ticks": 2,
            "maker_entry_timeout_bars": 4,
            "execution_delay_probability": 0.45,
            "initial_balance": 5000,
        },
    )
    assert config["entry_order"] == "post_only"
    assert config["maker_offset_ticks"] == 2
    assert config["maker_entry_timeout_bars"] == 4
    assert config["execution_delay_probability"] == 0.45
    assert "initial_balance" not in config


@pytest.mark.parametrize("key", ["purpose", "note", "of_filters"])
def test_validation_free_form_param_fails_run(tmp_path: Path, key: str) -> None:
    """Free-form row keys are not a whitelist loophole (user ruling 2026-10-27)."""
    db = tmp_path / "s.db"
    run_id = _create_run(db, "1m", {key: "free form"})
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    with pytest.raises(ValidationRunnerError, match=key):
        runner.run(run_id=run_id)
    runner.close()
    assert _FakeNode.seen == []
    assert _run_status(db, run_id) == "failed"


@pytest.mark.parametrize(
    "params",
    [
        {"entry.order": "post_only", "maker.offset.ticks": 2},
        {"entry.order": "post_only", "maker_offset_ticks": 2, "maker_entry_timeout_bars": 5},
    ],
)
def test_validation_dotted_post_only_resolves_to_post_only_precheck(
    tmp_path: Path, params: dict[str, object]
) -> None:
    """The maker precheck sees the mode the compiled config will get (market twin wiring)."""
    runner = ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")
    try:
        assert runner._resolve_maker_params({"parameters": params}) == "post_only"
    finally:
        runner.close()


def test_validation_dotted_maker_knob_is_range_checked(tmp_path: Path) -> None:
    runner = ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")
    try:
        with pytest.raises(ValidationRunnerError, match="maker_offset_ticks"):
            runner._resolve_maker_params(
                {"parameters": {"entry.order": "post_only", "maker.offset.ticks": 0}}
            )
    finally:
        runner.close()


def test_validation_duplicate_maker_spellings_rejected(tmp_path: Path) -> None:
    runner = ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")
    try:
        with pytest.raises(ValidationRunnerError, match="entry_order"):
            runner._resolve_maker_params(
                {"parameters": {"entry.order": "post_only", "entry_order": "market"}}
            )
    finally:
        runner.close()


def test_validation_dotted_post_only_run_gets_market_twin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post_only mode spelled ``entry.order`` still runs the market twin and maker notes."""
    from tests.unit.test_validation_maker_entries import _VALID_DSL, _primary_result

    db = tmp_path / "s.db"
    state = StateManager(db)
    sid = state.create_strategy("maker_runner", _VALID_DSL)
    run_id = state.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        parameters={
            "entry.order": "post_only",
            "maker_offset_ticks": 2,
            "maker_entry_timeout_bars": 5,
        },
    )
    state.close()
    monkeypatch.setattr(ValidationRunner, "_clamp_run_window", lambda self, rc, tf: (rc, None))
    monkeypatch.setattr(
        ValidationRunner, "_resolve_detail_timeframe", lambda self, rc, tf, ov: None
    )
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    modes: list[object] = []

    def fake(**kw: Any) -> Any:
        modes.append(kw.get("entry_order_override", "primary"))
        return _primary_result(**kw)

    monkeypatch.setattr(runner, "_run_backtest", fake)
    try:
        result = runner.run(run_id=run_id)
    finally:
        runner.close()
    assert modes == ["primary", "market"]
    maker = result.notes["maker_entries"]
    assert isinstance(maker, dict)
    assert "market_twin" in maker
