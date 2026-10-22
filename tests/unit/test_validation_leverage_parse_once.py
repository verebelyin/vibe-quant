"""Validation parses the run's liquidation settings exactly once (vibe-quant-6wq6w, R5(2)).

``run`` and ``run_walk_forward`` resolve ``liquidation_mode`` + ``leverage`` once during setup
and hand the SAME settings to the venue config and to every window's extraction (a re-parse per
caller could one day disagree with the validated value). The real ``_run_backtest`` runs here;
only the NT node and the result extraction are replaced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.launch_settings import PARSE_ORDER, Field
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.validation.results import ValidationResult
from vibe_quant.validation.runner import (
    ValidationRunner,
    ValidationRunnerError,
    _parse_launch_fields,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

_COVERED = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2026, 3, 17, tzinfo=UTC))


class _OkNode:
    """A BacktestNode whose run finishes instantly with an empty result."""

    def __init__(self, configs: list[Any]) -> None:
        self._id = configs[0].id

    def build(self) -> None:
        pass

    def run(self) -> None:
        pass

    def get_engine(self, _id: object) -> SimpleNamespace:
        result = SimpleNamespace(
            iterations=1, total_events=1, total_orders=0, total_positions=0, elapsed_time=86_400.0
        )
        return SimpleNamespace(get_result=lambda: result)

    def get_engines(self) -> list[Any]:
        return []

    def dispose(self) -> None:
        pass


@pytest.fixture
def extracted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    """Offline validation setup; the kwargs of every ``_extract_results`` call are recorded."""
    import nautilus_trader.backtest.node as nt_node

    import vibe_quant.data.catalog as catalog_mod
    import vibe_quant.validation.extraction as extraction_mod

    def date_range(
        self: CatalogManager, symbol: str, interval: str
    ) -> tuple[datetime, datetime] | None:
        return _COVERED if interval == "1m" else None

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", date_range)
    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", tmp_path / "catalog")
    monkeypatch.setattr(nt_node, "BacktestNode", _OkNode)
    monkeypatch.setattr(ValidationRunner, "_register_statistics", lambda self, node: None)
    monkeypatch.setattr(extraction_mod, "maker_entry_stats", lambda *a, **kw: {})
    calls: list[dict[str, Any]] = []

    def fake_extract(self: ValidationRunner, **kw: Any) -> ValidationResult:
        calls.append(kw)
        return ValidationResult(run_id=kw["run_id"], strategy_name="p", total_trades=3)

    monkeypatch.setattr(ValidationRunner, "_extract_results", fake_extract)
    return calls


@pytest.fixture
def parses(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object]]:
    """One entry per ``LiquidationSettings`` validation (every parse path ends in __post_init__)."""
    seen: list[tuple[str, object]] = []
    original = LiquidationSettings.__post_init__

    def counting(self: LiquidationSettings) -> None:
        seen.append((self.mode, float(self.leverage)))  # type: ignore[arg-type]
        original(self)

    monkeypatch.setattr(LiquidationSettings, "__post_init__", counting)
    return seen


@pytest.fixture
def orders(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Field, ...]]:
    """One entry per ``iter_fields`` call, recording the order the runner walked (R40)."""
    import vibe_quant.validation.runner as runner_mod

    seen: list[tuple[Field, ...]] = []
    original = runner_mod.iter_fields

    def recording(
        params: Mapping[str, object],
        order: tuple[Field, ...],
        *,
        parsed: Mapping[Field, object] | None = None,
    ) -> Iterator[tuple[Field, object]]:
        seen.append(order)
        if parsed is None:
            return original(params, order)
        return original(params, order, parsed=parsed)

    monkeypatch.setattr(runner_mod, "iter_fields", recording)
    return seen


def _launch(tmp_path: Path, parameters: dict[str, object]) -> int:
    dsl = {
        "name": "parse_once_probe",
        "timeframe": "4h",
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi < 30"]},
        "exit_conditions": {"long": ["rsi > 70"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }
    state = StateManager(tmp_path / "s.db")
    sid = state.create_strategy(name="parse_once_probe", dsl_config=dsl)
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="validation", symbols=["BTCUSDT"], timeframe="4h",
        start_date="2025-01-01", end_date="2025-02-01", parameters=parameters,
    )
    state.close()
    return run_id


def _runner(tmp_path: Path) -> ValidationRunner:
    return ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"leverage": 3},
        {"liquidation_mode": "isolated", "leverage": 20},
        {"liquidation_mode": "cross", "leverage": 20, "entry_order": "post_only"},
    ],
    ids=["default", "off-3x", "isolated-20x", "cross-20x-with-market-twin"],
)
def test_run_parses_the_liquidation_settings_once(
    tmp_path: Path,
    extracted: list[dict[str, Any]],
    parses: list[tuple[str, object]],
    orders: list[tuple[Field, ...]],
    parameters: dict[str, object],
) -> None:
    run_id = _launch(tmp_path, parameters)
    runner = _runner(tmp_path)
    try:
        runner.run(run_id=run_id)
    finally:
        runner.close()
    expected_mode = str(parameters.get("liquidation_mode", "off"))
    expected_leverage = float(str(parameters.get("leverage", 10)))
    assert parses == [(expected_mode, expected_leverage)]
    # R40: the launch fields are walked ONCE per phase, in the table's order.
    assert orders == [
        PARSE_ORDER["validation_pre_detail"],
        PARSE_ORDER["validation_post_detail"],
    ]
    assert extracted, "the backtest never reached the extraction"
    assert {kw["liquidation_mode"] for kw in extracted} == {expected_mode}
    if parameters.get("entry_order") == "post_only":
        assert len(extracted) == 2  # primary + market twin, still one parse


def test_walk_forward_parses_the_liquidation_settings_once_for_all_windows(
    tmp_path: Path,
    extracted: list[dict[str, Any]],
    parses: list[tuple[str, object]],
    orders: list[tuple[Field, ...]],
) -> None:
    run_id = _launch(tmp_path, {"liquidation_mode": "isolated", "leverage": 20})
    runner = _runner(tmp_path)
    try:
        runner.run_walk_forward(run_id=run_id, train_days=7, test_days=7, step_days=7)
    finally:
        runner.close()
    assert len(extracted) > 1, "expected several walk-forward windows"
    assert parses == [("isolated", 20.0)]
    assert orders == [
        PARSE_ORDER["validation_pre_detail"],
        PARSE_ORDER["validation_post_detail"],
    ]
    assert {kw["liquidation_mode"] for kw in extracted} == {"isolated"}


def test_direct_venue_config_parses_only_its_own_phase(
    tmp_path: Path, orders: list[tuple[Field, ...]]
) -> None:
    """R40: a direct ``_create_venue_config`` (no parsed settings) walks the venue row once."""
    runner = _runner(tmp_path)
    try:
        runner._create_venue_config({"parameters": {}}, None)
    finally:
        runner.close()
    assert orders == [PARSE_ORDER["validation_direct_venue"]]


@pytest.mark.parametrize("bad", [10**400, -(10**400)])
def test_huge_integer_leverage_fails_the_run_not_with_an_overflow(
    tmp_path: Path, extracted: list[dict[str, Any]], bad: int
) -> None:
    run_id = _launch(tmp_path, {"leverage": bad})
    runner = _runner(tmp_path)
    try:
        with pytest.raises(ValidationRunnerError, match=r"\[1, 125\]"):
            runner.run(run_id=run_id)
        with pytest.raises(ValidationRunnerError, match=r"\[1, 125\]"):
            _parse_launch_fields({"leverage": bad}, PARSE_ORDER["validation_pre_detail"])
    finally:
        runner.close()
    state = StateManager(tmp_path / "s.db")
    try:
        row = state.get_backtest_run(run_id)
    finally:
        state.close()
    assert row is not None and row["status"] == "failed"
    assert extracted == []
