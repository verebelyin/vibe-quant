"""Validation releases each strategy's orders on its OWN next detail bar (yul7u.11).

NT's LatencyModel releases pending commands venue-wide on the next datum of
ANY instrument: in a BTC+ETH run the ETH strategy's orders were released by
the BTC 1m bar processed first in the same timestamp batch and filled at the
ETH signal-bar close (93 stale fills, 239 BTC+ETH 0.8079 -> 0.4956 fixed).
With detail data validation now uses the generated strategy's command outbox
(``command_release="bar"`` on the strategy's own detail bar type) and no venue
latency model. The fill model is still chosen from the latency preset, so the
single-symbol numbers are bit-identical.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.latency import LatencyPreset
from vibe_quant.validation.results import ValidationResult
from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError
from vibe_quant.validation.venue import (
    create_backtest_venue_config,
    create_venue_config_for_validation,
)

if TYPE_CHECKING:
    from pathlib import Path


def _dsl(timeframe: str) -> dict[str, object]:
    return {
        "name": f"own_release_probe_{timeframe}",
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

    def date_range(
        self: CatalogManager, symbol: str, interval: str
    ) -> tuple[datetime, datetime] | None:
        return _COVERED if interval == "1m" else None

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", date_range)
    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", tmp_path / "catalog")
    monkeypatch.setattr(nt_node, "BacktestNode", _FakeNode)
    _FakeNode.seen = []


def _run_config(
    tmp_path: Path, timeframe: str, symbols: list[str], parameters: dict[str, object] | None = None
) -> Any:
    """Launch a validation run and return the captured BacktestRunConfig."""
    db = tmp_path / "s.db"
    state = StateManager(db)
    sid = state.create_strategy(name=f"own_release_probe_{timeframe}", dsl_config=_dsl(timeframe))
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="validation", symbols=symbols, timeframe=timeframe,
        start_date="2025-01-01", end_date="2025-02-01", parameters=parameters or {},
    )
    state.close()
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    with pytest.raises(ValidationRunnerError, match="_Captured"):
        runner.run(run_id=run_id)
    runner.close()
    assert len(_FakeNode.seen) == 1
    return _FakeNode.seen[0]


def _strategy_configs(bt_run_config: Any) -> dict[str, dict[str, Any]]:
    return {s.config["instrument_id"]: s.config for s in bt_run_config.engine.strategies}


@pytest.mark.parametrize("symbols", [["BTCUSDT"], ["BTCUSDT", "ETHUSDT"], ["ETHUSDT", "BTCUSDT"]])
def test_detail_run_releases_on_each_strategys_own_detail_bar(
    tmp_path: Path, symbols: list[str]
) -> None:
    cfg = _run_config(tmp_path, "4h", symbols)
    configs = _strategy_configs(cfg)
    assert sorted(configs) == sorted(f"{s}-PERP.BINANCE" for s in symbols)
    for instrument_id, config in configs.items():
        assert config["command_release"] == "bar"
        assert config["command_release_bar_type"] == f"{instrument_id}-1-MINUTE-LAST-EXTERNAL"
        assert config["execution_delay_probability"] == 0.0


def test_detail_run_has_no_latency_model_and_todays_fill_model(tmp_path: Path) -> None:
    cfg = _run_config(tmp_path, "4h", ["BTCUSDT", "ETHUSDT"])
    (venue,) = cfg.venues
    assert venue.latency_model is None
    # Fill model still chosen from the (default cloud) latency preset.
    today = create_backtest_venue_config(
        create_venue_config_for_validation(latency_preset=LatencyPreset.CLOUD)
    )
    assert venue.fill_model == today.fill_model
    assert venue.fill_model.config["prob_best_price_fill"] == 1.0


def test_detail_run_overrides_explicit_execution_delay(tmp_path: Path) -> None:
    """A stale per-run delay would make the generated on_start raise."""
    cfg = _run_config(tmp_path, "4h", ["BTCUSDT"], {"execution_delay_probability": 0.45})
    (config,) = _strategy_configs(cfg).values()
    assert config["execution_delay_probability"] == 0.0


def test_1m_strategy_without_detail_unchanged(tmp_path: Path) -> None:
    cfg = _run_config(tmp_path, "1m", ["BTCUSDT", "ETHUSDT"])
    for config in _strategy_configs(cfg).values():
        assert "command_release" not in config
        assert "command_release_bar_type" not in config
        assert config["execution_delay_probability"] == 1.0
    (venue,) = cfg.venues
    assert venue.latency_model is None
    no_latency = create_backtest_venue_config(create_venue_config_for_validation(latency_preset=None))
    assert venue.fill_model == no_latency.fill_model


def _create_run(
    tmp_path: Path, timeframe: str, parameters: dict[str, object] | None = None
) -> int:
    state = StateManager(tmp_path / "s.db")
    sid = state.create_strategy(name=f"own_release_probe_{timeframe}", dsl_config=_dsl(timeframe))
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="validation", symbols=["BTCUSDT"], timeframe=timeframe,
        start_date="2025-01-01", end_date="2025-02-01", parameters=parameters or {},
    )
    state.close()
    return run_id


def _stored_notes(tmp_path: Path, run_id: int) -> dict[str, object]:
    state = StateManager(tmp_path / "s.db")
    stored = state.get_backtest_result(run_id)
    state.close()
    assert stored is not None
    return json.loads(str(stored["notes"] or "{}"))  # type: ignore[no-any-return]


def _notes_after_run(
    tmp_path: Path, timeframe: str, parameters: dict[str, object] | None = None
) -> dict[str, object]:
    db = tmp_path / "s.db"
    run_id = _create_run(tmp_path, timeframe, parameters)
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    runner._run_backtest = lambda **kw: ValidationResult(  # type: ignore[method-assign]
        run_id=run_id, strategy_name="p", total_trades=3
    )
    runner.run(run_id=run_id)
    runner.close()
    state = StateManager(db)
    stored = state.get_backtest_result(run_id)
    state.close()
    assert stored is not None
    return json.loads(str(stored["notes"] or "{}"))  # type: ignore[no-any-return]


def test_detail_run_records_execution_release_note(tmp_path: Path) -> None:
    assert _notes_after_run(tmp_path, "4h")["execution_release"] == "own_detail_bar"


def test_no_detail_run_records_no_execution_release_note(tmp_path: Path) -> None:
    assert "execution_release" not in _notes_after_run(tmp_path, "1m")


def test_detail_run_records_delay_override_note(tmp_path: Path) -> None:
    notes = _notes_after_run(tmp_path, "4h", {"execution_delay_probability": 0.45})
    assert notes["execution_delay_override"] == {"requested": 0.45, "applied": 0.0}


def test_no_delay_override_note_without_explicit_delay(tmp_path: Path) -> None:
    assert "execution_delay_override" not in _notes_after_run(tmp_path, "4h")


def _start_event(tmp_path: Path) -> dict[str, Any]:
    (path,) = (tmp_path / "logs").glob("*.jsonl")
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("data", {}).get("event") == "BACKTEST_START":
            return event["data"]  # type: ignore[no-any-return]
    raise AssertionError("no BACKTEST_START event")


def test_start_event_logs_fill_model_preset_and_release(tmp_path: Path) -> None:
    _notes_after_run(tmp_path, "4h")
    data = _start_event(tmp_path)
    assert data["latency_preset"] == "cloud"
    assert data["execution_release"] == "own_detail_bar"


def test_start_event_without_detail_logs_no_preset(tmp_path: Path) -> None:
    _notes_after_run(tmp_path, "1m")
    data = _start_event(tmp_path)
    assert data["latency_preset"] is None
    assert data["execution_release"] is None


def test_walk_forward_records_execution_release_note(tmp_path: Path) -> None:
    run_id = _create_run(tmp_path, "4h")
    runner = ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")
    runner._run_backtest = lambda **kw: ValidationResult(  # type: ignore[method-assign]
        run_id=run_id, strategy_name="p", total_trades=3
    )
    runner.run_walk_forward(run_id=run_id, train_days=7, test_days=7, step_days=7)
    runner.close()
    assert _stored_notes(tmp_path, run_id)["execution_release"] == "own_detail_bar"


class TestDetailTimeframeOverride:
    """Explicit detail overrides must be loadable and not coarser than the
    strategy timeframe (a 1h release bar under 15m bars = orders up to 1h late)."""

    def _runner(self, tmp_path: Path) -> ValidationRunner:
        return ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")

    def _cfg(self, detail: str | None = None) -> dict[str, object]:
        cfg: dict[str, object] = {
            "symbols": json.dumps(["BTCUSDT"]), "start_date": "2025-01-01", "end_date": "2025-02-01",
        }
        if detail is not None:
            cfg["parameters"] = {"detail_timeframe": detail}
        return cfg

    def test_unknown_detail_param_fails_run(self, tmp_path: Path) -> None:
        run_id = _create_run(tmp_path, "4h", {"detail_timeframe": "3m"})
        runner = self._runner(tmp_path)
        with pytest.raises(ValidationRunnerError, match="Unknown detail timeframe '3m'"):
            runner.run(run_id=run_id)
        runner.close()
        state = StateManager(tmp_path / "s.db")
        run = state.get_backtest_run(run_id)
        state.close()
        assert run is not None
        assert run["status"] == "failed"

    def test_run_backtest_rejects_unknown_detail(self, tmp_path: Path) -> None:
        """Defensive guard: no release bar would mean signal-close fills."""
        runner = self._runner(tmp_path)
        with pytest.raises(ValidationRunnerError, match="Unknown detail timeframe '3m'"):
            runner._run_backtest(
                run_id=1,
                strategy_name="own_release_probe_4h",
                dsl=runner._validate_dsl(_dsl("4h"), strategy_name="own_release_probe_4h"),
                venue_config=create_venue_config_for_validation(latency_preset=None),
                run_config=self._cfg(),
                writer=None,  # type: ignore[arg-type]
                detail_timeframe="3m",
            )
        assert _FakeNode.seen == []
        runner.close()

    @pytest.mark.parametrize("via", ["argument", "parameters"])
    def test_resolve_rejects_unknown_override(self, tmp_path: Path, via: str) -> None:
        """Fail at setup (before the window clamp/compile), not in _run_backtest."""
        runner = self._runner(tmp_path)
        cfg = self._cfg("3m" if via == "parameters" else None)
        override = "3m" if via == "argument" else None
        with pytest.raises(ValidationRunnerError, match="Unknown detail timeframe '3m'"):
            runner._resolve_detail_timeframe(cfg, "4h", override)
        runner.close()

    @pytest.mark.parametrize("via", ["argument", "parameters"])
    def test_coarser_override_raises(self, tmp_path: Path, via: str) -> None:
        runner = self._runner(tmp_path)
        cfg = self._cfg("1h" if via == "parameters" else None)
        override = "1h" if via == "argument" else None
        with pytest.raises(ValidationRunnerError, match="coarser than the 15m strategy"):
            runner._resolve_detail_timeframe(cfg, "15m", override)
        runner.close()

    @pytest.mark.parametrize("detail", ["15m", "1m"])
    def test_equal_or_finer_override_allowed(self, tmp_path: Path, detail: str) -> None:
        runner = self._runner(tmp_path)
        assert runner._resolve_detail_timeframe(self._cfg(detail), "15m", None) == detail
        assert runner._resolve_detail_timeframe(self._cfg(), "15m", detail) == detail
        runner.close()
