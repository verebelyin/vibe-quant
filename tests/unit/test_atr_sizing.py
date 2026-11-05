"""ATR sizing mode (vibe-quant-9sost): the risk budget divides by ATR * multiplier.

Unit rows run the generated ``_calculate_position_size`` on a fake strategy with literal
numbers; the warm-up and causality rows run REAL NautilusTrader screening on a toy catalog.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.objects import Price, Quantity

from tests.unit.test_audit_dsl_runtime import _BTC, _Acct, _fake
from tests.unit.test_costed_sizing import _archive, _dsl, _write_catalog, run_screening
from tests.unit.test_fast_tier import END, METRICS, START, _fills, _genomes, needs_catalog
from tests.unit.test_fast_tier import scratch_env as scratch_env  # noqa: F401 (fixture)
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.sizing_settings import SizingSettings

if TYPE_CHECKING:
    from pathlib import Path


class _ATR:
    def __init__(self, value: float, initialized: bool = True) -> None:
        self.value = value
        self.initialized = initialized


def _atr_sizer(
    balance: float, *, atr: _ATR | None, max_pos: float = 0.5
) -> types.SimpleNamespace:
    cfg = types.SimpleNamespace(
        risk_per_trade=0.02, max_position_pct=max_pos, stop_loss_type="fixed_pct",
        stop_loss_percent=1.0, sizing_mode="atr", sizing_atr_period=14,
        sizing_atr_multiplier=2.0,
    )
    return _fake(
        config=cfg, instrument=_BTC, instrument_id=_BTC.id, _sizing_atr=atr,
        cache=types.SimpleNamespace(account_for_venue=lambda venue: _Acct({USDT: balance})),
    )


_BAR_1000 = types.SimpleNamespace(close=Price.from_str("1000.0"), ts_init=0)


def test_atr_size_is_risk_over_atr_times_multiplier() -> None:
    # 1000 * 0.02 / (50 * 2) = 0.2 BTC; cap 0.5 * 1000 / 1000 = 0.5 does not bind
    s = _atr_sizer(1000.0, atr=_ATR(50.0))
    assert s._calculate_position_size(_BAR_1000, True) == Quantity.from_str("0.200")
    assert s._calculate_position_size(_BAR_1000, False) == Quantity.from_str("0.200")


def test_atr_size_is_capped_by_max_position_pct() -> None:
    # same risk size 0.2, cap 0.1 * 1000 / 1000 = 0.1 binds
    s = _atr_sizer(1000.0, atr=_ATR(50.0), max_pos=0.1)
    assert s._calculate_position_size(_BAR_1000, True) == Quantity.from_str("0.100")


def test_atr_size_at_the_cap_equality_edge() -> None:
    # risk size 0.2 == cap 0.2 * 1000 / 1000; one tick of ATR either side moves only one rule
    assert _atr_sizer(1000.0, atr=_ATR(50.0), max_pos=0.2)._calculate_position_size(
        _BAR_1000, True
    ) == Quantity.from_str("0.200")
    assert _atr_sizer(1000.0, atr=_ATR(40.0), max_pos=0.2)._calculate_position_size(
        _BAR_1000, True
    ) == Quantity.from_str("0.200")  # risk 0.25 > cap: cap
    assert _atr_sizer(1000.0, atr=_ATR(62.5), max_pos=0.2)._calculate_position_size(
        _BAR_1000, True
    ) == Quantity.from_str("0.160")  # risk 0.16 < cap: risk


def test_atr_mode_ignores_the_stop_loss_distance() -> None:
    # risk mode with a 1% stop: 20 / 10 = 2 BTC -> cap 0.5; ATR mode: 20 / 100 = 0.2
    s = _atr_sizer(1000.0, atr=_ATR(50.0))
    s.config.sizing_mode = "risk"
    assert s._calculate_position_size(_BAR_1000, True) == Quantity.from_str("0.500")
    s.config.sizing_mode = "atr"
    assert s._calculate_position_size(_BAR_1000, True) == Quantity.from_str("0.200")


def test_atr_not_warmed_up_skips_the_entry_with_its_reason() -> None:
    for atr in (_ATR(50.0, initialized=False), None):
        s = _atr_sizer(1000.0, atr=atr)
        assert s._calculate_position_size(_BAR_1000, True) is None
        assert s._sizing_skips == {"atr_warmup": 1}
        assert s.log.warnings == ["Sizing: ATR(14) not warmed up -- entry skipped"]


def test_zero_atr_skips_the_entry_never_a_fallback_size() -> None:
    s = _atr_sizer(1000.0, atr=_ATR(0.0))
    assert s._calculate_position_size(_BAR_1000, True) is None
    assert s._sizing_skips == {"atr_zero": 1}


# ------------------------------------------------------------- real NT screening (toy catalog)
# Flat 1m bars, 100 on even hours and 110 on odd ones, so every 1h bar's true range is 10
# (the first is 0). Always-long 1h strategy, 50% SL / 10% TP (only a spike reaches the TP).
# risk 0.05 of 1000 = 50; cap 0.5 * 1000 / 100 = 5 BTC.

_ATR_SIZING = {"max_position_pct": 0.5, "risk_per_trade": 0.05, "sizing_mode": "atr",
               "sizing_atr_period": 5}


def _oscillating(spike_minute: int | None = None) -> Any:
    def price(i: int) -> float:
        if i == spike_minute:
            return 150.0
        return 100.0 + 10.0 * ((i // 60) % 2)

    return price


def _first_buy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sizing: dict[str, Any],
               spike_minute: int | None = None) -> tuple[tuple[Any, ...], dict[str, int]]:
    catalog = _write_catalog(tmp_path / f"cat-{spike_minute}", _oscillating(spike_minute),
                             n_min=2 * 1440)
    run = run_screening(
        monkeypatch, catalog, _archive(tmp_path, rate=0.0), _dsl(),
        launch=LaunchSettings(sizing=SizingSettings.from_params(sizing)), end="2024-01-03",
    )
    buys = [f for f in run.fills if f[1] == "BUY"]
    assert buys
    return buys[0], run.skips[0]


_MINUTE_NS = 60_000_000_000
_DAY0_NS = 1_704_067_200_000 * 1_000_000


def test_screening_atr_warmup_skips_entries_then_sizes_on_atr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Risk mode enters at the first ready dispatch (03:00, 50 / (100 * 50%) = 1 BTC). ATR(5)
    is not ready on the 03:00 and 04:00 dispatches (2 skips); the 05:00 dispatch sizes on
    ATR(5) of hours 0-4 = (0 + 10 + 10 + 10 + 10) / 5 = 8 -> 50 / 16 = 3.125 BTC."""
    risk, risk_skips = _first_buy(tmp_path, monkeypatch, {"max_position_pct": 0.5,
                                                          "risk_per_trade": 0.05})
    assert (risk[2], risk[4]) == (1.0, _DAY0_NS + 181 * _MINUTE_NS - 1_000_000)
    assert risk_skips == {}
    atr, atr_skips = _first_buy(tmp_path, monkeypatch, _ATR_SIZING)
    assert (atr[2], atr[4]) == (3.125, _DAY0_NS + 301 * _MINUTE_NS - 1_000_000)
    assert atr_skips == {"atr_warmup": 2}


def test_screening_atr_reads_only_closed_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry is sized on the 04:00 dispatch bar's close and fills at 05:00:59.999. A spike
    later in the fill hour (05:20) cannot change its size; the same spike in the dispatch
    hour (04:30) can: TR 150 - 100 = 50 -> ATR (0 + 10 + 10 + 10 + 50) / 5 = 16 -> 1.562.
    (The 05:20 spike hits the take-profit, so that run re-enters later; only the first
    entry is compared.)"""
    base, _ = _first_buy(tmp_path, monkeypatch, _ATR_SIZING)
    after, _ = _first_buy(tmp_path, monkeypatch, _ATR_SIZING, spike_minute=5 * 60 + 20)
    during, _ = _first_buy(tmp_path, monkeypatch, _ATR_SIZING, spike_minute=4 * 60 + 30)
    assert after == base
    assert (during[2], during[4]) == (1.562, base[4])


def test_validation_sizes_the_atr_entry_like_screening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored run parameters reach validation's compiled strategy: same entry instant,
    same 3.125 BTC as screening (the price may differ by validation's own fill model)."""
    from nautilus_trader.model.events import OrderFilled

    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.validation.runner import ValidationRunner

    screening, _ = _first_buy(tmp_path, monkeypatch, _ATR_SIZING)
    catalog = _write_catalog(tmp_path / "val-cat", _oscillating(), n_min=2 * 1440)
    dsl = _dsl()
    db = tmp_path / "state.db"
    state = StateManager(db)
    run_id = state.create_backtest_run(
        strategy_id=state.create_strategy(name=str(dsl["name"]), dsl_config=dsl),
        run_mode="validation", symbols=["BTCUSDT"], timeframe="1h",
        start_date="2024-01-01", end_date="2024-01-03", parameters=dict(_ATR_SIZING),
    )
    state.close()
    buys: list[tuple[float, int]] = []
    orig = ValidationRunner._extract_results

    def spy(self: ValidationRunner, **kwargs: Any) -> Any:
        for order in kwargs["engine"].kernel.cache.orders():
            buys.extend(
                (float(ev.last_qty), int(ev.ts_event)) for ev in order.events
                if isinstance(ev, OrderFilled) and ev.order_side.name == "BUY"
            )
        return orig(self, **kwargs)

    monkeypatch.setattr(ValidationRunner, "_extract_results", spy)
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs", catalog_path=catalog)
    try:
        runner.run(run_id=run_id)
    finally:
        runner.close()
    assert sorted(buys, key=lambda b: b[1])[0] == (screening[2], screening[4]) == (
        3.125, _DAY0_NS + 301 * _MINUTE_NS - 1_000_000
    )


# ------------------------------------------------------------------ fast tier == NT (real 1m)


@needs_catalog
def test_fast_tier_sizes_atr_entries_exactly_like_nt(
    scratch_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay feeds each dispatch bar's recorded ATR to the generated sizing: fills and
    metrics equal NT's bit for bit, and the ATR (not the cap) sets the sizes (they vary)."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn
    from vibe_quant.screening import nt_runner
    from vibe_quant.validation.extraction import all_positions

    captured: dict[str, list[Any]] = {}
    original = nt_runner.NTScreeningRunner._extract_metrics

    def grab(self: Any, params: Any, bt_result: Any, engine: Any, *a: Any, **k: Any) -> Any:
        captured["positions"] = all_positions(engine)
        return original(self, params, bt_result, engine, *a, **k)

    monkeypatch.setattr(nt_runner.NTScreeningRunner, "_extract_metrics", grab)
    sizing = SizingSettings.from_params(
        {"max_position_pct": 5.0, "risk_per_trade": 0.001, "sizing_mode": "atr"}
    )
    chromosome = _genomes()[0]
    fast_fn = FastBacktestFn(["BTCUSDT"], "1m", START, END, sizing=sizing)
    fast = fast_fn._run_single(chromosome, START, END)
    nt = NTBacktestFn(["BTCUSDT"], "1m", START, END, sizing=sizing)._run_single(
        chromosome, START, END
    )
    assert nt["total_trades"] > 5
    assert _fills(fast_fn.last_positions) == _fills(captured["positions"])
    assert {m: fast[m] for m in METRICS} == {m: nt[m] for m in METRICS}
    entries = [str(pos.events[0].last_qty) for pos in fast_fn.last_positions]
    assert len(set(entries)) > 3, entries


def test_an_indicator_whose_field_collides_with_a_sizing_setting_is_refused() -> None:
    """Indicator ``sizing_atr`` generates ``sizing_atr_period``: compiled, the ATR-sizing
    setting would silently set the EMA's period. The compiler refuses the duplicate field."""
    from vibe_quant.dsl.compiler import CompilerError, StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict

    dsl = {**_dsl(), "name": "collide", "indicators": {"sizing_atr": {"type": "EMA", "period": 3}},
           "entry_conditions": {"long": ["sizing_atr > 0"]}}
    with pytest.raises(CompilerError) as excinfo:
        StrategyCompiler().compile(validate_strategy_dict(dsl))
    assert str(excinfo.value) == (
        "strategy config field(s) ['sizing_atr_period'] generated twice: an indicator name "
        "collides with a built-in setting; rename the indicator"
    )
