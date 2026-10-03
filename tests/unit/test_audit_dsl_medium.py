"""vibe-quant-e70tl.23 (DSL medium/low items).

* time filters / funding avoidance evaluated at bar CLOSE time and gating
  entries only (exits keep running);
* WMA is a real linearly weighted MA (NT without weights == SMA);
* ICHIMOKU outputs labelled correctly;
* unsupported ``source`` rejected instead of silently ignored;
* compute_fn warmup is NaN (not-ready), never 0;
* compiler_version_hash covers compute_builtins/derived/schema/plugins.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from vibe_quant.data.catalog import create_instrument
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name, compiler_version_hash
from vibe_quant.dsl.compute_builtins import compute_ichimoku
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.nt_compat import retain_log_guard

_IID = "BTCUSDT-PERP.BINANCE"
_H = 3_600_000_000_000
_T0 = 1_704_067_200_000_000_000  # 2024-01-01 00:00 UTC


def _hour_bars(closes: list[float]) -> list[Bar]:
    bt = BarType.from_str(f"{_IID}-1-HOUR-LAST-EXTERNAL")
    out = []
    for i, c in enumerate(closes):
        out.append(
            Bar(
                bar_type=bt,
                open=Price.from_str(f"{c:.1f}"),
                high=Price.from_str(f"{c + 0.5:.1f}"),
                low=Price.from_str(f"{c - 0.5:.1f}"),
                close=Price.from_str(f"{c:.1f}"),
                volume=Quantity.from_str("10.000"),
                ts_event=_T0 + i * _H,
                ts_init=_T0 + (i + 1) * _H - 1_000_000,  # Binance close_time (hh:59:59.999)
            )
        )
    return out


def _run(dsl: dict[str, Any], bars: list[Bar], probe_mixin: type | None = None) -> Any:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        bar_execution=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    engine.add_data(bars)
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(dsl))
    cn = _to_class_name(dsl["name"])
    base = getattr(module, f"{cn}Strategy")
    cls = type("Probe", (probe_mixin, base), {}) if probe_mixin else base
    strat = cls(getattr(module, f"{cn}Config")(instrument_id=_IID))
    engine.add_strategy(strat)
    engine.run()
    engine.dispose()
    return strat


class _Recorder:
    """Records the bar-close hour of every entry / exit submission; never trades."""

    force_open = False

    def on_start(self) -> None:
        super().on_start()  # type: ignore[misc]
        self.entry_hours: list[int] = []
        self.exit_hours: list[int] = []

    def _hour(self, bar: Bar) -> int:
        return int(((bar.ts_init + 1_000_000 - _T0) // _H) % 24)

    def _submit_long_entry(self, bar: Bar) -> None:
        self.entry_hours.append(self._hour(bar))

    def _submit_exit(self, bar: Bar) -> None:
        self.exit_hours.append(self._hour(bar))

    def _sync_position_state(self) -> None:
        if self.force_open:  # pretend a long is open so exits are evaluated
            self._position_open = True
            self._position_side = OrderSide.BUY


class _ExitRecorder(_Recorder):
    force_open = True


def _filter_dsl(name: str, **time_filters: Any) -> dict[str, Any]:
    return {
        "name": name,
        "timeframe": "1h",
        "indicators": {"sma": {"type": "SMA", "period": 2}},
        "entry_conditions": {"long": ["close > 0"]},
        "exit_conditions": {"long": ["close > 0"]},
        "time_filters": time_filters,
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


class TestTimeFilters:
    BARS = _hour_bars([100.0] * 48)

    def test_funding_avoidance_blocks_entries_at_close_time(self) -> None:
        dsl = _filter_dsl("tf_funding", avoid_around_funding={"enabled": True})
        strat = _run(dsl, self.BARS, _Recorder)
        # Entries are decided at the bar close: closes at 00:00/08:00/16:00 are
        # blocked (old code used ts_event = OPEN and blocked 01/09/17 instead).
        assert {0, 8, 16}.isdisjoint(strat.entry_hours)
        assert {1, 9, 17} <= set(strat.entry_hours)

    def test_session_filter_uses_close_time(self) -> None:
        dsl = _filter_dsl(
            "tf_session", allowed_sessions=[{"start": "09:00", "end": "17:00", "timezone": "UTC"}]
        )
        strat = _run(dsl, self.BARS, _Recorder)
        assert set(strat.entry_hours) == set(range(9, 18))

    def test_filters_never_block_exits(self) -> None:
        dsl = _filter_dsl(
            "tf_exits",
            avoid_around_funding={"enabled": True},
            allowed_sessions=[{"start": "09:00", "end": "10:00", "timezone": "UTC"}],
            blocked_days=["Monday"],  # 2024-01-01 is a Monday
        )
        strat = _run(dsl, self.BARS, _ExitRecorder)
        assert strat.entry_hours == []  # position "open": entries not evaluated
        # every ready bar evaluates the exit, blocked hours included
        assert len(strat.exit_hours) == len(self.BARS) - 1
        assert {0, 8, 16} <= set(strat.exit_hours)


class TestWma:
    def test_wma_is_linearly_weighted(self) -> None:
        rng = np.random.default_rng(3)
        closes = list(np.round(100 + np.cumsum(rng.normal(0, 1, 60)), 1))
        dsl = {
            "name": "wma_probe",
            "timeframe": "1h",
            "indicators": {"wma": {"type": "WMA", "period": 10}},
            "entry_conditions": {"long": ["wma > 100000"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 4.0},
        }
        strat = _run(dsl, _hour_bars(closes))
        s = pd.Series(closes)
        assert strat._get_indicator_value("wma") == pytest.approx(float(ta.wma(s, 10).iloc[-1]), rel=1e-12)
        assert strat._get_indicator_value("wma") != pytest.approx(float(s.tail(10).mean()), rel=1e-6)


class TestIchimoku:
    def test_outputs_labelled_by_meaning(self) -> None:
        rng = np.random.default_rng(1)
        c = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
        df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0})
        out = compute_ichimoku(df, {"tenkan": 9, "kijun": 26, "senkou": 52})

        def mid(n: int) -> pd.Series:
            return (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2

        tenkan, kijun, senkou_b = mid(9), mid(26), mid(52)
        np.testing.assert_allclose(out["conversion"].iloc[-1], tenkan.iloc[-1])
        np.testing.assert_allclose(out["base"].iloc[-1], kijun.iloc[-1])
        # cloud at the current bar = spans computed kijun (26) bars ago
        np.testing.assert_allclose(out["span_a"].iloc[-1], (0.5 * (tenkan + kijun)).iloc[-27])
        np.testing.assert_allclose(out["span_b"].iloc[-1], senkou_b.iloc[-27])


class TestSource:
    def test_unsupported_source_rejected(self) -> None:
        from vibe_quant.dsl.schema import IndicatorConfig

        with pytest.raises(ValueError, match="source 'hl2' is not supported"):
            IndicatorConfig(type="EMA", period=20, source="hl2")
        with pytest.raises(ValueError, match="source 'volume' is not supported"):
            IndicatorConfig(type="RSI", period=14, source="volume")

    def test_close_and_volume_for_volume_indicators_ok(self) -> None:
        from vibe_quant.dsl.schema import IndicatorConfig

        IndicatorConfig(type="EMA", period=20, source="close")
        IndicatorConfig(type="VOLSMA", period=20, source="volume")


class TestWarmupNotReady:
    def test_kama_short_period_never_reads_zero(self) -> None:
        # KAMA(5): compiler lookback 15 but pandas-ta needs 30 bars -> the old
        # code stored KAMA = 0 for bars 15..29, so `close > k` fired in warmup.
        closes = [100.0 + i * 0.1 for i in range(60)]

        class FirstFire:
            def on_start(self) -> None:
                super().on_start()  # type: ignore[misc]
                self.fired: list[int] = []
                self.n = -1

            def on_bar(self, bar: Bar) -> None:
                self.n += 1
                super().on_bar(bar)  # type: ignore[misc]

            def _submit_long_entry(self, bar: Bar) -> None:
                self.fired.append(self.n)

        dsl = {
            "name": "kama_warm_probe",
            "timeframe": "1h",
            "indicators": {"k": {"type": "KAMA", "period": 5}},
            "entry_conditions": {"long": ["close > k"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 4.0},
        }
        strat = _run(dsl, _hour_bars(closes), FirstFire)
        assert strat.fired and min(strat.fired) >= 29  # needs 30 bars (index 29)


def test_compiler_version_hash_covers_all_runtime_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    read: list[str] = []
    orig = Path.read_bytes

    def spy(self: Path) -> bytes:
        read.append(self.name)
        return orig(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    compiler_version_hash()
    for name in ("compute_builtins.py", "derived.py", "schema.py", "kama.py", "frama.py"):
        assert name in read
