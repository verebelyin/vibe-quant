"""vibe-quant-e70tl.8: compute_fn (pandas-path) indicators.

* params must come from ``self.config`` so sweeps / WFA overrides apply
  (before: ``compute_adx(_df, {"period": 14})`` literals -> overrides no-op);
* an indicator's ``timeframe`` must be honoured (before: the buffer only got
  primary-TF bars, so a "4h ADX" on a 1h strategy was a 1h ADX);
* unknown sweep/override keys must fail loudly instead of being dropped.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from vibe_quant.data.catalog import create_instrument
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.compute_builtins import compute_adx, compute_macd
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.screening.nt_runner import (
    NTScreeningRunner,
    UnknownStrategyParamError,
    resolve_strategy_params,
)

_IID = "BTCUSDT-PERP.BINANCE"
_H = 3_600_000_000_000


def _ohlc(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.round(30_000.0 * np.exp(np.cumsum(rng.normal(0, 0.006, n))), 1)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.round(np.abs(rng.normal(0, 60, n)), 1) + 1.0
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = np.round(rng.uniform(1, 50, n), 3)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def _bars(df: pd.DataFrame, step_h: int, spec: str) -> list[Bar]:
    bt = BarType.from_str(f"{_IID}-{spec}-LAST-EXTERNAL")
    out = []
    for i, r in enumerate(df.itertuples(index=False)):
        out.append(
            Bar(
                bar_type=bt,
                open=Price.from_str(f"{r.open:.1f}"),
                high=Price.from_str(f"{r.high:.1f}"),
                low=Price.from_str(f"{r.low:.1f}"),
                close=Price.from_str(f"{r.close:.1f}"),
                volume=Quantity.from_str(f"{r.volume:.3f}"),
                ts_event=i * step_h * _H,
                ts_init=(i + 1) * step_h * _H - 1,
            )
        )
    return out


def _agg_4h(df1h: pd.DataFrame) -> pd.DataFrame:
    g = np.arange(len(df1h)) // 4
    return pd.DataFrame(
        {
            "open": df1h["open"].groupby(g).first(),
            "high": df1h["high"].groupby(g).max(),
            "low": df1h["low"].groupby(g).min(),
            "close": df1h["close"].groupby(g).last(),
            "volume": df1h["volume"].groupby(g).sum(),
        }
    ).reset_index(drop=True)


def _run_strategy(dsl: dict[str, Any], bars: list[Bar], **cfg_overrides: Any) -> Any:
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
    strat = getattr(module, f"{cn}Strategy")(
        getattr(module, f"{cn}Config")(instrument_id=_IID, **cfg_overrides)
    )
    engine.add_strategy(strat)
    engine.run()
    engine.dispose()
    return strat


def _adx_dsl(name: str, timeframe_override: str | None = None) -> dict[str, Any]:
    ind: dict[str, Any] = {"type": "ADX", "period": 14}
    d: dict[str, Any] = {
        "name": name,
        "timeframe": "1h",
        "indicators": {"adx": ind},
        "entry_conditions": {"long": ["adx > 1000"]},  # never trades
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
    if timeframe_override:
        ind["timeframe"] = timeframe_override
        d["additional_timeframes"] = [timeframe_override]
    return d


class TestParamsFromConfig:
    def test_config_override_changes_compute_fn_params(self) -> None:
        df = _ohlc(600, seed=3)
        bars = _bars(df, 1, "1-HOUR")
        default = _run_strategy(_adx_dsl("pta_cfg_default"), bars)
        swept = _run_strategy(_adx_dsl("pta_cfg_swept"), bars, adx_period=40)

        def expected(strat: Any, period: int) -> float:
            # Same window the strategy's (capped) rolling buffer holds.
            window = df.tail(len(strat._pta_bufs["1h"]["close"])).reset_index(drop=True)
            return float(compute_adx(window, {"period": period}).iloc[-1])

        assert swept._pta_params["adx"] == {"period": 40}
        assert swept._pta_lookback["adx"] == 40
        assert default._pta_values["adx"] == pytest.approx(expected(default, 14), rel=1e-12)
        assert swept._pta_values["adx"] == pytest.approx(expected(swept, 40), rel=1e-12)
        assert expected(default, 14) != pytest.approx(expected(swept, 40), rel=1e-3)

    def test_macd_fast_slow_override(self) -> None:
        df = _ohlc(600, seed=5)
        bars = _bars(df, 1, "1-HOUR")
        dsl = {
            "name": "pta_cfg_macd",
            "timeframe": "1h",
            "indicators": {"macd": {"type": "MACD", "fast_period": 12, "slow_period": 26}},
            "entry_conditions": {"long": ["macd.histogram > 1000000"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 4.0},
        }
        strat = _run_strategy(dsl, bars, macd_fast_period=5, macd_slow_period=50)
        window = df.tail(len(strat._pta_bufs["1h"]["close"])).reset_index(drop=True)
        exp = compute_macd(window, {"fast_period": 5, "slow_period": 50, "signal_period": 9})
        assert strat._pta_values["macd_histogram"] == pytest.approx(
            float(exp["histogram"].iloc[-1]), rel=1e-12
        )
        default = compute_macd(window, {"fast_period": 12, "slow_period": 26, "signal_period": 9})
        assert float(default["histogram"].iloc[-1]) != pytest.approx(
            strat._pta_values["macd_histogram"], rel=1e-3
        )

    def test_generated_source_reads_config(self) -> None:
        src = StrategyCompiler().compile(validate_strategy_dict(_adx_dsl("pta_cfg_src")))
        assert '"period": config.adx_period' in src
        assert 'compute_adx(_df, {"period": 14})' not in src

    def test_plugin_extra_param_is_a_config_field(self) -> None:
        dsl = validate_strategy_dict(
            {
                "name": "pta_cfg_plugin",
                "timeframe": "1h",
                "indicators": {"arsi": {"type": "ADAPTIVE_RSI", "period": 14, "alpha": 0.7}},
                "entry_conditions": {"long": ["arsi > 50"]},
                "stop_loss": {"type": "fixed_pct", "percent": 2.0},
                "take_profit": {"type": "fixed_pct", "percent": 4.0},
            }
        )
        mod = StrategyCompiler().compile_to_module(dsl)
        cfg = mod.PtaCfgPluginConfig(instrument_id=_IID)
        assert cfg.arsi_alpha == 0.7
        strat = mod.PtaCfgPluginStrategy(mod.PtaCfgPluginConfig(instrument_id=_IID, arsi_alpha=0.3))
        assert strat._pta_params["arsi"]["alpha"] == 0.3


class TestTimeframeHonoured:
    def test_4h_adx_on_1h_strategy_uses_4h_bars(self) -> None:
        df1h = _ohlc(1200, seed=11)
        df4h = _agg_4h(df1h)
        bars = _bars(df1h, 1, "1-HOUR") + _bars(df4h, 4, "4-HOUR")
        bars.sort(key=lambda b: b.ts_init)
        strat = _run_strategy(_adx_dsl("pta_mtf_adx", timeframe_override="4h"), bars)

        exp_4h = float(compute_adx(df4h, {"period": 14}).iloc[-1])
        exp_1h = float(compute_adx(df1h, {"period": 14}).iloc[-1])
        assert strat._pta_values["adx"] == pytest.approx(exp_4h, rel=1e-9)
        assert exp_4h != pytest.approx(exp_1h, rel=1e-3)
        # 4h buffer only holds 4h bars
        assert len(strat._pta_bufs["4h"]["close"]) == len(df4h)


class TestStochAliases:
    def test_period_k_d_normalized_to_dsl_fields(self) -> None:
        from vibe_quant.dsl.schema import IndicatorConfig

        cfg = IndicatorConfig(type="STOCH", period_k=10, period_d=5)
        assert (cfg.period, cfg.d_period) == (10, 5)
        assert not cfg.model_extra  # not a silently ignored extra

    def test_conflicting_alias_rejected(self) -> None:
        from vibe_quant.dsl.schema import IndicatorConfig

        with pytest.raises(ValueError, match="Conflicting 'period_k'"):
            IndicatorConfig(type="STOCH", period=14, period_k=10)

    def test_compute_stoch_prefers_dsl_fields(self) -> None:
        from vibe_quant.dsl.compute_builtins import compute_stoch

        df = _ohlc(300, seed=2)
        a = compute_stoch(df, {"period_k": 14, "period_d": 3, "period": 9, "d_period": 5})
        b = compute_stoch(df, {"period_k": 9, "period_d": 5})
        pd.testing.assert_series_equal(a["k"], b["k"])
        pd.testing.assert_series_equal(a["d"], b["d"])


class TestUnknownOverrideKeys:
    FIELDS = (
        "instrument_id",
        "rsi_period",
        "stoch_period",
        "stoch_d_period",
        "take_profit_risk_reward",
        "take_profit_long_risk_reward",
        "stop_loss_percent",
    )

    def test_dot_and_underscore_keys_map(self) -> None:
        out = resolve_strategy_params(
            {"rsi.period": 10, "stop_loss_percent": 1.5}, self.FIELDS
        )
        assert out == {"rsi_period": 10, "stop_loss_percent": 1.5}

    def test_known_aliases_map(self) -> None:
        out = resolve_strategy_params(
            {
                "take_profit.risk_reward_ratio": 2.0,
                "take_profit_long.risk_reward_ratio": 3.0,
                "stoch.period_k": 9,
                "stoch.period_d": 5,
            },
            self.FIELDS,
        )
        assert out == {
            "take_profit_risk_reward": 2.0,
            "take_profit_long_risk_reward": 3.0,
            "stoch_period": 9,
            "stoch_d_period": 5,
        }

    def test_unknown_key_raises_with_valid_list(self) -> None:
        with pytest.raises(UnknownStrategyParamError, match=r"RSI_0\.period") as ei:
            resolve_strategy_params({"RSI_0.period": 10}, self.FIELDS)
        assert "rsi_period" in str(ei.value)

    def test_runner_call_raises_instead_of_silent_drop(self) -> None:
        dsl = {
            "name": "unknown_key_probe",
            "timeframe": "4h",
            "indicators": {"rsi": {"type": "RSI", "period": 14}},
            "entry_conditions": {"long": ["rsi < 30"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 4.0},
        }
        runner = NTScreeningRunner(dsl, ["BTCUSDT"], "2024-01-01", "2024-02-01")
        with pytest.raises(UnknownStrategyParamError, match="rsi.lenght"):
            runner({"rsi.lenght": 10})

    def test_unknown_sweep_key_fails_at_construction(self) -> None:
        dsl = {
            "name": "unknown_sweep_probe",
            "timeframe": "4h",
            "indicators": {"rsi": {"type": "RSI", "period": 14}},
            "entry_conditions": {"long": ["rsi < 30"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 4.0},
            "sweep": {"rsi_period": [10, 14], "RSI_0.period": [7, 14]},
        }
        with pytest.raises(UnknownStrategyParamError, match=r"RSI_0\.period"):
            NTScreeningRunner(dsl, ["BTCUSDT"], "2024-01-01", "2024-02-01")
