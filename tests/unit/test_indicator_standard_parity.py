"""NT indicators, fast tier and compute_fn compute the pandas-ta-classic definition.

A compiled DSL strategy holding RSI, ATR (14 and 30), NATR, ROC, STOCH, BBANDS and KC runs
in a real ``BacktestEngine`` over catalog BTCUSDT bars; the value ``_get_indicator_value``
returns for every output on every bar is compared with ``pandas_ta_classic`` on the same
bars, and with each spec's ``compute_fn``. The fast tier's replay (``build_strategy`` +
``handle_bar`` per bar) must give the engine's values bit for bit. NT seeds its averages
from the first value, TradingView from a simple average of the first ``n`` values, so the
ta comparison starts after a warm-up (Wilder 30n, EMA 15n, windows at their first defined
row). Real catalog bars: skipped when data/catalog is absent.
"""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.nt_compat import retain_log_guard

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "data" / "catalog"
_BAR_DIR = CATALOG / "data" / "bar"
_IID = "BTCUSDT-PERP.BINANCE"
needs_catalog = pytest.mark.skipif(
    not (_BAR_DIR / f"{_IID}-1-HOUR-LAST-EXTERNAL").is_dir()
    or not (_BAR_DIR / f"{_IID}-1-MINUTE-LAST-EXTERNAL").is_dir(),
    reason="needs the BTCUSDT 1h and 1m catalog",
)

# DSL indicator name -> DSL fields; every output is read as ``name`` or ``name_output``.
_DSL_INDICATORS: dict[str, dict[str, Any]] = {
    "rsi": {"type": "RSI", "period": 14},
    "atr14": {"type": "ATR", "period": 14},
    "atr30": {"type": "ATR", "period": 30},
    "natr": {"type": "NATR", "period": 14},
    "roc": {"type": "ROC", "period": 10},
    "stoch": {"type": "STOCH", "period": 14, "d_period": 3},
    "bb": {"type": "BBANDS", "period": 20, "std_dev": 2.0},
    "kc": {"type": "KC", "period": 20, "atr_multiplier": 2.0},
}
_KEYS = (
    "rsi", "atr14", "atr30", "natr", "roc", "stoch_k", "stoch_d",
    "bb_upper", "bb_middle", "bb_lower", "kc_upper", "kc_middle", "kc_lower",
)  # fmt: skip

# R1-R7 per DSL output: (type, first compared row W, tolerance), design § 4.
# Wilder 30n (RSI/ATR/NATR 14 -> 420, ATR 30 -> 900), EMA 15n (KC 20 -> 300), windows at
# their first defined row (ROC 10, STOCH 14 + 3 - 2 rows, BBANDS 20 - 1).
_ROWS: dict[str, tuple[str, int, float]] = {
    "rsi": ("RSI", 420, 1e-9),
    "atr14": ("ATR", 420, 1e-9),
    "atr30": ("ATR", 900, 1e-9),
    "natr": ("NATR", 420, 1e-9),
    "roc": ("ROC", 10, 1e-9),
    "stoch_k": ("STOCH", 15, 1e-12),
    "stoch_d": ("STOCH", 15, 1e-12),
    "bb_upper": ("BBANDS", 19, 1e-9),
    "bb_middle": ("BBANDS", 19, 1e-9),
    "bb_lower": ("BBANDS", 19, 1e-9),
    "kc_upper": ("KC", 300, 1e-9),
    "kc_middle": ("KC", 300, 1e-9),
    "kc_lower": ("KC", 300, 1e-9),
}


def _ta_reference(df: pd.DataFrame) -> dict[str, pd.Series]:
    h, low, c = df["high"], df["low"], df["close"]
    st = ta.stoch(h, low, c, k=14, d=3, smooth_k=1)
    bb = ta.bbands(c, length=20, std=2.0)
    kc = ta.kc(h, low, c, length=20, scalar=2.0)
    return {
        "rsi": ta.rsi(c, 14),
        "atr14": ta.atr(h, low, c, 14),
        "atr30": ta.atr(h, low, c, 30),
        "natr": ta.natr(h, low, c, 14),
        "roc": ta.roc(c, 10),
        "stoch_k": st.iloc[:, 0],
        "stoch_d": st.iloc[:, 1],
        "bb_lower": bb.iloc[:, 0],
        "bb_middle": bb.iloc[:, 1],
        "bb_upper": bb.iloc[:, 2],
        "kc_lower": kc.iloc[:, 0],
        "kc_middle": kc.iloc[:, 1],
        "kc_upper": kc.iloc[:, 2],
    }


def _compile(timeframe: str) -> tuple[Any, str, str]:
    dsl: dict[str, Any] = {
        "name": f"std_parity_{timeframe}",
        "timeframe": timeframe,
        "indicators": _DSL_INDICATORS,
        "entry_conditions": {"long": ["rsi > 1000"]},  # never fires: the run only reads indicators
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(dsl))
    cn = _to_class_name(str(dsl["name"]))
    return module, f"{cn}Strategy", f"{cn}Config"


def _read(strategy: Any, bar: Any) -> dict[str, float]:
    strategy._last_close = float(bar.close)
    return {k: float(strategy._get_indicator_value(k)) for k in _KEYS}


def _bars(timeframe: str, start: str, end: str) -> list[Any]:
    spec = {"1h": "1-HOUR", "1m": "1-MINUTE"}[timeframe]
    return list(
        ParquetDataCatalog(str(CATALOG)).bars(
            bar_types=[f"{_IID}-{spec}-LAST-EXTERNAL"], start=start, end=end
        )
    )


def _engine_values(
    timeframe: str, start: str, end: str
) -> tuple[list[Any], pd.DataFrame, pd.DataFrame]:
    """(bars, OHLCV frame, indicator values after each bar) from a real BacktestEngine run."""
    bars = _bars(timeframe, start, end)
    module, strategy_name, config_name = _compile(timeframe)
    rows: list[dict[str, float]] = []
    base = getattr(module, strategy_name)

    class Recording(base):  # type: ignore[misc, valid-type]
        def on_bar(self, bar: Any) -> None:
            super().on_bar(bar)
            rows.append(_read(self, bar))

    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(10_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        bar_execution=True,
    )
    engine.add_instrument(ParquetDataCatalog(str(CATALOG)).instruments(instrument_ids=[_IID])[0])
    engine.add_data(bars)
    engine.add_strategy(Recording(getattr(module, config_name)(instrument_id=_IID)))
    engine.run()
    engine.dispose()
    df = pd.DataFrame(
        {
            k: [float(getattr(b, k)) for b in bars]
            for k in ("open", "high", "low", "close", "volume")
        }
    )
    return bars, df, pd.DataFrame(rows)


@pytest.fixture(scope="module")
def engine_1h() -> tuple[list[Any], pd.DataFrame, pd.DataFrame]:
    return _engine_values("1h", "2026-01-01", "2026-04-01")


def _max_diff(nt: pd.Series, ref: pd.Series, warm: int) -> float:
    return float((nt.iloc[warm:] - ref.iloc[warm:]).abs().max())


@needs_catalog
@pytest.mark.parametrize("indicator", ["RSI", "ATR", "NATR", "ROC", "STOCH", "BBANDS", "KC"])
def test_engine_matches_ta(
    indicator: str, engine_1h: tuple[list[Any], pd.DataFrame, pd.DataFrame]
) -> None:
    """R1-R7: the engine's value equals the pandas-ta-classic definition after the warm-up."""
    bars, df, nt = engine_1h
    assert len(nt) == len(bars) == 2160
    ref = _ta_reference(df)
    keys = [k for k, (typ, _, _) in _ROWS.items() if typ == indicator]
    assert keys
    for key in keys:
        _, warm, tol = _ROWS[key]
        assert len(nt) > warm + 500, key  # the compared span is real
        diff = _max_diff(nt[key], ref[key], warm)
        assert diff <= tol, f"{indicator} {key}: max|NT - ta| from row {warm} = {diff!r} > {tol}"


def _compute_fn_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Every DSL output through its spec's ``compute_fn`` (NATR has none: a helper over ATR)."""
    out: dict[str, pd.Series] = {}
    for name, params in _DSL_INDICATORS.items():
        spec = indicator_registry.get(str(params["type"]))
        assert spec is not None
        if spec.compute_fn is None:
            continue
        res = spec.compute_fn(df, {k: v for k, v in params.items() if k != "type"})
        if isinstance(res, dict):
            for out_name, series in res.items():
                out[f"{name}_{out_name}"] = series
        else:
            out[name] = res
    return out


@needs_catalog
@pytest.mark.parametrize("key", [k for k in _ROWS if k != "natr"])
def test_engine_matches_compute_fn(
    key: str, engine_1h: tuple[list[Any], pd.DataFrame, pd.DataFrame]
) -> None:
    """R10: the NT path and the compute_fn path agree to 1e-9 from the warm-up on."""
    _, df, nt = engine_1h
    fn = _compute_fn_series(df)
    _, warm, _ = _ROWS[key]
    diff = _max_diff(nt[key], fn[key], warm)
    assert diff <= 1e-9, f"{key}: max|NT - compute_fn| from row {warm} = {diff!r}"


def test_natr_has_no_compute_fn() -> None:
    """R10 skips NATR only because it is ATR/close scaled by a helper, not a compute_fn."""
    spec = indicator_registry.get("NATR")
    assert (
        spec is not None
        and spec.compute_fn is None
        and spec.primary_helper == "compute_natr_hourly"
    )


def _fast_tier_values(bars: list[Any]) -> pd.DataFrame:
    """The fast tier's indicator path: ``build_strategy`` + ``handle_bar`` per bar (``compute_signals``)."""
    from vibe_quant.fast_tier.signals import build_strategy, load_instrument

    module, strategy_name, config_name = _compile("1m")
    instrument = load_instrument(str(CATALOG), _IID)
    strategy, indicators, _ = build_strategy(
        module.__name__, strategy_name, config_name, {"instrument_id": _IID}, instrument
    )
    rows = []
    for bar in bars:
        for indicator in indicators:
            indicator.handle_bar(bar)
        rows.append(_read(strategy, bar))
    return pd.DataFrame(rows)


@needs_catalog
def test_fast_tier_matches_engine() -> None:
    """R9: fast tier == NT engine, diff 0.0, on 1m BTC 2026-03-02..03-09 for all seven types."""
    bars, _, nt = _engine_values("1m", "2026-03-02", "2026-03-09")
    fast = _fast_tier_values(bars)
    assert len(nt) == len(fast) == len(bars) > 9000
    for key in _KEYS:
        assert not nt[key].isna().any() and not fast[key].isna().any(), key
        diff = float((nt[key] - fast[key]).abs().max())
        assert diff == 0.0, f"{key}: max|NT - fast tier| = {diff!r}"
        assert math.isfinite(float(nt[key].iloc[-1])), key
        assert np.ptp(nt[key].to_numpy()) > 0.0, key  # a constant column would pass vacuously
