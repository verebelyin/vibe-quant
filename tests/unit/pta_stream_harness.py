"""Stream harness: drive a COMPILED strategy's real ``_feed_pta_buffer`` path.

``run_stream`` pushes bar events through the generated ``_feed_pta_buffer`` /
``_update_pta_indicators`` (buffer append, 25%-slack trim, per-bar recompute)
without an NT engine and records the ``_pta_values`` snapshot after every bar,
so a test can compare indicator value streams (memo on vs off) across buffer
trims.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.indicators import pta_buffer_cap, pta_lookback
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.prefix_memo import MEMO, refresh_enabled

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# (timeframe, open, high, low, close, volume)
BarEvent = tuple[str, float, float, float, float, float]


def compile_indicators(
    indicators: dict[str, dict[str, Any]],
    timeframe: str = "1h",
    additional_timeframes: list[str] | None = None,
) -> tuple[ModuleType, type]:
    """Compile a strategy whose only content is ``indicators`` (a raw DSL dict)."""
    names = list(indicators)
    dsl = validate_strategy_dict(
        {
            "name": "pta_stream",
            "timeframe": timeframe,
            "indicators": indicators,
            "additional_timeframes": additional_timeframes or [],
            "entry_conditions": {"long": [f"{names[0]} > 0"]},
            "exit_conditions": {"long": [f"{names[0]} < 0"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 4.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    cls = next(v for k, v in vars(module).items() if k.endswith("Strategy") and k != "Strategy")
    return module, cls


@contextmanager
def memo_env(enabled: bool) -> Iterator[None]:
    """Set the kill switch and start from an empty global memo."""
    old = os.environ.get("VIBE_QUANT_PTA_MEMO")
    os.environ["VIBE_QUANT_PTA_MEMO"] = "1" if enabled else "0"
    refresh_enabled()
    MEMO.clear()
    try:
        yield
    finally:
        if old is None:
            del os.environ["VIBE_QUANT_PTA_MEMO"]
        else:
            os.environ["VIBE_QUANT_PTA_MEMO"] = old
        refresh_enabled()


def run_stream(
    compiled_cls: type,
    bars: Sequence[BarEvent],
    memo: bool,
    *,
    specs: dict[str, tuple[str, dict[str, object], str]],
) -> list[dict[str, float]]:
    """Feed ``bars`` and return the ``_pta_values`` snapshot after each event.

    ``specs``: indicator name -> (registry type, params, timeframe), mirroring
    what the generated ``__init__`` derives from the config.
    """
    harness: Any = type("Harness", (compiled_cls,), {"config": SimpleNamespace(instrument_id="X")})
    inst: Any = harness.__new__(harness)
    tfs = sorted({tf for _, _, tf in specs.values()})
    inst._pta_bufs = {tf: {k: [] for k in ("open", "high", "low", "close", "volume")} for tf in tfs}
    inst._pta_values = {}
    inst._pta_params = {name: params for name, (_, params, _) in specs.items()}
    inst._pta_lookback = {name: pta_lookback(t, params) for name, (t, params, _) in specs.items()}
    inst._pta_buffer_cap = {
        tf: pta_buffer_cap(
            [inst._pta_lookback[n] for n, (_, _, t) in specs.items() if t == tf],
            full_history=False,
        )
        for tf in tfs
    }
    out: list[dict[str, float]] = []
    with memo_env(memo):
        for tf, o, h, lo, c, v in bars:
            bar = SimpleNamespace(open=o, high=h, low=lo, close=c, volume=v)
            inst._feed_pta_buffer(tf, bar)
            out.append(dict(inst._pta_values))
    return out


def random_walk_bars(n: int, seed: int = 7, tf: str = "1h") -> list[BarEvent]:
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + np.abs(rng.randn(n)) * 0.4
    low = close - np.abs(rng.randn(n)) * 0.4
    return [
        (tf, float(close[i]), float(high[i]), float(low[i]), float(close[i]), 1.0) for i in range(n)
    ]
