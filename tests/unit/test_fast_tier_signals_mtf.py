"""``compute_signals`` over a primary timeframe plus merged additional timeframes.

R43: the vector replay of a strategy that reads a ``1d`` additional timeframe equals what the
real screening runner's generated strategy sees at every primary bar, including the midnight
ties: a ``1d`` bar and the day's final intraday bar share ``ts_init``, NT hands them to the
strategy in the order the data was added (``sorted(timeframes)``), so ``1m``/``5m``/``1h``/
``4h`` primaries see the new daily value ON their final bar and ``15m`` (sorted before
``1d``) one bar late (user ruling D5, pinned by ``test_mtf_midnight_order.py``).

The reference is the runtime itself: a recorder on the generated ``_check_long_entry`` /
``_check_short_entry`` stores each raw result (and returns False so nothing trades), the
real ``NTScreeningRunner`` runs over a tmp catalog, and the replay's booleans must equal the
recorded ones bar for bar. The fixture's daily ``open`` differs from its ``close`` (the final
minute of every day closes 50 higher), so a replay that read the close for ``PRICE(open)``
fails.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.catalog import (
    CatalogManager,
    aggregate_bars,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.screening.nt_runner import NTScreeningRunner

if TYPE_CHECKING:
    from pathlib import Path

SYMBOL = "BTCUSDT"
START, END = "2024-01-01", "2024-01-06"  # five whole UTC days 0..4
DAY_MS = 86_400_000
MINUTE_MS = 60_000
DAY0_MS = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()) * 1000
LEVELS = (100.0, 200.0, 300.0, 400.0, 500.0)  # daily OPEN of day 0..4
CLOSE_BUMP = 50.0  # the day's final minute closes this much above its open
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}

#: (ts_init ns, long-entry raw result, short-entry raw result) per ready primary bar
Record = tuple[int, bool, bool]
_RECORDS: list[Record] = []


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("mtf_signals") / "catalog"
    manager = CatalogManager(path)
    instrument = create_instrument(SYMBOL)
    manager.write_instrument(instrument)
    klines = []
    for day, level in enumerate(LEVELS):
        for minute in range(DAY_MS // MINUTE_MS):
            open_ms = DAY0_MS + day * DAY_MS + minute * MINUTE_MS
            last = minute == DAY_MS // MINUTE_MS - 1
            close = level + CLOSE_BUMP if last else level
            klines.append(
                {
                    "open_time": open_ms,
                    "close_time": open_ms + MINUTE_MS - 1,
                    "open": level,
                    "high": max(level, close),
                    "low": level,
                    "close": close,
                    "volume": 1.0,
                }
            )
    bars_1m = klines_to_bars(
        klines,  # type: ignore[arg-type]  # dict rows stand in for sqlite3.Row
        instrument.id,
        get_bar_type(SYMBOL, "1m"),
        instrument.size_precision,
        instrument.price_precision,
    )
    manager.write_bars(bars_1m)
    for interval in ("5m", "15m", "1h", "4h", "1d"):
        manager.write_bars(
            aggregate_bars(
                bars_1m,
                get_bar_type(SYMBOL, interval),
                TF_MINUTES.get(interval, 1440),
                instrument.size_precision,
                instrument.price_precision,
            )
        )
    return path


def _dsl(kind: str, timeframe: str) -> dict[str, Any]:
    """Entries read the daily level; ``price`` through PRICE(open) 1d, ``sma`` through NT SMA(close) 1d."""
    if kind == "price":
        indicators: dict[str, Any] = {
            "d_open": {"type": "PRICE", "src": "open", "offset": 0, "timeframe": "1d"},
            "d_open_prev": {"type": "PRICE", "src": "open", "offset": 1, "timeframe": "1d"},
        }
        # D1 open > 250: day 2's level 300 on (and after) day 2's final bar; D2 open > 150: 200
        long_cond, short_cond = "d_open > 250", "d_open_prev > 150"
    else:
        indicators = {
            "d_sma1": {"type": "SMA", "period": 1, "timeframe": "1d"},
            "d_sma2": {"type": "SMA", "period": 2, "timeframe": "1d"},
        }
        long_cond, short_cond = "d_sma1 > 340", "d_sma2 > 250"
    return {
        "name": f"mtf_signals_{kind}_{timeframe}",
        "timeframe": timeframe,
        "additional_timeframes": ["1d"],
        "indicators": indicators,
        "entry_conditions": {"long": [long_cond], "short": [short_cond]},
        "exit_conditions": {"long": ["close < 0"], "short": ["close < 0"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }


def _record_engine_checks(
    monkeypatch: pytest.MonkeyPatch, catalog: Path, dsl: dict[str, Any]
) -> list[Record]:
    """Run the real screening runner with a recorder on the generated entry checks."""
    from vibe_quant.dsl.compiler import StrategyCompiler

    original = StrategyCompiler.compile_to_module
    pending: dict[int, bool] = {}

    def compile_to_module(self: StrategyCompiler, dsl_obj: Any, *args: Any, **kwargs: Any) -> Any:
        module = original(self, dsl_obj, *args, **kwargs)
        cls = getattr(module, "".join(w.capitalize() for w in dsl_obj.name.split("_")) + "Strategy")
        if not getattr(cls, "_mtf_signal_recorder", False):
            check_long, check_short = cls._check_long_entry, cls._check_short_entry

            def long_entry(self: Any, bar: Any) -> bool:
                pending[int(bar.ts_init)] = bool(check_long(self, bar))
                return False

            def short_entry(self: Any, bar: Any) -> bool:
                raw = bool(check_short(self, bar))
                _RECORDS.append((int(bar.ts_init), pending.pop(int(bar.ts_init)), raw))
                return False

            cls._mtf_originals = (check_long, check_short)
            cls._check_long_entry, cls._check_short_entry = long_entry, short_entry
            cls._mtf_signal_recorder = True
        return module

    monkeypatch.setattr(StrategyCompiler, "compile_to_module", compile_to_module)
    _RECORDS.clear()
    runner = NTScreeningRunner(dsl, [SYMBOL], START, END, catalog_path=str(catalog))
    metrics = runner({})
    assert metrics.sharpe_ratio != float("-inf"), "screening run failed"
    recorded = list(_RECORDS)
    # un-record: the replay below calls the same generated class
    runner._ensure_compiled()
    import sys

    cls = getattr(sys.modules[runner._module_path], runner._strategy_cls_name)
    cls._check_long_entry, cls._check_short_entry = cls._mtf_originals
    cls._mtf_signal_recorder = False
    _RECORDS.clear()
    _MODULE_RUNNERS.append(runner)
    return recorded


_MODULE_RUNNERS: list[NTScreeningRunner] = []


def _replayed(timeframe: str, kind: str, *, coarse_first: bool = False) -> list[Record]:
    from vibe_quant.fast_tier.signals import signals_for_runner

    runner = _MODULE_RUNNERS[-1]
    run = signals_for_runner(runner, SYMBOL, coarse_first=coarse_first)
    sig = run.signals
    return [
        (int(bar.ts_init), bool(sig.long_entry[i]), bool(sig.short_entry[i]))
        for i, bar in enumerate(run.bars[timeframe])
        if sig.ready[i]
    ]


def _final_bar_ns(day: int, timeframe: str) -> int:
    """ts_init of the last primary bar of ``day`` (close - 1 ms)."""
    return (DAY0_MS + (day + 1) * DAY_MS - 1) * 1_000_000


def _at(records: list[Record], ts_ns: int) -> Record:
    return next(r for r in records if r[0] == ts_ns)


@pytest.mark.parametrize("kind", ["price", "sma"])
def test_compute_signals_1h_with_1d_matches_runtime(
    kind: str, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R43: a 1h strategy reading 1d switches on the day's FINAL hourly bar, replay == runtime."""
    engine = _record_engine_checks(monkeypatch, catalog, _dsl(kind, "1h"))
    replay = _replayed("1h", kind)

    assert len(engine) >= 72  # 3 ready days of hourly bars, not an empty record
    assert replay == engine

    # Literal pins at the day 2 -> 3 midnight. Before the final hourly bar D1 is day 1 (open 200,
    # close 250) and D2 is day 0: neither condition holds. On the final bar the 1d bar of day 2 is
    # merged (D1 open 300 / close 350, D2 open 200 / mean of the last two closes 300): both hold.
    one_hour_ns = 3_600_000_000_000
    assert _at(replay, _final_bar_ns(2, "1h") - one_hour_ns)[1:] == (False, False)
    assert _at(replay, _final_bar_ns(2, "1h"))[1:] == (True, True)


def test_compute_signals_15m_with_1d_switches_one_bar_late(
    catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D5: a 15m primary sorts before 1d, so the daily value lands one primary bar later."""
    engine = _record_engine_checks(monkeypatch, catalog, _dsl("price", "15m"))
    replay = _replayed("15m", "price")

    assert replay == engine
    final = _final_bar_ns(2, "15m")
    assert _at(replay, final)[1:] == (False, False)  # TradingView would already show D1 = 300 here
    assert _at(replay, final + 15 * 60 * 1_000_000_000)[1:] == (True, True)


@pytest.mark.parametrize("timeframe", ["15m", "1h"])
def test_coarse_first_merges_the_daily_value_on_the_final_bar_for_every_primary(
    timeframe: str, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TradingView's order (``coarse_first``): the 1d bar goes first at the midnight tie, so even
    a 15m primary sees day 2's value ON its final bar, one bar before the engine's order does."""
    _record_engine_checks(monkeypatch, catalog, _dsl("price", timeframe))
    tv = _replayed(timeframe, "price", coarse_first=True)
    engine_order = _replayed(timeframe, "price")

    final = _final_bar_ns(2, timeframe)
    assert _at(tv, final)[1:] == (True, True)
    step_ns = TF_MINUTES[timeframe] * 60 * 1_000_000_000
    assert _at(tv, final - step_ns)[1:] == (False, False)
    # The two orders differ only at midnight ties of a 15m primary (1d already sorts before 1h):
    # day 1's final bar (the first ready bar in TV order, one bar before the engine's) and day 2's
    # final bar (the flip itself).
    tv_by_ts = {r[0]: r[1:] for r in tv}
    engine_by_ts = {r[0]: r[1:] for r in engine_order}
    differing = sorted(
        {ts for ts, vals in tv_by_ts.items() if engine_by_ts.get(ts) != vals}
        | (engine_by_ts.keys() - tv_by_ts.keys())
    )
    expected = [_final_bar_ns(1, timeframe), final] if timeframe == "15m" else []
    assert differing == expected
