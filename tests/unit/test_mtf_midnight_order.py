"""A ``1d`` additional timeframe switches on the primary bars at midnight, on both real runners.

TradingView (Pine v4 ``security('D', open, lookahead_off)`` on historical bars) merges the
daily value at the LAST intraday bar of the day: D1 (the last completed daily bar) turns into
the new day's value on that final bar. In NT a 1d bar and the day's final intraday bar share
``ts_init`` (open + length - 1 ms), so the strategy sees them in the order the data was
added to the engine. Both runners add data in ``sorted(timeframes)`` order, and ``"1d"`` sorts
before ``"1m"``, ``"1h"``, ``"4h"`` and ``"5m"``, so those primaries already see the new
daily value on their final bar. ``"15m"`` sorts before ``"1d"``: its final bar still sees the
old value and the switch lands one bar late (design-v2 § 3.3, user ruling D5: the engine's
data order is kept, port 14 refuses a 15m primary).

Mechanism: the DSL cannot read a daily ``open`` yet (``source`` other than close is refused;
the PRICE plugin lands with T1), so the daily levels come from two NT ``SimpleMovingAverage``
indicators on the ``1d`` bars: period 1 = the last completed daily close (D1), period 2 =
the mean of the last two (D2 = 2 * mean2 - D1). The fixture's daily bars have open == close ==
level, so the close IS the open. A recording wrapper on the generated strategy's ``on_bar``
captures (ts_init, D1, D2) for every PRIMARY bar the real runner delivers.
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
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.validation.runner import ValidationRunner

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SYMBOL = "BTCUSDT"
START, END = "2024-01-01", "2024-01-04"  # three whole UTC days: 0, 1, 2
DAY_MS = 86_400_000
MINUTE_MS = 60_000
DAY0_MS = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()) * 1000
LEVELS = (100.0, 200.0, 300.0)  # daily open == close of day 0 / 1 / 2
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}

# (ts_init ns, D1, D2) per primary bar the strategy received; D1/D2 None until initialized.
Record = tuple[int, float | None, float | None]
_RECORDS: list[Record] = []


def _install_recorder(module: Any, strategy_cls_name: str) -> None:
    """Wrap the generated strategy's ``on_bar`` once: record the daily levels at every primary bar."""
    cls = getattr(module, strategy_cls_name)
    if getattr(cls, "_mtf_recorder", False):
        return
    original = cls.on_bar

    def on_bar(self: Any, bar: Any) -> None:
        if bar.bar_type == self.primary_bar_type:
            one, two = self.ind_daily_1, self.ind_daily_2
            d1 = float(one.value) if one.initialized else None
            d2 = 2 * float(two.value) - float(one.value) if two.initialized else None
            _RECORDS.append((int(bar.ts_init), d1, d2))
        original(self, bar)

    cls.on_bar = on_bar
    cls._mtf_recorder = True


@pytest.fixture(autouse=True)
def _record_generated_strategies(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Both runners compile through ``StrategyCompiler.compile_to_module``: hook the recorder in there."""
    original = StrategyCompiler.compile_to_module

    def compile_to_module(self: StrategyCompiler, dsl: Any, *args: Any, **kwargs: Any) -> Any:
        module = original(self, dsl, *args, **kwargs)
        class_name = "".join(word.capitalize() for word in dsl.name.split("_"))
        _install_recorder(module, f"{class_name}Strategy")
        return module

    monkeypatch.setattr(StrategyCompiler, "compile_to_module", compile_to_module)
    _RECORDS.clear()
    yield
    _RECORDS.clear()


def _dsl(runner: str, timeframe: str) -> dict[str, object]:
    return {
        "name": f"mtf_midnight_{runner}_{timeframe}",
        "timeframe": timeframe,
        "additional_timeframes": ["1d"],
        "indicators": {
            "daily_1": {"type": "SMA", "period": 1, "timeframe": "1d"},
            "daily_2": {"type": "SMA", "period": 2, "timeframe": "1d"},
        },
        # never true: the strategy only observes (no order, so fills play no part)
        "entry_conditions": {"long": ["daily_1 > 1000000"]},
        "exit_conditions": {"long": ["daily_1 < 0"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tmp catalog: 3 days of flat 1m bars at the daily level, aggregated like production."""
    path = tmp_path_factory.mktemp("mtf_midnight") / "catalog"
    manager = CatalogManager(path)
    instrument = create_instrument(SYMBOL)
    manager.write_instrument(instrument)
    klines = []
    for day, level in enumerate(LEVELS):
        for minute in range(DAY_MS // MINUTE_MS):
            open_ms = DAY0_MS + day * DAY_MS + minute * MINUTE_MS
            klines.append(
                {
                    "open_time": open_ms,
                    "close_time": open_ms + MINUTE_MS - 1,
                    "open": level,
                    "high": level,
                    "low": level,
                    "close": level,
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


def _run_screening(catalog: Path, timeframe: str) -> None:
    runner = NTScreeningRunner(
        _dsl("screening", timeframe), [SYMBOL], START, END, catalog_path=str(catalog)
    )
    metrics = runner({})
    assert metrics.sharpe_ratio != float("-inf"), "screening run failed"


def _run_validation(catalog: Path, timeframe: str, tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    state = StateManager(db)
    dsl = _dsl("validation", timeframe)
    strategy_id = state.create_strategy(name=str(dsl["name"]), dsl_config=dsl)
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="validation",
        symbols=[SYMBOL],
        timeframe=timeframe,
        start_date=START,
        end_date=END,
        parameters={},
    )
    state.close()
    runner = ValidationRunner(
        db_path=db,
        logs_path=tmp_path / "logs",
        catalog_path=catalog,
        funding_archive_path=tmp_path / "no_funding.db",
    )
    try:
        runner.run(run_id=run_id)
    finally:
        runner.close()


def _run(runner: str, catalog: Path, timeframe: str, tmp_path: Path) -> list[Record]:
    _RECORDS.clear()
    if runner == "screening":
        _run_screening(catalog, timeframe)
    else:
        _run_validation(catalog, timeframe, tmp_path)
    return list(_RECORDS)


def _expected_levels(ts_init: int, switch_on_final_bar: bool) -> tuple[float | None, float | None]:
    """(D1, D2) for a primary bar under the stated switch rule, from the fixture alone."""
    day = (ts_init // 1_000_000 - DAY0_MS) // DAY_MS
    in_day_ms = (ts_init // 1_000_000 - DAY0_MS) % DAY_MS
    is_final = in_day_ms == DAY_MS - 1  # ts_init = close - 1 ms
    # daily bars already delivered when this primary bar arrives
    completed = day + 1 if (is_final and switch_on_final_bar) else day
    d1 = LEVELS[completed - 1] if completed >= 1 else None
    d2 = LEVELS[completed - 2] if completed >= 2 else None
    return d1, d2


def _day2(records: list[Record], tf: str) -> tuple[Record, Record, Record]:
    """First, middle and final primary bar of day 2."""
    start_ns = (DAY0_MS + 2 * DAY_MS) * 1_000_000
    bars = [r for r in records if r[0] >= start_ns]
    assert len(bars) == DAY_MS // (TF_MINUTES[tf] * MINUTE_MS)
    return bars[0], bars[len(bars) // 2], bars[-1]


@pytest.mark.parametrize("tf", ["1m", "5m", "1h", "4h"])
@pytest.mark.parametrize("runner", ["screening", "validation"])
def test_daily_merge_switches_on_final_intraday_bar(
    runner: str, tf: str, catalog: Path, tmp_path: Path
) -> None:
    """R40/R41: D1/D2 turn over on the day's FINAL intraday bar, on both real runners."""
    records = _run(runner, catalog, tf, tmp_path)
    assert len(records) == 3 * DAY_MS // (TF_MINUTES[tf] * MINUTE_MS), "every primary bar recorded"

    # R40: day 2 first / middle / final bar, literal values.
    first, middle, final = _day2(records, tf)
    assert (first[1], first[2]) == (LEVELS[1], LEVELS[0])
    assert (middle[1], middle[2]) == (LEVELS[1], LEVELS[0])
    assert (final[1], final[2]) == (LEVELS[2], LEVELS[1])

    # Every bar of the 3 days follows the final-bar rule (also day 0 -> 1 midnight).
    got = list(records)
    want = [(ts, *_expected_levels(ts, switch_on_final_bar=True)) for ts, _, _ in records]
    assert got == want


@pytest.mark.parametrize("runner", ["screening", "validation"])
def test_daily_merge_15m_switches_one_bar_late_pinned(
    runner: str, catalog: Path, tmp_path: Path
) -> None:
    """R42, user ruling D5 (2026-11-08): the engine keeps its data order, so a 15m primary
    sees the new daily value one bar late.

    ``sorted(timeframes)`` puts ``"15m"`` before ``"1d"``: at the midnight tie the 15m bar is
    delivered first, so its final bar still carries the old D1/D2 and the first bar of the next
    day carries the new ones. TradingView switches on the final bar; port 14 refuses a 15m
    primary for exactly this reason. This test pins today's behaviour: if it turns red, the
    data order changed (a comparability break for every 15m multi-timeframe strategy) and
    D5 / the port 14 refusal must be revisited.
    """
    records = _run(runner, catalog, "15m", tmp_path)
    assert len(records) == 3 * DAY_MS // (15 * MINUTE_MS)

    first, middle, final = _day2(records, "15m")
    assert (first[1], first[2]) == (LEVELS[1], LEVELS[0])
    assert (middle[1], middle[2]) == (LEVELS[1], LEVELS[0])
    # TradingView would show (LEVELS[2], LEVELS[1]) here; the engine is one bar late.
    assert (final[1], final[2]) == (LEVELS[1], LEVELS[0])

    got = list(records)
    want = [(ts, *_expected_levels(ts, switch_on_final_bar=False)) for ts, _, _ in records]
    assert got == want
    # The late switch is visible at the day 1 -> 2 midnight: the first bar of day 2 is the
    # first bar carrying day 1's value, one bar after TradingView would show it.
    day1_final = [r for r in records if r[0] == (DAY0_MS + 2 * DAY_MS) * 1_000_000 - 1_000_000][0]
    assert (day1_final[1], day1_final[2]) == (LEVELS[0], None)
