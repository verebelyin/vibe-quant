"""GA exit genes: screening and validation fill every new exit type the same way.

Each test builds a hand-made extended chromosome, renders it with the GA's own
``chromosome_to_dsl`` (so the DSL under test is what a discovery run would trade; only the
entry rule, timeframe and name are swapped for the toy's), and runs it through both public
entry points on a toy catalog with real NautilusTrader. Fills are read from each engine's
cache (every position's open/close ``ts_event``, prices and closing order type) plus every
protective order's trigger/limit level, never from serialized results.

Toy path (``_path``): BTC 10 000 / ETH 20 000, flat +-20 for 480 min, then +0.4167 per
minute to 11 000 at 2024-01-03 00:00 and -0.347 per minute afterwards; 1m high/low =
max/min(open, close) +- 3 (BTC) / +- 6 (ETH). Window 2024-01-01..2024-01-05, 4h strategy
timeframe, entry ``ema(3) > 0``, ATR(3). Every number below is MEASURED through these
helpers and equals the architect's probe (``probe_exit_parity.py`` / ``probe_rearm_gap.py``).

Price parity: timestamps of entries are exact; each engine may draw up to one tick of
slippage per fill independently, so one fill's prices differ by at most 2 ticks (chief
ruling R16c). Exit timestamps differ by design: screening stamps a protective exit at its
4h bar close, validation at the 1m minute it fills (the existing protective-exit stamping).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

from tests.unit.test_max_bars_exit import _utc
from tests.unit.test_max_bars_exit_parity import _Fills, _screening, _validation
from tests.unit.test_screening_fill_mode import _DAY0_MS, _MIN_MS, _N_MIN
from vibe_quant.data.catalog import (
    CatalogManager,
    aggregate_bars,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.discovery.exit_genes import ExitBlock, ExitRules, canonical_block
from vibe_quant.discovery.fitness import _evaluate_single
from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.dsl.schema import StrategyDSL
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.validation.runner import ValidationRunner

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from vibe_quant.screening.types import BacktestMetrics

_BTC_TICK = 0.1
_ETH_TICK = 0.01
_ENTRY_ATR = "exit_atr"  # the GA's indicator name for the long/only block's ATR
_H4_MIN = 240


# ---------------------------------------------------------------------------
# Toy catalogs
# ---------------------------------------------------------------------------


def _path(i: int, base: float) -> float:
    """Toy close of 1m bar ``i``: flat wiggle, a 40 h rise, then a fall."""
    s = base / 10_000.0
    if i < 480:
        return s * (10_000 + 20 * math.sin(i / 7))
    if i < 2880:
        return s * (10_000 + (i - 480) * (1000 / 2400))
    return s * (11_000 - (i - 2880) * (1500 / 4320))


def _crash_path(i: int, base: float) -> float:
    """Rise to 11 199.6, then the 4h bar opening at minute 2880 closes 300 lower."""
    s = base / 10_000.0
    if i < 2880:
        return s * (10_000 + i * (1000 / 2400))
    top = 10_000 + 2879 * (1000 / 2400)
    return s * (top - 300 + (i - 2880) * 0.01)


def _wild_path(i: int, base: float) -> float:
    """Price 3000..6000 with a 4h half-period: every 4h bar spans about 3000 (ATR ~ price)."""
    s = base / 10_000.0
    return s * (4500 + 1500 * math.sin(2 * math.pi * i / 480))


def _swing_path(i: int, base: float) -> float:
    """A 24 h sine around 10 000 (+-400): long entries on the way up, short ones on the way down."""
    s = base / 10_000.0
    return s * (10_000 + 400 * math.sin(2 * math.pi * i / 1440))


def _write_path_catalog(
    root: Path, path: Callable[[int, float], float], symbols: tuple[str, ...]
) -> Path:
    cm = CatalogManager(root)
    for symbol in symbols:
        inst = create_instrument(symbol)
        cm.write_instrument(inst)
        base = 20_000.0 if symbol == "ETHUSDT" else 10_000.0
        klines: list[Any] = []
        for i in range(_N_MIN):
            o, c = path(max(i - 1, 0), base), path(i, base)
            pad = 3 * (1 + (symbol == "ETHUSDT"))
            klines.append(
                {
                    "open_time": _DAY0_MS + i * _MIN_MS,
                    "open": o,
                    "high": max(o, c) + pad,
                    "low": min(o, c) - pad,
                    "close": c,
                    "volume": 10.0,
                    "close_time": _DAY0_MS + i * _MIN_MS + 59_999,
                }
            )
        bars_1m = klines_to_bars(
            klines, inst.id, get_bar_type(symbol, "1m"), inst.size_precision, inst.price_precision
        )
        cm.write_bars(bars_1m)
        cm.write_bars(
            aggregate_bars(
                bars_1m,
                get_bar_type(symbol, "4h"),
                _H4_MIN,
                inst.size_precision,
                inst.price_precision,
            )
        )
    return root


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_path_catalog(tmp_path_factory.mktemp("toy"), _path, ("BTCUSDT", "ETHUSDT"))


@pytest.fixture(scope="module")
def crash(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_path_catalog(tmp_path_factory.mktemp("crash"), _crash_path, ("BTCUSDT",))


@pytest.fixture(scope="module")
def wild(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_path_catalog(tmp_path_factory.mktemp("wild"), _wild_path, ("BTCUSDT",))


@pytest.fixture(scope="module")
def swing(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_path_catalog(tmp_path_factory.mktemp("swing"), _swing_path, ("BTCUSDT",))


# ---------------------------------------------------------------------------
# The GA's DSL for a hand-built extended chromosome
# ---------------------------------------------------------------------------


def _block(
    stop: str, stop_v: float | None, target: str, target_v: float | None, atr: int | None
) -> ExitBlock:
    return canonical_block(
        ExitBlock(
            stop_type=stop,  # type: ignore[arg-type]
            stop_value=stop_v,
            target_type=target,  # type: ignore[arg-type]
            target_value=target_v,
            atr_period=atr,
        )
    )


def _gene(period: float) -> StrategyGene:
    return StrategyGene(
        indicator_type="RSI",
        parameters={"period": period},
        condition=ConditionType.LT,
        threshold=30.0,
    )


def _ga_dsl(
    name: str,
    long: ExitBlock,
    *,
    short: ExitBlock | None = None,
    max_bars: int | None = None,
    entry: dict[str, list[str]] | None = None,
    direction: Direction = Direction.LONG,
    extra_indicators: dict[str, object] | None = None,
) -> dict[str, Any]:
    """``chromosome_to_dsl`` of an extended genome, with the toy's entry rule and 4h timeframe.

    Only what is not an exit gene is swapped: name, timeframe, the always-true ``ema(3) > 0``
    entry and the genome's RSI genes. The stop, target, ATR indicators and time exit stay
    exactly as the GA renders them.
    """
    both = direction == Direction.BOTH
    chrom = StrategyChromosome(
        entry_genes=[_gene(14.0)],
        exit_genes=[_gene(10.0)],
        stop_loss_pct=2.0,
        take_profit_pct=3.0,
        direction=direction,
        short_entry_genes=[_gene(9.0)] if both else [],
        short_exit_genes=[_gene(8.0)] if both else [],
        exit_rules=ExitRules(long=long, short=short, max_bars=max_bars),
    )
    dsl: dict[str, Any] = dict(chromosome_to_dsl(chrom))
    ga_atrs = {k: v for k, v in dsl["indicators"].items() if k.startswith("exit_atr")}
    dsl["indicators"] = {"ema": {"type": "EMA", "period": 3}, **ga_atrs, **(extra_indicators or {})}
    dsl["name"] = name
    dsl["timeframe"] = "4h"
    side = {Direction.LONG: "long", Direction.SHORT: "short"}.get(direction)
    dsl["entry_conditions"] = entry or {side or "long": ["ema > 0"]}
    del dsl["exit_conditions"]
    StrategyDSL(**dsl)  # the GA's DSL must be parser-valid before an engine sees it
    return dsl


# ---------------------------------------------------------------------------
# Runs: positions and protective levels from the engine caches
# ---------------------------------------------------------------------------


class _Level(NamedTuple):
    """One reduce-only STOP_MARKET (trigger) or LIMIT (price) order of the run."""

    ts_init: int
    type: str
    level: float
    status: str


class _Tap(NamedTuple):
    protective: list[_Level]
    sizing_skips: dict[str, int]


class _Run(NamedTuple):
    fills: _Fills
    tap: _Tap


def _read_tap(engine: Any) -> _Tap:
    cache = engine.kernel.cache
    levels = sorted(
        _Level(
            int(o.ts_init),
            o.order_type.name,
            float(o.trigger_price if o.order_type.name == "STOP_MARKET" else o.price),
            o.status.name,
        )
        for o in cache.orders()
        if o.is_reduce_only and o.order_type.name in ("STOP_MARKET", "LIMIT")
    )
    skips: dict[str, int] = {}
    for strategy in engine.trader.strategies():
        for reason, n in getattr(strategy, "_sizing_skips", {}).items():
            skips[reason] = skips.get(reason, 0) + n
    return _Tap(levels, skips)


@contextmanager
def _tapping() -> Iterator[list[_Tap]]:
    """Record every engine's protective levels and sizing skips next to the parity helpers' own spy."""
    taps: list[_Tap] = []
    with pytest.MonkeyPatch.context() as mp:
        orig_scr = NTScreeningRunner._extract_metrics
        orig_val = ValidationRunner._extract_results

        def scr(self: NTScreeningRunner, *args: Any, **kwargs: Any) -> Any:
            taps.append(_read_tap(kwargs.get("engine", args[2] if len(args) > 2 else None)))
            return orig_scr(self, *args, **kwargs)

        def val(self: ValidationRunner, **kwargs: Any) -> Any:
            taps.append(_read_tap(kwargs["engine"]))
            return orig_val(self, **kwargs)

        mp.setattr(NTScreeningRunner, "_extract_metrics", scr)
        mp.setattr(ValidationRunner, "_extract_results", val)
        yield taps


def _scr(
    catalog: Path, dsl: dict[str, Any], symbol: str = "BTCUSDT"
) -> tuple[_Run, BacktestMetrics]:
    with _tapping() as taps:
        fills, metrics = _screening(catalog, dsl, symbols=(symbol,))
    return _Run(fills, taps[0]), metrics


def _val(tmp_path: Path, catalog: Path, dsl: dict[str, Any], symbol: str = "BTCUSDT") -> _Run:
    with _tapping() as taps:
        fills, _ = _validation(tmp_path, catalog, dsl, symbols=(symbol,))
    return _Run(fills, taps[0])


def _rows(run: _Run) -> list[tuple[str, str, float, float | None, str | None]]:
    """(open ts, close ts or OPEN, open px, close px, closing order type) per position."""
    return [
        (
            _utc(p.ts_open),
            _utc(p.ts_close),
            round(p.px_open, 2),
            None if p.px_close is None else round(p.px_close, 2),
            p.closing,
        )
        for p in run.fills.positions
    ]


def _levels(run: _Run, kind: str, *, before_position: int | None = None) -> list[float]:
    """Levels of ``kind`` orders in submission order; optionally only the first N positions'."""
    cut = run.fills.positions[before_position].ts_open if before_position is not None else None
    return [
        lv.level
        for lv in run.tap.protective
        if lv.type == kind and (cut is None or lv.ts_init < cut)
    ]


def _assert_tier_parity(scr: _Run, val: _Run, tick: float) -> None:
    """Same positions, same entry stamps, same closer, every price within two ticks."""
    s, v = scr.fills.positions, val.fills.positions
    assert len(s) == len(v)
    assert [p.ts_open for p in s] == [p.ts_open for p in v]
    assert [p.closing for p in s] == [p.closing for p in v]
    assert [p.ts_close is None for p in s] == [p.ts_close is None for p in v]
    for a, b in zip(s, v, strict=True):
        assert a.px_open == pytest.approx(b.px_open, abs=2 * tick + 1e-6)
        if a.px_close is not None:
            assert b.px_close is not None
            assert a.px_close == pytest.approx(b.px_close, abs=2 * tick + 1e-6)


def _path_4h_bars(
    path: Callable[[int, float], float], base: float, n: int
) -> list[tuple[float, float, float]]:
    """(high, low, close) of the first ``n`` 4h bars built from ``path``'s BTC 1m klines."""
    out = []
    pad = 3.0
    for b in range(n):
        highs, lows = [], []
        for i in range(b * _H4_MIN, (b + 1) * _H4_MIN):
            o, c = path(max(i - 1, 0), base), path(i, base)
            highs.append(max(o, c) + pad)
            lows.append(min(o, c) - pad)
        out.append((max(highs), min(lows), path((b + 1) * _H4_MIN - 1, base)))
    return out


def _wilder_atr(bars: list[tuple[float, float, float]], period: int, last: int) -> float:
    """Wilder ATR after bar ``last``: seeded with the first true range, then (w*(n-1)+tr)/n."""
    atr = 0.0
    for k in range(last + 1):
        high, low, _ = bars[k]
        if k == 0:
            atr = high - low
            continue
        prev_close = bars[k - 1][2]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        atr = (atr * (period - 1) + tr) / period
    return atr


# ---------------------------------------------------------------------------
# R11: atr_trailing ratchets and both tiers agree
# ---------------------------------------------------------------------------


_TRAIL_TRIGGERS = [
    9959.8,
    10035.5,
    10119.5,
    10208.9,
    10301.8,
    10397.1,
    10493.9,
    10591.8,
    10690.4,
    10789.5,
]


@pytest.mark.slow(reason="screening and validation NT engine runs on the same data")
def test_trailing_ratchets_and_matches(toy: Path, tmp_path: Path) -> None:
    dsl = _ga_dsl("trail_m2", _block("atr_trailing", 2.0, "fixed_pct", 90.0, 3))
    assert dsl["stop_loss"] == {
        "type": "atr_trailing",
        "atr_multiplier": 2.0,
        "indicator": _ENTRY_ATR,
    }
    scr, _ = _scr(toy, dsl)
    val = _val(tmp_path, toy, dsl)

    # the trigger moves up every bar, never loosens, and is the same level in both tiers
    for run in (scr, val):
        triggers = _levels(run, "STOP_MARKET", before_position=1)
        assert triggers == _TRAIL_TRIGGERS
        assert triggers == sorted(triggers)

    assert _rows(scr) == [
        ("01-01 12:00:59.999", "01-03 11:59:59.999", 10100.0, 10789.4, "STOP_MARKET"),
        ("01-03 12:00:59.999", "01-03 23:59:59.999", 10750.1, 10562.1, "STOP_MARKET"),
        ("01-04 00:00:59.999", "01-04 11:59:59.999", 10500.0, 10318.6, "STOP_MARKET"),
        ("01-04 12:00:59.999", "01-04 23:59:59.999", 10250.0, 10070.4, "STOP_MARKET"),
    ]
    assert _rows(val) == [
        ("01-01 12:00:59.999", "01-03 09:58:59.999", 10100.0, 10789.4, "STOP_MARKET"),
        ("01-03 12:00:59.999", "01-03 20:53:59.999", 10750.0, 10562.0, "STOP_MARKET"),
        ("01-04 00:00:59.999", "01-04 08:34:59.999", 10500.0, 10318.5, "STOP_MARKET"),
        ("01-04 12:00:59.999", "01-04 20:29:59.999", 10250.0, 10070.4, "STOP_MARKET"),
    ]
    _assert_tier_parity(scr, val, _BTC_TICK)


# ---------------------------------------------------------------------------
# R10: the ATR at a fill is the signal bar's, never the fill bar's
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason="screening and validation NT engine runs on the same data")
def test_atr_stop_uses_signal_bar_atr(toy: Path, tmp_path: Path) -> None:
    dsl = _ga_dsl("atr_stop_m2", _block("atr_fixed", 2.0, "fixed_pct", 90.0, 3))
    scr, _ = _scr(toy, dsl)
    val = _val(tmp_path, toy, dsl)

    # independent ATR(3): the 4h bars 00-04, 04-08, 08-12 (closed when the signal fires)
    bars = _path_4h_bars(_path, 10_000.0, 4)
    through_signal_bar = _wilder_atr(bars, 3, 2)
    through_fill_bar = _wilder_atr(bars, 3, 3)  # what a look-ahead into 12:00-16:00 would use
    assert through_signal_bar == pytest.approx(70.1, abs=0.05)
    assert abs(through_fill_bar - through_signal_bar) > 5.0

    for run in (scr, val):
        (position,) = run.fills.positions
        assert (_utc(position.ts_open), position.px_open, position.ts_close) == (
            "01-01 12:00:59.999",
            10100.0,
            None,
        )
        assert _levels(run, "STOP_MARKET") == [9959.8]  # 10100.0 - 2 * 70.1
        assert _levels(run, "STOP_MARKET")[0] == pytest.approx(
            position.px_open - 2 * through_signal_bar, abs=_BTC_TICK
        )
        assert _levels(run, "LIMIT") == [19190.0]  # TP 90 %
    _assert_tier_parity(scr, val, _BTC_TICK)


# ---------------------------------------------------------------------------
# R12 / R13: risk_reward and ATR targets
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason="screening and validation NT engine runs on the same data")
def test_risk_reward_target(toy: Path, tmp_path: Path) -> None:
    dsl = _ga_dsl("rr_1_5", _block("atr_fixed", 2.0, "risk_reward", 1.5, 3))
    assert dsl["take_profit"] == {"type": "risk_reward", "risk_reward_ratio": 1.5}
    scr, _ = _scr(toy, dsl)
    val = _val(tmp_path, toy, dsl)

    for run in (scr, val):
        # trade 1: stop 10100 - 140.2, target 10100 + 1.5 * 140.2 (an R:R target is relative to the stop)
        assert _levels(run, "LIMIT")[0] == 10310.3
        assert _levels(run, "STOP_MARKET")[0] == 9959.8
    assert _rows(scr) == [
        ("01-01 12:00:59.999", "01-01 23:59:59.999", 10100.0, 10310.2, "LIMIT"),
        ("01-02 00:00:59.999", "01-02 11:59:59.999", 10400.1, 10686.1, "LIMIT"),
        ("01-02 12:00:59.999", "01-04 03:59:59.999", 10700.0, 10494.3, "STOP_MARKET"),
        ("01-04 04:00:59.999", "01-04 15:59:59.999", 10416.7, 10236.1, "STOP_MARKET"),
        ("01-04 16:00:59.999", "OPEN", 10166.8, None, None),
    ]
    assert _rows(val) == [
        ("01-01 12:00:59.999", "01-01 20:18:59.999", 10100.0, 10310.3, "LIMIT"),
        ("01-02 00:00:59.999", "01-02 11:20:59.999", 10400.0, 10686.1, "LIMIT"),
        ("01-02 12:00:59.999", "01-04 00:08:59.999", 10700.0, 10494.2, "STOP_MARKET"),
        ("01-04 04:00:59.999", "01-04 12:31:59.999", 10416.7, 10236.1, "STOP_MARKET"),
        ("01-04 16:00:59.999", "OPEN", 10166.7, None, None),
    ]
    _assert_tier_parity(scr, val, _BTC_TICK)


@pytest.mark.slow(reason="screening and validation NT engine runs on the same data")
def test_atr_target(toy: Path, tmp_path: Path) -> None:
    dsl = _ga_dsl("atr_tp_m3", _block("fixed_pct", 10.0, "atr_fixed", 3.0, 3))
    scr, _ = _scr(toy, dsl)
    val = _val(tmp_path, toy, dsl)

    for run in (scr, val):
        assert _levels(run, "LIMIT")[0] == 10310.3  # 10100 + 3 * 70.1
        assert _levels(run, "STOP_MARKET")[0] == 9090.0  # 10 % below 10100
    assert _rows(scr) == [
        ("01-01 12:00:59.999", "01-01 23:59:59.999", 10100.0, 10310.2, "LIMIT"),
        ("01-02 00:00:59.999", "01-02 11:59:59.999", 10400.1, 10686.1, "LIMIT"),
        ("01-02 12:00:59.999", "OPEN", 10700.0, None, None),
    ]
    assert _rows(val) == [
        ("01-01 12:00:59.999", "01-01 20:18:59.999", 10100.0, 10310.3, "LIMIT"),
        ("01-02 00:00:59.999", "01-02 11:20:59.999", 10400.0, 10686.1, "LIMIT"),
        ("01-02 12:00:59.999", "OPEN", 10700.0, None, None),
    ]
    _assert_tier_parity(scr, val, _BTC_TICK)


# ---------------------------------------------------------------------------
# R14 / R14b: safety stop + none target + time exit
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason="screening and validation NT engine runs on the same data")
def test_safety_none_time_exit(toy: Path, tmp_path: Path) -> None:
    dsl = _ga_dsl(
        "safety_none_time",
        _block("safety", None, "none", None, None),
        max_bars=3,
    )
    assert dsl["max_bars_in_trade"] == 3
    assert "exit_atr" not in dsl["indicators"]
    scr, _ = _scr(toy, dsl, "ETHUSDT")
    val = _val(tmp_path, toy, dsl, "ETHUSDT")

    for run in (scr, val):
        # SL 50 % below 20200 and TP 100 % above (a 100 % target never fills a long)
        assert _levels(run, "STOP_MARKET")[0] == 10100.0
        assert _levels(run, "LIMIT")[0] == 40400.0
        assert {lv.status for lv in run.tap.protective} <= {"CANCELED", "ACCEPTED"}
        closed = [p for p in run.fills.positions if p.ts_close is not None]
        assert len(run.fills.positions) == 6
        # every close is the time exit, 12 h (3 x 4h) after its entry
        assert {p.closing for p in closed} == {"MARKET"}
        assert {p.ts_close - p.ts_open for p in closed if p.ts_close} == {12 * 3_600 * 10**9}
    assert _rows(scr) == [
        ("01-01 12:00:59.999", "01-02 00:00:59.999", 20200.0, 20799.99, "MARKET"),
        ("01-02 04:00:59.999", "01-02 16:00:59.999", 21000.01, 21599.99, "MARKET"),
        ("01-02 20:00:59.999", "01-03 08:00:59.999", 21800.0, 21666.67, "MARKET"),
        ("01-03 12:00:59.999", "01-04 00:00:59.999", 21500.0, 20999.99, "MARKET"),
        ("01-04 04:00:59.999", "01-04 16:00:59.999", 20833.34, 20333.32, "MARKET"),
        ("01-04 20:00:59.999", "OPEN", 20166.68, None, None),
    ]
    assert _rows(val) == [
        ("01-01 12:00:59.999", "01-02 00:00:59.999", 20200.0, 20800.0, "MARKET"),
        ("01-02 04:00:59.999", "01-02 16:00:59.999", 21000.0, 21600.0, "MARKET"),
        ("01-02 20:00:59.999", "01-03 08:00:59.999", 21800.0, 21666.67, "MARKET"),
        ("01-03 12:00:59.999", "01-04 00:00:59.999", 21500.0, 21000.0, "MARKET"),
        ("01-04 04:00:59.999", "01-04 16:00:59.999", 20833.33, 20333.33, "MARKET"),
        ("01-04 20:00:59.999", "OPEN", 20166.67, None, None),
    ]
    _assert_tier_parity(scr, val, _ETH_TICK)


@pytest.mark.slow(reason="screening and validation NT engine runs on the same data")
def test_none_target_on_a_short_places_no_take_profit(toy: Path, tmp_path: Path) -> None:
    """A short's 100 % target is price 0: no TP order exists; the 50 % stop sits above entry."""
    dsl = _ga_dsl(
        "safety_none_short",
        _block("safety", None, "none", None, None),
        max_bars=3,
        direction=Direction.SHORT,
    )
    scr, _ = _scr(toy, dsl, "ETHUSDT")
    val = _val(tmp_path, toy, dsl, "ETHUSDT")

    for run in (scr, val):
        assert [lv for lv in run.tap.protective if lv.type == "LIMIT"] == []
        stops = _levels(run, "STOP_MARKET")
        assert len(stops) == len(run.fills.positions) == 6
        for stop, position in zip(stops, run.fills.positions, strict=True):
            assert stop == pytest.approx(position.px_open * 1.5, abs=2 * _ETH_TICK)
        closed = [p for p in run.fills.positions if p.ts_close is not None]
        assert {p.closing for p in closed} == {"MARKET"}
    _assert_tier_parity(scr, val, _ETH_TICK)


# ---------------------------------------------------------------------------
# R15: a safety stop at default sizing never reaches BTC's minimum notional
# ---------------------------------------------------------------------------


def _zero_trade_chromosome(rules: ExitRules) -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_gene(14.0)],
        exit_genes=[_gene(10.0)],
        stop_loss_pct=2.0,
        take_profit_pct=3.0,
        direction=Direction.LONG,
        exit_rules=rules,
    )


def _fitness_of(chrom: StrategyChromosome, metrics: BacktestMetrics) -> Any:
    bt: dict[str, float | int] = {
        "sharpe_ratio": metrics.sharpe_ratio,
        "max_drawdown": metrics.max_drawdown,
        "profit_factor": metrics.profit_factor,
        "total_trades": metrics.total_trades,
        "total_return": metrics.total_return,
    }
    return _evaluate_single(chrom, lambda _c: bt, None, 10, "4h")


@pytest.mark.slow(reason="screening and validation NT engine runs on the same data")
def test_safety_on_btc_default_sizing_skips(toy: Path, tmp_path: Path) -> None:
    """2 % risk / 50 % stop = 4 % of 1000 = 40 USDT < BTC's 50 USDT minimum: every entry skips."""
    rules = ExitRules(long=_block("safety", None, "none", None, None), short=None, max_bars=None)
    dsl = _ga_dsl("safety_btc", rules.long)
    scr, metrics = _scr(toy, dsl)
    val = _val(tmp_path, toy, dsl)

    for run in (scr, val):
        assert run.fills.positions == []
        assert run.fills.orders == []
        assert run.tap.sizing_skips == {"min_notional": 22}
    assert metrics.total_trades == 0
    fitness = _fitness_of(_zero_trade_chromosome(rules), metrics)
    assert fitness.adjusted_score == 0.0
    assert fitness.error is None


# ---------------------------------------------------------------------------
# R17: known first-minute divergence (pinned so a fix moves it on purpose)
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason="screening and validation NT engine runs on the same data")
def test_first_minute_crash_divergence_is_pinned(crash: Path, tmp_path: Path) -> None:
    """The 4h bar's first minute crashes through the resting stop AND the re-armed one.

    Both tiers exit on the OLD stop (10993.6) at 00:00:59.999 and reject the re-armed
    11093.6 (reduce-only, flat). They differ on the FILL PRICE: screening's fill tick gaps
    through the stop (tick price 10899.5), validation's 1m path fills at the trigger minus
    a tick (10993.5). Pre-existing for every stop type; real-data hit rate 0 of 73.
    """
    dsl = _ga_dsl("crash_trail_m1", _block("atr_trailing", 1.0, "fixed_pct", 90.0, 3))
    scr, _ = _scr(crash, dsl)
    val = _val(tmp_path, crash, dsl)

    for run in (scr, val):
        second_open = run.fills.positions[1].ts_open
        first_stops = [
            lv for lv in run.tap.protective if lv.type == "STOP_MARKET" and lv.ts_init < second_open
        ]
        # the old stop fills, the ratcheted one (armed on the last bar before the crash) is rejected
        assert [(lv.level, lv.status) for lv in first_stops[-2:]] == [
            (10993.6, "FILLED"),
            (11093.6, "REJECTED"),
        ]
        first = run.fills.positions[0]
        assert (_utc(first.ts_open), _utc(first.ts_close), first.px_open, first.closing) == (
            "01-01 12:00:59.999",
            "01-03 00:00:59.999",
            10300.0,
            "STOP_MARKET",
        )
    assert scr.fills.positions[0].px_close == 10899.5
    assert val.fills.positions[0].px_close == 10993.5
    assert scr.fills.positions[0].px_close != val.fills.positions[0].px_close  # the divergence
    assert _rows(scr)[1] == ("01-03 04:00:59.999", "OPEN", 10902.1, None, None)
    assert _rows(val)[1] == ("01-03 04:00:59.999", "OPEN", 10902.0, None, None)


# ---------------------------------------------------------------------------
# R18: an ATR that never warms up trades nothing
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason="screening and validation NT engine runs on the same data")
def test_unready_atr_scores_zero_trades(toy: Path, tmp_path: Path) -> None:
    """ATR(30) needs 30 4h bars; the window has 24: no entry, no exception, fitness 0."""
    rules = ExitRules(
        long=_block("atr_trailing", 2.0, "fixed_pct", 4.0, 30), short=None, max_bars=1
    )
    dsl = _ga_dsl("atr30_never_ready", rules.long, max_bars=1)
    assert dsl["indicators"]["exit_atr"] == {"type": "ATR", "period": 30}
    scr, metrics = _scr(toy, dsl)
    val = _val(tmp_path, toy, dsl)

    for run in (scr, val):
        assert run.fills.positions == []
        assert run.fills.orders == []
        assert run.tap.sizing_skips == {}  # not a sizing skip: the entry was never evaluated
    assert metrics.total_trades == 0
    fitness = _fitness_of(_zero_trade_chromosome(rules), metrics)
    assert fitness.adjusted_score == 0.0
    assert fitness.error is None

    # control: the same genome with a warm ATR(3) trades, so the zero above is the ATR gate
    warm = _ga_dsl("atr3_ready", _block("atr_trailing", 2.0, "fixed_pct", 4.0, 3), max_bars=1)
    warm_run, warm_metrics = _scr(toy, warm)
    assert len(warm_run.fills.positions) > 0
    assert warm_metrics.total_trades > 0


# ---------------------------------------------------------------------------
# R18b: a stop that would sit at or below zero is not placed
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason="screening and validation NT engine runs on the same data")
def test_atr_stop_below_zero_places_no_stop_order(wild: Path, tmp_path: Path) -> None:
    """m x ATR >= entry: the long's stop price is <= 0, so no STOP_MARKET exists; no exception."""
    dsl = _ga_dsl("atr_stop_below_zero", _block("atr_fixed", 6.0, "fixed_pct", 90.0, 3))
    scr, _ = _scr(wild, dsl)
    val = _val(tmp_path, wild, dsl)

    for run in (scr, val):
        assert run.fills.positions
        assert [lv for lv in run.tap.protective if lv.type == "STOP_MARKET"] == []
        assert [lv.type for lv in run.tap.protective] == ["LIMIT"] * len(run.tap.protective)
        (position,) = run.fills.positions
        (entry,) = [o for o in run.fills.orders if not o.reduce_only and o.filled > 0]
        # entry sized at the cap: 1000 USDT balance x max_position_pct 0.5 = 500 USDT, rounded
        # down to the 0.01 BTC step (0.11 at 4500); a stop-based size (2 % risk) would be 0.22
        assert (position.px_open, entry.filled) == (4500.0, 0.11)
        assert entry.filled * entry.px <= 500.0 < (entry.filled + 0.01) * entry.px


# ---------------------------------------------------------------------------
# R19: direction=both, each side exits by its own rule (screening only)
# ---------------------------------------------------------------------------


@pytest.mark.slow(reason="screening NT engine run on the toy data")
def test_both_sides_exit_by_own_rule(swing: Path) -> None:
    long_block = _block("atr_trailing", 2.0, "risk_reward", 2.0, 14)
    short_block = _block("fixed_pct", 3.0, "atr_fixed", 2.0, 7)
    dsl = _ga_dsl(
        "both_own_rules",
        long_block,
        short=short_block,
        direction=Direction.BOTH,
        entry={"long": ["close > ema"], "short": ["close < ema"]},
    )
    assert dsl["indicators"]["exit_atr"] == {"type": "ATR", "period": 14}
    assert dsl["indicators"]["exit_atr_short"] == {"type": "ATR", "period": 7}
    scr, _ = _scr(swing, dsl)

    assert _rows(scr) == [
        ("01-03 08:00:59.999", "01-03 15:59:59.999", 10346.4, 9782.5, "STOP_MARKET"),
        ("01-03 16:00:59.999", "01-03 23:59:59.999", 9653.5, 9943.2, "STOP_MARKET"),
        ("01-04 00:00:59.999", "01-04 15:59:59.999", 10000.0, 9808.0, "STOP_MARKET"),
        ("01-04 16:00:59.999", "01-04 23:59:59.999", 9653.6, 9943.3, "STOP_MARKET"),
    ]
    entries = [
        o.side
        for o in sorted(scr.fills.orders, key=lambda o: o.ts_last)
        if not o.reduce_only and o.filled > 0
    ]
    assert entries == ["BUY", "SELL", "BUY", "SELL"]

    opens = [p.ts_open for p in scr.fills.positions] + [10**30]
    by_position = [
        [lv for lv in scr.tap.protective if opens[k] <= lv.ts_init < opens[k + 1]] for k in range(4)
    ]
    bars = _path_4h_bars(_swing_path, 10_000.0, 16)
    atr14 = {13: _wilder_atr(bars, 14, 13)}  # signal bar of the 01-03 08:00 long
    atr7 = {15: _wilder_atr(bars, 7, 15)}  # signal bar of the 01-03 16:00 short
    # had the short side read the long block's ATR(14), its target would sit >= 1 price unit away
    assert abs(_wilder_atr(bars, 14, 15) - atr7[15]) * 2 > 1.0

    # long 1 (08:00): stop = entry - 2 x ATR(14), target = entry + 2 x that stop distance
    long1 = {lv.type: lv for lv in by_position[0]}
    entry1 = scr.fills.positions[0].px_open
    assert long1["STOP_MARKET"].level == 9782.6
    assert long1["STOP_MARKET"].level == pytest.approx(entry1 - 2 * atr14[13], abs=0.15)
    assert long1["LIMIT"].level == 11474.0
    assert long1["LIMIT"].level == pytest.approx(
        entry1 + 2 * (entry1 - long1["STOP_MARKET"].level), abs=_BTC_TICK + 1e-6
    )

    # long 3 (01-04 00:00): the trigger is re-armed on later bars and never loosens
    trail = [lv.level for lv in by_position[2] if lv.type == "STOP_MARKET"]
    assert trail == [9438.9, 9774.0, 9808.0]

    # shorts: a static fixed 3 % stop above entry; target = entry - 2 x the SHORT block's ATR(7)
    for k in (1, 3):
        stops = [lv.level for lv in by_position[k] if lv.type == "STOP_MARKET"]
        (tp,) = [lv.level for lv in by_position[k] if lv.type == "LIMIT"]
        entry = scr.fills.positions[k].px_open
        assert stops == [pytest.approx(entry * 1.03, abs=0.1)]
        assert tp < entry
    short1_tp = next(lv.level for lv in by_position[1] if lv.type == "LIMIT")
    assert short1_tp == 9097.9
    assert short1_tp == pytest.approx(scr.fills.positions[1].px_open - 2 * atr7[15], abs=0.15)
