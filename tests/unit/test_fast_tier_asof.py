"""TAKER_IMBALANCE on the fast tier: as-of context columns equal the generated feed.

The fast tier replays ``compute_fn`` indicators from columns computed once over the whole
run frame. That is exact only for specs flagged ``asof_context`` (the value at row ``k``
depends on that row's ``bar_close_ns``, ``symbol``, params and archive contents only).
TAKER_IMBALANCE is the first such spec.

Every comparison here is against the REAL generated ``_feed_pta_buffer`` +
``_update_pta_indicators`` (the NT source of truth), bitwise and NaN included, so a
wrongly flagged spec, a NaN that is held, or a wrong first written row fails. The
end-to-end test additionally proves the genome never silently falls back to NT.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl import aux_data
from vibe_quant.dsl.aux_data import AuxDataUnavailableError
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.indicators import (
    IndicatorSpec,
    indicator_registry,
    pta_buffer_cap,
    pta_lookback,
)
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.pta_buffer import PtaBuffer
from vibe_quant.dsl.pta_replay import PtaReplayUnsupported, asof_context_columns, bar_close_ns
from vibe_quant.fast_tier import fallbacks
from vibe_quant.fast_tier.backtest_fn import FastBacktestFn
from vibe_quant.fast_tier.signals import Signals, compute_signals, load_bars

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "data" / "catalog"
ARCHIVE = REPO / "data" / "archive" / "raw_data.db"
INSTRUMENT_ID = "ZECUSDT-PERP.BINANCE"
BAR_TYPE = f"{INSTRUMENT_ID}-1-MINUTE-LAST-EXTERNAL"
MIN = 60_000_000_000

needs_catalog = pytest.mark.skipif(
    not (CATALOG / "data" / "bar" / BAR_TYPE).is_dir(), reason="needs the ZECUSDT 1m catalog"
)
needs_archive = pytest.mark.skipif(not ARCHIVE.is_file(), reason="needs the taker kline archive")


@pytest.fixture
def _aux_data() -> Any:
    """Point aux lookups at the repo catalog/archive and restore the globals after."""
    aux_data.configure(ARCHIVE, CATALOG)
    try:
        yield
    finally:
        aux_data.configure(None, None)


# ---------------------------------------------------------------------------
# Harness: a compiled strategy wired onto the generated context path.
# ---------------------------------------------------------------------------


def _compiled(
    indicators: dict[str, dict[str, Any]], entry: str, exit_: str
) -> tuple[type, Any]:
    """The generated 1m strategy class and its config instance for these conditions."""
    dsl = validate_strategy_dict(
        {
            "name": "taker_asof",
            "timeframe": "1m",
            "indicators": indicators,
            "entry_conditions": {"long": [entry]},
            "exit_conditions": {"long": [exit_]},
            "stop_loss": {"type": "fixed_pct", "percent": 0.8},
            "take_profit": {"type": "fixed_pct", "percent": 1.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    cls = next(v for k, v in vars(module).items() if k.endswith("Strategy") and k != "Strategy")
    config_cls = getattr(module, cls.__name__.removesuffix("Strategy") + "Config")
    return cls, config_cls(instrument_id=INSTRUMENT_ID)


def _strategy(
    indicators: dict[str, dict[str, Any]],
    entry: str,
    exit_: str,
    cap: int,
) -> Any:
    """A generated strategy whose ``_pta_*`` state mirrors what its ``__init__`` builds."""
    cls, config = _compiled(indicators, entry, exit_)
    harness = type("Wired", (cls,), {"config": config})
    inst: Any = harness.__new__(harness)
    params = {
        name: {k: v for k, v in cfg.items() if k != "type"} for name, cfg in indicators.items()
    }
    lookbacks = {name: pta_lookback(cfg["type"], params[name]) for name, cfg in indicators.items()}
    inst._pta_values = {}
    inst._prev_values = {}
    inst._last_close = 0.0
    inst._pta_params = params
    inst._pta_lookback = lookbacks
    inst._pta_buffer_cap = {"1m": cap}
    inst._pta_bufs = {"1m": PtaBuffer(cap, with_close_ns=True)}
    inst._pta_step_ns = {"1m": MIN}
    return inst


def _taker_strategy(period: int) -> Any:
    return _strategy(
        {"ti": {"type": "TAKER_IMBALANCE", "period": period}},
        "ti > 0.05",
        "ti < -0.05",
        cap=400,
    )


def _zec_bars(start: str, end: str, limit: int | None = None) -> list[Bar]:
    bars = load_bars(str(CATALOG), BAR_TYPE, start, end)
    return bars if limit is None else bars[:limit]


def _frame(bars: list[Bar]) -> pd.DataFrame:
    return pd.DataFrame(
        {c: [float(getattr(b, c)) for b in bars] for c in ("open", "high", "low", "close", "volume")}
    )


def _ts_init(bars: list[Bar]) -> np.ndarray:
    return np.fromiter((int(b.ts_init) for b in bars), dtype=np.int64, count=len(bars))


def _bits(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.int64)


def _drive_generated(strategy: Any, bars: list[Bar]) -> list[dict[str, float]]:
    """``_pta_values`` snapshot after each bar's REAL generated ``_feed_pta_buffer``."""
    out: list[dict[str, float]] = []
    for bar in bars:
        strategy._feed_pta_buffer("1m", bar)
        out.append(dict(strategy._pta_values))
    return out


def _asof_column(
    bars: list[Bar], period: int, frame: pd.DataFrame | None = None
) -> tuple[np.ndarray, int]:
    frame = _frame(bars) if frame is None else frame
    [(key, column, first)] = asof_context_columns(
        {"ti": {"period": period}},
        {"ti": ("TAKER_IMBALANCE", "1m")},
        "1m",
        frame,
        bar_close_ns(_ts_init(bars), MIN),
        INSTRUMENT_ID,
    )
    assert key == "ti"
    assert frame.attrs == {}, "the shared frame must not be mutated"
    return column, first


def _feed_reference(strategy: Any, bars: list[Bar]) -> Signals:
    """``compute_signals``' loop with the values the generated feed writes (NT's own)."""
    n = len(bars)
    ready = np.zeros(n, dtype=bool)
    le = np.zeros(n, dtype=bool)
    se = np.zeros(n, dtype=bool)
    lx = np.zeros(n, dtype=bool)
    sx = np.zeros(n, dtype=bool)
    checks = [
        getattr(strategy, name, None)
        for name in ("_check_long_entry", "_check_short_entry", "_check_long_exit", "_check_short_exit")
    ]
    for i, bar in enumerate(bars):
        strategy._feed_pta_buffer("1m", bar)
        strategy._last_close = float(bar.close)
        if not strategy._indicators_ready():
            continue
        ready[i] = True
        for out, check in zip((le, se, lx, sx), checks, strict=True):
            if check is not None:
                out[i] = check(bar)
        strategy._update_prev_values(bar)
    return Signals(ready, le, se, lx, sx)


# ---------------------------------------------------------------------------
# R1 / R13: the as-of column == the generated per-bar feed, bit for bit.
# ---------------------------------------------------------------------------


@needs_catalog
@needs_archive
def test_asof_column_equals_generated_feed_bar_for_bar(_aux_data: None) -> None:
    """ZEC 1m 2026-03-02..05 (4320 bars, cap 400 so trims run): same value on every bar."""
    bars = _zec_bars("2026-03-02", "2026-03-05")
    assert len(bars) == 4320
    strategy = _taker_strategy(60)
    assert strategy._pta_buffer_cap["1m"] == 400
    stream = _drive_generated(strategy, bars)

    frame = _frame(bars)
    column, first = _asof_column(bars, 60, frame)

    assert first == 0
    assert all("ti" in values for values in stream), "the generated feed skipped a bar"
    fed = np.array([values["ti"] for values in stream], dtype=np.float64)
    assert np.array_equal(_bits(fed), _bits(column)), "as-of column diverged from NT's feed"
    assert int(np.isnan(column).sum()) == 0


# ---------------------------------------------------------------------------
# R2: the as-of close time is the generated ``((ts_init + step//2)//step)*step``.
# ---------------------------------------------------------------------------


@needs_catalog
@needs_archive
def test_bar_close_ns_matches_generated_feed(_aux_data: None) -> None:
    """The generated buffer's ``bar_close_ns`` is ``bar_close_ns(ts_init, 1m step)``, per bar."""
    bars = _zec_bars("2026-03-02", "2026-03-03", limit=200)
    strategy = _taker_strategy(60)
    expected = bar_close_ns(_ts_init(bars), MIN)
    assert expected.dtype == np.int64
    for i, bar in enumerate(bars):
        strategy._feed_pta_buffer("1m", bar)
        last = strategy._pta_bufs["1m"].frame().attrs["bar_close_ns"][-1]
        assert int(last) == int(expected[i])
    # MEASURED: a 1m bar's ts_init is its open + 59.999s, so the close is the next minute
    assert int(expected[0]) - int(bars[0].ts_init) == 1_000_000
    assert int(expected[0]) % MIN == 0


def test_bar_close_ns_rounds_15m_and_1h_closes() -> None:
    """15m: 00:14:59.999 -> 00:15:00; 1h: 00:59:59.999 -> 01:00:00 (the generated rule)."""
    quarter = bar_close_ns(np.array([899_999_000_000], dtype=np.int64), 15 * MIN)
    assert [int(quarter[0])] == [900_000_000_000]
    hourly = bar_close_ns(np.array([3_599_999_000_000], dtype=np.int64), 60 * MIN)
    assert [int(hourly[0])] == [3_600_000_000_000]


def test_bar_close_ns_floors_at_and_rounds_up_half_a_step() -> None:
    """The rounding edges: exactly on a boundary stays, half a step over rolls up."""
    cases = [120 * MIN, 120 * MIN + 1, 120 * MIN + MIN // 2 - 1, 120 * MIN + MIN // 2]
    got = [int(v) for v in bar_close_ns(np.array(cases, dtype=np.int64), MIN)]
    assert got == [120 * MIN, 120 * MIN, 120 * MIN, 121 * MIN]


# ---------------------------------------------------------------------------
# R4 / R5: NaN is a value (never held) and readiness starts at ``lookback - 1``.
# ---------------------------------------------------------------------------


@needs_catalog
@needs_archive
def test_nan_bar_is_not_held_and_blocks_entry(_aux_data: None) -> None:
    """ZEC 2025-08-29 has 15 zero-volume klines: exactly 15 NaN bars, and no entry on them."""
    bars = _zec_bars("2025-08-29", "2025-08-30")
    assert len(bars) == 1440
    reference = _feed_reference(_taker_strategy(5), bars)
    column, first = _asof_column(bars, 5)

    nan_rows = np.flatnonzero(np.isnan(column))
    assert nan_rows.tolist() == list(range(382, 397))
    assert first == 0
    assert reference.long_entry[nan_rows].sum() == 0

    fast = compute_signals(_taker_strategy(5), [], bars, {"ti": "TAKER_IMBALANCE"})
    assert np.array_equal(fast.ready, reference.ready)
    assert np.array_equal(fast.long_entry, reference.long_entry)
    assert np.array_equal(fast.short_entry, reference.short_entry)
    assert np.array_equal(fast.long_exit, reference.long_exit)
    assert np.array_equal(fast.short_exit, reference.short_exit)
    assert fast.long_entry.any()
    # TAKER's lookback is 1: the first bar is ready on both paths
    assert bool(reference.ready[0]) is True
    assert bool(fast.ready[0]) is True


# ---------------------------------------------------------------------------
# R8 / R12: unflagged context specs still fall back; missing aux data fails the run.
# ---------------------------------------------------------------------------


def test_unflagged_context_indicator_still_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A context spec WITHOUT the as-of flag: the fast tier refuses it and records one fallback.

    Every registered context spec is flagged now, so the fallback path is held open by an
    unflagged copy of BTC_TREND registered for this test only: the refusal must keep
    coming from the missing flag, not from the indicator's name.
    """
    import dataclasses

    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    real = indicator_registry.get("BTC_TREND")
    assert real is not None
    monkeypatch.setitem(
        indicator_registry._indicators, "BTC_TREND", dataclasses.replace(real, asof_context=False)
    )

    genome = StrategyChromosome(
        entry_genes=[StrategyGene("BTC_TREND", {"period": 200.0}, ConditionType.GT, 0.0)],
        exit_genes=[StrategyGene("WILLR", {"period": 14.0}, ConditionType.GT, -20.0)],
        stop_loss_pct=0.8,
        take_profit_pct=1.0,
        direction=Direction.LONG,
    )
    fallbacks.clear()
    fn = FastBacktestFn(["BTCUSDT"], "1m", "2026-03-06", "2026-03-07", nt_fallback=True)
    fn._run_single(genome, "2026-03-06", "2026-03-07")

    assert fallbacks.summary() == {
        "total": 1,
        "by_reason": {"compute_fn indicator btc_trend_entry_N cannot be replayed": 1},
    }


def test_missing_archive_fails_run_not_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing kline archive raises before any engine, and records no fallback."""
    import vibe_quant.data.archive as archive_mod
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    monkeypatch.setattr(archive_mod, "DEFAULT_ARCHIVE_PATH", tmp_path / "missing.db")
    genome = StrategyChromosome(
        entry_genes=[StrategyGene("TAKER_IMBALANCE", {"period": 60.0}, ConditionType.GT, 0.05)],
        exit_genes=[StrategyGene("WILLR", {"period": 14.0}, ConditionType.GT, -20.0)],
        stop_loss_pct=0.8,
        take_profit_pct=1.0,
        direction=Direction.LONG,
    )
    fallbacks.clear()
    fn = FastBacktestFn(["ZECUSDT"], "1m", "2026-03-02", "2026-03-04", nt_fallback=True)
    with pytest.raises(AuxDataUnavailableError, match="kline archive"):
        fn._run_single(genome, "2026-03-02", "2026-03-04")
    assert fallbacks.summary() == {"total": 0, "by_reason": {}}


# ---------------------------------------------------------------------------
# R16: every asof_context spec in the registry, default params, long run.
# ---------------------------------------------------------------------------


def _asof_specs() -> list[str]:
    return [
        name
        for name in sorted(indicator_registry.list_indicators())
        if (spec := indicator_registry.get(name)) is not None and spec.asof_context
    ]


_ASOF = _asof_specs()


def test_registry_has_an_asof_context_spec() -> None:
    """Guard: the registry-wide parametrisation must not silently shrink to nothing."""
    assert _ASOF == ["BTC_ROC", "BTC_TREND", "FUNDING", "FUNDING_Z", "OWN_TREND", "TAKER_IMBALANCE"]


@needs_catalog
@needs_archive
@pytest.mark.parametrize("name", _ASOF)
def test_every_asof_spec_equals_generated_stream(name: str, _aux_data: None) -> None:
    """Registry-wide: an asof spec's whole-frame column == its generated per-bar stream."""
    spec = indicator_registry.get(name)
    assert spec is not None and spec.compute_fn is not None
    params = dict(spec.default_params)
    lookback = pta_lookback(spec, params)
    cap = pta_buffer_cap([lookback], full_history=spec.requires_full_history)
    bars = _zec_bars("2026-03-02", "2026-03-03")
    assert len(bars) >= 1.25 * cap + 1, "the run must include a buffer trim"

    strategy = _strategy({"ind": {"type": name, **params}}, "ind > 0", "ind < 0", cap=cap)
    stream = _drive_generated(strategy, bars)
    [(key, column, first)] = asof_context_columns(
        {"ind": params},
        {"ind": (name, "1m")},
        "1m",
        _frame(bars),
        bar_close_ns(_ts_init(bars), MIN),
        INSTRUMENT_ID,
    )

    assert key == "ind"
    assert first == lookback - 1
    assert all("ind" in values for values in stream), "the generated feed skipped a bar"
    fed = np.array([values["ind"] for values in stream], dtype=np.float64)
    assert np.array_equal(_bits(fed), _bits(column)), f"{name} diverged from NT's feed"
    assert int(np.isnan(column).sum()) == 0


# ---------------------------------------------------------------------------
# R14: end to end against the NT engine, zero fallbacks.
# ---------------------------------------------------------------------------


def _taker_willr() -> Any:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    return StrategyChromosome(
        entry_genes=[StrategyGene("TAKER_IMBALANCE", {"period": 60.0}, ConditionType.GT, 0.05)],
        exit_genes=[StrategyGene("WILLR", {"period": 14.0}, ConditionType.GT, -20.0)],
        stop_loss_pct=0.8,
        take_profit_pct=1.0,
        direction=Direction.LONG,
    )


def _taker_roc_crosses() -> Any:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    return StrategyChromosome(
        entry_genes=[
            StrategyGene("TAKER_IMBALANCE", {"period": 60.0}, ConditionType.CROSSES_ABOVE, 0.02),
            StrategyGene("ROC", {"period": 20.0}, ConditionType.CROSSES_ABOVE, 0.0),
        ],
        exit_genes=[
            StrategyGene("TAKER_IMBALANCE", {"period": 60.0}, ConditionType.CROSSES_BELOW, -0.02)
        ],
        stop_loss_pct=0.8,
        take_profit_pct=1.0,
        direction=Direction.LONG,
    )


METRICS = ("total_trades", "total_return", "sharpe_ratio", "profit_factor", "max_drawdown")


def _fills(positions: list[Any]) -> list[tuple[Any, ...]]:
    out = []
    for pos in sorted(positions, key=lambda p: int(p.ts_opened)):
        for f in pos.events:
            out.append(
                (
                    int(f.ts_event),
                    f.order_side.name,
                    f.order_type.name,
                    str(f.last_qty),
                    str(f.last_px),
                    str(f.commission),
                )
            )
    return out


@needs_catalog
@needs_archive
@pytest.mark.usefixtures("_aux_data")
@pytest.mark.parametrize(
    ("genome", "start", "end"),
    [
        (_taker_willr(), "2026-03-06", "2026-03-07"),
        (_taker_roc_crosses(), "2026-03-02", "2026-03-05"),
    ],
    ids=["A_long_willr_1d", "B_both_cross_roc_3d"],
)
def test_taker_genome_matches_nt_with_no_fallback(
    genome: Any, start: str, end: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fills, every result value, MTM bytes equal to NT's; no NT fallback recorded."""
    from vibe_quant.screening import nt_runner
    from vibe_quant.validation.extraction import all_positions

    captured: dict[str, list[Any]] = {}
    original = nt_runner.NTScreeningRunner._extract_metrics

    def grab(self: Any, params: Any, bt_result: Any, engine: Any, *a: Any, **k: Any) -> Any:
        captured["positions"] = all_positions(engine)
        return original(self, params, bt_result, engine, *a, **k)

    monkeypatch.setattr(nt_runner.NTScreeningRunner, "_extract_metrics", grab)

    fallbacks.clear()
    fast_fn = FastBacktestFn(["ZECUSDT"], "1m", start, end, collect_daily_returns=True)
    fast = fast_fn._run_single(genome, start, end)
    fast_fills = _fills(fast_fn.last_positions)

    nt_fn = fast_fn.nt_twin()
    captured.clear()
    nt = nt_fn._run_single(genome, start, end)
    nt_fills = _fills(captured.get("positions", []))

    assert fallbacks.summary() == {"total": 0, "by_reason": {}}
    assert nt["total_trades"] > 0
    assert nt_fills == fast_fills
    assert {m: nt[m] for m in METRICS} == {m: fast[m] for m in METRICS}
    assert fast.keys() == nt.keys()
    for key in fast:
        if key == "mtm_daily_returns":
            continue
        assert repr(fast[key]) == repr(nt[key]), key
    a = fast["mtm_daily_returns"]
    b = nt["mtm_daily_returns"]
    assert a.first_day == b.first_day
    assert a.values.tobytes() == b.values.tobytes()


# ---------------------------------------------------------------------------
# P2: one archive read per process per (archive, symbol), whatever the genome count.
# ---------------------------------------------------------------------------


@needs_catalog
def test_three_taker_genomes_read_the_archive_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three TAKER genomes in one process hit the kline archive exactly once.

    The archive is a private three-day ZECUSDT table covering the window and its preflight margin (what is counted is reads, not values),
    and its source stamp is pinned: a shared archive's ``-wal`` file appears and vanishes
    between genomes, which correctly forces a preflight reload. An UNCHANGED archive must
    never be re-read.
    """
    import sqlite3
    from datetime import UTC, datetime

    from vibe_quant.data.archive import RawDataArchive
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    day_ms = int(datetime(2026, 3, 5, tzinfo=UTC).timestamp()) * 1000
    archive = RawDataArchive(tmp_path / "raw_data.db")
    try:
        archive.conn.executemany(
            "INSERT INTO raw_klines (symbol, interval, open_time, open, high, low, close, "
            "volume, close_time, taker_buy_volume, source) "
            "VALUES ('ZECUSDT', '1m', ?, 1, 1, 1, 1, 2.0, ?, 1.0, 'test')",
            [(day_ms + i * 60_000, day_ms + i * 60_000 + 59_999) for i in range(-2, 3 * 1440)],
        )
        archive.conn.commit()
    finally:
        archive.close()
    import vibe_quant.data.archive as archive_mod

    monkeypatch.setattr(archive_mod, "DEFAULT_ARCHIVE_PATH", tmp_path / "raw_data.db")

    reads: list[str] = []
    real_connect = sqlite3.connect

    def counting_connect(database: str, *a: Any, **k: Any) -> Any:
        if str(database).endswith("raw_data.db?mode=ro"):
            reads.append(str(database))
        return real_connect(database, *a, **k)

    monkeypatch.setattr(sqlite3, "connect", counting_connect)
    monkeypatch.setattr(aux_data, "_taker_source_stamp", lambda path: ((1, 1), (1, 1)))
    aux_data._TAKER_KLINES.clear()
    fallbacks.clear()

    for period in (5, 30, 60):
        genome = StrategyChromosome(
            entry_genes=[
                StrategyGene("TAKER_IMBALANCE", {"period": float(period)}, ConditionType.GT, 0.05)
            ],
            exit_genes=[StrategyGene("WILLR", {"period": 14.0}, ConditionType.GT, -20.0)],
            stop_loss_pct=0.8,
            take_profit_pct=1.0,
            direction=Direction.LONG,
        )
        fn = FastBacktestFn(["ZECUSDT"], "1m", "2026-03-06", "2026-03-07")
        fn._run_single(genome, "2026-03-06", "2026-03-07")

    assert len(reads) == 1, reads
    assert len(aux_data._TAKER_KLINES) == 1
    assert fallbacks.summary() == {"total": 0, "by_reason": {}}
    aux_data._TAKER_KLINES.clear()


# ---------------------------------------------------------------------------
# The as-of path refuses malformed compute output instead of inventing values.
# ---------------------------------------------------------------------------


def test_asof_context_columns_rejects_malformed_compute_output() -> None:
    """A compute_fn that is not one value per bar is unsupported, never silently padded."""
    frames = 5
    spec = IndicatorSpec(
        name="BAD_ASOF",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 2},
        param_schema={"period": int},
        compute_fn=lambda df, params: np.zeros(frames - 1),
        needs_context=True,
        asof_context=True,
    )
    indicator_registry.register_spec(spec)
    try:
        frame = pd.DataFrame({"close": np.arange(frames, dtype=np.float64)})
        with pytest.raises(PtaReplayUnsupported, match=r"bad returned shape \(4,\) for 5 bars"):
            asof_context_columns(
                {"bad": {"period": 2}},
                {"bad": ("BAD_ASOF", "1m")},
                "1m",
                frame,
                np.arange(frames, dtype=np.int64),
                INSTRUMENT_ID,
            )
    finally:
        indicator_registry.unregister("BAD_ASOF")
