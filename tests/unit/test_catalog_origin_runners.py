"""Both runners, the precompute fallback and the fast tier start the catalog-origin kinds
(ADX period > 16, VIDYA, ADAPTIVE_RSI) from the bar type's first catalog bar.

Design d42m5 T2: a run started at S or S+k sees the same value at every shared bar, the
two tiers agree across validation's coverage clamp, the live per-bar path equals the
precompute, a sweep override of ADX's period selects the origin the way a stored DSL does,
and an origin query that does not end with the run's bars fails the run. A temp
ParquetDataCatalog of synthetic 1m bars (with flat stretches: VIDYA's 0/0 hold) keeps every
run under a second; funding comes from the real archive (skipped when absent).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from nautilus_trader.model.data import Bar
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.db.state_manager import StateManager
from vibe_quant.errors import DataUnavailableError
from vibe_quant.fast_tier import signals as fast_signals
from vibe_quant.fast_tier.errors import FastTierUnsupported
from vibe_quant.screening import nt_runner
from vibe_quant.screening.nt_runner import NTScreeningRunner, resolve_strategy_params
from vibe_quant.validation import runner as validation_runner
from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "data" / "archive" / "raw_data.db"
pytestmark = pytest.mark.skipif(not ARCHIVE.is_file(), reason="needs the funding archive")

MIN = 60_000_000_000
DAY = 1440 * MIN
DAY0 = 1_704_067_200_000_000_000  # 2024-01-01T00:00:00Z
INIT = 59_999_000_000  # a 1m bar's ts_init = open + 59.999 s
BTC = "BTCUSDT-PERP.BINANCE"
HYPE = "HYPEUSDT-PERP.BINANCE"
BTC_ORIGIN = DAY0 + INIT
HYPE_ORIGIN = DAY0 + 3 * DAY + INIT  # HYPE lists three days later
END = "2024-01-07"
_EXITS = {
    "stop_loss": {"type": "fixed_pct", "percent": 0.5},
    "take_profit": {"type": "fixed_pct", "percent": 0.6},
}


def origin_dsl() -> dict[str, Any]:
    """The three origin kinds on one 1m strategy (ADX(30) is past the period-16 boundary)."""
    return {
        "name": "origin_runners",
        "timeframe": "1m",
        "indicators": {
            "arsi": {"type": "ADAPTIVE_RSI", "period": 14, "alpha": 0.7},
            "vidya": {"type": "VIDYA", "period": 14},
            "adx": {"type": "ADX", "period": 30},
        },
        "entry_conditions": {"long": ["close > vidya", "arsi < 60"], "short": ["close < vidya"]},
        **_EXITS,
    }


def adx_dsl(period: int) -> dict[str, Any]:
    return {
        "name": f"origin_adx_{period}",
        "timeframe": "1m",
        "indicators": {"adx": {"type": "ADX", "period": period}},
        "entry_conditions": {"long": ["adx > 25"]},
        **_EXITS,
    }


def _bars(
    symbol: str, first_open: int, n: int, base: float, step: float, seed: int, every: int = 1
) -> list[Bar]:
    """``n`` 1m bars (one per ``every`` minutes) of a random walk with two 40-bar flat stretches."""
    inst = create_instrument(symbol)
    prec = inst.price_precision
    bar_type = get_bar_type(symbol, "1m")
    walk = base + np.cumsum(np.random.default_rng(seed).normal(0.0, step, n))
    flat = np.zeros(n, dtype=bool)
    for at in (n // 2, n - n // 4):
        flat[at : at + 40] = True
        walk[at : at + 40] = walk[at - 1]
    out: list[Bar] = []
    prev = round(float(walk[0]), prec)
    for i in range(n):
        close = round(float(walk[i]), prec)
        open_ = close if flat[i] else prev
        wick = 0.0 if flat[i] else 3 * step / 20
        high = round(max(open_, close) + wick, prec)
        low = round(min(open_, close) - wick, prec)
        t = first_open + i * every * MIN
        out.append(
            Bar(
                bar_type,
                Price(open_, prec),
                Price(high, prec),
                Price(low, prec),
                Price(close, prec),
                Quantity(1 + i % 7, inst.size_precision),
                t,
                t + INIT,
            )
        )
        prev = close
    return out


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """BTC 1m from 2024-01-01 (7 days), HYPE 1m from 2024-01-04 (4 days)."""
    root = tmp_path_factory.mktemp("origin") / "catalog"
    cat = ParquetDataCatalog(str(root))
    cat.write_data([create_instrument("BTCUSDT"), create_instrument("HYPEUSDT")])
    cat.write_data(_bars("BTCUSDT", DAY0, 7 * 1440, 42_000.0, 20.0, 7))
    cat.write_data(_bars("HYPEUSDT", DAY0 + 3 * DAY, 4 * 1440, 20.0, 0.01, 11))
    return root


@pytest.fixture(scope="module")
def sparse_catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """BTC 1m bars every 10 minutes from 2024-01-01 (5 days): a cheap live per-bar path."""
    root = tmp_path_factory.mktemp("origin_sparse") / "catalog"
    cat = ParquetDataCatalog(str(root))
    cat.write_data([create_instrument("BTCUSDT")])
    cat.write_data(_bars("BTCUSDT", DAY0, 5 * 144, 42_000.0, 60.0, 5, every=10))
    return root


Rows = dict[str, dict[int, dict[str, float]]]


@contextmanager
def recording() -> Iterator[Rows]:
    """``{instrument: {ts_init: _pta_values after that bar's feed}}`` of every engine run.

    Wraps ``run_engine`` in both tiers: after the runner's own ``prepare`` (origin, then
    precompute) each strategy's ``_feed_pta_buffer`` is wrapped to copy ``_pta_values``.
    """
    rows: Rows = {}
    real = nt_runner.run_engine

    def run_engine(spec: Any, consume: Any, *, costs: Any, prepare: Any = None) -> Any:
        def prep(node: Any) -> None:
            if prepare is not None:
                prepare(node)
            for engine in node.get_engines():
                for strategy in engine.trader.strategies():
                    log = rows.setdefault(str(strategy.config.instrument_id), {})
                    feed = strategy._feed_pta_buffer

                    def rec(tf: str, bar: Any, *, _s: Any = strategy, _f: Any = feed,
                            _log: dict[int, dict[str, float]] = log) -> None:
                        _f(tf, bar)
                        _log[int(bar.ts_init)] = dict(_s._pta_values)

                    strategy._feed_pta_buffer = rec

        return real(spec, consume, costs=costs, prepare=prep)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nt_runner, "run_engine", run_engine)
        mp.setattr(validation_runner, "run_engine", run_engine)
        yield rows


def screen(
    catalog: Path, symbols: list[str], start: str, dsl: dict[str, Any] | None = None,
    params: dict[str, float | int] | None = None, end: str = END,
) -> tuple[NTScreeningRunner, Rows]:
    runner = NTScreeningRunner(
        dsl or origin_dsl(), symbols, start, end,
        catalog_path=str(catalog), funding_archive_path=str(ARCHIVE),
    )
    with recording() as rows:
        metrics = runner(params or {})
    assert metrics.total_trades > 0  # the run filled orders: not an empty engine
    return runner, rows


def validate(
    catalog: Path, tmp_path: Path, symbols: list[str], start: str, end: str = END,
    dsl: dict[str, Any] | None = None,
) -> tuple[ValidationRunner, Any, Rows, int, Path]:
    db = tmp_path / f"state-{start}-{len(symbols)}.db"
    state = StateManager(db)
    run_id = state.create_backtest_run(
        strategy_id=state.create_strategy(name="origin_runners", dsl_config=dsl or origin_dsl()),
        run_mode="validation", symbols=symbols, timeframe="1m",
        start_date=start, end_date=end, parameters={},
    )
    state.close()
    runner = ValidationRunner(
        db_path=db, logs_path=tmp_path / "logs", catalog_path=catalog,
        funding_archive_path=ARCHIVE,
    )
    try:
        with recording() as rows:
            result = runner.run(run_id=run_id)
    finally:
        runner.close()
    return runner, result, rows, run_id, db


def assert_same_at_shared(a: dict[int, dict[str, float]], b: dict[int, dict[str, float]],
                          keys: tuple[str, ...] = ("arsi", "vidya", "adx"),
                          *, ready: bool = True) -> None:
    """Bit-equal ``_pta_values`` at every shared bar; ``ready``: every key set at both first bars."""
    shared = sorted(set(a) & set(b))
    assert len(shared) > 500
    for ts in shared:
        assert a[ts] == b[ts], ts
    for run in (a, b) if ready else ():
        first = run[min(run)]
        assert all(k in first for k in keys), sorted(first)  # ready at run bar 0


def row(instrument: str, origin_ns: int, prefill: int) -> dict[str, object]:
    return {"instrument": instrument, "tf": "1m", "origin_ns": origin_ns, "prefill": prefill}


@pytest.mark.slow(reason="eight NT runs (six screening, two validation) on a temp catalog, ~6 s")
def test_catalog_origin_start_independent(catalog: Path, tmp_path: Path) -> None:
    """Runs from S and S+k (k in 1, 7, 37, 101, 1000 bars): one origin, bit-equal values."""
    base_runner, base = screen(catalog, ["BTCUSDT"], "2024-01-05")
    for k in (1, 7, 37, 101, 1000):
        start = f"2024-01-05T{k // 60:02d}:{k % 60:02d}"
        runner, rows = screen(catalog, ["BTCUSDT"], start)
        assert min(rows[BTC]) == DAY0 + (5760 + k) * MIN + INIT
        assert_same_at_shared(base[BTC], rows[BTC], ready=False)
        assert runner.last_pta_origin == [row(BTC, BTC_ORIGIN, 5760 + k)], k
    assert_same_at_shared(base[BTC], rows[BTC])  # and ready at both runs' first bar
    # 2024-01-05 is 4 days = 5760 bars after the origin bar
    assert base_runner.last_pta_origin == [row(BTC, BTC_ORIGIN, 5760)]

    for start, prefill in (("2024-01-05", 5760), ("2024-01-06", 7200)):  # validation: whole days
        vrunner, result, vrows, _, _ = validate(catalog, tmp_path, ["BTCUSDT"], start)
        assert vrunner.last_pta_origin == [row(BTC, BTC_ORIGIN, prefill)]
        assert result.notes["pta_origin"] == {BTC: {"1m": "2024-01-01T00:00:59.999000+00:00"}}
        assert_same_at_shared(base[BTC], vrows[BTC])


@pytest.mark.slow(reason="three NT runs on two symbols, one clamped by validation, ~4 s")
def test_tier_parity_across_clamp(catalog: Path, tmp_path: Path) -> None:
    """HYPE lists on 01-04: validation clamps 01-02 to 01-04; BTC's origin stays BTC's own."""
    symbols = ["BTCUSDT", "HYPEUSDT"]
    requested_runner, requested = screen(catalog, symbols, "2024-01-02")
    clamped_runner, clamped = screen(catalog, symbols, "2024-01-04")
    vrunner, result, vrows, _, _ = validate(catalog, tmp_path, symbols, "2024-01-02")

    assert result.notes["data_window"]["effective_start"] == "2024-01-04"
    assert sorted(requested_runner.last_pta_origin, key=lambda r: str(r["instrument"])) == [
        row(BTC, BTC_ORIGIN, 1440), row(HYPE, HYPE_ORIGIN, 0),
    ]
    for origin_rows in (clamped_runner.last_pta_origin, vrunner.last_pta_origin):
        assert sorted(origin_rows, key=lambda r: str(r["instrument"])) == [
            row(BTC, BTC_ORIGIN, 4320), row(HYPE, HYPE_ORIGIN, 0),
        ]
    assert result.notes["pta_origin"] == {
        BTC: {"1m": "2024-01-01T00:00:59.999000+00:00"},
        HYPE: {"1m": "2024-01-04T00:00:59.999000+00:00"},
    }
    assert_same_at_shared(requested[BTC], clamped[BTC])
    assert_same_at_shared(requested[BTC], vrows[BTC])
    # HYPE's run starts at its first catalog bar (prefill 0): it warms up, no readiness claim
    assert_same_at_shared(clamped[HYPE], vrows[HYPE], keys=())
    assert "arsi" in clamped[HYPE][max(clamped[HYPE])]


@pytest.mark.slow(reason="live per-bar recompute from a prefilled buffer in both tiers, ~3 s")
def test_origin_precompute_off_equals_on(
    sparse_catalog: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VIBE_QUANT_PTA_PRECOMPUTE=0 (the live per-bar path from the prefilled buffer) == on."""
    start, end = "2024-01-04", "2024-01-05"
    on_runner, on = screen(sparse_catalog, ["BTCUSDT"], start, end=end)
    assert on_runner.last_pta_precompute[0]["fed"] == 144
    monkeypatch.setenv("VIBE_QUANT_PTA_PRECOMPUTE", "0")
    off_runner, off = screen(sparse_catalog, ["BTCUSDT"], start, end=end)
    assert off_runner.last_pta_precompute == []
    assert off_runner.last_pta_origin == on_runner.last_pta_origin == [row(BTC, BTC_ORIGIN, 432)]
    assert off[BTC] == on[BTC] and len(on[BTC]) == 144
    vrunner, _, voff, _, _ = validate(sparse_catalog, tmp_path, ["BTCUSDT"], start, end)
    assert vrunner.last_pta_precompute == []
    assert vrunner.last_pta_origin == [row(BTC, BTC_ORIGIN, 432)]
    assert voff[BTC] == on[BTC]


@pytest.mark.slow(reason="four NT screening runs on a temp catalog, ~3 s")
def test_origin_follows_effective_params(catalog: Path) -> None:
    """A sweep override of ADX's period selects the origin like the stored DSL: 16 no, 17 yes."""
    start = "2024-01-05"
    r16, _ = screen(catalog, ["BTCUSDT"], start, dsl=adx_dsl(14), params={"adx_period": 16})
    assert r16.last_pta_origin == []
    r17, swept = screen(catalog, ["BTCUSDT"], start, dsl=adx_dsl(14), params={"adx_period": 17})
    assert r17.last_pta_origin == [row(BTC, BTC_ORIGIN, 5760)]
    stored_runner, stored = screen(catalog, ["BTCUSDT"], start, dsl=adx_dsl(17))
    assert stored_runner.last_pta_origin == r17.last_pta_origin
    assert swept[BTC] == stored[BTC]
    shifted_runner, shifted = screen(
        catalog, ["BTCUSDT"], "2024-01-05T01:41", dsl=adx_dsl(14), params={"adx_period": 17}
    )
    assert shifted_runner.last_pta_origin == [row(BTC, BTC_ORIGIN, 5861)]
    assert_same_at_shared(swept[BTC], shifted[BTC], keys=("adx",))


def _break_origin(monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    """The origin query (start None) disagrees with the run's: last close moved, or it fails."""
    real = nt_runner._pta_bar_arrays

    def fake(catalog: str, bar_type: str, start: str | None, end: str) -> dict[str, np.ndarray]:
        arrays = real(catalog, bar_type, start, end)
        if start is not None:
            return arrays
        if how == "raise":
            raise OSError("parquet gone")
        moved = dict(arrays)
        moved["close"] = arrays["close"].copy()
        moved["close"][-1] += 1.0
        return moved

    monkeypatch.setattr(nt_runner, "_pta_bar_arrays", fake)


_MISMATCH = "catalog-origin history for BTCUSDT-PERP.BINANCE 1m does not end with the run's bars"
_UNREADABLE = "catalog-origin history for BTCUSDT-PERP.BINANCE 1m cannot be read: OSError: parquet gone"


@pytest.mark.parametrize(("how", "message"), [("close", _MISMATCH), ("raise", _UNREADABLE)])
def test_origin_mismatch_fails_run(
    catalog: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str, message: str
) -> None:
    """Both tiers fail before any metric: screening raises (never a -inf row), validation fails."""
    _break_origin(monkeypatch, how)
    runner = NTScreeningRunner(
        origin_dsl(), ["BTCUSDT"], "2024-01-05", END,
        catalog_path=str(catalog), funding_archive_path=str(ARCHIVE),
    )
    with pytest.raises(DataUnavailableError, match=message):
        runner({})
    assert runner.last_pta_origin == []

    with pytest.raises(ValidationRunnerError, match=message):
        validate(catalog, tmp_path, ["BTCUSDT"], "2024-01-05")
    (db,) = tmp_path.glob("state-*.db")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        status, error = con.execute("select status, error_message from backtest_runs").fetchone()
        results = con.execute("select count(*) from backtest_results").fetchone()[0]
    finally:
        con.close()
    assert status == "failed" and message in error
    assert results == 0


def _fast_tier_values(
    runner: NTScreeningRunner, bars: list[Bar], *, merged: bool
) -> list[dict[str, float]]:
    """``_pta_values`` after every bar of the fast-tier replay (legacy 1m loop or merged path)."""
    runner._ensure_compiled()
    params = resolve_strategy_params({}, runner._config_fields())
    config = {"instrument_id": BTC, **params, "execution_delay_probability": 1.0}
    instrument = fast_signals.load_instrument(str(runner._resolved_catalog_path.resolve()), BTC)
    strategy, indicators, _ = fast_signals.build_strategy(
        runner._module_path, runner._strategy_cls_name, runner._config_cls_name,
        config, instrument, any_timeframe=merged,
    )
    seen: list[dict[str, float]] = []
    ready = strategy._indicators_ready

    def record() -> bool:
        seen.append(dict(strategy._pta_values))
        return bool(ready())

    strategy._indicators_ready = record
    types = {"arsi": "ADAPTIVE_RSI", "vidya": "VIDYA", "adx": "ADX"}
    fast_signals.compute_signals(
        strategy, indicators, bars, types, timeframe="1m", record=("arsi",) if merged else ()
    )
    return seen


def test_fast_tier_matches_nt_with_origin(catalog: Path) -> None:
    """Legacy 1m loop and merged path write NT's origin values at every bar."""
    start = "2024-01-05T00:37"
    runner, nt_rows = screen(catalog, ["BTCUSDT"], start)
    bars = fast_signals.load_bars(
        str(catalog.resolve()), f"{BTC}-1-MINUTE-LAST-EXTERNAL", start, END
    )
    nt = [nt_rows[BTC][int(b.ts_init)] for b in bars]
    assert all("vidya" in v for v in nt)
    assert _fast_tier_values(runner, bars, merged=False) == nt
    assert _fast_tier_values(runner, bars, merged=True) == nt

    # bars that are no load_bars query cannot be traced to their origin: refused
    with pytest.raises(FastTierUnsupported, match="catalog-origin 1m"):
        _fast_tier_values(runner, list(bars), merged=False)


@contextmanager
def counting_replays() -> Iterator[list[int]]:
    """Rows of every replay compute call (one ``_segment_arrays`` per compute_fn call), on an
    empty process memo of origin replays."""
    from collections import OrderedDict

    from vibe_quant.dsl import pta_replay

    sizes: list[int] = []
    real = pta_replay._segment_arrays

    def count(result: object, names: tuple[str, ...], n_rows: int) -> dict[str, np.ndarray]:
        sizes.append(n_rows)
        return real(result, names, n_rows)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pta_replay, "_segment_arrays", count)
        mp.setattr(pta_replay, "_ORIGIN_SERIES", OrderedDict(), raising=False)
        yield sizes


@pytest.mark.slow(reason="two NT screening runs and one validation run on a sparse catalog, ~3 s")
def test_one_origin_replay_per_run(
    sparse_catalog: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prefill seed and the precompute share ONE cap-0 replay of the origin frame per
    indicator (432 prefill + 144 run bars), precompute on or off, in both tiers."""
    start, end = "2024-01-04", "2024-01-05"
    with counting_replays() as on:
        on_runner, _ = screen(sparse_catalog, ["BTCUSDT"], start, end=end)
    assert on_runner.last_pta_precompute[0]["fed"] == 144
    assert on == [576, 576, 576]  # arsi, vidya, adx: one call each over the origin frame
    with counting_replays() as validation:
        validate(sparse_catalog, tmp_path, ["BTCUSDT"], start, end)
    assert validation == [576, 576, 576]
    monkeypatch.setenv("VIBE_QUANT_PTA_PRECOMPUTE", "0")
    with counting_replays() as off:
        off_runner, _ = screen(sparse_catalog, ["BTCUSDT"], start, end=end)
    assert off_runner.last_pta_precompute == []
    assert off == [576, 576, 576]


@pytest.mark.slow(reason="three NT screening runs on a sparse catalog, ~3 s")
def test_origin_replays_shared_across_runs(sparse_catalog: Path) -> None:
    """A later run of the process with an earlier end (another genome with the same indicator
    params) replays nothing: its values are cut from the first run's replays and equal the
    same run on an empty memo, bit for bit, with no bar after its end."""
    start = "2024-01-04"
    with counting_replays() as calls:
        screen(sparse_catalog, ["BTCUSDT"], start, end="2024-01-05")
        assert calls == [576, 576, 576]
        _, served = screen(sparse_catalog, ["BTCUSDT"], start, end="2024-01-04T12:00")
        assert calls == [576, 576, 576]  # served from the memo
    with counting_replays() as fresh_calls:
        _, fresh = screen(sparse_catalog, ["BTCUSDT"], start, end="2024-01-04T12:00")
    assert fresh_calls == [504, 504, 504]  # 432 prefill + 72 run bars
    assert served[BTC] == fresh[BTC]
    assert max(served[BTC]) == DAY0 + 3 * DAY + 71 * 10 * MIN + INIT  # 11:50 opens the last


def test_origin_identity_error_after_the_reads_fails_the_run(
    sparse_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bar file vanishing between the origin reads and the identity check after them is a
    data error of the run (CatalogOriginError), never a -inf metrics row."""
    read = nt_runner._pta_bar_arrays
    identity = nt_runner._bar_files_identity
    armed: list[bool] = []

    def read_then_arm(path: str, bar_type: str, start: str | None, end: str) -> Any:
        result = read(path, bar_type, start, end)
        if start is not None:  # the run's read: the second of the origin's two reads
            armed.append(True)
        return result

    def vanishing(path: str, bar_type: str) -> Any:
        if armed:
            raise FileNotFoundError("MARKER parquet replaced between scandir and stat")
        return identity(path, bar_type)

    monkeypatch.setattr(nt_runner, "_pta_bar_arrays", read_then_arm)
    monkeypatch.setattr(nt_runner, "_bar_files_identity", vanishing)
    runner = NTScreeningRunner(
        origin_dsl(), ["BTCUSDT"], "2024-01-04", "2024-01-05",
        catalog_path=str(sparse_catalog), funding_archive_path=str(ARCHIVE),
    )
    with pytest.raises(nt_runner.CatalogOriginError, match="cannot be read: FileNotFoundError"):
        runner({})
    assert armed


BTC_1M = f"{BTC}-1-MINUTE-LAST-EXTERNAL"
HYPE_1M = f"{HYPE}-1-MINUTE-LAST-EXTERNAL"
_COLUMNS = ("ts_init", "open", "high", "low", "close", "volume")


@contextmanager
def origin_loads() -> Iterator[list[tuple[str, str | None]]]:
    """(bar type, end) of every catalog read from the bar type's first bar, on fresh caches."""
    from collections import OrderedDict

    from vibe_quant.dsl import pta_replay

    loads: list[tuple[str, str | None]] = []
    real = ParquetDataCatalog.bars
    real_decode = nt_runner._decode_bar_file

    def bars(self: ParquetDataCatalog, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("start") is None:
            loads.append((kwargs["bar_types"][0], kwargs.get("end")))
        return real(self, *args, **kwargs)

    def decode(catalog_path: str, bar_type: str) -> dict[str, np.ndarray] | None:
        arrays = real_decode(catalog_path, bar_type)
        if arrays is not None:  # the fast read of every bar (else the NT query is recorded)
            loads.append((bar_type, None))
        return arrays

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ParquetDataCatalog, "bars", bars)
        mp.setattr(nt_runner, "_decode_bar_file", decode)
        mp.setattr(pta_replay, "_ORIGIN_SERIES", OrderedDict())
        mp.setattr(nt_runner, "_PTA_ARRAYS", OrderedDict())
        mp.setattr(nt_runner, "_ORIGIN_HISTORY", OrderedDict(), raising=False)
        yield loads


def fresh_query(catalog: Path, bar_type: str, end: str) -> dict[str, np.ndarray]:
    """The origin query straight from NT (no cache): first catalog bar to ``end``."""
    bars = ParquetDataCatalog(str(catalog)).bars(bar_types=[bar_type], end=end)
    return {
        "ts_init": np.array([b.ts_init for b in bars], dtype=np.int64),
        **{c: np.array([float(getattr(b, c)) for b in bars]) for c in _COLUMNS[1:]},
    }


def assert_bit_equal(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> None:
    for column in _COLUMNS:
        assert got[column].dtype == want[column].dtype, column
        assert got[column].tobytes() == want[column].tobytes(), column


@pytest.mark.slow(reason="three NT screening runs and one validation run on a temp catalog, ~5 s")
def test_origin_history_loads_once_across_backtests(catalog: Path, tmp_path: Path) -> None:
    """Two screenings and a validation in one process, three different ends: ONE origin load;
    the later, shorter run equals the same run on fresh caches and never sees a later bar."""
    with origin_loads() as loads:
        screen(catalog, ["BTCUSDT"], "2024-01-05")  # end 2024-01-07
        short_runner, short = screen(catalog, ["BTCUSDT"], "2024-01-05", end="2024-01-06")
        vrunner, _, vrows, _, _ = validate(catalog, tmp_path, ["BTCUSDT"], "2024-01-04", "2024-01-06")
        assert len(loads) == 1
    # the last bar opens 2024-01-05 23:59; the 2024-01-06 00:00 bar (ts_init 00:00:59.999) is past
    assert max(short[BTC]) == DAY0 + 5 * DAY - MIN + INIT
    assert short_runner.last_pta_origin == [row(BTC, BTC_ORIGIN, 5760)]
    assert vrunner.last_pta_origin == [row(BTC, BTC_ORIGIN, 4320)]
    with origin_loads() as fresh_loads:
        fresh_runner, fresh = screen(catalog, ["BTCUSDT"], "2024-01-05", end="2024-01-06")
    assert fresh_loads == [(BTC_1M, None)]  # loaded to the catalog end, sliced to the run's
    assert fresh[BTC] == short[BTC]
    assert_same_at_shared(fresh[BTC], vrows[BTC])
    assert fresh_runner.last_pta_origin == short_runner.last_pta_origin


@pytest.mark.parametrize(
    ("end", "last_ts"),
    [
        ("2024-01-05T00:00:59.999", DAY0 + 4 * DAY + INIT),  # end == a bar's ts_init: included
        ("2024-01-05T00:00:59.998999999", DAY0 + 4 * DAY - MIN + INIT),  # 1 ns before: excluded
        ("2024-01-02T13:07", DAY0 + DAY + 786 * MIN + INIT),
        ("2030-01-01", DAY0 + 7 * DAY - MIN + INIT),  # past the catalog end: every bar
    ],
)
def test_origin_history_slices_to_the_run_end(catalog: Path, end: str, last_ts: int) -> None:
    """A history cached to a later end serves an earlier end bit-equal to NT's own query,
    with no bar after the end (look-ahead), from the one load."""
    with origin_loads() as loads:
        nt_runner._pta_bar_arrays(str(catalog), BTC_1M, None, END)
        got = nt_runner._pta_bar_arrays(str(catalog), BTC_1M, None, end)
        assert len(loads) == 1
    assert int(got["ts_init"][-1]) == last_ts
    assert int(got["ts_init"][0]) == BTC_ORIGIN
    assert_bit_equal(got, fresh_query(catalog, BTC_1M, end))
    with pytest.raises(ValueError, match="read-only"):
        got["close"][0] = 0.0  # a caller cannot corrupt the shared history


def test_origin_history_reloads_after_a_catalog_update(tmp_path: Path) -> None:
    """Rewriting the bar type's parquet (a catalog update) at the same path and end reloads."""
    root = tmp_path / "catalog"
    cat = ParquetDataCatalog(str(root))
    cat.write_data([create_instrument("BTCUSDT")])
    cat.write_data(_bars("BTCUSDT", DAY0, 1440, 42_000.0, 20.0, 7))
    with origin_loads() as loads:
        before = nt_runner._pta_bar_arrays(str(root), BTC_1M, None, END)
        nt_runner._pta_bar_arrays(str(root), BTC_1M, None, END)
        assert len(loads) == 1  # unchanged files: a hit
        bar_dir = root / "data" / "bar" / BTC_1M
        for f in bar_dir.iterdir():
            f.unlink()
        cat.write_data(_bars("BTCUSDT", DAY0, 1440, 42_000.0, 20.0, 8))  # same bars, new prices
        after = nt_runner._pta_bar_arrays(str(root), BTC_1M, None, END)
        assert len(loads) == 2
    assert before["close"].tobytes() != after["close"].tobytes()
    assert_bit_equal(after, fresh_query(root, BTC_1M, END))


def test_origin_history_memory_is_bounded(
    catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Least-recently-used histories leave once the byte cap is reached; one over the cap
    is served uncached."""
    btc_bytes = 7 * 1440 * 6 * 8
    hype_bytes = 4 * 1440 * 6 * 8
    with origin_loads() as loads:
        monkeypatch.setattr(nt_runner, "_ORIGIN_HISTORY_MAX_BYTES", btc_bytes, raising=False)
        nt_runner._pta_bar_arrays(str(catalog), BTC_1M, None, END)
        nt_runner._pta_bar_arrays(str(catalog), HYPE_1M, None, END)  # evicts BTC
        assert [key[1] for key in nt_runner._ORIGIN_HISTORY] == [HYPE_1M]
        nt_runner._pta_bar_arrays(str(catalog), HYPE_1M, None, END)  # hit
        nt_runner._pta_bar_arrays(str(catalog), BTC_1M, None, END)  # reload, evicts HYPE
        assert [key[1] for key in nt_runner._ORIGIN_HISTORY] == [BTC_1M]
        assert len(loads) == 3
        monkeypatch.setattr(nt_runner, "_ORIGIN_HISTORY_MAX_BYTES", hype_bytes, raising=False)
        nt_runner._pta_bar_arrays(str(catalog), BTC_1M, None, "2024-01-02")  # cached: a hit
        assert len(loads) == 3
        nt_runner._pta_bar_arrays(str(catalog), HYPE_1M, None, END)
        nt_runner._pta_bar_arrays(str(catalog), BTC_1M, None, END)  # over the cap: uncached
        assert [key[1] for key in nt_runner._ORIGIN_HISTORY] == [HYPE_1M]
        nt_runner._pta_bar_arrays(str(catalog), BTC_1M, None, END)
        assert len(loads) == 6
        monkeypatch.setattr(nt_runner, "_ORIGIN_HISTORY_MAX_BYTES", btc_bytes + hype_bytes)
        nt_runner._pta_bar_arrays(str(catalog), BTC_1M, None, END)  # exactly at the cap: both stay
        assert [key[1] for key in nt_runner._ORIGIN_HISTORY] == [HYPE_1M, BTC_1M]
        monkeypatch.setattr(nt_runner, "_ORIGIN_HISTORY_MAX_BYTES", btc_bytes + hype_bytes - 1)
        nt_runner._ORIGIN_HISTORY.clear()
        nt_runner._pta_bar_arrays(str(catalog), HYPE_1M, None, END)
        nt_runner._pta_bar_arrays(str(catalog), BTC_1M, None, END)  # one byte over: HYPE leaves
        assert [key[1] for key in nt_runner._ORIGIN_HISTORY] == [BTC_1M]


# --- one bar-file generation per run (a backfill, a correction, a file swapped mid-run) ----


def _sparse(root: Path, *days: tuple[int, int]) -> Path:
    """A catalog of 144-bar (every 10 min) BTC 1m days ``(day offset, seed)``."""
    cat = ParquetDataCatalog(str(root))
    cat.write_data([create_instrument("BTCUSDT")])
    for offset, seed in days:
        cat.write_data(_bars("BTCUSDT", DAY0 + offset * DAY, 144, 42_000.0, 50.0, seed, every=10))
    return root


def _replacement_file(tmp_path: Path, bars: list[Bar]) -> Path:
    """``bars`` written to a scratch catalog: its one parquet file, named as the catalog names it."""
    root = tmp_path / "replacement"
    ParquetDataCatalog(str(root)).write_data(bars)
    (new_file,) = (root / "data" / "bar" / BTC_1M).glob("*.parquet")
    return new_file


def test_fast_tier_follows_a_backfill_at_the_same_path_and_dates(tmp_path: Path) -> None:
    """Earlier bars backfilled into the catalog (the run window's file untouched): the next
    fast-tier replay in the same process starts from the new origin, as a fresh screening does."""
    root = _sparse(tmp_path / "catalog", (1, 7))
    start, end = "2024-01-02", "2024-01-03"
    runner, _ = screen(root, ["BTCUSDT"], start, end=end)
    catalog = str(root.resolve())
    before = _fast_tier_values(runner, fast_signals.load_bars(catalog, BTC_1M, start, end), merged=True)
    assert before[0] == {}  # the run is the whole catalog: warms up from its own first bar

    ParquetDataCatalog(str(root)).write_data(
        _bars("BTCUSDT", DAY0, 144, 42_000.0, 50.0, 8, every=10)
    )
    bars = fast_signals.load_bars(catalog, BTC_1M, start, end)
    after_merged = _fast_tier_values(runner, bars, merged=True)
    after_legacy = _fast_tier_values(runner, bars, merged=False)
    fresh_runner, nt_rows = screen(root, ["BTCUSDT"], start, end=end)
    assert fresh_runner.last_pta_origin == [row(BTC, BTC_ORIGIN, 144)]
    nt = [nt_rows[BTC][int(b.ts_init)] for b in bars]
    assert len(nt) == 144 and sorted(nt[0]) == ["adx", "arsi", "vidya"]
    assert after_merged == nt
    assert after_legacy == nt


def test_fast_tier_refuses_bars_older_than_the_catalog(tmp_path: Path) -> None:
    """Bars loaded before a catalog rewrite are never replayed against the new origin."""
    root = _sparse(tmp_path / "catalog", (0, 7), (1, 8))
    start, end = "2024-01-02", "2024-01-03"
    runner, _ = screen(root, ["BTCUSDT"], start, end=end)
    bars = fast_signals.load_bars(str(root.resolve()), BTC_1M, start, end)
    ParquetDataCatalog(str(root)).write_data(
        _bars("BTCUSDT", DAY0 - DAY, 144, 42_000.0, 50.0, 9, every=10)
    )
    with pytest.raises(nt_runner.CatalogOriginError, match="changed since the run's bars were loaded"):
        _fast_tier_values(runner, bars, merged=False)
    fresh = fast_signals.load_bars(str(root.resolve()), BTC_1M, start, end)
    assert fresh is not bars and [b.ts_init for b in fresh] == [b.ts_init for b in bars]
    with pytest.raises(FastTierUnsupported, match="not a load_bars query"):
        _fast_tier_values(runner, bars, merged=False)  # the new load replaced this list


def test_fast_tier_origin_frame_and_memo_bytes_after_a_second_start(tmp_path: Path) -> None:
    """A second start with the same end reuses the verified origin frame (no reinsertion), and
    the memo's byte count stays the sum of its entries."""
    from collections import OrderedDict

    root = _sparse(tmp_path / "catalog", (0, 7), (1, 8))
    end = "2024-01-03"
    runner, _ = screen(root, ["BTCUSDT"], "2024-01-02", end=end)
    catalog = str(root.resolve())
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fast_signals, "_MEMO", OrderedDict())
        mp.setattr(fast_signals, "_memo_bytes", 0)
        _fast_tier_values(runner, fast_signals.load_bars(catalog, BTC_1M, "2024-01-02", end),
                          merged=False)
        (frame_key,) = [k for k in fast_signals._MEMO if k[:2] == ("frame", "origin")]
        frame = fast_signals._MEMO[frame_key][1]
        _fast_tier_values(
            runner, fast_signals.load_bars(catalog, BTC_1M, "2024-01-02T06:00", end), merged=False
        )
        assert fast_signals._MEMO[frame_key][1] is frame
        assert fast_signals._memo_bytes == sum(e[2] for e in fast_signals._MEMO.values())

        bars = fast_signals._MEMO[frame_key][0]
        fast_signals._memo_put(("x",), bars, 1, 100)
        fast_signals._memo_put(("x",), bars, 2, 40)  # a replacement deducts the old entry
        assert fast_signals._MEMO[("x",)][1:] == (2, 40)
        assert fast_signals._memo_bytes == sum(e[2] for e in fast_signals._MEMO.values())


def test_fast_tier_seeds_a_slower_origin_timeframe_before_its_first_bar(tmp_path: Path) -> None:
    """1m ARSI/ADX + 5m VIDYA: from the run's first 1m bar on, the fast tier's readiness, the
    values its checks read and its entry signals equal NT's bar by bar (the 5m prefill's held
    VIDYA is read before the first in-window 5m bar)."""
    root = tmp_path / "catalog"
    cat = ParquetDataCatalog(str(root))
    cat.write_data([create_instrument("BTCUSDT")])
    cat.write_data(_bars("BTCUSDT", DAY0, 2880, 42_000.0, 50.0, 7))
    coarse = _bars("BTCUSDT", DAY0, 576, 42_000.0, 100.0, 8, every=5)
    cat.write_data([
        Bar(get_bar_type("BTCUSDT", "5m"), b.open, b.high, b.low, b.close, b.volume, b.ts_event,
            b.ts_init + 4 * MIN)  # a 5m bar closes 5 minutes after it opens
        for b in coarse
    ])
    dsl = origin_dsl()
    dsl["additional_timeframes"] = ["5m"]
    dsl["indicators"]["vidya"]["timeframe"] = "5m"
    start, end = "2024-01-02", "2024-01-03"

    ready: dict[int, tuple[bool, dict[str, float]]] = {}
    checks: dict[tuple[str, int], bool] = {}
    real = nt_runner.run_engine

    def run_engine(spec: Any, consume: Any, *, costs: Any, prepare: Any = None) -> Any:
        def prep(node: Any) -> None:
            prepare(node)
            for engine in node.get_engines():
                for s in engine.trader.strategies():
                    def is_ready(*, _s: Any = s, _f: Any = s._indicators_ready) -> bool:
                        out = bool(_f())
                        ts = int(_s._last_bar_ts)
                        ready[ts] = (out, dict(_s._pta_values))
                        return out

                    def feed(tf: str, bar: Any, *, _s: Any = s, _f: Any = s._feed_pta_buffer) -> None:
                        _f(tf, bar)
                        if tf == "1m":
                            _s._last_bar_ts = int(bar.ts_init)

                    def wrap(name: str, _s: Any = s) -> None:
                        check = getattr(_s, name)

                        def recorded(bar: Any) -> bool:
                            out = bool(check(bar))
                            checks[(name, int(bar.ts_init))] = out
                            return out

                        setattr(_s, name, recorded)

                    s._indicators_ready, s._feed_pta_buffer = is_ready, feed
                    wrap("_check_long_entry")
                    wrap("_check_short_entry")

        return real(spec, consume, costs=costs, prepare=prep)

    runner = NTScreeningRunner(dsl, ["BTCUSDT"], start, end, catalog_path=str(root),
                               funding_archive_path=str(ARCHIVE))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nt_runner, "run_engine", run_engine)
        assert runner({}).total_trades > 0
    assert sorted(r["tf"] for r in runner.last_pta_origin) == ["1m", "5m"]

    run = fast_signals.signals_for_runner(runner, "BTCUSDT")
    times = [int(b.ts_init) for b in run.bars["1m"]]
    assert sorted(ready) == times and len(times) == 1440
    first = ready[times[0]]
    assert first[0] and sorted(first[1]) == ["adx", "arsi", "vidya"]  # NT: ready at bar 0
    assert run.signals.ready.tolist() == [ready[t][0] for t in times]
    columns = {"_check_long_entry": run.signals.long_entry, "_check_short_entry": run.signals.short_entry}
    index = {t: i for i, t in enumerate(times)}
    assert len(checks) > 100
    for (name, ts), value in checks.items():
        assert bool(columns[name][index[ts]]) == value, (name, ts)

    # the values the checks read: the merged replay's _pta_values at every ready call
    strategy, indicators, _ = fast_signals.build_strategy(
        runner._module_path, runner._strategy_cls_name, runner._config_cls_name,
        {"instrument_id": BTC, **resolve_strategy_params({}, runner._config_fields()),
         "execution_delay_probability": 1.0},
        run.instrument, any_timeframe=True,
    )
    seen: list[dict[str, float]] = []
    fast_ready = strategy._indicators_ready

    def record() -> bool:
        seen.append(dict(strategy._pta_values))
        return bool(fast_ready())

    strategy._indicators_ready = record
    fast_signals.compute_signals(
        strategy, indicators, run.bars["1m"], {"arsi": "ADAPTIVE_RSI", "vidya": "VIDYA", "adx": "ADX"},
        extra_bars={"5m": run.bars["5m"]}, timeframe="1m",
    )
    assert seen == [ready[t][1] for t in times]


@contextmanager
def swapped_after_prepare(new_file: Path, live_file: Path) -> Iterator[None]:
    """Both tiers: the bar file is atomically replaced after ``prepare`` (origin + precompute
    read it), before the engine reads its bars."""
    import os

    real = nt_runner.run_engine

    def run_engine(spec: Any, consume: Any, *, costs: Any, prepare: Any = None) -> Any:
        def prep(node: Any) -> None:
            prepare(node)
            os.replace(new_file, live_file)

        return real(spec, consume, costs=costs, prepare=prep)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nt_runner, "run_engine", run_engine)
        mp.setattr(validation_runner, "run_engine", run_engine)
        yield


def _highs_moved(bars: list[Bar]) -> list[Bar]:
    """Same ts_init and close, highs raised: invisible to the precompute's ts/close guard."""
    return [
        Bar(b.bar_type, b.open, Price(float(b.high) + 100 * (i % 5), b.high.precision),
            b.low, b.close, b.volume, b.ts_event, b.ts_init)
        for i, b in enumerate(bars)
    ]


_CHANGED = "catalog bar files changed during the run"


def test_a_bar_file_swapped_after_prepare_fails_both_tiers(tmp_path: Path) -> None:
    """A same-close OHLCV correction lands between the origin/precompute read and the engine's
    read: screening raises, validation fails the run and persists no result."""
    root = _sparse(tmp_path / "catalog", (0, 7), (2, 9))  # validation needs data past the end
    suffix = _bars("BTCUSDT", DAY0 + DAY, 144, 42_000.0, 50.0, 8, every=10)
    ParquetDataCatalog(str(root)).write_data(suffix)
    start, end = "2024-01-02", "2024-01-03"
    new_file = _replacement_file(tmp_path / "s", _highs_moved(suffix))
    live_file = root / "data" / "bar" / BTC_1M / new_file.name
    assert live_file.is_file()
    runner = NTScreeningRunner(origin_dsl(), ["BTCUSDT"], start, end, catalog_path=str(root),
                               funding_archive_path=str(ARCHIVE))
    with swapped_after_prepare(new_file, live_file), pytest.raises(DataUnavailableError, match=_CHANGED):
        runner({})

    new_file = _replacement_file(tmp_path / "v", _highs_moved(suffix))
    with swapped_after_prepare(new_file, live_file), pytest.raises(ValidationRunnerError, match=_CHANGED):
        validate(root, tmp_path, ["BTCUSDT"], start, end)
    (db,) = tmp_path.glob("state-*.db")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        status, error = con.execute("select status, error_message from backtest_runs").fetchone()
        results = con.execute("select count(*) from backtest_results").fetchone()[0]
    finally:
        con.close()
    assert status == "failed" and _CHANGED in error
    assert results == 0


def test_a_run_without_pta_arrays_ignores_a_bar_file_swap(tmp_path: Path) -> None:
    """No origin timeframe, no compute_fn indicator: the run reads no PTA arrays, so nothing is
    checked and a swap before the engine's read is simply the data the engine runs on."""
    root = _sparse(tmp_path / "catalog", (0, 7))
    suffix = _bars("BTCUSDT", DAY0 + DAY, 144, 42_000.0, 50.0, 8, every=10)
    ParquetDataCatalog(str(root)).write_data(suffix)
    new_file = _replacement_file(tmp_path / "s", _highs_moved(suffix))
    live_file = root / "data" / "bar" / BTC_1M / new_file.name
    dsl = {**adx_dsl(14), "name": "native_ema",
           "indicators": {"ema": {"type": "EMA", "period": 5}},
           "entry_conditions": {"long": ["close > ema"]}}
    runner = NTScreeningRunner(dsl, ["BTCUSDT"], "2024-01-02", "2024-01-03",
                               catalog_path=str(root), funding_archive_path=str(ARCHIVE))
    with swapped_after_prepare(new_file, live_file):
        metrics = runner({})
    assert metrics.total_trades > 0 and runner.last_pta_origin == []
    assert not new_file.exists()  # the swap ran


def test_a_completed_catalog_refresh_runs_again_at_the_same_path_and_dates(tmp_path: Path) -> None:
    """The run window's file is replaced by a valid, close-changing one between two screenings:
    the second run reads origin and window from the new files and equals a cold-cache run."""
    import os
    from collections import OrderedDict

    root = _sparse(tmp_path / "catalog", (0, 7), (1, 8))
    start, end = "2024-01-02", "2024-01-03"
    screen(root, ["BTCUSDT"], start, end=end)
    new_file = _replacement_file(
        tmp_path / "s", _bars("BTCUSDT", DAY0 + DAY, 144, 42_000.0, 50.0, 9, every=10)
    )
    os.replace(new_file, root / "data" / "bar" / BTC_1M / new_file.name)
    warm_runner, warm = screen(root, ["BTCUSDT"], start, end=end)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nt_runner, "_PTA_ARRAYS", OrderedDict())
        mp.setattr(nt_runner, "_ORIGIN_HISTORY", OrderedDict())
        cold_runner, cold = screen(root, ["BTCUSDT"], start, end=end)
    assert len(warm[BTC]) == 144
    assert warm == cold
    assert warm_runner.last_pta_origin == cold_runner.last_pta_origin == [row(BTC, BTC_ORIGIN, 144)]
    assert warm_runner.last_pta_precompute == cold_runner.last_pta_precompute
    assert warm_runner.last_pta_precompute[0]["fed"] == 144
