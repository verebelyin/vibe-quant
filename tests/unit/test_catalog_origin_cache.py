"""Catalog-origin speed (vibe-quant-mjw5l) without moving a single value.

A 1m catalog-origin run replays ADAPTIVE_RSI / VIDYA / ADX over every catalog bar of the coin
(~3.25M for BTC). Three levers keep that fast, each pinned bit for bit here:

- the origin history is decoded from the bar type's parquet file with pyarrow, equal to NT's
  own query (``nt_runner._decode_bar_file``), with the NT query as the fallback;
- ADAPTIVE_RSI's per-bar loops run on plain floats, equal to the original numpy loops;
- the process memo ``pta_replay._ORIGIN_SERIES`` serves a later request with an earlier end
  by slicing, never a value computed from a bar after the requested end.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from tests.unit.test_catalog_origin import _arrays, _flat_to_moving, _slice
from tests.unit.test_catalog_origin_runners import DAY, DAY0, _bars
from vibe_quant.data.catalog import create_instrument
from vibe_quant.dsl import pta_replay
from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.dsl.plugins.example_adaptive_rsi import _efficiency_ratio, compute_adaptive_rsi
from vibe_quant.dsl.pta_replay import OriginReplay, replay_pta_series
from vibe_quant.screening import nt_runner

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

REPO = Path(__file__).resolve().parents[2]
REAL_CATALOG = REPO / "data" / "catalog"
BTC_1M = "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
HYPE_1M = "HYPEUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
FIELDS = ("ts_init", "open", "high", "low", "close", "volume")


def _bits(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes()


def assert_arrays_bit_equal(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> None:
    assert sorted(got) == sorted(want)
    for name in want:
        assert got[name].dtype == want[name].dtype, name
        assert _bits(got[name]) == _bits(want[name]), name


# --- the fast history decode ------------------------------------------------------------


def _fixed_column(raws: list[int]) -> pa.FixedSizeBinaryArray:
    return pa.array([r.to_bytes(16, "little", signed=True) for r in raws], type=pa.binary(16))


def test_fixed_raw_decode_rounds_like_nt() -> None:
    """int128 raws -> NT's float: correctly rounded (ties to even) then / 1e16, on both sides
    of 2**64 and at the exact-tie and sticky-bit edges of the 53-bit rounding."""
    base = (2**53 + 1) << 40  # odd 53-bit significand, 40 bits below it
    raws = [
        0, 1, 2**53 + 1, 2**63, 2**64 - 1, 2**64, 2**64 + 1, 2**65 - 1,
        base + (1 << 39),  # exact tie: rounds to the even neighbour (up)
        base - (2 << 40) + (1 << 39),  # exact tie below an even significand: rounds down
        base + (1 << 39) + 1,  # just above the tie: the sticky bit rounds up
        base + (1 << 39) - 1,  # just below the tie: rounds down
        420001234500000000000,  # BTC 42000.12345 at precision 16
    ]
    rng = np.random.default_rng(5)
    raws += [int(rng.integers(0, 2**62)) << int(rng.integers(0, 34)) for _ in range(2000)]
    got = nt_runner._fixed_raw_to_f64(_fixed_column(raws))
    assert got is not None
    want = np.array([float(Price.from_raw(r, 16)) for r in raws])
    assert _bits(got) == _bits(want)
    assert _bits(got) == _bits(np.array([float(r) / 1e16 for r in raws]))
    quantities = nt_runner._fixed_raw_to_f64(_fixed_column(raws[:9]))
    assert quantities is not None
    assert _bits(quantities) == _bits(np.array([float(Quantity.from_raw(r, 16)) for r in raws[:9]]))
    # an offset slice of the buffer decodes its own rows
    sliced = nt_runner._fixed_raw_to_f64(_fixed_column(raws).slice(8, 4))
    assert sliced is not None and _bits(sliced) == _bits(want[8:12])
    assert nt_runner._fixed_raw_to_f64(_fixed_column([5, -1])) is None  # negative: NT path


@pytest.fixture(scope="module")
def fixture_catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """BTC 1m (3 days, price precision 1) and HYPE 1m (2 days, another precision) via NT."""
    root = tmp_path_factory.mktemp("decode") / "catalog"
    cat = ParquetDataCatalog(str(root))
    cat.write_data([create_instrument("BTCUSDT"), create_instrument("HYPEUSDT")])
    cat.write_data(_bars("BTCUSDT", DAY0, 3 * 1440, 42_000.0, 20.0, 7))
    cat.write_data(_bars("HYPEUSDT", DAY0 + DAY, 2 * 1440, 20.0, 0.01, 11))
    return root


@pytest.mark.parametrize(("bar_type", "rows"), [(BTC_1M, 3 * 1440), (HYPE_1M, 2 * 1440)])
def test_fast_decode_equals_the_nt_query(fixture_catalog: Path, bar_type: str, rows: int) -> None:
    got = nt_runner._decode_bar_file(str(fixture_catalog), bar_type)
    assert got is not None and len(got["ts_init"]) == rows
    assert_arrays_bit_equal(got, nt_runner._load_bar_arrays(str(fixture_catalog), bar_type, None, None))


@pytest.mark.slow(reason="decodes ~3.6M real BTC 1m bars twice (NT objects ~5 s)")
@pytest.mark.skipif(not (REAL_CATALOG / "data" / "bar" / BTC_1M).is_dir(), reason="no real catalog")
def test_fast_decode_equals_the_nt_query_on_the_real_btc_1m_catalog() -> None:
    got = nt_runner._decode_bar_file(str(REAL_CATALOG), BTC_1M)
    assert got is not None and len(got["ts_init"]) > 3_000_000
    assert_arrays_bit_equal(got, nt_runner._load_bar_arrays(str(REAL_CATALOG), BTC_1M, None, None))


def _as_int64(table: pa.Table, column: str) -> pa.Table:
    idx = table.schema.get_field_index(column)
    return table.set_column(idx, column, table.column(idx).cast(pa.int64()))


def _ts_init(values: Callable[[list[int]], list[int]]) -> Callable[[pa.Table], pa.Table]:
    def rewrite(table: pa.Table) -> pa.Table:
        idx = table.schema.get_field_index("ts_init")
        ts = values(table.column(idx).to_pylist())
        return table.set_column(idx, table.schema.field(idx), pa.array(ts, type=pa.uint64()))

    return rewrite


def _swap(ts: list[int]) -> list[int]:
    ts[5], ts[6] = ts[6], ts[5]
    return ts


@pytest.mark.parametrize(
    "rewrite",
    [
        pytest.param(lambda t: _as_int64(t, "ts_event"), id="signed-ts_event"),
        pytest.param(lambda t: _as_int64(t, "ts_init"), id="signed-ts_init"),
        # a uint64 past int64 would wrap to a negative ts_init: refused before the cast
        pytest.param(_ts_init(lambda ts: [2**64 - 1, *ts[1:]]), id="first-ts-wraps"),
        pytest.param(_ts_init(lambda ts: [2**63, *ts[1:]]), id="first-ts-at-2^63"),
        pytest.param(_ts_init(_swap), id="unsorted"),
    ],
)
def test_fast_decode_falls_back_on_a_layout_surprise(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    rewrite: Callable[[pa.Table], pa.Table],
) -> None:
    """Any schema, type, range or sort surprise takes NT's query (stubbed here: NT is never
    fed a raw outside its range), logged once above DEBUG, never a fast-decoded value."""
    root = tmp_path / "catalog"
    cat = ParquetDataCatalog(str(root))
    cat.write_data([create_instrument("BTCUSDT")])
    cat.write_data(_bars("BTCUSDT", DAY0, 20, 42_000.0, 20.0, 3))
    (path,) = (root / "data" / "bar" / BTC_1M).glob("*.parquet")
    assert nt_runner._decode_bar_file(str(root), BTC_1M) is not None  # the verified layout
    pq.write_table(rewrite(pq.read_table(path)), path)
    sentinel = {"ts_init": np.array([7], dtype=np.int64)}
    monkeypatch.setattr(nt_runner, "_load_bar_arrays", lambda *args: sentinel)
    monkeypatch.setattr(nt_runner, "_FAST_DECODE_FALLBACKS", set())
    with caplog.at_level(logging.INFO, logger=nt_runner.logger.name):
        assert nt_runner._load_origin_arrays(str(root), BTC_1M) is sentinel
        assert nt_runner._load_origin_arrays(str(root), BTC_1M) is sentinel
    assert caplog.text.count("fast decode off: layout not the verified one") == 1


def test_fast_decode_falls_back_to_nt_on_several_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two parquet files for one bar type: the NT query (the reference), logged once."""
    root = tmp_path / "catalog"
    cat = ParquetDataCatalog(str(root))
    cat.write_data([create_instrument("BTCUSDT")])
    bars = _bars("BTCUSDT", DAY0, 2 * 1440, 42_000.0, 20.0, 3)
    cat.write_data(bars[:1440])
    cat.write_data(bars[1440:])
    assert len(list((root / "data" / "bar" / BTC_1M).glob("*.parquet"))) == 2
    assert nt_runner._decode_bar_file(str(root), BTC_1M) is None
    monkeypatch.setattr(nt_runner, "_FAST_DECODE_FALLBACKS", set())
    with caplog.at_level(logging.INFO, logger=nt_runner.logger.name):
        first = nt_runner._load_origin_arrays(str(root), BTC_1M)
        second = nt_runner._load_origin_arrays(str(root), BTC_1M)
    want = nt_runner._load_bar_arrays(str(root), BTC_1M, None, None)
    assert len(want["ts_init"]) == 2 * 1440
    assert_arrays_bit_equal(first, want)
    assert_arrays_bit_equal(second, want)
    assert caplog.text.count("read through NT bars (fast decode off") == 1


def test_fast_decode_error_falls_back_to_nt(
    fixture_catalog: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A decode error takes the NT query; the log names the error type, never its text."""

    def broken(catalog_path: str, bar_type: str) -> None:
        raise RuntimeError("MARKER-decode-text")

    monkeypatch.setattr(nt_runner, "_decode_bar_file", broken)
    monkeypatch.setattr(nt_runner, "_FAST_DECODE_FALLBACKS", set())
    with caplog.at_level(logging.INFO, logger=nt_runner.logger.name):
        got = nt_runner._load_origin_arrays(str(fixture_catalog), HYPE_1M)
    assert_arrays_bit_equal(got, nt_runner._load_bar_arrays(str(fixture_catalog), HYPE_1M, None, None))
    assert len(got["ts_init"]) == 2 * 1440
    assert "fast decode off: RuntimeError" in caplog.text
    assert "MARKER" not in caplog.text


# --- ADAPTIVE_RSI: plain-float loops equal the original numpy loops ---------------------


def _adaptive_rsi_reference(close: np.ndarray, period: int, alpha: float) -> np.ndarray:
    """main's compute_adaptive_rsi body before vibe-quant-mjw5l, verbatim (per-element numpy)."""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    er = np.full(n, 0.5)
    er[period:] = _efficiency_ratio(close, period)[: n - period]
    fast_sc = 2.0 / (period + 1)
    slow_sc = 2.0 / (2 * period + 1)
    sc = np.full(n - 1, fast_sc)
    for i in range(period, n - 1):
        sc[i] = (er[i + 1] * (fast_sc - slow_sc) + slow_sc) ** (1.0 / max(alpha, 0.01))
    avg_gain = np.full(n - 1, np.nan)
    avg_loss = np.full(n - 1, np.nan)
    avg_gain[period - 1] = gains[:period].mean()
    avg_loss[period - 1] = losses[:period].mean()
    for i in range(period, n - 1):
        k = sc[i]
        avg_gain[i] = avg_gain[i - 1] * (1 - k) + gains[i] * k
        avg_loss[i] = avg_loss[i - 1] * (1 - k) + losses[i] * k
    rsi = np.full(n, np.nan)
    for i in range(period - 1, n - 1):
        ag = avg_gain[i]
        al = avg_loss[i]
        if np.isnan(ag) or np.isnan(al):
            continue
        if al == 0:
            rsi[i + 1] = 100.0
        else:
            rs = ag / al
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _closes(n: int, seed: int) -> np.ndarray:
    """A walk opening with a rising run (zero losses from the seed on: the RSI-100 branch),
    a flat run and a NaN close."""
    walk = 100.0 + np.cumsum(np.random.default_rng(seed).normal(0.0, 0.7, n))
    walk[:80] = walk[80] - np.arange(80, 0, -1) * 0.25
    walk[n // 2 : n // 2 + 30] = walk[n // 2 - 1]
    walk[3 * n // 4] = np.nan
    return walk


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.7, 0.9, 1.0, 2.0])
@pytest.mark.parametrize("period", [2, 14, 30])
def test_adaptive_rsi_equals_the_numpy_loops(period: int, alpha: float) -> None:
    """alpha 0.5 and 2 pin the scalar pow (np.power would take sqrt / square)."""
    for n, seed in ((1500, period), (period + 1, 1), (period + 2, 2), (period, 3)):
        close = _closes(n, seed) if n >= 40 else 100.0 + np.arange(n, dtype=np.float64) % 3
        frame = pd.DataFrame({"close": close})
        got = compute_adaptive_rsi(frame, {"period": period, "alpha": alpha}).to_numpy()
        want = _adaptive_rsi_reference(close, period, alpha)
        assert _bits(got) == _bits(want), (n, seed)
        if n == 1500:  # the long case reaches the RSI-100 branch and real values
            assert (want == 100.0).any() and ((want > 0) & (want < 100)).sum() > 500


# --- the process memo of origin replays (_ORIGIN_SERIES) ---------------------------------

KINDS = [
    ("ADAPTIVE_RSI", {"period": 14, "alpha": 0.7}),
    ("VIDYA", {"period": 14}),
    ("ADX", {"period": 26}),
]
KEY = ("catalog", "BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL", (("f.parquet", 1, 2, 3),))
E = 300  # the request's end bar: ts_init[E] == end


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Length of every replay the memo runs; an empty process memo."""
    seen: list[int] = []
    real = pta_replay.replay_pta_series

    def counting(spec: Any, ohlcv: pd.DataFrame, params: Any, cap: int) -> dict[str, np.ndarray]:
        seen.append(len(ohlcv))
        return real(spec, ohlcv, params, cap)

    monkeypatch.setattr(pta_replay, "replay_pta_series", counting)
    monkeypatch.setattr(pta_replay, "_ORIGIN_SERIES", OrderedDict())
    yield seen


def _hist(scale_after: int | None = None) -> dict[str, np.ndarray]:
    arrays = {name: array.copy() for name, array in _arrays(_flat_to_moving()).items()}
    if scale_after is not None:
        for name in ("open", "high", "low", "close"):
            arrays[name][scale_after + 1 :] *= 1.5
    return arrays


def _served(
    arrays: dict[str, np.ndarray], kind: str, params: dict[str, Any],
    memo_key: tuple[object, ...] | None = KEY, cap: int = 0,
) -> dict[str, np.ndarray]:
    spec = indicator_registry.get(kind)
    assert spec is not None
    origin = OriginReplay(arrays, arrays["ts_init"], memo_key)
    return origin.replay(spec, origin.frame, params, cap)


def _fresh(arrays: dict[str, np.ndarray], kind: str, params: dict[str, object]) -> dict[str, np.ndarray]:
    spec = indicator_registry.get(kind)
    assert spec is not None
    return replay_pta_series(spec, OriginReplay(arrays).frame, params, 0)


@pytest.mark.parametrize(("kind", "params"), KINDS)
def test_memo_never_serves_a_bar_after_the_end(
    calls: list[int], kind: str, params: dict[str, object]
) -> None:
    """Stored from a history whose bars AFTER the end are scaled 1.5x, a request ending at
    ``end`` is served from the memo and equals a fresh replay of its own bars, bit for bit."""
    _served(_slice(_hist(scale_after=E), 0, 400), kind, params)
    assert calls == [400]
    request = _slice(_hist(), 0, E + 1)
    got = _served(request, kind, params)
    assert calls == [400]  # served: no replay
    want = _fresh(request, kind, params)
    assert_arrays_bit_equal(got, want)
    assert not np.isnan(next(iter(want.values()))[E])  # the end bar holds a value


@pytest.mark.parametrize(("kind", "params"), KINDS)
def test_memo_edges(calls: list[int], kind: str, params: dict[str, object]) -> None:
    """Stored to ``end``: a request to ``end`` is served, one to ``end`` + 1 bar replays."""
    hist = _hist()
    at_end, one_more = _slice(hist, 0, E + 1), _slice(hist, 0, E + 2)
    first = _served(at_end, kind, params)
    again = _served(at_end, kind, params)
    assert calls == [E + 1]
    assert_arrays_bit_equal(again, first)
    assert_arrays_bit_equal(first, _fresh(at_end, kind, params))
    longer = _served(one_more, kind, params)
    assert calls == [E + 1, E + 2]  # past the stored end: replayed, entry replaced
    assert_arrays_bit_equal(longer, _fresh(one_more, kind, params))
    assert_arrays_bit_equal(_served(at_end, kind, params), first)  # sliced from the longer
    assert calls == [E + 1, E + 2]
    shifted = {**at_end, "ts_init": at_end["ts_init"] + 1}
    _served(shifted, kind, params)  # same key, another ts_init prefix: never served
    _served(at_end, kind, params, memo_key=None)  # no key: this run only
    _served(at_end, kind, params, cap=500)  # a capped replay never uses the memo
    assert calls == [E + 1, E + 2, E + 1, E + 1, E + 1]


def test_memo_keys_on_params_and_bar_files(calls: list[int]) -> None:
    hist = _slice(_hist(), 0, E + 1)
    arsi: dict[str, object] = {"period": 14, "alpha": 0.7}
    _served(hist, "ADAPTIVE_RSI", arsi)
    _served(hist, "ADAPTIVE_RSI", {"period": 14, "alpha": 0.9})
    _served(hist, "ADAPTIVE_RSI", arsi, memo_key=(*KEY[:2], (("f.parquet", 1, 2, 4),)))
    assert calls == [E + 1] * 3
    _served(hist, "ADAPTIVE_RSI", arsi)
    assert calls == [E + 1] * 3


def test_memo_is_bounded_least_recently_used(
    calls: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the byte cap the least recently USED entry leaves: A, B stored, A served again,
    then C evicts B (FIFO would evict A); an entry over the cap is not stored."""
    hist = _slice(_hist(), 0, E + 1)
    one = 2 * hist["ts_init"].nbytes  # ts_init + one output column
    monkeypatch.setattr(pta_replay, "_ORIGIN_SERIES_MAX_BYTES", 2 * one)

    def arsi(alpha: float) -> dict[str, np.ndarray]:
        return _served(hist, "ADAPTIVE_RSI", {"period": 14, "alpha": alpha})

    arsi(0.7)
    arsi(0.8)
    arsi(0.7)  # hit: A becomes the most recently used
    assert calls == [E + 1, E + 1]
    arsi(0.9)  # C: evicts B
    assert calls == [E + 1] * 3
    arsi(0.7)  # A kept
    assert calls == [E + 1] * 3
    arsi(0.8)  # B evicted: replayed
    assert calls == [E + 1] * 4
    monkeypatch.setattr(pta_replay, "_ORIGIN_SERIES_MAX_BYTES", one - 1)
    pta_replay._ORIGIN_SERIES.clear()
    arsi(0.7)
    arsi(0.7)
    assert calls == [E + 1] * 6  # over the cap: never stored


def test_memo_stays_bounded_under_concurrent_misses(
    calls: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two threads miss different keys at once (both past the replay and the ts_init copy
    before either publishes): the memo still holds at most the cap."""
    hist = _slice(_hist(), 0, E + 1)
    rendezvous = Barrier(2, timeout=10)

    class PausingTimestamps(np.ndarray):
        def copy(self, *args: Any, **kwargs: Any) -> np.ndarray:  # type: ignore[override]
            rendezvous.wait()
            return super().copy(*args, **kwargs)

    hist["ts_init"] = hist["ts_init"].view(PausingTimestamps)
    cap = 2 * hist["ts_init"].nbytes  # one entry: ts_init + one output column
    monkeypatch.setattr(pta_replay, "_ORIGIN_SERIES_MAX_BYTES", cap)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_served, hist, "ADAPTIVE_RSI", {"period": 14, "alpha": alpha})
            for alpha in (0.7, 0.9)
        ]
        results = [future.result() for future in futures]
    for alpha, got in zip((0.7, 0.9), results, strict=True):
        assert_arrays_bit_equal(got, _fresh(hist, "ADAPTIVE_RSI", {"period": 14, "alpha": alpha}))
    assert len(pta_replay._ORIGIN_SERIES) == 1
    calls.clear()
    monkeypatch.setattr(pta_replay, "_ORIGIN_SERIES_MAX_BYTES", 2 * cap)  # a miss evicts none
    plain = {**hist, "ts_init": hist["ts_init"].view(np.ndarray)}  # no rendezvous now
    _served(plain, "ADAPTIVE_RSI", {"period": 14, "alpha": 0.7})
    _served(plain, "ADAPTIVE_RSI", {"period": 14, "alpha": 0.9})
    assert calls == [E + 1]  # exactly one of the two was kept


def test_memo_follows_a_registry_replacement(calls: list[int]) -> None:
    """A same-name spec registered over ADAPTIVE_RSI is replayed, never served the values
    of the replaced spec."""
    hist = _slice(_hist(), 0, E + 1)
    params: dict[str, object] = {"period": 14, "alpha": 0.7}
    original = indicator_registry.get("ADAPTIVE_RSI")
    assert original is not None and original.compute_fn is not None
    compute = original.compute_fn
    before = _served(hist, "ADAPTIVE_RSI", params)

    def halved(df: pd.DataFrame, p: dict[str, Any]) -> Any:
        return compute(df, p) / 2

    try:
        indicator_registry.register_spec(
            dataclasses.replace(original, compute_fn=halved), override=True
        )
        after = _served(hist, "ADAPTIVE_RSI", params)
        want = _fresh(hist, "ADAPTIVE_RSI", params)
    finally:
        indicator_registry.register_spec(original, override=True)
    assert calls == [E + 1, E + 1]  # the replacement replayed, not served
    assert_arrays_bit_equal(after, want)
    (name,) = before
    assert np.nanmax(np.abs(after[name] * 2 - before[name])) == 0.0
    assert np.isfinite(after[name]).sum() > 200
