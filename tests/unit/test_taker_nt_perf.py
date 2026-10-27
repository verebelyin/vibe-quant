"""TAKER_IMBALANCE on the NT path: per-bar scalar lookup, values bit-equal to the vector form.

The generated ``_update_pta_indicators`` used to build a DataFrame and a Series per bar
(~190 us/bar). A flagged ``asof_context`` spec now reads one scalar, ``compute_fn.asof_scalar``.
Every value here is compared BITWISE (NaN included) against ``taker_imbalance_asof``, the
vectorised reference, through the real generated ``_feed_pta_buffer``.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import vibe_quant.dsl.plugins.taker as taker_plugin
from vibe_quant.dsl import aux_data
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.indicators import pta_buffer_cap, pta_lookback
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.pta_buffer import PtaBuffer

from .test_taker_flow import DAY0, MIN, _make_kline_archive, _sqlite_klines

if TYPE_CHECKING:
    from collections.abc import Iterator

HOUR = 60 * MIN
INSTRUMENT_ID = "ZECUSDT-PERP.BINANCE"


@pytest.fixture
def clean() -> Iterator[None]:
    yield
    aux_data.configure(None, None)


def _bits(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.int64)


def _strategy(tf: str, periods: dict[str, int], instrument_id: str = INSTRUMENT_ID) -> Any:
    """Generated strategy of ``tf`` with one TAKER_IMBALANCE per ``periods`` entry (name -> N)."""
    indicators = {n: {"type": "TAKER_IMBALANCE", "period": p} for n, p in periods.items()}
    first = next(iter(periods))
    dsl = validate_strategy_dict(
        {
            "name": "taker_nt_perf",
            "timeframe": tf,
            "indicators": indicators,
            "entry_conditions": {"long": [f"{first} > 0.05"]},
            "exit_conditions": {"long": [f"{first} < -0.05"]},
            "stop_loss": {"type": "fixed_pct", "percent": 0.8},
            "take_profit": {"type": "fixed_pct", "percent": 1.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    cls: Any = next(
        v for k, v in vars(module).items() if k.endswith("Strategy") and k != "Strategy"
    )
    cfg = type("Cfg", (), {"instrument_id": instrument_id})()
    wired: Any = type("Wired", (cls,), {"config": cfg})
    inst: Any = wired.__new__(wired)
    params: dict[str, dict[str, object]] = {n: {"period": p} for n, p in periods.items()}
    lookbacks = {n: pta_lookback("TAKER_IMBALANCE", params[n]) for n in periods}
    cap = pta_buffer_cap(list(lookbacks.values()), full_history=False)
    step = {"1m": MIN, "1h": HOUR}[tf]
    inst._pta_values = {}
    inst._prev_values = {}
    inst._last_close = 0.0
    inst._pta_params = params
    inst._pta_lookback = lookbacks
    inst._pta_buffer_cap = {tf: cap}
    inst._pta_bufs = {tf: PtaBuffer(cap, with_close_ns=True)}
    inst._pta_step_ns = {tf: step}
    return inst


def _bar(close_ns: int, i: int) -> Any:
    """Backtest-style bar: ts_init is the exchange close_time, 1 ms before the boundary."""
    px = 100.0 + (i % 17)
    return SimpleNamespace(
        open=px, high=px + 1, low=px - 1, close=px + 0.5, volume=3.0, ts_init=close_ns - 1_000_000
    )


def _archive(path: Path, n: int, gaps: tuple[int, ...] = ()) -> Path:
    """``n`` 1m klines from DAY0 with 3-decimal volumes; minutes in ``gaps`` are missing."""
    rng = np.random.default_rng(11)
    minutes = [m for m in range(n) if m not in gaps]
    volumes = np.round(rng.uniform(0.001, 40.0, len(minutes)), 3)
    takers = np.round(volumes * rng.uniform(0.1, 0.9, len(minutes)), 3)
    return _make_kline_archive(path, volumes.tolist(), takers.tolist(), minute_indices=minutes)


def _feed_all(inst: Any, tf: str, closes: list[int], name: str) -> list[float]:
    out = []
    for i, c in enumerate(closes):
        inst._feed_pta_buffer(tf, _bar(c, i))
        out.append(inst._pta_values.get(name, float("nan")))
    return out


@pytest.mark.parametrize(
    ("tf", "step", "period", "n_bars", "gaps"),
    [
        ("1m", MIN, 5, 300, (40, 41, 130)),  # holes: NaN windows, never carried forward
        ("1m", MIN, 30, 300, ()),
        ("1h", HOUR, 30, 40, (700, 1500)),
    ],
)
def test_generated_feed_equals_vector_reference(
    tmp_path: Path, clean: None, tf: str, step: int, period: int, n_bars: int, gaps: tuple[int, ...]
) -> None:
    """Every stored value is bit-equal to ``taker_imbalance_asof`` at the rounded bar close."""
    n_klines = n_bars * (step // MIN) + 5
    aux_data.configure(_archive(tmp_path / "a.db", n_klines, gaps), None)
    closes = [DAY0 + (j + 1) * step for j in range(n_bars)]
    expected = aux_data.taker_imbalance_asof("ZECUSDT", np.array(closes, dtype=np.int64), period)
    assert np.isfinite(expected).any()
    if gaps:  # a hole after warmup: NaN windows exist and are stored, not skipped
        warmup = period - 1 if tf == "1m" else 0
        assert int(np.isnan(expected).sum()) > warmup

    inst = _strategy(tf, {"ti": period})
    got = _feed_all(inst, tf, closes, "ti")

    assert np.array_equal(_bits(got), _bits(expected))


def test_two_unrelated_indicators_forward_their_own_period_and_column(
    tmp_path: Path, clean: None
) -> None:
    """Two TAKER indicators of different periods each store their own series."""
    aux_data.configure(_archive(tmp_path / "two.db", 400), None)
    closes = [DAY0 + (j + 1) * MIN for j in range(300)]
    arr = np.array(closes, dtype=np.int64)
    inst = _strategy("1m", {"fast": 5, "slow": 45})
    fast, slow = [], []
    for i, c in enumerate(closes):
        inst._feed_pta_buffer("1m", _bar(c, i))
        fast.append(inst._pta_values.get("fast", float("nan")))
        slow.append(inst._pta_values.get("slow", float("nan")))

    ref_fast = aux_data.taker_imbalance_asof("ZECUSDT", arr, 5)
    ref_slow = aux_data.taker_imbalance_asof("ZECUSDT", arr, 45)
    assert np.array_equal(_bits(fast), _bits(ref_fast))
    assert np.array_equal(_bits(slow), _bits(ref_slow))
    assert not np.array_equal(_bits(ref_fast), _bits(ref_slow))


def test_each_instrument_reads_its_own_symbol_at_the_same_close(
    tmp_path: Path, clean: None
) -> None:
    """Two symbols in one archive: the forwarded instrument id picks the symbol.

    At the same close both have exactly 5 klines: ZECUSDT volume 2.0 / taker 1.0
    -> 0.0, BTCUSDT volume 2.0 / taker 1.5 -> 0.25. ZEC is fed again after BTC so
    the per-symbol kline cache is covered, not just the first load.
    """
    archive = _make_kline_archive(tmp_path / "both.db", [2.0] * 5, [1.0] * 5, symbol="ZECUSDT")
    _make_kline_archive(tmp_path / "both.db", [2.0] * 5, [1.5] * 5, symbol="BTCUSDT")
    aux_data.configure(archive, None)
    close = DAY0 + 5 * MIN
    zec = _strategy("1m", {"ti": 5}, instrument_id="ZECUSDT-PERP.BINANCE")
    btc = _strategy("1m", {"ti": 5}, instrument_id="BTCUSDT-PERP.BINANCE")

    def feed(inst: Any) -> float:
        inst._feed_pta_buffer("1m", _bar(close, 5))
        return float(inst._pta_values["ti"])

    assert feed(zec) == 0.0
    assert feed(btc) == 0.25  # never ZEC's 0.0
    assert feed(zec) == 0.0  # cached, still ZEC's own klines


def test_taker_only_feed_builds_no_dataframe(
    tmp_path: Path, clean: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scalar path never materialises the buffer frame (the per-bar pandas cost)."""
    aux_data.configure(_archive(tmp_path / "f.db", 200), None)
    built: list[int] = []
    real = PtaBuffer.frame

    def counting_frame(self: PtaBuffer) -> Any:
        built.append(1)
        return real(self)

    monkeypatch.setattr(PtaBuffer, "frame", counting_frame)
    inst = _strategy("1m", {"ti": 30})
    closes = [DAY0 + (j + 1) * MIN for j in range(100)]
    got = _feed_all(inst, "1m", closes, "ti")
    assert np.isfinite(got).any()
    assert built == []


# -- scalar vs vector at every edge of the window -----------------------------------------


def _scalar(instrument_id: str, close_ns: int, period: int) -> float:
    scalar: Any = taker_plugin.compute_taker_imbalance.asof_scalar  # type: ignore[attr-defined]
    return float(scalar(instrument_id, close_ns, {"period": period}))


def _vector(close_ns: int, period: int) -> float:
    return float(
        aux_data.taker_imbalance_asof("ZECUSDT", np.array([close_ns], dtype=np.int64), period)[0]
    )


def test_scalar_equals_vector_at_window_edges(tmp_path: Path, clean: None) -> None:
    """Before/after coverage, exact first/last window, a gap inside, a NULL taker, zero volume."""
    volumes = [2.0] * 12 + [0.0] * 4 + [3.0] * 4 + [1.5] * 8
    takers: list[float | None] = [*([1.0] * 12), *([0.0] * 4), *([1.0] * 4), *([0.5] * 8)]
    takers[22] = None  # NULL taker_buy_volume inside the last window
    archive = _make_kline_archive(tmp_path / "e.db", volumes, takers)
    aux_data.configure(archive, None)
    last_close = DAY0 + len(volumes) * MIN
    for period in (1, 3, 4, 5):
        for k in range(-3, len(volumes) + 4):
            t = DAY0 + k * MIN
            assert (
                _bits([_scalar(INSTRUMENT_ID, t, period)])[0] == _bits([_vector(t, period)])[0]
            ), (
                period,
                k,
            )
    # a bar between two kline closes finds no kline ending at t: NaN, never the previous one
    assert np.isnan(_scalar(INSTRUMENT_ID, DAY0 + 5 * MIN + 1, 3))
    assert np.isnan(_scalar(INSTRUMENT_ID, last_close + MIN, 1))  # after the last kline
    assert np.isnan(_scalar(INSTRUMENT_ID, DAY0, 1))  # before the first kline closes


def test_scalar_window_boundary_excludes_kline_opening_at_t(tmp_path: Path, clean: None) -> None:
    volumes = [round(1.001 * (i + 1), 3) for i in range(11)]
    takers = [round(v * 0.5, 3) for v in volumes]
    takers[10] = 999.0
    aux_data.configure(_make_kline_archive(tmp_path / "b.db", volumes, takers), None)
    t = DAY0 + 10 * MIN
    assert _scalar(INSTRUMENT_ID, t, 3) == sum(takers[7:10]) / sum(volumes[7:10]) - 0.5
    assert _scalar(INSTRUMENT_ID, t + MIN, 3) > 30


def test_scalar_follows_archive_switch_between_runs(tmp_path: Path, clean: None) -> None:
    """A new configure() (new run) must not serve the previous archive's klines."""
    rows_a = [(i * 60_000, 2.0, 1.0) for i in range(5)]
    rows_b = [(i * 60_000, 2.0, 1.5) for i in range(5)]
    a = _sqlite_klines(tmp_path / "a.db", rows_a)
    b = _sqlite_klines(tmp_path / "b.db", rows_b)
    aux_data.configure(a, None)
    assert _scalar(INSTRUMENT_ID, 5 * MIN, 5) == 0.0
    aux_data.configure(b, None)
    assert _scalar(INSTRUMENT_ID, 5 * MIN, 5) == 0.25


def test_scalar_follows_default_archive_patch(
    tmp_path: Path, clean: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vibe_quant.data.archive as archive_mod

    a = _sqlite_klines(tmp_path / "a.db", [(i * 60_000, 2.0, 1.0) for i in range(5)])
    b = _sqlite_klines(tmp_path / "b.db", [(i * 60_000, 2.0, 1.5) for i in range(5)])
    aux_data.configure(None, None)
    monkeypatch.setattr(archive_mod, "DEFAULT_ARCHIVE_PATH", a)
    assert _scalar(INSTRUMENT_ID, 5 * MIN, 5) == 0.0
    monkeypatch.setattr(archive_mod, "DEFAULT_ARCHIVE_PATH", b)
    assert _scalar(INSTRUMENT_ID, 5 * MIN, 5) == 0.25


def test_scalar_relative_archive_follows_cwd_between_runs(
    tmp_path: Path, clean: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative archive path resolves against the cwd at the start of each run.

    A run starts with ``configure`` and a launch check (``_taker_klines(revalidate=True)``);
    either one makes the next per-bar lookup resolve the path again.
    """
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    (tmp_path / "z").mkdir()
    _sqlite_klines(tmp_path / "x" / "k.db", [(i * 60_000, 2.0, 1.0) for i in range(5)])
    _sqlite_klines(tmp_path / "y" / "k.db", [(i * 60_000, 2.0, 1.5) for i in range(5)])
    _sqlite_klines(tmp_path / "z" / "k.db", [(i * 60_000, 2.0, 0.5) for i in range(5)])
    aux_data.configure("k.db", None)
    monkeypatch.chdir(tmp_path / "x")
    assert _scalar(INSTRUMENT_ID, 5 * MIN, 5) == 0.0
    monkeypatch.chdir(tmp_path / "y")
    aux_data.configure("k.db", None)  # next run, same relative name
    assert _scalar(INSTRUMENT_ID, 5 * MIN, 5) == 0.25
    monkeypatch.chdir(tmp_path / "z")
    aux_data._taker_klines("ZECUSDT", revalidate=True)  # launch check of the next run
    assert _scalar(INSTRUMENT_ID, 5 * MIN, 5) == -0.25

    # the default archive path object survives configure(None): the reset must not rely on it
    import vibe_quant.data.archive as archive_mod

    monkeypatch.setattr(archive_mod, "DEFAULT_ARCHIVE_PATH", Path("k.db"))
    aux_data.configure(None, None)
    monkeypatch.chdir(tmp_path / "x")
    assert _scalar(INSTRUMENT_ID, 5 * MIN, 5) == 0.0
    monkeypatch.chdir(tmp_path / "y")
    aux_data.configure(None, None)
    assert _scalar(INSTRUMENT_ID, 5 * MIN, 5) == 0.25


def test_scalar_missing_archive_raises(tmp_path: Path, clean: None) -> None:
    aux_data.configure(tmp_path / "missing.db", None)
    with pytest.raises(aux_data.AuxDataUnavailableError, match="kline archive"):
        _scalar(INSTRUMENT_ID, DAY0, 5)


def test_scalar_period_defaults_to_60_and_symbol_is_stripped(tmp_path: Path, clean: None) -> None:
    aux_data.configure(_archive(tmp_path / "d.db", 100), None)
    t = DAY0 + 70 * MIN
    scalar: Any = taker_plugin.compute_taker_imbalance.asof_scalar  # type: ignore[attr-defined]
    default = scalar(INSTRUMENT_ID, t, {})
    assert default == _vector(t, 60)
    assert default != _vector(t, 30)


# -- speed ----------------------------------------------------------------------------------


@pytest.mark.timing
def test_taker_feed_under_50_us_per_bar(tmp_path: Path, clean: None) -> None:
    """Generated feed of two TAKER indicators (entry + exit style) stays under 50 us/bar."""
    n = 6000
    aux_data.configure(_archive(tmp_path / "t.db", n + 5), None)
    closes = [DAY0 + (j + 1) * MIN for j in range(n)]
    bars = [_bar(c, i) for i, c in enumerate(closes)]
    inst = _strategy("1m", {"ti_entry": 30, "ti_exit": 30})
    inst._feed_pta_buffer("1m", bars[0])  # warm: loads the archive cache
    per_bar = []
    for lo in range(1, n, 1000):
        t0 = time.perf_counter()
        for bar in bars[lo : lo + 1000]:
            inst._feed_pta_buffer("1m", bar)
        per_bar.append((time.perf_counter() - t0) / 1000 * 1e6)
    assert statistics.median(per_bar) < 50.0, per_bar
