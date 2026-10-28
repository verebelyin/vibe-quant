"""MFI / VIDYA / SQUEEZE_RATIO fast paths must be bit-identical to the library.

The discovery/screening ``compute_fn`` path re-runs these indicators on a capped
rolling buffer once per bar, so pandas' DataFrame/``.loc`` prep dominates the
engine. The lean ports keep every float operation of the library (hlc3 ->
raw money flow -> rolling sums; CMO -> seed -> recurrence; true range -> Wilder
RMA) and hand the rolling ops to pandas itself, so equality is asserted EXACTLY
(no tolerance, NaN positions and bit patterns included) on:

* BTCUSDT 4h and 1d full catalog history, at the registry's param-range edges
  and three interior values;
* every buffer slice a compiled strategy feeds them -- ``PtaBuffer`` capped and
  trimmed once per ``cap // 4`` bars, over a stream that crosses its trims, at the
  param-range edges and the middle value;
* short inputs (warmup / ``None`` normalization) and the library fallbacks.

References: ``vibe_quant/dsl/compute_builtins.py`` (``_mfi_port``),
``vibe_quant/dsl/plugins/vidya.py`` (``_vidya_port``) and
``vibe_quant/dsl/plugins/squeeze.py`` (``_squeeze_ratio_port``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import pytest

from vibe_quant.dsl.compute_builtins import _mfi_port, compute_mfi, nan_like
from vibe_quant.dsl.indicators import (
    indicator_registry,
    invoke_compute_fn,
)
from vibe_quant.dsl.plugins.squeeze import (
    _squeeze_ratio_library,
    _squeeze_ratio_port,
    compute_squeeze_ratio,
)
from vibe_quant.dsl.plugins.vidya import _vidya_library, _vidya_port, compute_vidya
from vibe_quant.dsl.pta_buffer import PtaBuffer

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG = REPO_ROOT / "data" / "catalog"

OHLCV = ("open", "high", "low", "close", "volume")


def _periods(name: str) -> list[int]:
    """Param-range edges plus three interior values (from the registry)."""
    spec = indicator_registry.get(name)
    assert spec is not None and spec.param_ranges is not None
    lo, hi = (int(x) for x in spec.param_ranges["period"])
    span = hi - lo
    return sorted({lo, hi, lo + span // 4, lo + span // 2, hi - span // 4})


MFI_PERIODS = _periods("MFI")
VIDYA_PERIODS = _periods("VIDYA")


@lru_cache(maxsize=4)
def _catalog_bars(interval: str) -> list[Any]:
    """BTCUSDT full catalog history; skip the test when the catalog is absent."""
    try:
        from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH, CatalogManager

        if not (CATALOG / "data").exists():
            pytest.skip("data/catalog absent")
        bars = CatalogManager(DEFAULT_CATALOG_PATH).get_bars("BTCUSDT", interval)
        if len(bars) < 300:
            pytest.skip(f"BTCUSDT {interval} catalog history too short ({len(bars)} bars)")
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # catalog symlink broken / unreadable
        pytest.skip(f"data/catalog unavailable: {exc}")
    return bars


def _frame(bars: list[Any], n: int | None = None) -> pd.DataFrame:
    rows = bars if n is None else bars[:n]
    return pd.DataFrame(
        {c: np.array([float(getattr(b, c)) for b in rows], dtype=np.float64) for c in OHLCV}
    )


def _assert_bit_equal(got: np.ndarray, expected: np.ndarray, where: str) -> None:
    assert np.array_equal(got, expected, equal_nan=True), f"{where}: value mismatch"
    assert np.array_equal(got.view(np.int64), expected.view(np.int64)), f"{where}: bit mismatch"


def _assert_series_equal(got: pd.Series, expected: pd.Series, where: str) -> None:
    assert got.index.equals(expected.index), f"{where}: index changed"
    _assert_bit_equal(got.to_numpy(), expected.to_numpy(), where)


# ---------------------------------------------------------------------------
# (a) full catalog history
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interval", ["4h", "1d"])
@pytest.mark.parametrize("period", MFI_PERIODS)
def test_mfi_port_matches_reference_on_full_catalog_history(interval: str, period: int) -> None:
    df = _frame(_catalog_bars(interval))
    expected = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=period)
    assert expected is not None
    got = _mfi_port(df, period)
    assert got is not None
    _assert_series_equal(got, expected, f"MFI {interval} period={period} n={len(df)}")


@pytest.mark.parametrize("interval", ["4h", "1d"])
@pytest.mark.parametrize("period", VIDYA_PERIODS)
def test_vidya_port_matches_reference_on_full_catalog_history(interval: str, period: int) -> None:
    df = _frame(_catalog_bars(interval))
    expected = ta.vidya(df["close"], length=period)
    assert expected is not None
    got = _vidya_port(df, period)
    assert got is not None
    _assert_series_equal(got, expected, f"VIDYA {interval} period={period} n={len(df)}")


# ---------------------------------------------------------------------------
# (b) every capped rolling-buffer slice the compiled strategy passes
# ---------------------------------------------------------------------------

# cap 400 is the standalone max(PTA_BUFFER_MIN_CAP, 10 x lookback); larger caps
# come from a strategy that also carries a longer-lookback compute_fn indicator.
# PtaBuffer's k-th trim lands at bar ``cap + k * (cap // 4 + 1)``: 627 bars cross the
# cap-400 trims at 501 (grown past the cap) and 602 (steady state) plus 25 bars of the
# post-trim stream; 700 and 1030 cross the first trim of caps 520 (651) and 800 (1001).
BUFFER_BARS = 400 + 2 * (400 // 4 + 1) + 25


def _edges_and_middle(periods: list[int]) -> list[int]:
    """The slice sweeps check both param-range edges and the middle value."""
    return [periods[0], periods[len(periods) // 2], periods[-1]]


# (cap, bars, period): the standalone cap at the edges + middle period, the larger caps at 14.
MFI_SLICE_CASES = [(400, BUFFER_BARS, p) for p in _edges_and_middle(MFI_PERIODS)] + [
    (520, 700, 14),
    (800, 1030, 14),
]
VIDYA_SLICE_CASES = [(400, BUFFER_BARS, p) for p in _edges_and_middle(VIDYA_PERIODS)] + [
    (520, 700, 14),
    (800, 1030, 14),
]


@pytest.mark.parametrize(("cap", "bars", "period"), MFI_SLICE_CASES)
def test_mfi_port_matches_every_buffer_slice(cap: int, bars: int, period: int) -> None:
    rows = _frame(_catalog_bars("4h"), bars)
    buf = PtaBuffer(cap)
    for i in range(len(rows)):
        buf.append(*(float(rows[c].iat[i]) for c in OHLCV))
        if len(buf) < period:
            continue
        df = buf.frame()
        expected = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=period)
        got = _mfi_port(df, period)
        assert got is not None
        _assert_bit_equal(
            got.to_numpy(), expected.to_numpy(), f"MFI cap={cap} n={len(df)} period={period}"
        )


@pytest.mark.parametrize(("cap", "bars", "period"), VIDYA_SLICE_CASES)
def test_vidya_port_matches_every_buffer_slice(cap: int, bars: int, period: int) -> None:
    rows = _frame(_catalog_bars("4h"), bars)
    buf = PtaBuffer(cap)
    for i in range(len(rows)):
        buf.append(*(float(rows[c].iat[i]) for c in OHLCV))
        if len(buf) < period:
            continue
        df = buf.frame()
        expected = ta.vidya(df["close"], length=period)
        got = _vidya_port(df, period)
        assert got is not None
        _assert_bit_equal(
            got.to_numpy(), expected.to_numpy(), f"VIDYA cap={cap} n={len(df)} period={period}"
        )


def test_buffer_trim_actually_happens() -> None:
    """Teeth check: the cap-400 simulation must pass through the 25% trim."""
    buf = PtaBuffer(400)
    rows = _frame(_catalog_bars("4h"), 600)
    lengths: list[int] = []
    for i in range(len(rows)):
        buf.append(*(float(rows[c].iat[i]) for c in OHLCV))
        lengths.append(len(buf))
    assert lengths[0] == 1
    assert max(lengths) > 400  # grows past the cap before trimming
    assert 400 in lengths  # trim resets it to the cap
    assert lengths[-1] == 600 - 501 + 400  # 99 bars past the first trim


# ---------------------------------------------------------------------------
# Warmup, literal contracts, and library fallbacks
# ---------------------------------------------------------------------------


def _synthetic(high: list[float], low: list[float], close: list[float], volume: list[float]):
    return pd.DataFrame(
        {
            "open": np.array(close, dtype=np.float64),
            "high": np.asarray(high, dtype=np.float64),
            "low": np.asarray(low, dtype=np.float64),
            "close": np.asarray(close, dtype=np.float64),
            "volume": np.asarray(volume, dtype=np.float64),
        }
    )


def test_mfi_port_none_below_lookback_and_nan_warmup() -> None:
    df = _synthetic([2, 3, 4], [1, 2, 3], [1.5, 2.5, 3.5], [10, 20, 30])
    assert _mfi_port(df.iloc[:2], 3) is None
    got = _mfi_port(df, 3)
    assert got is not None
    assert np.isnan(got.to_numpy()[:2]).all()  # warmup stays NaN, never 0


def test_vidya_port_none_below_lookback_and_nan_warmup() -> None:
    df = _synthetic([2, 3, 4], [1, 2, 3], [1.5, 2.5, 3.5], [10, 20, 30])
    assert _vidya_port(df.iloc[:2], 3) is None
    got = _vidya_port(df, 3)
    assert got is not None
    assert np.isnan(got.to_numpy()[:2]).all()


def test_mfi_public_contract_on_two_unrelated_inputs() -> None:
    """Hand-computed literals: up-days only -> 100, down-days only -> 0."""
    up = _synthetic([2, 3, 4], [1, 2, 3], [1.5, 2.5, 3.5], [10, 20, 30])
    # tp = (h+l+c)/3 = 1.5, 2.5, 3.5; rmf = 15, 50, 105; both diffs > 0.
    np.testing.assert_array_equal(compute_mfi(up, {"period": 3}).to_numpy(), [np.nan, np.nan, 100.0])
    down = _synthetic([3, 2, 1], [2, 1, 0.5], [2.5, 1.5, 0.75], [10, 20, 30])
    # tp = 2.5, 1.5, 0.75; rmf = 25, 30, 22.5; both diffs < 0.
    np.testing.assert_array_equal(compute_mfi(down, {"period": 3}).to_numpy(), [np.nan, np.nan, 0.0])


def test_vidya_public_contract_on_two_unrelated_inputs() -> None:
    """Hand-computed literals: |CMO| = 1 both directions, alpha = 0.5."""
    up = _synthetic([2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 4], [1, 1, 1, 1])
    # seed = mean(1,2,3) = 2; next = 0.5*1*4 + 2*(1-0.5) = 3.
    np.testing.assert_array_equal(
        compute_vidya(up, {"period": 3}).to_numpy(), [np.nan, np.nan, 2.0, 3.0]
    )
    down = _synthetic([5, 4, 3, 2], [4, 3, 2, 1], [4, 3, 2, 1], [1, 1, 1, 1])
    # seed = mean(4,3,2) = 3; next = 0.5*1*1 + 3*(1-0.5) = 2.
    np.testing.assert_array_equal(
        compute_vidya(down, {"period": 3}).to_numpy(), [np.nan, np.nan, 3.0, 2.0]
    )


def test_mfi_short_input_normalizes_to_nan_series() -> None:
    """The compiled path feeds short frames the library answers with None."""
    df = _synthetic([2, 3], [1, 2], [1.5, 2.5], [10, 20])
    assert compute_mfi(df, {"period": 14}) is None
    out = invoke_compute_fn(indicator_registry.get("MFI"), df, {"period": 14})  # type: ignore[arg-type]
    assert isinstance(out, pd.Series)
    assert np.isnan(out.to_numpy()).all()


def test_vidya_short_input_returns_nan_like() -> None:
    df = _synthetic([2, 3], [1, 2], [1.5, 2.5], [10, 20])
    out = compute_vidya(df, {"period": 14})
    expected = nan_like(df)
    _assert_series_equal(out, expected, "VIDYA short input")


def test_mfi_public_falls_back_to_library_on_non_float64() -> None:
    df = _synthetic([2, 3, 4, 5], [1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5], [10, 20, 30, 40])
    for col in ("high", "low", "close", "volume"):
        df[col] = df[col].astype("float32")
    expected = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=3)
    assert expected is not None
    _assert_bit_equal(compute_mfi(df, {"period": 3}).to_numpy(), expected.to_numpy(), "MFI f32")


def test_vidya_public_falls_back_to_library_on_non_float64() -> None:
    df = _synthetic([2, 3, 4, 5], [1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5], [10, 20, 30, 40])
    df["close"] = df["close"].astype("float32")
    expected = ta.vidya(df["close"], length=3)
    assert expected is not None
    _assert_bit_equal(compute_vidya(df, {"period": 3}).to_numpy(), expected.to_numpy(), "VIDYA f32")


def test_mfi_public_matches_library_error_on_non_unique_index() -> None:
    """The library's label-based ``.loc`` assignments reject a duplicate index;
    the public entry point must keep raising instead of returning values."""
    df = _synthetic([2, 3, 4, 5, 6], [1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5], [1, 2, 3, 4, 5])
    labels = np.arange(len(df))
    labels[3] = 1
    df.index = pd.Index(labels)
    assert not df.index.is_unique
    with pytest.raises(ValueError, match="duplicate labels"):
        ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=3)
    with pytest.raises(ValueError, match="duplicate labels"):
        compute_mfi(df, {"period": 3})


def test_vidya_public_falls_back_on_leading_nan() -> None:
    df = _synthetic(
        [2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6], [1.5, 2.5, 3.5, 4.5, 5.5, 6.5], [1] * 6
    )
    df.loc[:1, "close"] = np.nan
    expected = ta.vidya(df["close"], length=3)
    assert expected is not None
    _assert_bit_equal(compute_vidya(df, {"period": 3}).to_numpy(), expected.to_numpy(), "VIDYA lead nan")


# ---------------------------------------------------------------------------
# SQUEEZE_RATIO: same port pattern (rolling std + Wilder ATR)
# ---------------------------------------------------------------------------


def _lengths(name: str) -> list[int]:
    spec = indicator_registry.get(name)
    assert spec is not None and spec.param_ranges is not None
    lo, hi = (int(x) for x in spec.param_ranges["length"])
    span = hi - lo
    return sorted({lo, hi, lo + span // 4, lo + span // 2, hi - span // 4})


SQUEEZE_LENGTHS = _lengths("SQUEEZE_RATIO")
SQUEEZE_MULTS = ((2.0, 1.5), (1.5, 1.0), (2.5, 2.0))


@pytest.mark.parametrize("interval", ["4h", "1d"])
@pytest.mark.parametrize("length", SQUEEZE_LENGTHS)
@pytest.mark.parametrize("bb_mult,kc_mult", SQUEEZE_MULTS)
def test_squeeze_ratio_port_matches_library_on_full_catalog_history(
    interval: str, length: int, bb_mult: float, kc_mult: float
) -> None:
    df = _frame(_catalog_bars(interval))
    expected = _squeeze_ratio_library(df, length, bb_mult, kc_mult)
    got = _squeeze_ratio_port(df, length, bb_mult, kc_mult)
    _assert_series_equal(got, expected, f"SQUEEZE_RATIO {interval} length={length}")


@pytest.mark.parametrize("length", _edges_and_middle(SQUEEZE_LENGTHS))
def test_squeeze_ratio_port_matches_every_buffer_slice(length: int) -> None:
    rows = _frame(_catalog_bars("4h"), BUFFER_BARS)
    buf = PtaBuffer(400)
    for i in range(len(rows)):
        buf.append(*(float(rows[c].iat[i]) for c in OHLCV))
        if len(buf) < length:
            continue
        df = buf.frame()
        expected = _squeeze_ratio_library(df, length, 2.0, 1.5)
        got = _squeeze_ratio_port(df, length, 2.0, 1.5)
        _assert_bit_equal(got.to_numpy(), expected.to_numpy(), f"SQUEEZE n={len(df)} len={length}")


def test_squeeze_ratio_public_short_input_returns_nan_like() -> None:
    df = _synthetic([2, 3], [1, 2], [1.5, 2.5], [10, 20])
    out = compute_squeeze_ratio(df, {"length": 10})
    _assert_series_equal(out, nan_like(df), "SQUEEZE short input")


def test_squeeze_ratio_public_contract_on_two_unrelated_inputs() -> None:
    up = _synthetic([2, 3, 4, 5, 6], [1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5], [10] * 5)
    down = _synthetic([6, 5, 4, 3, 2], [5, 4, 3, 2, 1], [5.5, 4.5, 3.5, 2.5, 1.5], [10] * 5)
    for frame in (up, down):
        got = compute_squeeze_ratio(frame, {"length": 3, "bb_mult": 1.5, "kc_mult": 1.0})
        ref = _squeeze_ratio_library(frame, 3, 1.5, 1.0)
        _assert_series_equal(got, ref, "SQUEEZE public")


def test_squeeze_ratio_public_falls_back_to_library_on_non_float64() -> None:
    df = _synthetic([2, 3, 4, 5], [1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5], [10, 20, 30, 40])
    for col in ("high", "low", "close"):
        df[col] = df[col].astype("float32")
    got = compute_squeeze_ratio(df, {"length": 3})
    _assert_bit_equal(
        got.to_numpy(), _squeeze_ratio_library(df, 3, 2.0, 1.5).to_numpy(), "SQUEEZE f32"
    )


# ---------------------------------------------------------------------------
# Public plugin-parameter boundaries: 0 / negative / non-integer lengths must
# behave exactly like the library (same NaNs, same exception class + message).
# ---------------------------------------------------------------------------


def _ramp(n: int = 40) -> pd.DataFrame:
    close = np.arange(1.0, n + 1.0)
    return _synthetic(
        list(close + 1), list(close - 1), list(close), [1.0] * n
    ).assign(volume=np.arange(1.0, n + 1.0) % 7 + 1.0)


def _outcome(fn: Any) -> tuple[str, Any]:
    try:
        out = fn()
    except Exception as exc:
        return ("raise", (type(exc), str(exc)))
    return ("ok", None if out is None else out.to_numpy())


def _assert_same_outcome(got: tuple[str, Any], ref: tuple[str, Any], where: str) -> None:
    assert got[0] == ref[0], f"{where}: {got} vs {ref}"
    if got[0] == "raise":
        assert got[1] == ref[1], where
    elif ref[1] is None:
        assert got[1] is None, where
    else:
        _assert_bit_equal(got[1], ref[1], where)


BOUNDARY_LENGTHS = (0, -1, 3.7)


def _mfi_reference(df: pd.DataFrame, length: int) -> pd.Series | None:
    return ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=length)


@pytest.mark.parametrize("length", BOUNDARY_LENGTHS)
def test_mfi_public_boundary_lengths_match_library(length: float) -> None:
    for df in (_ramp(), _frame(_catalog_bars("4h"), 200)):
        ref = _outcome(lambda df=df: _mfi_reference(df, int(length)))
        got = _outcome(lambda df=df: compute_mfi(df, {"period": length}))
        _assert_same_outcome(got, ref, f"MFI period={length}")


@pytest.mark.parametrize("length", BOUNDARY_LENGTHS)
def test_vidya_public_boundary_lengths_match_library(length: float) -> None:
    for df in (_ramp(), _frame(_catalog_bars("4h"), 200)):
        ref = _outcome(lambda df=df: _vidya_library(df, int(length)))
        got = _outcome(lambda df=df: compute_vidya(df, {"period": length}))
        _assert_same_outcome(got, ref, f"VIDYA period={length}")


@pytest.mark.parametrize("length", BOUNDARY_LENGTHS)
def test_squeeze_ratio_public_boundary_lengths_match_library(length: float) -> None:
    spec = indicator_registry.get("SQUEEZE_RATIO")
    assert spec is not None
    for df in (_ramp(), _frame(_catalog_bars("4h"), 200)):
        ref = _outcome(lambda df=df: _squeeze_ratio_library(df, int(length), 2.0, 1.5))
        got = _outcome(lambda df=df: compute_squeeze_ratio(df, {"length": length}))
        _assert_same_outcome(got, ref, f"SQUEEZE direct length={length}")
        via = _outcome(lambda df=df: invoke_compute_fn(spec, df, {"length": length}))  # type: ignore[arg-type]
        _assert_same_outcome(via, ref, f"SQUEEZE invoke length={length}")


def test_squeeze_ratio_zero_length_is_not_defaulted_to_20() -> None:
    """Regression: length=0 once returned 20 finite values instead of the library's NaNs."""
    out = compute_squeeze_ratio(_ramp(), {"length": 0})
    ref = _squeeze_ratio_library(_ramp(), 0, 2.0, 1.5)
    _assert_bit_equal(out.to_numpy(), ref.to_numpy(), "SQUEEZE length=0")


def test_squeeze_ratio_port_requires_runtime_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform where the RMA port is not bit-identical must use the library."""
    import vibe_quant.dsl.plugins.squeeze as sq

    monkeypatch.setattr(sq, "_squeeze_ratio_port", lambda *a: pytest.fail("port used"))
    monkeypatch.setattr(sq, "_squeeze_port_ok", lambda: False)
    df = _ramp()
    _assert_bit_equal(
        sq.compute_squeeze_ratio(df, {"length": 5}).to_numpy(),
        _squeeze_ratio_library(df, 5, 2.0, 1.5).to_numpy(),
        "SQUEEZE guard fallback",
    )


def test_squeeze_port_self_check_passes_here() -> None:
    import vibe_quant.dsl.plugins.squeeze as sq

    sq._squeeze_port_ok.cache_clear()
    assert sq._squeeze_port_ok() is True
