"""Indicator tail memo through the COMPILED stream path (R4, R8, R20-R22).

The compiler emits ``memo_lookup`` / ``memo_store`` around every non-context
``compute_fn`` call, so these tests drive the real generated
``_feed_pta_buffer`` / ``_update_pta_indicators`` code: a memo hit must return
the exact scalar the uncached stream produced (bitwise, NaN included), must be
invalidated by a content change in ANY OHLCV column, and must never be used for
``needs_context`` specs (their outputs depend on ``df.attrs``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from tests.unit.pta_stream_harness import (
    compile_indicators,
    compile_source,
    random_walk_bars,
    run_stream,
)
from vibe_quant.dsl import indicator_memo
from vibe_quant.dsl.indicators import (
    indicator_registry,
    pta_buffer_cap,
    pta_lookback,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

BarEvent = tuple[str, float, float, float, float, float]


def _pta_candidates() -> list[str]:
    """Registry types that could reach the generated compute_fn buffer path."""
    names: list[str] = []
    for name in sorted(indicator_registry.list_indicators()):
        spec = indicator_registry.get(name)
        if spec is not None and spec.compute_fn is not None and not spec.needs_context:
            names.append(name)
    return names


_PTA_CANDIDATES = _pta_candidates()

# compute_fns whose pandas_ta call raises on a PRODUCTION context frame
# (``df.attrs["bar_close_ns"]`` is an int64 array and pandas' concat compares
# attrs dicts with ``==``). Both are NT-class specs on the compiled path.
_PANDAS_CONCAT_ATTRS_BREAK = frozenset({"ATR", "KC"})


def _compiled_or_skip(name: str) -> type:
    """Compile ``name`` alone; skip specs the compiler does not buffer."""
    try:
        _, cls = compile_indicators({"ind": {"type": name}})
    except Exception as exc:  # noqa: BLE001 — a registry spec may need aux context
        pytest.skip(f"{name}: does not compile as a PTA-path strategy ({type(exc).__name__})")
    if not hasattr(cls, "_feed_pta_buffer"):
        pytest.skip(f"{name}: compiles through the NT path, not the compute_fn buffer path")
    return cls


def _same(left: float, right: float) -> bool:
    both_nan = (
        isinstance(left, float)
        and isinstance(right, float)
        and math.isnan(left)
        and math.isnan(right)
    )
    return both_nan or left == right


def _assert_streams_equal(
    actual: Sequence[dict[str, float]], expected: Sequence[dict[str, float]], label: str
) -> None:
    assert len(actual) == len(expected), f"{label}: {len(actual)} != {len(expected)} bars"
    for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert list(got) == list(want), f"{label} bar {i}: keys {list(got)} != {list(want)}"
        for key, value in want.items():
            assert _same(got[key], value), f"{label} bar {i} {key}: {got[key]!r} != {value!r}"


def _assert_key_equal(
    actual: Sequence[dict[str, float]],
    expected: Sequence[dict[str, float]],
    key: str,
    label: str,
) -> None:
    assert len(actual) == len(expected)
    for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert _same(got.get(key, math.nan), want.get(key, math.nan)), (
            f"{label} bar {i} {key}: {got.get(key)!r} != {want.get(key)!r}"
        )


def _count(snap: dict[str, Any], key: str) -> int:
    return int(snap[key])


def _delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int:
    return _count(after, key) - _count(before, key)


# ---------------------------------------------------------------------------
# R4: caps that hold the same bars share a hit; equal length is not enough.
# ---------------------------------------------------------------------------


def test_cross_cap_hits_only_on_equal_content() -> None:
    """Cap 400 warmed, then cap 500: only bars 13..499 carry identical content."""
    bars = random_walk_bars(1500, seed=13)
    _, cls_a = compile_indicators({"w": {"type": "WILLR", "period": 14}})
    _, cls_b = compile_indicators(
        {"w": {"type": "WILLR", "period": 14}, "w50": {"type": "WILLR", "period": 50}}
    )
    _, cls_w50 = compile_indicators({"w50": {"type": "WILLR", "period": 50}})
    specs_a: dict[str, tuple[str, dict[str, object], str]] = {
        "w": ("WILLR", {"period": 14}, "1h")
    }
    specs_b: dict[str, tuple[str, dict[str, object], str]] = {
        "w": ("WILLR", {"period": 14}, "1h"),
        "w50": ("WILLR", {"period": 50}, "1h"),
    }
    assert pta_buffer_cap([pta_lookback("WILLR", {"period": 14})], full_history=False) == 400
    assert pta_buffer_cap([pta_lookback("WILLR", {"period": 50})], full_history=False) == 500

    # Memo-off reference for B, on a clean store.
    indicator_memo.clear()
    off_b = run_stream(cls_b, bars, memo=False, specs=specs_b)

    # Warm the store with A's capped stream, then replay B on top of it.
    indicator_memo.clear()
    run_stream(cls_a, bars, memo=False, ind_memo=True, specs=specs_a)
    before = indicator_memo.snapshot()
    on_b = run_stream(cls_b, bars, memo=False, ind_memo=True, specs=specs_b)
    after_b = indicator_memo.snapshot()
    total_hits = _delta(before, after_b, "hits")
    total_misses = _delta(before, after_b, "misses")

    _assert_streams_equal(on_b, off_b, "B memo-on")

    # w50 shares no params bucket with the warm stream: 0 hits over 1451 calls.
    indicator_memo.clear()
    run_stream(cls_a, bars, memo=False, ind_memo=True, specs=specs_a)
    before_w50 = indicator_memo.snapshot()
    on_w50 = run_stream(cls_w50, bars, memo=False, ind_memo=True, specs={"w50": specs_b["w50"]})
    after_w50 = indicator_memo.snapshot()
    w50_hits = _delta(before_w50, after_w50, "hits")
    w50_misses = _delta(before_w50, after_w50, "misses")

    assert (w50_hits, w50_misses) == (0, 1451)
    assert (total_hits - w50_hits, total_misses - w50_misses) == (487, 1000)
    _assert_key_equal(on_w50, off_b, "w50", "B w50 memo-on")


# ---------------------------------------------------------------------------
# R8: a content change in ANY column must miss from the changed bar on.
# ---------------------------------------------------------------------------


def _shift_column(
    bars: Sequence[BarEvent], column: str, start: int, delta: float
) -> list[BarEvent]:
    """Same bars, except ``column`` differs from bar ``start`` on."""
    index = ("open", "high", "low", "close", "volume").index(column) + 1
    out: list[BarEvent] = []
    for i, bar in enumerate(bars):
        if i < start:
            out.append(bar)
            continue
        values = [float(value) for value in bar[1:]]
        values[index - 1] += delta
        out.append(
            (
                bar[0],
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
            )
        )
    return out


@pytest.mark.parametrize(
    ("column", "ind_type", "delta"),
    [("high", "WILLR", 1.25), ("low", "WILLR", -1.25), ("volume", "MFI", 1.25)],
)
def test_column_difference_misses(column: str, ind_type: str, delta: float) -> None:
    """Hits stop at the first bar whose window touches the changed column."""
    spec = indicator_registry.get(ind_type)
    assert spec is not None and spec.compute_fn is not None
    params = dict(spec.default_params)
    lookback = pta_lookback(spec, params)
    _, cls = compile_indicators({"ind": {"type": ind_type}})
    assert hasattr(cls, "_feed_pta_buffer")

    base = random_walk_bars(1500, seed=13)
    changed = _shift_column(base, column, start=100, delta=delta)
    specs = {"ind": (ind_type, params, "1h")}

    indicator_memo.clear()
    run_stream(cls, base, memo=False, ind_memo=True, specs=specs)
    before = indicator_memo.snapshot()
    on_changed = run_stream(cls, changed, memo=False, ind_memo=True, specs=specs)
    after = indicator_memo.snapshot()

    assert _delta(before, after, "hits") == 101 - lookback

    indicator_memo.clear()
    off_changed = run_stream(cls, changed, memo=False, specs=specs)
    _assert_streams_equal(on_changed, off_changed, f"{ind_type} {column} changed")


# ---------------------------------------------------------------------------
# R20: every registry compute spec, memo on (cold then warm) == memo off.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _PTA_CANDIDATES)
def test_registry_memo_on_equals_off(name: str) -> None:
    spec = indicator_registry.get(name)
    assert spec is not None and spec.compute_fn is not None
    params = dict(spec.default_params)
    cls = _compiled_or_skip(name)
    bars = random_walk_bars(1500, seed=13)
    specs = {"ind": (name, params, "1h")}
    cap = pta_buffer_cap([pta_lookback(spec, params)], full_history=spec.requires_full_history)

    indicator_memo.clear()
    off = run_stream(cls, bars, memo=False, specs=specs)
    cold = run_stream(cls, bars, memo=False, ind_memo=True, specs=specs)
    after_cold = indicator_memo.snapshot()
    warm = run_stream(cls, bars, memo=False, ind_memo=True, specs=specs)
    after_warm = indicator_memo.snapshot()

    _assert_streams_equal(cold, off, f"{name} cold")
    _assert_streams_equal(warm, off, f"{name} warm")

    calls = _count(after_cold, "hits") + _count(after_cold, "misses")
    if cap == 0:
        # Unbounded buffers bypass the memo entirely (R6): no lookups at all.
        assert calls == 0
        assert _count(after_cold, "bypassed") > 0
    else:
        assert calls > 0, f"{name}: the memo took no lookups"
    assert _delta(after_cold, after_warm, "misses") == 0
    assert _delta(after_cold, after_warm, "hits") == calls


# ---------------------------------------------------------------------------
# R21: needs_context specs keep today's emission and never memoise.
# ---------------------------------------------------------------------------


def test_context_specs_never_memoised() -> None:
    src = compile_source(
        {"funding": {"type": "FUNDING"}, "willr": {"type": "WILLR", "period": 14}}
    )
    lines = src.splitlines()
    marker = next(
        i
        for i, line in enumerate(lines)
        if "via compute_funding" in line and "default lookback" in line
    )
    funding_block = lines[marker : marker + 9]
    assert not any("memo_lookup(compute_funding" in line for line in funding_block)
    assert not any("memo_store(compute_funding" in line for line in funding_block)

    # The context call path is exactly the pre-memo emission.
    assert funding_block == [
        "            # funding (FUNDING) via compute_funding — default lookback 1",
        '            if _n >= self._pta_lookback["funding"]:',
        "                if _df is None:",
        "                    _df = _buf.frame()",
        '                    _df.attrs["symbol"] = self.config.instrument_id',
        '                _res = compute_funding(_df, self._pta_params["funding"])',
        "                if _res is not None and len(_res) > 0:",
        "                    _v = _res.iloc[-1]",
        '                    self._pta_values["funding"] = float(_v)',
    ]

    # The plain spec next to it still gets the memo calls.
    assert 'memo_lookup(compute_willr, _buf, self._pta_params["willr"])' in src
    assert 'memo_store(compute_willr, _buf, self._pta_params["willr"], _res)' in src


def _indicator_block(src: str, fn_name: str) -> list[str]:
    """Generated lines of one compute_fn block: its comment through its last body line."""
    lines = src.splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if f"via {fn_name}" in line and "default lookback" in line
    )
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if ("via compute_" in lines[i] and "default lookback" in lines[i])
            or (lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) < indent)
        ),
        len(lines),
    )
    return lines[start:end]


def test_memo_miss_sets_symbol_for_later_context_spec() -> None:
    """A non-context miss builds ``_df`` first, so it must set the symbol attr.

    ``_df`` is shared by every compute_fn of a timeframe. With WILLR declared
    before FUNDING, a WILLR memo miss builds ``_df`` and FUNDING then reads
    ``df.attrs["symbol"]`` in ``aux_data.context_of``; without the assignment a
    mixed strategy raises a bare ``ValueError`` on the first miss.
    """
    src = compile_source(
        {"willr": {"type": "WILLR", "period": 14}, "funding": {"type": "FUNDING"}}
    )
    assert src.index("# willr (WILLR) via compute_willr") < src.index(
        "# funding (FUNDING) via compute_funding"
    )

    assert _indicator_block(src, "compute_willr")[:9] == [
        "            # willr (WILLR) via compute_willr — default lookback 14",
        '            if _n >= self._pta_lookback["willr"]:',
        '                _res = memo_lookup(compute_willr, _buf, self._pta_params["willr"])',
        "                if _res is MEMO_MISS:",
        "                    if _df is None:",
        "                        _df = _buf.frame()",
        '                        _df.attrs["symbol"] = self.config.instrument_id',
        '                    _res = compute_willr(_df, self._pta_params["willr"])',
        '                    memo_store(compute_willr, _buf, self._pta_params["willr"], _res)',
    ]

    # Non-context specs must not touch df.attrs when no context spec shares _df.
    solo = compile_source({"willr": {"type": "WILLR", "period": 14}})
    assert not any("_df.attrs[" in line for line in solo.splitlines())


# ---------------------------------------------------------------------------
# R22: every compute_fn is a pure function of df's values and params.
# ---------------------------------------------------------------------------


def _ohlcv(bars: Sequence[BarEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [b[1] for b in bars],
            "high": [b[2] for b in bars],
            "low": [b[3] for b in bars],
            "close": [b[4] for b in bars],
            "volume": [b[5] for b in bars],
        }
    )


def _tail(out: object) -> object:
    if out is None:
        return None
    if isinstance(out, dict):
        return {key: _tail(value) for key, value in out.items()}
    assert isinstance(out, pd.Series)
    assert len(out) > 0
    return float(out.iloc[-1])


def _assert_tails_equal(left: object, right: object, label: str) -> None:
    if isinstance(left, dict):
        assert isinstance(right, dict), label
        assert list(left) == list(right), label
        for key in left:
            _assert_tails_equal(left[key], right[key], f"{label}.{key}")
        return
    if left is None or right is None:
        assert left is None and right is None, label
        return
    assert isinstance(left, float) and isinstance(right, float), label
    assert _same(left, right), f"{label}: tail {left!r} != {right!r}"


@pytest.mark.parametrize("name", _PTA_CANDIDATES)
def test_compute_fns_do_not_mutate_inputs(name: str) -> None:
    spec = indicator_registry.get(name)
    assert spec is not None and spec.compute_fn is not None
    fn = spec.compute_fn
    bars = random_walk_bars(600, seed=23)

    df = _ohlcv(bars)
    columns = list(df.columns)
    values = {column: df[column].to_numpy(dtype=np.float64, copy=True) for column in columns}
    params = dict(spec.default_params)
    params_before = dict(params)

    _tail(fn(df, params))

    assert list(df.columns) == columns
    for column in columns:
        np.testing.assert_array_equal(
            df[column].to_numpy(dtype=np.float64), values[column], err_msg=f"{name}.{column}"
        )
    assert dict(df.attrs) == {}, f"{name} read/added df.attrs"
    assert params == params_before and list(params) == list(params_before)


def _attrs_params() -> list[Any]:
    return [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True,
                reason="pandas concat compares attrs dicts with `==` and the production "
                "bar_close_ns attr is an int64 array (ValueError inside pandas_ta "
                "true_range); ATR/KC are NT-class specs, so their compute_fn is not on "
                "the ctx+pta memo path",
            ),
        )
        if name in _PANDAS_CONCAT_ATTRS_BREAK
        else pytest.param(name)
        for name in _PTA_CANDIDATES
    ]


@pytest.mark.parametrize("name", _attrs_params())
def test_compute_fns_ignore_df_attrs(name: str) -> None:
    """The tail must not depend on df.attrs: only needs_context specs may read them."""
    spec = indicator_registry.get(name)
    assert spec is not None and spec.compute_fn is not None
    fn = spec.compute_fn
    bars = random_walk_bars(600, seed=23)
    params = dict(spec.default_params)

    plain = _tail(fn(_ohlcv(bars), params))

    with_attrs = _ohlcv(bars)
    with_attrs.attrs["symbol"] = "X"
    with_attrs.attrs["bar_close_ns"] = np.arange(len(bars), dtype=np.int64)
    _assert_tails_equal(plain, _tail(fn(with_attrs, params)), name)
