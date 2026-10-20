"""``indicator_memo``: keying, tail wrapping, byte bound, switch-off (T1).

Row ids (design-memo.md § 6) are named in each docstring so the contract and
the test suite stay one list.
"""

from __future__ import annotations

import resource
import sys
import tracemalloc
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl import indicator_memo
from vibe_quant.dsl.compute_builtins import compute_rsi, compute_willr
from vibe_quant.dsl.indicators import pta_lookback
from vibe_quant.dsl.pta_buffer import PtaBuffer

if TYPE_CHECKING:
    from collections.abc import Iterator

SNAPSHOT_KEYS = {
    "proc_id",
    "pid",
    "enabled",
    "state",
    "max_bytes",
    "entries",
    "est_bytes",
    "hits",
    "misses",
    "bypassed",
    "evictions",
    "sessions",
    "switch_offs",
    "peak_rss_bytes",
    "rss_at_first_session_bytes",
}


@pytest.fixture(autouse=True)
def _clean_memo(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("VIBE_QUANT_IND_MEMO", "1")
    monkeypatch.delenv("VIBE_QUANT_IND_MEMO_MAX_MB", raising=False)
    indicator_memo.refresh_enabled()
    indicator_memo.clear()
    yield
    indicator_memo.clear()


def _bars(n: int, seed: int = 13) -> list[tuple[float, float, float, float, float]]:
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    return [
        (float(close[i]), float(close[i]) + 0.2, float(close[i]) - 0.2, float(close[i]), 1.0)
        for i in range(n)
    ]


def _bits(values: list[object]) -> list[bytes]:
    return [np.asarray(v, dtype=np.float64).tobytes() for v in values]


class _Buf:
    """Duck-typed ``PtaBuffer`` (``cap`` + ``fingerprint()``) for non-stream rows."""

    def __init__(self, cap: int = 400, fp: bytes = b"fp") -> None:
        self.cap = cap
        self._fp = fp
        self.fp_calls = 0

    def fingerprint(self) -> bytes:
        self.fp_calls += 1
        return self._fp


def _obj_series(*values: object) -> pd.Series:
    return pd.Series(list(values), dtype=object)


def _willr_stream(
    buf: PtaBuffer,
    bars: list[tuple[float, float, float, float, float]],
    params: dict[str, object],
    *,
    memoised: bool,
) -> list[object]:
    lookback = pta_lookback("WILLR", params)
    out: list[object] = []
    for bar in bars:
        buf.append(*bar)
        if len(buf) < lookback:
            continue
        res: object
        if memoised:
            res = indicator_memo.memo_lookup(compute_willr, buf, params)
            if res is indicator_memo.MEMO_MISS:
                res = compute_willr(buf.frame(), params)
                indicator_memo.memo_store(compute_willr, buf, params, res)
        else:
            res = compute_willr(buf.frame(), params)
        out.append(res.iloc[-1])  # type: ignore[attr-defined]
    return out


def test_rerun_same_genome_all_hits() -> None:
    """R1: same fn + params + content: run 1 all misses, run 2 all hits, tails bitwise equal."""
    params = {"period": 14}
    bars = _bars(1500)
    assert pta_lookback("WILLR", params) == 14
    expected = _willr_stream(PtaBuffer(400), bars, params, memoised=False)

    with indicator_memo.session():
        first = _willr_stream(PtaBuffer(400), bars, params, memoised=True)
        snap = indicator_memo.snapshot()
        assert (snap["hits"], snap["misses"], snap["entries"]) == (0, 1487, 1487)

        second = _willr_stream(PtaBuffer(400), bars, params, memoised=True)
        snap = indicator_memo.snapshot()
        assert (snap["hits"], snap["misses"]) == (1487, 1487)

    assert _bits(first) == _bits(expected)
    assert _bits(second) == _bits(expected)


def test_params_repr_separates_keys() -> None:
    """R2: 14, 14.0 and 15 are three different keys (repr, not ``==``)."""
    bars = _bars(40)
    with indicator_memo.session():
        buf = PtaBuffer(400)
        for bar in bars:
            buf.append(*bar)
        assert indicator_memo.memo_lookup(compute_willr, buf, {"period": 14}) is (
            indicator_memo.MEMO_MISS
        )
        res = compute_willr(buf.frame(), {"period": 14})
        indicator_memo.memo_store(compute_willr, buf, {"period": 14}, res)
        assert indicator_memo.memo_lookup(compute_willr, buf, {"period": 14}) is not (
            indicator_memo.MEMO_MISS
        )
        assert (
            indicator_memo.memo_lookup(compute_willr, buf, {"period": 14.0})
            is indicator_memo.MEMO_MISS
        )
        assert (
            indicator_memo.memo_lookup(compute_willr, buf, {"period": 15})
            is indicator_memo.MEMO_MISS
        )


def test_fn_identity_separates_keys() -> None:
    """R3: the function OBJECT is part of the key; a second fn misses."""
    params = {"period": 14}
    bars = _bars(60)
    with indicator_memo.session():
        buf = PtaBuffer(400)
        for bar in bars:
            buf.append(*bar)
        willr = compute_willr(buf.frame(), params)
        indicator_memo.memo_store(compute_willr, buf, params, willr)
        assert indicator_memo.memo_lookup(compute_rsi, buf, params) is indicator_memo.MEMO_MISS
        rsi = compute_rsi(buf.frame(), params)
        indicator_memo.memo_store(compute_rsi, buf, params, rsi)

        hit_w = indicator_memo.memo_lookup(compute_willr, buf, params)
        hit_r = indicator_memo.memo_lookup(compute_rsi, buf, params)

    assert _bits([hit_w.iloc[-1]]) == _bits([willr.iloc[-1]])  # type: ignore[attr-defined]
    assert _bits([hit_r.iloc[-1]]) == _bits([rsi.iloc[-1]])  # type: ignore[attr-defined]
    assert float(hit_w.iloc[-1]) != float(hit_r.iloc[-1])  # type: ignore[attr-defined]


def test_cap_zero_bypasses() -> None:
    """R6: an unbounded buffer never hashes and never stores."""
    buf = _Buf(cap=0)

    def fn(_df: object, _params: object) -> pd.Series:
        return _obj_series(1.0, 2.0)

    with indicator_memo.session():
        assert indicator_memo.memo_lookup(fn, buf, {}) is indicator_memo.MEMO_MISS
        indicator_memo.memo_store(fn, buf, {}, fn(None, {}))
        snap = indicator_memo.snapshot()

    assert buf.fp_calls == 0
    assert snap["bypassed"] == 1
    assert snap["entries"] == 0
    assert snap["hits"] == snap["misses"] == 0


def test_result_shapes_round_trip() -> None:
    """R7: only Series/dict-of-Series/None are stored, and the tail object is reused."""
    scalar = 1.5
    nan_obj = float("nan")
    tail_a = 2.0
    one = _obj_series(1.0, scalar)
    with_nan = _obj_series(1.0, nan_obj)
    empty = pd.Series([], dtype=float)
    multi = {"a": _obj_series(0.0, tail_a), "b": None, "c": empty}
    bad_dict = {"a": 1.0}
    array = np.array([1.0, 2.0])

    seq = iter(range(100))
    with indicator_memo.session():

        def fresh(buf_fp: bytes = b"") -> _Buf:
            return _Buf(fp=b"fp-%d" % next(seq))

        hit = _round_trip(lambda _df, _p: one, fresh(), indicator_memo)
        assert hit.iloc[-1] is scalar  # type: ignore[attr-defined]
        assert len(hit) == 1  # type: ignore[arg-type]

        hit = _round_trip(lambda _df, _p: with_nan, fresh(), indicator_memo)
        assert hit.iloc[-1] is nan_obj  # type: ignore[attr-defined]

        hit = _round_trip(lambda _df, _p: empty, fresh(), indicator_memo)
        assert len(hit) == 0  # type: ignore[arg-type]

        hit = _round_trip(lambda _df, _p: None, fresh(), indicator_memo)
        assert hit is None

        hit = _round_trip(lambda _df, _p: multi, fresh(), indicator_memo)
        assert list(hit) == ["a", "b", "c"]  # type: ignore[arg-type]
        assert hit["a"].iloc[-1] is tail_a  # type: ignore[index]
        assert hit["b"] is None  # type: ignore[index]
        assert len(hit["c"]) == 0  # type: ignore[index]

        for not_stored in (bad_dict, array):
            bypassed_before = indicator_memo.snapshot()["bypassed"]
            fn = lambda _df, _p, obj=not_stored: obj  # noqa: E731
            buf = fresh()
            assert indicator_memo.memo_lookup(fn, buf, {}) is indicator_memo.MEMO_MISS
            original = fn(None, {})
            assert indicator_memo.memo_store(fn, buf, {}, original) is None
            assert fn(None, {}) is original  # the caller's result is untouched
            assert indicator_memo.memo_lookup(fn, buf, {}) is indicator_memo.MEMO_MISS
            assert indicator_memo.snapshot()["bypassed"] == bypassed_before + 1


def _round_trip(fn: Any, buf: _Buf, module: Any) -> Any:
    assert module.memo_lookup(fn, buf, {}) is module.MEMO_MISS
    res = fn(None, {})
    module.memo_store(fn, buf, {}, res)
    return module.memo_lookup(fn, buf, {})


def test_tail_scalars_keep_their_numpy_type() -> None:
    """R7b: a numpy scalar tail comes back as the very same object, never ``float(x)``."""
    f64 = np.float64(1.5)
    f32 = np.float32(2.5)
    seq = iter(range(100))

    def fresh() -> _Buf:
        return _Buf(fp=b"np-%d" % next(seq))

    with indicator_memo.session():
        hit = _round_trip(lambda _df, _p: _obj_series(0.0, f64), fresh(), indicator_memo)
        assert hit.iloc[-1] is f64
        assert type(hit.iloc[-1]) is np.float64

        hit = _round_trip(lambda _df, _p: _obj_series(0.0, f32), fresh(), indicator_memo)
        assert hit.iloc[-1] is f32
        assert type(hit.iloc[-1]) is np.float32

        multi = {"a": _obj_series(0.0, f32), "b": _obj_series(0.0, f64)}
        hit = _round_trip(lambda _df, _p: multi, fresh(), indicator_memo)
        assert hit["a"].iloc[-1] is f32
        assert hit["b"].iloc[-1] is f64


def test_raise_is_not_cached() -> None:
    """R9: an exception propagates untouched, stores nothing, and is raised again."""
    err = ValueError("x")

    def boom(_df: object, _params: object) -> pd.Series:
        raise err

    buf = _Buf()
    with indicator_memo.session():
        assert indicator_memo.memo_lookup(boom, buf, {}) is indicator_memo.MEMO_MISS
        with pytest.raises(ValueError) as excinfo:
            boom(None, {})  # the generated code's compute step; memo_store never runs
        assert excinfo.value is err
        assert indicator_memo.snapshot()["entries"] == 0

        assert indicator_memo.memo_lookup(boom, buf, {}) is indicator_memo.MEMO_MISS
        with pytest.raises(ValueError) as excinfo:
            boom(None, {})
        assert excinfo.value is err
        assert indicator_memo.snapshot()["entries"] == 0


def test_inactive_outside_session() -> None:
    """R10: no session -> no hashing, no stores, no counters (paper/validation/screening)."""
    buf = _Buf()

    def fn(_df: object, _params: object) -> pd.Series:
        return _obj_series(1.0)

    assert indicator_memo.is_active() is False
    assert indicator_memo.memo_lookup(fn, buf, {}) is indicator_memo.MEMO_MISS
    indicator_memo.memo_store(fn, buf, {}, fn(None, {}))
    snap = indicator_memo.snapshot()
    assert buf.fp_calls == 0
    assert snap["entries"] == 0
    assert snap["hits"] + snap["misses"] == 0
    assert snap["state"] == "on"  # configured on, but only a session activates it


def test_kill_switch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """R11: VIBE_QUANT_IND_MEMO=0 disables; any other value keeps it on."""
    monkeypatch.setenv("VIBE_QUANT_IND_MEMO", "0")
    indicator_memo.refresh_enabled()
    assert indicator_memo.snapshot()["state"] == "disabled"
    with indicator_memo.session():
        assert indicator_memo.is_active() is False
        assert indicator_memo.memo_lookup(compute_willr, _Buf(), {}) is (indicator_memo.MEMO_MISS)
    assert indicator_memo.snapshot()["sessions"] == 0  # a no-op, not a session

    for raw in ("1", "", "false"):
        monkeypatch.setenv("VIBE_QUANT_IND_MEMO", raw)
        indicator_memo.refresh_enabled()
        assert indicator_memo.snapshot()["enabled"] is True, raw
        with indicator_memo.session():
            assert indicator_memo.is_active() is True, raw
        assert indicator_memo.snapshot()["state"] == "on", raw


def test_two_generation_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    """R12: two generations at a 600-byte cap, promotion on an old-generation hit."""
    monkeypatch.setattr(indicator_memo, "_max_bytes_from_env", lambda: 600)

    def fn(_df: object, params: dict[str, object]) -> pd.Series:
        return _obj_series(float(params["k"]))

    buf = _Buf()

    def bucket(k: int) -> object:
        return (fn, (("k", repr(k)),))

    def store(k: int) -> None:
        indicator_memo.memo_store(fn, buf, {"k": k}, fn(None, {"k": k}))

    with indicator_memo.session():
        store(1)
        store(2)
        store(3)
        snap = indicator_memo.snapshot()
        assert set(indicator_memo._young) == {bucket(3)}
        assert set(indicator_memo._old) == {bucket(1), bucket(2)}
        assert snap["entries"] == 3
        assert snap["evictions"] == 0
        assert snap["est_bytes"] <= 750

        store(4)
        store(5)
        snap = indicator_memo.snapshot()
        assert snap["evictions"] == 2
        assert snap["entries"] == 3
        assert snap["est_bytes"] <= 750

        assert indicator_memo.memo_lookup(fn, buf, {"k": 3}) is not (indicator_memo.MEMO_MISS)
        snap = indicator_memo.snapshot()
        assert snap["hits"] == 1
        assert set(indicator_memo._young) == {bucket(5), bucket(3)}
        assert set(indicator_memo._old) == {bucket(4)}

        store(6)
        snap = indicator_memo.snapshot()
        assert snap["evictions"] == 3
        assert snap["entries"] == 3
        assert snap["est_bytes"] <= 750

        assert indicator_memo.memo_lookup(fn, buf, {"k": 1}) is indicator_memo.MEMO_MISS
        snap = indicator_memo.snapshot()
        assert snap["entries"] == 3
        assert snap["est_bytes"] <= 750


def _store_all(fn: Any, fps: list[bytes]) -> None:
    buf = _Buf()
    for fp in fps:
        buf._fp = fp
        indicator_memo.memo_store(fn, buf, {}, fn(None, {}))


def _single(_df: object, _params: object) -> pd.Series:
    return _obj_series(1.0)


def _dict_one(_df: object, _params: object) -> dict[str, pd.Series]:
    return {"a": _obj_series(1.0)}  # 150 + 105 = 255 estimated bytes


def test_rotation_uses_projected_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """R12b: rotate BEFORE the insert that would push young past cap // 2 (non-divisible)."""
    monkeypatch.setattr(indicator_memo, "_max_bytes_from_env", lambda: 650)  # half = 325
    with indicator_memo.session():
        _store_all(_single, [b"a", b"b"])
        assert indicator_memo._young_bytes == 300  # 300 <= 325: no rotation
        _store_all(_single, [b"c"])  # 450 > 325: rotate first
        snap = indicator_memo.snapshot()
        assert (indicator_memo._young_bytes, indicator_memo._old_bytes) == (150, 300)
        assert snap["est_bytes"] == 450
        assert snap["entries"] == 3


def test_rotation_equality_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    """R12c: young + new == cap // 2 stays; one byte over rotates (mixed entry sizes)."""
    for cap, rotated in ((810, False), (808, True)):  # half = 405 and 404
        indicator_memo.clear()
        monkeypatch.setattr(indicator_memo, "_max_bytes_from_env", lambda cap=cap: cap)
        with indicator_memo.session():
            _store_all(_single, [b"a"])
            _store_all(_dict_one, [b"b"])  # 150 + 255 = 405
            assert (indicator_memo._old_bytes > 0) is rotated, cap
            assert indicator_memo._young_bytes == (255 if rotated else 405), cap


@pytest.mark.parametrize("mixed", [False, True], ids=["single", "mixed"])
def test_bytes_never_exceed_cap_at_one_mb(monkeypatch: pytest.MonkeyPatch, mixed: bool) -> None:
    """R19c: at VIBE_QUANT_IND_MEMO_MAX_MB=1 neither generation passes half the cap."""
    monkeypatch.setenv("VIBE_QUANT_IND_MEMO_MAX_MB", "1")
    fns = [_single, _dict_one] if mixed else [_single]
    with indicator_memo.session():
        for k in range(8000):
            _store_all(fns[k % len(fns)], [k.to_bytes(4, "little")])
            assert indicator_memo._young_bytes <= 500_000, k
            assert indicator_memo._old_bytes <= 500_000, k
            assert indicator_memo.snapshot()["est_bytes"] <= 1_000_000, k
        assert indicator_memo.snapshot()["evictions"] > 0


def test_evictions_count_dropped_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """R12d: ``evictions`` = entries in the dropped generation, not its buckets."""
    monkeypatch.setattr(indicator_memo, "_max_bytes_from_env", lambda: 600)
    with indicator_memo.session():
        _store_all(_single, [b"a", b"b", b"c", b"d"])  # old {a, b} in ONE bucket
        assert indicator_memo.snapshot()["evictions"] == 0
        _store_all(_single, [b"e"])  # drops the old generation: 2 entries, 1 bucket
        snap = indicator_memo.snapshot()
        assert snap["evictions"] == 2
        assert snap["entries"] == 3
        assert indicator_memo.run_summary()["evictions"] == 2


@contextmanager
def _judged(*, judge: bool = True, hits: int = 0, misses: int = 0) -> Iterator[None]:
    """A session whose per-session counters are set directly (state-machine rows)."""
    with indicator_memo.session(judge=judge):
        indicator_memo._hits += hits
        indicator_memo._misses += misses
        yield


def test_switch_off_threshold_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """R13: exactly MIN_RATE stays ON (the rule is strict ``<``)."""
    monkeypatch.setattr(indicator_memo, "WINDOW", 4)
    monkeypatch.setattr(indicator_memo, "MIN_RATE", 0.10)
    for _ in range(4):
        with _judged(hits=1, misses=9):
            pass
    snap = indicator_memo.snapshot()
    assert snap["state"] == "on"
    assert snap["switch_offs"] == 0
    assert len(indicator_memo._window) == 4


def test_switch_off_and_reprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """R14: below MIN_RATE switches off for OFF_INITIAL sessions, then re-probes."""
    monkeypatch.setattr(indicator_memo, "WINDOW", 4)
    monkeypatch.setattr(indicator_memo, "MIN_RATE", 0.10)
    monkeypatch.setattr(indicator_memo, "OFF_INITIAL", 2)
    for _ in range(4):
        with _judged(hits=1, misses=9):
            pass
    with _judged(hits=0, misses=10):  # 3/40 = 0.075
        pass
    snap = indicator_memo.snapshot()
    assert snap["state"] == "off"
    assert snap["switch_offs"] == 1
    assert indicator_memo._off_left == 2
    assert len(indicator_memo._window) == 0

    for _ in range(2):
        buf = _Buf()
        with indicator_memo.session():  # judge is irrelevant while OFF
            assert indicator_memo.is_active() is False
            assert indicator_memo.memo_lookup(compute_willr, buf, {"period": 14}) is (
                indicator_memo.MEMO_MISS
            )
            assert indicator_memo.snapshot()["state"] == "off"
        assert buf.fp_calls == 0

    assert indicator_memo.snapshot()["state"] == "on"
    assert len(indicator_memo._window) == 0


def test_switch_off_backoff_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """R15: OFF doubles up to OFF_MAX; a full window >= MIN_RATE resets to OFF_INITIAL."""
    monkeypatch.setattr(indicator_memo, "WINDOW", 4)
    monkeypatch.setattr(indicator_memo, "MIN_RATE", 0.10)
    monkeypatch.setattr(indicator_memo, "OFF_INITIAL", 2)
    monkeypatch.setattr(indicator_memo, "OFF_MAX", 8)

    def zero_window() -> None:
        for _ in range(4):
            with _judged(hits=0, misses=10):
                pass

    zero_window()
    assert (indicator_memo._off_left, indicator_memo._backoff) == (2, 4)
    for _ in range(2):
        with indicator_memo.session():
            pass
    assert indicator_memo.snapshot()["state"] == "on"

    zero_window()
    assert (indicator_memo._off_left, indicator_memo._backoff) == (4, 8)
    for _ in range(4):
        with indicator_memo.session():
            pass

    zero_window()
    assert (indicator_memo._off_left, indicator_memo._backoff) == (8, 8)
    for _ in range(8):
        with indicator_memo.session():
            pass

    zero_window()
    assert (indicator_memo._off_left, indicator_memo._backoff) == (8, 8)  # capped
    for _ in range(8):
        with indicator_memo.session():
            pass

    for _ in range(4):
        with _judged(hits=1, misses=9):
            pass
    assert indicator_memo._backoff == 2  # OFF_INITIAL, after a healthy window
    assert indicator_memo.snapshot()["state"] == "on"


def _switch_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(indicator_memo, "WINDOW", 4)
    monkeypatch.setattr(indicator_memo, "MIN_RATE", 0.10)
    monkeypatch.setattr(indicator_memo, "OFF_INITIAL", 2)
    for _ in range(4):
        with _judged(hits=0, misses=10):
            pass
    assert indicator_memo.snapshot()["state"] == "off"


def test_cap_env_validated_while_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """R17b: a cap changed to garbage while OFF still raises at session() entry."""
    _switch_off(monkeypatch)
    monkeypatch.setenv("VIBE_QUANT_IND_MEMO_MAX_MB", "abc")
    with pytest.raises(ValueError) as excinfo, indicator_memo.session():
        pytest.fail("session body must not run")
    assert str(excinfo.value) == ("VIBE_QUANT_IND_MEMO_MAX_MB must be an integer >= 1, got 'abc'")
    assert indicator_memo.is_active() is False
    assert indicator_memo.snapshot()["state"] == "off"

    monkeypatch.setenv("VIBE_QUANT_IND_MEMO_MAX_MB", "8")
    ran = False
    with indicator_memo.session():
        ran = True
        assert indicator_memo.is_active() is False  # still OFF
    assert ran
    assert indicator_memo.snapshot()["max_bytes"] == 8_000_000
    assert indicator_memo._off_left == 1


def test_cap_env_not_parsed_when_disabled_or_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    """R17c: disabled and nested sessions stay no-ops: a bad cap does not raise there."""
    with indicator_memo.session():
        monkeypatch.setenv("VIBE_QUANT_IND_MEMO_MAX_MB", "abc")
        with indicator_memo.session():  # nested
            assert indicator_memo.is_active() is True
    monkeypatch.setenv("VIBE_QUANT_IND_MEMO", "0")
    indicator_memo.refresh_enabled()
    with indicator_memo.session():
        assert indicator_memo.is_active() is False


def test_failed_off_sessions_advance_reprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """F4: a runner exception inside an OFF session still counts it down, unchanged."""
    _switch_off(monkeypatch)
    sessions = indicator_memo.snapshot()["sessions"]
    errors = [OSError("first"), OSError("second")]
    for err in errors:
        with pytest.raises(OSError) as excinfo, indicator_memo.session():
            raise err
        assert excinfo.value is err
        assert indicator_memo.is_active() is False
        if err is errors[0]:
            assert indicator_memo._off_left == 1
    snap = indicator_memo.snapshot()
    assert snap["sessions"] == sessions + 2
    assert snap["state"] == "on"  # re-probe after the final OFF session, though it failed
    assert indicator_memo._off_left == 0
    assert len(indicator_memo._window) == 0
    with indicator_memo.session():
        assert indicator_memo.is_active() is True


def test_zero_call_sessions_not_judged() -> None:
    """R16: sessions with 0 calls never fill the window."""
    for _ in range(10):
        with _judged():
            pass
    snap = indicator_memo.snapshot()
    assert snap["state"] == "on"
    assert len(indicator_memo._window) == 0


def test_unjudged_sessions_never_switch_off() -> None:
    """R16b: generation-1 (judge=False) sessions are invisible to the window."""
    for _ in range(300):
        with _judged(judge=False, hits=0, misses=10):
            pass
    snap = indicator_memo.snapshot()
    assert snap["switch_offs"] == 0
    assert snap["state"] == "on"
    assert len(indicator_memo._window) == 0

    for _ in range(99):
        with _judged(hits=0, misses=10):
            pass
    assert indicator_memo.snapshot()["state"] == "on"  # not full yet
    with _judged(hits=0, misses=10):
        pass
    assert indicator_memo.snapshot()["state"] == "off"  # exactly the 100th
    assert indicator_memo.snapshot()["switch_offs"] == 1


@pytest.mark.parametrize("raw", ["abc", "0", "-5", "2.5"])
def test_max_mb_env_parsing_rejects(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """R17: a bad cap raises at session() entry, before the runner runs."""
    monkeypatch.setenv("VIBE_QUANT_IND_MEMO_MAX_MB", raw)
    with pytest.raises(ValueError) as excinfo, indicator_memo.session():
        pytest.fail("session body must not run")
    assert str(excinfo.value) == (
        f"VIBE_QUANT_IND_MEMO_MAX_MB must be an integer >= 1, got {raw!r}"
    )


@pytest.mark.parametrize(("raw", "expected"), [(None, 64), ("", 64), ("1", 1), ("8", 8)])
def test_max_mb_env_parsing_accepts(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int
) -> None:
    """R17: unset/empty -> 64 MB, an integer -> that many MB, as bytes."""
    if raw is None:
        monkeypatch.delenv("VIBE_QUANT_IND_MEMO_MAX_MB", raising=False)
    else:
        monkeypatch.setenv("VIBE_QUANT_IND_MEMO_MAX_MB", raw)
    with indicator_memo.session():
        assert indicator_memo.snapshot()["max_bytes"] == expected * 1_000_000


def _fake_nt_fn(monkeypatch: pytest.MonkeyPatch, *, raises: bool = False) -> Any:
    from vibe_quant.discovery import backtest_fn as bf

    fn = bf.NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-06-01")
    monkeypatch.setattr("vibe_quant.discovery.genome.chromosome_to_dsl", lambda _chrom: {})
    observed: list[bool] = []

    class _Runner:
        last_starting_balance = 0.0
        last_days = 1

        def __call__(self, _params: dict[str, object]) -> SimpleNamespace:
            observed.append(indicator_memo.is_active())
            if raises:
                raise OSError("boom")
            return SimpleNamespace(
                sharpe_ratio=1.0,
                max_drawdown=0.1,
                profit_factor=1.2,
                total_trades=3,
            )

    monkeypatch.setattr(fn, "_make_runner", lambda *_a, **_k: _Runner())
    return fn, observed


def test_run_single_opens_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """R18: ``_run_single`` is the activation site, and ``judge`` is forwarded."""
    judges: list[bool] = []
    real_session = indicator_memo.session

    @contextmanager
    def spy(*, judge: bool = True) -> Iterator[None]:
        judges.append(judge)
        with real_session(judge=judge):
            yield

    monkeypatch.setattr(indicator_memo, "session", spy)
    chrom = SimpleNamespace(uid="x")

    fn, observed = _fake_nt_fn(monkeypatch)
    fn.ind_memo_judge = True
    fn._run_single(chrom, "2024-01-01", "2024-06-01")
    assert observed == [True]
    assert indicator_memo.is_active() is False
    assert judges == [True]

    fn, observed = _fake_nt_fn(monkeypatch)
    fn.ind_memo_judge = False
    fn._eval_symbols(chrom, None)
    assert observed == [True]  # the session is open; judge only gates the verdict
    assert judges[-1] is False
    assert indicator_memo.is_active() is False

    fn, observed = _fake_nt_fn(monkeypatch, raises=True)
    with pytest.raises(OSError, match="boom"):
        fn._run_single(chrom, "2024-01-01", "2024-06-01")
    assert observed == [True]
    assert indicator_memo.is_active() is False
    with indicator_memo.session():
        assert indicator_memo.is_active() is True  # not left stuck open


def test_bytes_never_exceed_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """R19: 1000 single stores at a 45 kB cap keep est_bytes <= cap + one entry."""
    monkeypatch.setattr(indicator_memo, "_max_bytes_from_env", lambda: 45_000)

    def fn(_df: object, params: dict[str, object]) -> pd.Series:
        return _obj_series(float(params["k"]))

    buf = _Buf()
    with indicator_memo.session():
        for k in range(1000):
            indicator_memo.memo_store(fn, buf, {"k": k}, fn(None, {"k": k}))
            assert indicator_memo.snapshot()["est_bytes"] <= 45_150, k


@pytest.mark.timing
@pytest.mark.parametrize("outputs", [1, 3], ids=["single", "dict"])
def test_est_bytes_tracks_tracemalloc(monkeypatch: pytest.MonkeyPatch, outputs: int) -> None:
    """R19b: the calibrated estimate is within [0.7, 1.1] of tracemalloc."""
    monkeypatch.setattr(indicator_memo, "_max_bytes_from_env", lambda: 10**12)
    buckets = 5
    # 5 buckets x 30k bars = 150k entries: the size the design's calibration
    # numbers (148/118 single, 460/429 dict B/entry) were measured at.
    bars = 30_000
    fns = [lambda _df, _p, i=i: None for i in range(buckets)]

    class _FpBuf:
        cap = 400

        def __init__(self) -> None:
            self.fp = b""

        def fingerprint(self) -> bytes:
            return self.fp

    buf = _FpBuf()
    with indicator_memo.session():
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            for bar in range(bars):
                buf.fp = bar.to_bytes(8, "little") + b"x" * 8  # one shared fp object per bar
                series = pd.Series([0.0, float(bar)])  # float64, like a real compute_fn
                value: object = series if outputs == 1 else {"a": series, "b": series, "c": series}
                for i in range(buckets):
                    indicator_memo.memo_store(fns[i], buf, {"i": i}, value)
            measured = tracemalloc.get_traced_memory()[0] - base
        finally:
            tracemalloc.stop()

    entries = bars * buckets
    est = indicator_memo.snapshot()["est_bytes"]
    assert isinstance(est, int)
    assert indicator_memo.snapshot()["entries"] == entries
    ratio = (measured / entries) / (est / entries)
    print(f"measured {measured / entries:.0f} B/entry, est {est / entries:.0f} B/entry")
    assert 0.7 <= ratio <= 1.1, f"measured/est = {ratio:.2f}"


def test_snapshot_keys_and_rss_units(monkeypatch: pytest.MonkeyPatch) -> None:
    """R25: exactly 15 keys; ru_maxrss is bytes on darwin and KiB elsewhere."""
    snap = indicator_memo.snapshot()
    assert set(snap) == SNAPSHOT_KEYS
    assert len(snap) == 15
    assert snap["rss_at_first_session_bytes"] is None
    assert isinstance(snap["proc_id"], str) and len(snap["proc_id"]) == 32

    def _usage(_who: int) -> SimpleNamespace:
        return SimpleNamespace(ru_maxrss=338_000_000)

    monkeypatch.setattr(resource, "getrusage", _usage)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert indicator_memo.snapshot()["peak_rss_bytes"] == 338_000_000

    def _usage_kib(_who: int) -> SimpleNamespace:
        return SimpleNamespace(ru_maxrss=330_078)

    monkeypatch.setattr(resource, "getrusage", _usage_kib)
    monkeypatch.setattr(sys, "platform", "linux")
    assert indicator_memo.snapshot()["peak_rss_bytes"] == 337_999_872

    with indicator_memo.session():
        assert indicator_memo.snapshot()["rss_at_first_session_bytes"] == 337_999_872


def _worker_snap(proc_id: str, max_bytes: int) -> dict[str, object]:
    snap = dict(indicator_memo.snapshot())
    snap["proc_id"] = proc_id
    snap["max_bytes"] = max_bytes
    snap["evictions"] = 1
    return snap


def test_run_summary_max_mb_is_the_workers_cap() -> None:
    """D12: the parent never opened a session (default cap), the workers ran with 8."""
    indicator_memo.record_worker(_worker_snap("w1", 8_000_000))
    summary = indicator_memo.run_summary()
    assert summary["parent_sessions"] == 0
    assert summary["max_mb"] == 8
    assert summary["evictions"] == 1


def test_run_summary_max_mb_is_the_largest_worker_cap() -> None:
    indicator_memo.record_worker(_worker_snap("w1", 8_000_000))
    indicator_memo.record_worker(_worker_snap("w2", 16_000_000))
    assert indicator_memo.run_summary()["max_mb"] == 16


def test_run_summary_max_mb_without_workers_is_the_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBE_QUANT_IND_MEMO_MAX_MB", "5")
    with indicator_memo.session():
        pass
    assert indicator_memo.run_summary()["max_mb"] == 5
