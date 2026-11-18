"""Paper warm seam of generated strategies (SEED -> install -> handover).

A compiled strategy gets an OFF-by-default warm mode: ``_warm_from(seed)`` installs an
REST seed through the validation precompute, after which ``_warm_ingress`` (first statement
of ``on_bar``) drops REST duplicates, queues gaps and asks the node for a top-up.

Every strategy here runs in a real BacktestEngine on synthetic bars whose ``ts_event`` is the
CLOSE time (``ts_init`` too), like live Binance bars. The reference is a cold run (indicators
registered in ``on_start``) over the SAME bars: warm values must equal it exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.objects import Price, Quantity

from tests.unit.test_fill_timing import _INSTRUMENT_ID, _new_engine
from vibe_quant.data.catalog import get_bar_type
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict

if TYPE_CHECKING:
    from collections.abc import Callable

_T0_MS = 1_735_689_600_000  # 2025-01-01 00:00:00 UTC
_STEP_MS = {"15m": 900_000, "4h": 14_400_000}
_NEVER = "close > 10000000"


def _bars(tf: str, n: int, seed: int) -> list[Bar]:
    """``n`` bars of ``tf``; ts_event == ts_init == close time (open + step - 1 ms)."""
    rng = np.random.default_rng(seed)
    closes = 60_000.0 + np.cumsum(rng.normal(0.0, 80.0, n))
    bar_type = get_bar_type("BTCUSDT", tf)
    out: list[Bar] = []
    prev = float(closes[0])
    for i in range(n):
        c = round(float(closes[i]), 1)
        hi = round(max(prev, c) + 25.0, 1)
        lo = round(min(prev, c) - 25.0, 1)
        ts = (_T0_MS + (i + 1) * _STEP_MS[tf] - 1) * 1_000_000
        out.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{prev:.1f}"),
                high=Price.from_str(f"{hi:.1f}"),
                low=Price.from_str(f"{lo:.1f}"),
                close=Price.from_str(f"{c:.1f}"),
                volume=Quantity.from_str("100.000"),
                ts_event=ts,
                ts_init=ts,
            )
        )
        prev = c
    return out


def _close_ts(tf: str, i: int) -> int:
    return (_T0_MS + (i + 1) * _STEP_MS[tf] - 1) * 1_000_000


def _compile(
    timeframe: str,
    indicators: dict[str, dict[str, Any]],
    exit_long: list[str] | None = None,
) -> tuple[type, type]:
    dsl = validate_strategy_dict(
        {
            "name": "warm_probe",
            "timeframe": timeframe,
            "indicators": indicators,
            "additional_timeframes": sorted(
                {c["timeframe"] for c in indicators.values() if "timeframe" in c} - {timeframe}
            ),
            "entry_conditions": {"long": [_NEVER]},
            "exit_conditions": {"long": exit_long} if exit_long else {},
            "stop_loss": {"type": "fixed_pct", "percent": 50.0},
            "take_profit": {"type": "fixed_pct", "percent": 50.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    camel = _to_class_name(dsl.name)
    return getattr(module, f"{camel}Strategy"), getattr(module, f"{camel}Config")


_IND_15M: dict[str, dict[str, Any]] = {"ema": {"type": "EMA", "period": 5}, "pmax": {"type": "PMAX"}}
_IND_15M_4H: dict[str, dict[str, Any]] = {
    "ema": {"type": "EMA", "period": 5},
    "pmax": {"type": "PMAX"},
    "ema4": {"type": "EMA", "period": 3, "timeframe": "4h"},
}
_IND_4H_15M: dict[str, dict[str, Any]] = {
    "ema4": {"type": "EMA", "period": 3},
    "ema15": {"type": "EMA", "period": 3, "timeframe": "15m"},
}


def _make_probe(base: type) -> type:
    """``base`` recording what each bar left behind and which signal paths ran."""

    class Probe(base):  # type: ignore[valid-type, misc]
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.rows: list[tuple[int, str, dict[str, float]]] = []
            self.entry_checks: list[tuple[int, dict[str, float]]] = []
            self.events: list[tuple[str, int]] = []
            self.last_ts = 0
            self.topups: list[tuple[str, int, int]] = []
            self.topup_script: dict[str, tuple[int, list[Bar]]] = {}
            self.delivered_since_request: dict[str, int] = {}
            self.stub_position = False
            self.replayed: list[tuple[str, int]] = []

        def on_start(self) -> None:
            super().on_start()
            if self.stub_position:
                self._position_open = True
                self._position_side = OrderSide.BUY

        def on_bar(self, bar: Bar) -> None:
            self.last_ts = int(bar.ts_event)
            super().on_bar(bar)
            values = {
                k: float(getattr(self, f"ind_{k}").value)
                for k in ("ema", "ema4", "ema15")
                if hasattr(self, f"ind_{k}")
            }
            values.update({f"pta_{k}": v for k, v in getattr(self, "_pta_values", {}).items()})
            self.rows.append((self.last_ts, str(bar.bar_type.spec), values))
            for name in list(self.delivered_since_request):
                self.delivered_since_request[name] += 1
                after, bars = self.topup_script[name]
                if self.delivered_since_request[name] >= after:
                    del self.delivered_since_request[name]
                    self._warm_feed_topup(self._bt(name), bars)

        def _bt(self, name: str) -> BarType:
            return getattr(self, f"bar_type_{name}")

        def fake_topup(self, bar_type: BarType, wm: int, ts: int) -> None:
            self.topups.append((str(bar_type.spec), wm, ts))
            name = "15m" if "15-MINUTE" in str(bar_type) else "4h"
            if name in self.topup_script:
                self.delivered_since_request[name] = 0

        def _check_long_entry(self, bar: Bar) -> bool:
            self.entry_checks.append((int(bar.ts_event), dict(self._prev_values)))
            return bool(super()._check_long_entry(bar))

        def _ensure_protection(self) -> bool:
            self.events.append(("protect", self.last_ts))
            self._rearm_protection = False
            return True

        def _submit_exit(self, bar: Bar) -> None:
            self.events.append(("exit", int(bar.ts_event)))

        def _warm_replay_bar(self, bar: Bar) -> None:
            self.replayed.append((str(bar.bar_type.spec), int(bar.ts_event)))
            super()._warm_replay_bar(bar)

    return Probe


def _run(strategy: Any, bars: list[Bar]) -> None:
    engine = _new_engine()
    # one add_data per bar type in tf order (what the runners' data configs do): equal-ts
    # bars of different timeframes then reach the strategy in the order NT itself picks
    for key in sorted({str(b.bar_type) for b in bars}):
        engine.add_data(sorted((b for b in bars if str(b.bar_type) == key), key=lambda b: b.ts_init))
    engine.add_strategy(strategy)
    try:
        engine.run()
    finally:
        engine.reset()
        engine.dispose()


def _build(
    indicators: dict[str, dict[str, Any]],
    *,
    timeframe: str = "15m",
    exit_long: list[str] | None = None,
    **config: Any,
) -> tuple[Callable[[], Any], type]:
    base, config_cls = _compile(timeframe, indicators, exit_long)
    probe = _make_probe(base)
    params = {"instrument_id": _INSTRUMENT_ID, "execution_delay_probability": 0.0, **config}

    def make() -> Any:
        return probe(config=config_cls(**params))

    return make, probe


def _seed_of(*groups: list[Bar]) -> dict[str, list[Bar]]:
    return {str(g[0].bar_type): g for g in groups}


def _by_ts(rows: list[tuple[int, str, dict[str, float]]]) -> dict[tuple[int, str], dict[str, float]]:
    return {(ts, spec): values for ts, spec, values in rows}


SEED_N = 300


def test_rest_bar_redelivered_live_is_dropped() -> None:
    """R7: the REST bar t arrives again from the websocket AFTER the install -> dropped, no double feed."""
    make, _ = _build(_IND_15M, ema_period=5)
    all_bars = _bars("15m", SEED_N + 20, seed=1)

    ref = make()
    _run(ref, all_bars)

    warm = make()
    warm._warm_from(_seed_of(all_bars[:SEED_N]))
    live = all_bars[SEED_N - 1 :]  # REST already holds bar SEED_N-1; the websocket sends it again
    _run(warm, live)

    ref_rows = _by_ts(ref.rows)
    wm = _close_ts("15m", SEED_N - 1)
    compared = mismatches = 0
    for ts, spec, values in warm.rows:
        if ts <= wm:
            continue
        compared += 1
        mismatches += values != ref_rows[(ts, spec)]
    assert compared == 20
    assert mismatches == 0
    assert warm.ind_ema.count == SEED_N + 20
    assert warm._warm_dropped == 1
    assert warm._warm_wm[warm.bar_type_15m] == _close_ts("15m", SEED_N + 19)


def test_redelivered_bar_with_other_close_is_counted_not_fed() -> None:
    make, _ = _build(_IND_15M, ema_period=5)
    bars = _bars("15m", SEED_N + 1, seed=1)
    warm = make()
    warm._warm_from(_seed_of(bars[:SEED_N]))
    before = warm.ind_ema.value
    dup = bars[SEED_N - 1]
    other = Bar(
        dup.bar_type, dup.open, dup.high, dup.low, dup.high, dup.volume, dup.ts_event, dup.ts_init
    )
    assert float(other.close) != float(dup.close)
    assert warm._warm_ingress(other) is False
    assert warm._warm_close_mismatches == 1
    assert warm.ind_ema.value == before
    assert warm._warm_ingress(dup) is False  # the same close as REST: dropped without a mismatch
    assert warm._warm_close_mismatches == 1
    assert warm._warm_dropped == 2


def test_ingress_edges_at_the_watermark() -> None:
    """ts == wm drops, ts == wm + step feeds, ts == wm + 2 steps queues and asks once."""
    make, _ = _build(_IND_15M, ema_period=5)
    bars = _bars("15m", SEED_N + 6, seed=1)
    warm = make()
    warm._warm_from(_seed_of(bars[:SEED_N]))
    asked: list[tuple[str, int, int]] = []
    warm._warm_request_topup = lambda bt, wm, ts: asked.append((str(bt.spec), wm, ts))
    bt = warm.bar_type_15m
    wm0 = _close_ts("15m", SEED_N - 1)
    assert warm._warm_wm[bt] == wm0

    assert warm._warm_ingress(bars[SEED_N - 1]) is False  # ts == wm
    assert warm._warm_ingress(bars[SEED_N - 2]) is False  # ts < wm
    assert warm.ind_ema.count == SEED_N

    assert warm._warm_ingress(bars[SEED_N]) is True  # ts == wm + step
    assert warm.ind_ema.count == SEED_N + 1
    assert warm._warm_wm[bt] == _close_ts("15m", SEED_N)
    assert asked == []

    assert warm.ind_ema.count == SEED_N + 1


def test_first_live_bar_two_steps_ahead_queues_and_asks_once() -> None:
    make, _ = _build(_IND_15M, ema_period=5)
    bars = _bars("15m", SEED_N + 6, seed=1)
    warm = make()
    warm._warm_from(_seed_of(bars[:SEED_N]))
    asked: list[tuple[str, int, int]] = []
    warm._warm_request_topup = lambda bt, wm, ts: asked.append((str(bt.spec), wm, ts))
    bt = warm.bar_type_15m
    assert warm._warm_ingress(bars[SEED_N + 1]) is False  # ts == wm + 2 steps
    assert warm._warm_pending == {bt}
    assert asked == [("15-MINUTE-LAST", _close_ts("15m", SEED_N - 1), _close_ts("15m", SEED_N + 1))]
    assert warm._warm_ingress(bars[SEED_N + 2]) is False  # queued, no second request
    assert len(asked) == 1
    assert warm.ind_ema.count == SEED_N  # nothing fed while pending
    assert len(warm._warm_queue) == 2


def test_gap_after_the_first_contiguous_live_bar_is_fed_as_is() -> None:
    """Later gaps (reconnects) keep today's behaviour: fed, counted, no top-up."""
    make, _ = _build(_IND_15M, ema_period=5)
    bars = _bars("15m", SEED_N + 6, seed=1)
    warm = make()
    warm._warm_from(_seed_of(bars[:SEED_N]))
    warm._warm_request_topup = lambda *_a: pytest.fail("no top-up after the first contiguous bar")
    assert warm._warm_ingress(bars[SEED_N]) is True
    assert warm._warm_ingress(bars[SEED_N + 3]) is True  # 2 bars missing
    assert warm._warm_late_gap_bars == 2
    assert warm.ind_ema.count == SEED_N + 2
    assert not warm._warm_pending


def _gap_case(topup_after: int, gap: list[int]) -> tuple[Any, Any, list[Bar]]:
    """Seed to SEED_N-1; live bars SEED_N+2.. arrive (bars SEED_N, SEED_N+1 missing)."""
    make, _ = _build(_IND_15M, ema_period=5)
    all_bars = _bars("15m", SEED_N + 12, seed=2)
    ref = make()
    _run(ref, all_bars)
    warm = make()
    warm._warm_from(_seed_of(all_bars[:SEED_N]))
    warm._warm_request_topup = warm.fake_topup
    warm.topup_script["15m"] = (topup_after, [all_bars[i] for i in gap])
    _run(warm, all_bars[SEED_N + 2 :])
    return ref, warm, all_bars


def test_gap_at_handover_requests_topup_and_matches_no_gap_reference() -> None:
    """R9: first live bar at wm + 3 steps -> top-up (wm, ts), then values == the no-gap run."""
    ref, warm, bars = _gap_case(topup_after=3, gap=[SEED_N, SEED_N + 1])
    assert warm.topups == [("15-MINUTE-LAST", _close_ts("15m", SEED_N - 1), _close_ts("15m", SEED_N + 2))]
    ref_rows = _by_ts(ref.rows)
    resumed = [(ts, spec, v) for ts, spec, v in warm.rows if ts > _close_ts("15m", SEED_N + 4)]
    assert len(resumed) == 7
    for ts, spec, values in resumed:
        assert values == ref_rows[(ts, spec)]
    assert warm.ind_ema.count == SEED_N + 12
    assert not warm._warm_pending
    assert warm._warm_queue == []
    # no signal evaluation while pending, signals resume on the next bar
    signal_ts = [ts for ts, _ in warm.entry_checks]
    assert min(signal_ts) == _close_ts("15m", SEED_N + 5)
    assert signal_ts == [_close_ts("15m", SEED_N + i) for i in range(5, 12)]


def test_queued_duplicate_of_topup_bar_is_fed_once() -> None:
    """R8: the REST top-up already holds bars the websocket queued -> each fed once, REST copy first."""
    gap = [SEED_N, SEED_N + 1, SEED_N + 2, SEED_N + 3]  # REST holds the closed bars 302 and 303 too
    ref, warm, bars = _gap_case(topup_after=3, gap=gap)
    assert warm.ind_ema.count == SEED_N + 12
    ref_rows = _by_ts(ref.rows)
    for ts, spec, values in warm.rows:
        if ts > _close_ts("15m", SEED_N + 4):
            assert values == ref_rows[(ts, spec)]
    fed = [ts for _, ts in warm.replayed]
    assert fed.count(_close_ts("15m", SEED_N + 2)) == 1
    assert fed.count(_close_ts("15m", SEED_N + 3)) == 1
    assert fed == sorted(fed)


def _two_tf_bars(n15: int) -> tuple[list[Bar], list[Bar]]:
    return _bars("15m", n15, seed=3), _bars("4h", n15 // 16, seed=4)


def test_secondary_only_gap_queues_the_primary_until_topup() -> None:
    """R10: 15m primary + 4h secondary, one 4h bar missing -> 15m bars queue until the 4h top-up lands."""
    n15 = 16 * 40
    b15, b4 = _two_tf_bars(n15)
    make, _ = _build(_IND_15M_4H, ema_period=5)
    ref = make()
    _run(ref, [*b15, *b4])

    seed15 = 16 * 30 + 2  # seed cut 30 min into the 4h bar 30
    k_seed = 30  # 4h bars 0..29 are closed at the cut
    warm = make()
    warm._warm_from(_seed_of(b15[:seed15], b4[:k_seed]))
    warm._warm_request_topup = warm.fake_topup
    # live: 15m continues; the 4h bar 30 never arrives, bar 31 does -> gap on 4h only
    live15 = b15[seed15:]
    live4 = b4[k_seed + 1 :]
    warm.topup_script["4h"] = (4, [b4[k_seed]])
    _run(warm, [*live15, *live4])

    bt4 = warm.bar_type_4h
    assert warm.topups == [("4-HOUR-LAST", _close_ts("4h", k_seed - 1), _close_ts("4h", k_seed + 1))]
    assert len(warm.topups) == 1
    ref_rows = _by_ts(ref.rows)
    boundary = _close_ts("4h", k_seed + 1)
    step_ns = 900_000 * 1_000_000
    checks = [ts for ts, _ in warm.entry_checks]
    assert boundary in checks  # the 15m bar tied with the 4h bar ran before the gap showed
    # the 4h bar + 3 later 15m bars were queued; the top-up landed after the third
    assert not [ts for ts in checks if boundary < ts <= boundary + 3 * step_ns]
    first_signal = boundary + 4 * step_ns
    assert first_signal in checks
    assert warm._warm_wm[bt4] == _close_ts("4h", len(b4) - 1)
    for ts, spec, values in warm.rows:
        if ts >= first_signal:
            assert values == ref_rows[(ts, spec)]
    # both timeframes were replayed through the non-signal sequence in (ts, rank) order
    gap_replay = [x for x in warm.replayed if x[1] > _close_ts("4h", k_seed - 1)]
    assert gap_replay == sorted(gap_replay, key=lambda x: (x[1], 0 if x[0].startswith("15") else 1))


def test_prev_values_captured_at_last_primary_bar() -> None:
    """R11: 4h primary + 15m secondary, start 02:00 -> the 15m prev value is the one at the 00:00 primary bar."""
    n15 = 16 * 40
    b15, b4 = _two_tf_bars(n15)
    make, _ = _build(_IND_4H_15M, timeframe="4h")
    ref = make()
    _run(ref, [*b15, *b4])

    k_seed = 21  # 4h bars 0..20 seeded: the last closes at 00:00; the cut is 02:00 (8 15m bars later)
    cut15 = 16 * k_seed + 8
    warm = make()
    warm._warm_from(_seed_of(b15[:cut15], b4[:k_seed]))
    ref_at_primary = dict(next(p for ts, p in ref.entry_checks if ts == _close_ts("4h", k_seed)))
    # the cold run saw the same bars before the 4h bar k_seed: its prev values at that bar's
    # check are the ones captured at bar k_seed - 1 (the 00:00 bar)
    expected_prev = ref_at_primary["ema15"]
    assert warm._prev_values["ema15"] == expected_prev
    assert warm._prev_values["ema15"] != float(warm.ind_ema15.value)  # not the 02:00 value

    warm._warm_request_topup = warm.fake_topup
    _run(warm, [*b15[cut15:], *b4[k_seed:]])
    warm_prev = dict(next(p for ts, p in warm.entry_checks if ts == _close_ts("4h", k_seed)))
    assert warm_prev == ref_at_primary


def test_nt_tie_order_equals_traversal_order() -> None:
    """R12: equal-ts bars of different timeframes replay in the order NT backtests deliver them."""
    n15 = 16 * 12
    b15, b4 = _two_tf_bars(n15)
    make, _ = _build(_IND_15M_4H, ema_period=5)
    ref = make()
    _run(ref, [*b15, *b4])
    engine_order = [(spec, ts) for ts, spec, _ in ref.rows]

    warm = make()
    warm._warm_from(_seed_of(b15, b4))
    tied = [ts for ts in {t for _, t in engine_order} if sum(1 for _, x in engine_order if x == ts) > 1]
    assert len(tied) == len(b4)
    assert warm.replayed == engine_order


def test_no_signal_exit_while_pending_and_protection_at_feed_time() -> None:
    """R16: adopted position, top-up pending -> no signal exit; _ensure_protection runs when the top-up lands."""
    make, _ = _build(_IND_15M, exit_long=["close > 0"], ema_period=5)
    all_bars = _bars("15m", SEED_N + 12, seed=2)
    warm = make()
    warm.stub_position = True
    warm._warm_from(_seed_of(all_bars[:SEED_N]))
    warm._rearm_protection = True
    warm._warm_request_topup = warm.fake_topup
    warm.topup_script["15m"] = (3, [all_bars[SEED_N], all_bars[SEED_N + 1]])
    _run(warm, all_bars[SEED_N + 2 :])

    pending_until = _close_ts("15m", SEED_N + 4)  # bars +2, +3, +4 queued; top-up lands after +4
    assert warm.events[0] == ("protect", pending_until)
    exits = [ts for kind, ts in warm.events if kind == "exit"]
    assert exits == [_close_ts("15m", SEED_N + i) for i in range(5, 12)]
    assert [e for e in warm.events if e[0] == "protect"] == [("protect", pending_until)]


def test_warmup_off_is_cold_path() -> None:
    """R2: without _warm_from nothing changes -- same registrations in order, no ingress."""
    make, probe = _build(_IND_15M_4H, ema_period=5, sizing_mode="atr", sizing_atr_period=7)
    registered: list[tuple[str, str]] = []

    class Spy(probe):  # type: ignore[valid-type, misc]
        def register_indicator_for_bars(self, bar_type: BarType, indicator: Any) -> None:
            registered.append((str(bar_type.spec), type(indicator).__name__))
            super().register_indicator_for_bars(bar_type, indicator)

        def _warm_ingress(self, bar: Bar) -> bool:
            raise AssertionError("ingress ran with the seam off")

    base_cfg = make().config
    strategy = Spy(config=base_cfg)
    assert strategy._warm_wm is None
    b15, b4 = _two_tf_bars(16 * 10)
    _run(strategy, [*b15, *b4])
    assert registered == [
        ("15-MINUTE-LAST", "ExponentialMovingAverage"),
        ("4-HOUR-LAST", "ExponentialMovingAverage"),
        ("15-MINUTE-LAST", "WilderAverageTrueRange"),
    ]
    assert strategy.ind_ema.count == len(b15)
    assert strategy.ind_ema4.count == len(b4)
    assert strategy._sizing_atr is not None and strategy._sizing_atr.initialized


def test_missing_primary_seed_is_refused() -> None:
    make, _ = _build(_IND_15M, ema_period=5)
    bars = _bars("15m", 10, seed=1)
    with pytest.raises(ValueError, match="primary"):
        make()._warm_from({"other": bars})


def _fresh_ema(period: int, bars: list[Bar]) -> float:
    from nautilus_trader.indicators import ExponentialMovingAverage

    ind = ExponentialMovingAverage(period)
    for b in bars:
        ind.handle_bar(b)
    return float(ind.value)


def test_completed_topups_accumulate_until_the_global_flush() -> None:
    """Both timeframes pending; 15m completes twice before 4h lands -> no fetched bar is discarded."""
    b15, b4 = _two_tf_bars(16 * 40)
    make, _ = _build(_IND_15M_4H, ema_period=5)
    k_seed, s15 = 30, 16 * 30 + 2
    warm = make()
    warm._warm_from(_seed_of(b15[:s15], b4[:k_seed]))
    asked: list[tuple[str, int, int]] = []
    warm._warm_request_topup = lambda bt, wm, ts: asked.append((str(bt.spec), wm, ts))
    assert warm._warm_ingress(b15[s15 + 2]) is False
    assert warm._warm_ingress(b4[k_seed + 1]) is False
    warm._warm_feed_topup(warm.bar_type_15m, [b15[s15], b15[s15 + 1]])
    assert warm._warm_pending == {warm.bar_type_4h}
    assert warm._warm_ingress(b15[s15 + 5]) is False  # a second 15m hole opens
    assert len(asked) == 3
    warm._warm_feed_topup(warm.bar_type_15m, [b15[s15 + 3], b15[s15 + 4]])
    assert warm._warm_pending == {warm.bar_type_4h}
    warm._warm_feed_topup(warm.bar_type_4h, [b4[k_seed]])
    assert not warm._warm_pending
    assert warm.ind_ema.count == s15 + 6
    assert warm.ind_ema.value == _fresh_ema(5, b15[: s15 + 6])
    assert warm.ind_ema4.value == _fresh_ema(3, b4[: k_seed + 2])


def test_growing_gap_while_pending_stays_pending_until_every_bar_is_present() -> None:
    """A hole that opens while the same timeframe is pending is requested; signals wait for it."""
    make, _ = _build(_IND_15M, ema_period=5)
    bars = _bars("15m", SEED_N + 8, seed=1)
    warm = make()
    warm._warm_from(_seed_of(bars[:SEED_N]))
    asked: list[tuple[str, int, int]] = []
    warm._warm_request_topup = lambda bt, wm, ts: asked.append((str(bt.spec), wm, ts))
    bt = warm.bar_type_15m
    assert warm._warm_ingress(bars[SEED_N + 2]) is False
    assert warm._warm_ingress(bars[SEED_N + 5]) is False  # 303, 304 are missing too
    assert len(asked) == 1
    warm._warm_feed_topup(bt, [bars[SEED_N], bars[SEED_N + 1]])
    assert warm._warm_pending == {bt}  # still pending: the new hole is not covered
    assert asked[1] == ("15-MINUTE-LAST", _close_ts("15m", SEED_N + 2), _close_ts("15m", SEED_N + 5))
    assert warm.ind_ema.count == SEED_N  # nothing replayed yet
    assert warm._warm_ingress(bars[SEED_N + 6]) is False  # still no signals
    warm._warm_feed_topup(bt, [bars[SEED_N + 3], bars[SEED_N + 4]])
    assert not warm._warm_pending
    assert len(asked) == 2
    assert warm.ind_ema.count == SEED_N + 7
    assert warm.ind_ema.value == _fresh_ema(5, bars[: SEED_N + 7])
    assert warm._warm_ingress(bars[SEED_N + 7]) is True
