"""Liquidation watch loop and per-symbol state (bead vibe-quant-2gyk1.1.11, task P3a2).

Contract rows R46 (per-cycle request window and the 7-day clamp), R47 / R47b
(emission order, then cursor advance), R48 (zero executedQty rows), R49 (the
watcher never touches strategy/guard/exec), R50 (state and restore), R51 (a full
page fails one symbol's cycle), R52 / R52b / R52c (per-symbol failures, backoff,
at-most-once delivery) and R53 (shutdown).

The watch loop is REPORT ONLY: it turns forceOrders rows into events and alerts
and never submits, cancels or halts anything. Every test drives a fake ``fetch``
and an injected ``now_ms`` clock — no network, no real sleeping.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from vibe_quant.paper.force_orders import ForceOrder, IncompleteForceOrders
from vibe_quant.paper.liquidation_watch import LiquidationWatcher

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

_START_MS = 1_700_000_000_000
_CLAMP_SLACK_MS = 60_000
_OVERLAP_MS = 300_000
_LOOKBACK_CAP_MS = 604_800_000
_EIGHT_DAYS_MS = 8 * 86_400_000


class _Clock:
    """Injected ``now_ms``: time only moves when the test moves it."""

    def __init__(self, ms: int = _START_MS) -> None:
        self.ms = ms

    def __call__(self) -> int:
        return self.ms

    def advance(self, seconds: int) -> None:
        self.ms += seconds * 1_000

    def advance_ms(self, ms: int) -> None:
        self.ms += ms


class _Recorder:
    """One watcher's handler calls, in call order."""

    def __init__(self) -> None:
        self.events: list[ForceOrder] = []
        self.warnings: list[str] = []
        self.recoveries: list[tuple[str, int]] = []

    async def on_liquidation(self, event: ForceOrder) -> None:
        self.events.append(event)

    async def on_warning(self, message: str) -> None:
        self.warnings.append(message)

    async def on_recovered(self, symbol: str, failures: int) -> None:
        self.recoveries.append((symbol, failures))


class _FakeFetch:
    """Stand-in for P3a1's ``Fetch``: records the window, replays rows or raises.

    Rows are filtered to the requested window because Binance's ``startTime`` and
    ``endTime`` are inclusive bounds: a row outside ``[start_ms, end_ms]`` would
    never come back from the endpoint the watcher is allowed to assume.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, int, int]] = []
        self.orders: dict[str, list[ForceOrder]] = {}
        self.errors: dict[str, Exception] = {}

    async def __call__(self, symbol: str, start_ms: int, end_ms: int) -> list[ForceOrder]:
        self.requests.append((symbol, start_ms, end_ms))
        error = self.errors.get(symbol)
        if error is not None:
            raise error
        return [
            order
            for order in self.orders.get(symbol, [])
            if start_ms <= order.update_time_ms <= end_ms
        ]


def _order(
    *,
    symbol: str = "BTCUSDT",
    order_id: int,
    update_time_ms: int,
    executed_qty: str = "0.002",
) -> ForceOrder:
    """One decoded forceOrders row (the shape P3a1 hands the watcher)."""
    return ForceOrder.from_binance(
        {
            "symbol": symbol,
            "orderId": order_id,
            "clientOrderId": "autoclose-1596107620040000020",
            "side": "SELL",
            "status": "FILLED",
            "avgPrice": "60000.0",
            "executedQty": executed_qty,
            "updateTime": update_time_ms,
            "autoCloseType": "LIQUIDATION",
        },
    )


async def test_hung_symbols_time_out_together_not_one_after_another() -> None:
    """qe6ye: one cycle fetches every due symbol concurrently, so N hung symbols cost
    one ``fetch_timeout_s``, not N of them, and a healthy symbol still delivers.

    Serial polling stretched the traded symbols' cadence to max(60, 30 N) s
    (MEASURED 241 s at N=8 in risk-watch-carried-1's scaled probe).
    """
    recorder = _Recorder()
    hung = [f"H{i}USDT" for i in range(8)]
    in_flight = 0
    peak = 0
    requests: list[str] = []

    async def fetch(symbol: str, start_ms: int, end_ms: int) -> list[ForceOrder]:
        nonlocal in_flight, peak
        requests.append(symbol)
        if symbol == "BTCUSDT":
            return [_order(order_id=61, update_time_ms=_START_MS - 1_000)]
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.Event().wait()  # never comes back
        finally:
            in_flight -= 1
        return []

    watcher = _make_watcher(
        fetch, recorder, symbols=("BTCUSDT", *hung), fetch_timeout_s=0.05
    )

    await asyncio.wait_for(watcher.poll_once(), timeout=2)

    assert peak == 8
    assert requests == ["BTCUSDT", *hung]
    assert [event.order_id for event in recorder.events] == [61]
    assert watcher.state()["cursors"]["BTCUSDT"] == _START_MS


def _make_watcher(
    fetch: Callable[[str, int, int], Awaitable[list[ForceOrder]]],
    recorder: _Recorder,
    *,
    symbols: Sequence[str] = ("BTCUSDT",),
    clock: _Clock | None = None,
    cursors: dict[str, int] | None = None,
    seen: dict[str, int] | None = None,
    on_liquidation: Callable[[ForceOrder], Awaitable[None]] | None = None,
    on_warning: Callable[[str], Awaitable[None]] | None = None,
    **kwargs: object,
) -> LiquidationWatcher:
    return LiquidationWatcher(
        fetch,
        list(symbols),
        on_liquidation=on_liquidation or recorder.on_liquidation,
        on_warning=on_warning or recorder.on_warning,
        on_recovered=recorder.on_recovered,
        now_ms=clock or _Clock(),
        cursors=cursors,
        seen=seen,
        **kwargs,
    )


async def test_failing_symbol_does_not_block_healthy_symbol_alerts() -> None:
    """R52: a fetch error fails only that symbol; the healthy symbol still emits."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [_order(order_id=7, update_time_ms=_START_MS - 5_000)]
    fetch.errors["ETHUSDT"] = RuntimeError("binance 503")
    clock = _Clock()
    watcher = _make_watcher(fetch, recorder, symbols=("BTCUSDT", "ETHUSDT"), clock=clock)

    await watcher.poll_once()

    assert [event.order_id for event in recorder.events] == [7]
    assert recorder.warnings == []  # one failure is below degraded_after
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS, "ETHUSDT": _START_MS}
    assert watcher.state()["seen"] == {"BTCUSDT:7": _START_MS - 5_000}
    assert [symbol for symbol, _, _ in fetch.requests] == ["BTCUSDT", "ETHUSDT"]

    clock.advance(60)
    await watcher.poll_once()

    # R52b: the healthy symbol's key is delivered at most once. R52: the failed
    # symbol is retried on its own backoff and still does not block the healthy one.
    assert [event.order_id for event in recorder.events] == [7]
    assert [symbol for symbol, _, _ in fetch.requests] == [
        "BTCUSDT",
        "ETHUSDT",
        "BTCUSDT",
        "ETHUSDT",
    ]
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS + 60_000, "ETHUSDT": _START_MS}
    assert recorder.warnings == []


async def test_request_window_is_the_cursor_overlap_and_this_cycles_end() -> None:
    """R46: startTime = max(cursor - overlap, end - cap + 60 s), endTime = end."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    clock = _Clock()
    watcher = _make_watcher(fetch, recorder, clock=clock)

    await watcher.poll_once()
    clock.advance(60)
    await watcher.poll_once()

    assert fetch.requests == [
        ("BTCUSDT", _START_MS - _OVERLAP_MS, _START_MS),
        ("BTCUSDT", _START_MS - _OVERLAP_MS, _START_MS + 60_000),
    ]


async def test_seven_day_clamp_warns_once_per_symbol_per_episode() -> None:
    """R46: the 7-day clamp applies every cycle and warns once per clamp episode."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = TimeoutError("binance timeout")
    clock = _Clock()
    restored_cursor = _START_MS - _EIGHT_DAYS_MS
    watcher = _make_watcher(fetch, recorder, clock=clock, cursors={"BTCUSDT": restored_cursor})
    clamped_start = _START_MS - _LOOKBACK_CAP_MS + _CLAMP_SLACK_MS
    clamped_warning = "liquidation detector lookback clamped to 7 days: BTCUSDT"

    # A cursor more than 7 days behind now: the window starts at the cap, once warned.
    await watcher.poll_once()
    assert fetch.requests == [("BTCUSDT", clamped_start, _START_MS)]
    assert recorder.warnings == [clamped_warning]

    # The failure kept the cursor where it was, so this is the same episode.
    clock.advance(60)
    await watcher.poll_once()
    assert fetch.requests[-1] == ("BTCUSDT", clamped_start + 60_000, _START_MS + 60_000)
    assert recorder.warnings == [clamped_warning]

    # A success advances the cursor out of the clamp; the warning is not repeated.
    fetch.errors.clear()
    clock.advance(600)
    await watcher.poll_once()
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS + 660_000}
    assert recorder.warnings == [clamped_warning]

    # A cycle inside the overlap is not clamped, which ends the episode (silently).
    clock.advance(60)
    await watcher.poll_once()
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS + 720_000}
    assert recorder.warnings == [clamped_warning]

    # ... so falling more than 7 days behind again is a new episode: one more warning.
    clock.advance_ms(_EIGHT_DAYS_MS)
    later = _START_MS + 720_000 + _EIGHT_DAYS_MS
    await watcher.poll_once()
    assert fetch.requests[-1] == ("BTCUSDT", later - _LOOKBACK_CAP_MS + _CLAMP_SLACK_MS, later)
    assert recorder.warnings == [clamped_warning, clamped_warning]


async def test_rows_are_emitted_in_update_time_order() -> None:
    """R47: emission order is updateTime order whatever the API returns."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [
        _order(order_id=3, update_time_ms=_START_MS - 1_000),
        _order(order_id=1, update_time_ms=_START_MS - 3_000),
        _order(order_id=2, update_time_ms=_START_MS - 2_000),
    ]
    watcher = _make_watcher(fetch, recorder)

    await watcher.poll_once()

    assert [event.order_id for event in recorder.events] == [1, 2, 3]
    assert recorder.events[1].ts_ns == (_START_MS - 2_000) * 1_000_000


async def test_zero_executed_qty_row_is_skipped_and_not_marked_seen() -> None:
    """R48: a zero executedQty row emits nothing, is not marked seen, does not fail."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [
        _order(order_id=5, update_time_ms=_START_MS - 2_000, executed_qty="0"),
        _order(order_id=6, update_time_ms=_START_MS - 1_000),
    ]
    watcher = _make_watcher(fetch, recorder)

    await watcher.poll_once()

    assert [event.order_id for event in recorder.events] == [6]
    assert watcher.state()["seen"] == {"BTCUSDT:6": _START_MS - 1_000}
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS}
    assert recorder.warnings == []


@pytest.mark.parametrize(
    "error",
    [
        IncompleteForceOrders("forceOrders for ETHUSDT returned a full page of 100 rows"),
        ValueError("forceOrders row field 'executedQty' is not numeric: 'x'"),
    ],
    ids=["full-page", "malformed-row"],
)
async def test_broken_response_fails_only_that_symbol_cycle(error: Exception) -> None:
    """R51/R52: a full page or a malformed row emits nothing and keeps the cursor."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [_order(order_id=8, update_time_ms=_START_MS - 1_000)]
    fetch.errors["ETHUSDT"] = error
    clock = _Clock()
    watcher = _make_watcher(fetch, recorder, symbols=("BTCUSDT", "ETHUSDT"), clock=clock)

    await watcher.poll_once()

    assert [event.order_id for event in recorder.events] == [8]
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS, "ETHUSDT": _START_MS}
    assert recorder.warnings == []  # one failure, below degraded_after

    # The failure is counted, so the symbol is retried on its own backoff.
    clock.advance(60)
    await watcher.poll_once()
    assert [request for request in fetch.requests if request[0] == "ETHUSDT"] == [
        ("ETHUSDT", _START_MS - _OVERLAP_MS, _START_MS),
        ("ETHUSDT", _START_MS - _OVERLAP_MS, _START_MS + 60_000),
    ]
    assert [event.order_id for event in recorder.events] == [8]


async def test_event_handler_failure_mid_symbol_replays_only_the_rest() -> None:
    """R52c: a handler error fails the cycle; delivered rows stay seen, cursor stays."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [
        _order(order_id=1, update_time_ms=_START_MS - 3_000),
        _order(order_id=2, update_time_ms=_START_MS - 2_000),
        _order(order_id=3, update_time_ms=_START_MS - 1_000),
    ]
    delivered: list[int] = []
    failing = True

    async def flaky_writer(event: ForceOrder) -> None:
        if failing and event.order_id == 2:
            raise RuntimeError("event writer down")
        recorder.events.append(event)
        delivered.append(event.order_id)

    clock = _Clock()
    watcher = _make_watcher(fetch, recorder, clock=clock, on_liquidation=flaky_writer)

    await watcher.poll_once()

    assert delivered == [1]
    assert watcher.state()["seen"] == {"BTCUSDT:1": _START_MS - 3_000}
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS}  # not advanced

    failing = False
    clock.advance(60)
    await watcher.poll_once()

    assert delivered == [1, 2, 3]
    assert watcher.state()["seen"] == {
        "BTCUSDT:1": _START_MS - 3_000,
        "BTCUSDT:2": _START_MS - 2_000,
        "BTCUSDT:3": _START_MS - 1_000,
    }
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS + 60_000}


async def test_backoff_schedule_and_single_degraded_warning() -> None:
    """R52: delay min(60*2**(n-1), 600) s, one on_warning when f reaches 3."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = TimeoutError("binance timeout")
    clock = _Clock()
    watcher = _make_watcher(fetch, recorder, clock=clock)

    await watcher.poll_once()  # failure 1 -> retry in 60 s
    assert recorder.warnings == []

    for index, delay_s in enumerate([60, 120, 240, 480, 600, 600]):
        clock.advance(delay_s - 1)
        await watcher.poll_once()
        assert len(fetch.requests) == index + 1  # not due a second before its backoff
        clock.advance(1)
        await watcher.poll_once()
        assert len(fetch.requests) == index + 2  # due exactly on its backoff
        expected_warnings = 1 if index >= 1 else 0
        assert len(recorder.warnings) == expected_warnings

    assert recorder.warnings == ["liquidation detector degraded: BTCUSDT TimeoutError"]


async def test_recovery_alert_fires_once_on_the_first_success_after_degraded() -> None:
    """R52: on_recovered(symbol, failures) on the first success after degraded."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = TimeoutError("binance timeout")
    clock = _Clock()
    watcher = _make_watcher(fetch, recorder, clock=clock)

    for _ in range(3):
        clock.advance(600)
        await watcher.poll_once()

    assert recorder.warnings == ["liquidation detector degraded: BTCUSDT TimeoutError"]
    assert recorder.recoveries == []

    fetch.errors.clear()
    clock.advance(600)
    await watcher.poll_once()
    assert recorder.recoveries == [("BTCUSDT", 3)]

    clock.advance(600)
    await watcher.poll_once()
    assert recorder.recoveries == [("BTCUSDT", 3)]


async def test_failing_alert_handler_never_stops_detection() -> None:
    """R52d(3): a raising on_warning is logged; detection and other symbols run on."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [_order(order_id=4, update_time_ms=_START_MS - 1_000)]
    fetch.errors["ETHUSDT"] = TimeoutError("binance timeout")
    clock = _Clock()

    async def raising_warning(message: str) -> None:
        raise RuntimeError("telegram down")

    watcher = _make_watcher(
        fetch,
        recorder,
        symbols=("BTCUSDT", "ETHUSDT"),
        clock=clock,
        on_warning=raising_warning,
    )

    for _ in range(3):
        clock.advance(600)
        await watcher.poll_once()

    assert [event.order_id for event in recorder.events] == [4]
    assert watcher.state()["cursors"]["ETHUSDT"] == _START_MS  # still failing, still tracked


async def test_state_restores_cursors_and_seen_and_prunes_stale_keys() -> None:
    """R50: restore the passed maps, then prune seen to min(cursor) - overlap."""
    restored_cursor = _START_MS - 100_000
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [
        _order(order_id=1, update_time_ms=restored_cursor - 1_000),  # already delivered
        _order(order_id=2, update_time_ms=restored_cursor + 1_000),
    ]
    clock = _Clock()
    watcher = _make_watcher(
        fetch,
        recorder,
        symbols=("BTCUSDT", "ETHUSDT"),
        clock=clock,
        cursors={"BTCUSDT": restored_cursor},
        seen={
            "BTCUSDT:1": restored_cursor - 1_000,
            "BTCUSDT:9": restored_cursor - 5_000_000,  # older than the window: pruned
        },
    )

    state = watcher.state()
    assert state["cursors"] == {"BTCUSDT": restored_cursor, "ETHUSDT": _START_MS}
    assert state["seen"] == {"BTCUSDT:1": restored_cursor - 1_000}

    await watcher.poll_once()

    assert [event.order_id for event in recorder.events] == [2]  # R52b across a restore
    assert watcher.state()["seen"] == {
        "BTCUSDT:1": restored_cursor - 1_000,
        "BTCUSDT:2": restored_cursor + 1_000,
    }
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS, "ETHUSDT": _START_MS}


async def test_stale_clamped_symbol_cannot_cause_a_duplicate_delivery() -> None:
    """R52b/R52e: the prune floor is min(cursor) - overlap, so even a >7-day-stale
    symbol (whose clamp window reaches back over an already-delivered key) never
    re-delivers it: pruning can only drop keys no later window can return."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    stale_cursor = _START_MS - _EIGHT_DAYS_MS
    fetch.errors["ETHUSDT"] = TimeoutError("binance timeout")  # its cursor stays stale
    fetch.orders["BTCUSDT"] = [_order(order_id=21, update_time_ms=_START_MS - 100_000)]
    clock = _Clock()
    watcher = _make_watcher(
        fetch,
        recorder,
        symbols=("BTCUSDT", "ETHUSDT"),
        clock=clock,
        cursors={"ETHUSDT": stale_cursor},
    )

    await watcher.poll_once()

    assert [event.order_id for event in recorder.events] == [21]
    assert fetch.requests[1] == (
        "ETHUSDT",
        _START_MS - _LOOKBACK_CAP_MS + _CLAMP_SLACK_MS,
        _START_MS,
    )
    assert watcher.state()["cursors"]["ETHUSDT"] == stale_cursor

    # BTCUSDT's window re-returns the row on every later cycle and it stays seen:
    # the prune floor is the stale ETHUSDT cursor, well below the row, not the
    # fresh cursor that pruning against BTCUSDT alone would have used.
    for _ in range(3):
        clock.advance(600)
        await watcher.poll_once()

    assert [event.order_id for event in recorder.events] == [21]
    assert watcher.state()["seen"] == {"BTCUSDT:21": _START_MS - 100_000}
    assert watcher.state()["cursors"]["ETHUSDT"] == stale_cursor


async def test_seen_key_on_the_exact_prune_floor_is_never_re_delivered() -> None:
    """R50/R52b: the prune floor is inclusive (``update_ms >= min(cursor) - overlap``).

    The floor is exactly where the next request window starts, and Binance's
    ``startTime`` is inclusive, so a row whose ``updateTime`` equals the floor is
    returned again. Pruning its key would re-deliver an already-alerted close and
    break at-most-once delivery (R52b).
    """
    restored_cursor = _START_MS - 60_000
    floor_ms = restored_cursor - _OVERLAP_MS
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [
        _order(order_id=41, update_time_ms=floor_ms),
        _order(order_id=42, update_time_ms=_START_MS - 1_000),
    ]
    clock = _Clock()
    watcher = _make_watcher(
        fetch,
        recorder,
        clock=clock,
        cursors={"BTCUSDT": restored_cursor},
        seen={"BTCUSDT:41": floor_ms},
    )

    # The key sits exactly on the floor, so the constructor's prune must keep it.
    assert watcher.state()["seen"] == {"BTCUSDT:41": floor_ms}

    # The window starts exactly on that floor, so the fetch hands row 41 back.
    await watcher.poll_once()
    assert fetch.requests == [("BTCUSDT", floor_ms, _START_MS)]
    assert [event.order_id for event in recorder.events] == [42]
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS}


async def test_hung_fetch_times_out_for_its_own_symbol_only() -> None:
    """R52f: a fetch that never returns is cut off after ``fetch_timeout_s``.

    The timeout counts as that symbol's failure (R52 backoff, one degraded warning
    at the third), the cursor stays put, and a healthy symbol emitting in the same
    cycle is unaffected. The injected timeout is short — production's is 30 s.
    """
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [_order(order_id=31, update_time_ms=_START_MS - 1_000)]
    clock = _Clock()

    async def hanging_fetch(symbol: str, start_ms: int, end_ms: int) -> list[ForceOrder]:
        fetch.requests.append((symbol, start_ms, end_ms))
        if symbol == "ETHUSDT":
            await asyncio.Event().wait()  # a request that never comes back
        return list(fetch.orders.get(symbol, []))

    watcher = _make_watcher(
        hanging_fetch,
        recorder,
        symbols=("BTCUSDT", "ETHUSDT"),
        clock=clock,
        fetch_timeout_s=0.05,
    )

    await asyncio.wait_for(watcher.poll_once(), timeout=2)

    assert [event.order_id for event in recorder.events] == [31]
    assert [symbol for symbol, _, _ in fetch.requests] == ["BTCUSDT", "ETHUSDT"]
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS, "ETHUSDT": _START_MS}

    for _ in range(2):
        clock.advance(600)
        await asyncio.wait_for(watcher.poll_once(), timeout=2)

    assert [event.order_id for event in recorder.events] == [31]  # R52b
    assert watcher.state()["cursors"]["ETHUSDT"] == _START_MS
    assert recorder.warnings == ["liquidation detector degraded: ETHUSDT TimeoutError"]


async def test_hung_first_symbol_times_out_without_skipping_the_next_symbol() -> None:
    """R52f/R52: a hung symbol is cut off, then the loop moves on to the next symbol.

    Symbols are polled in order, so leaving the per-symbol loop when a fetch times
    out — instead of recording that symbol's failure and continuing — would silently
    drop every symbol after the hung one for the whole cycle. The hung symbol comes
    FIRST here: ETHUSDT's row is still delivered in the same cycle, and BTCUSDT still
    counts exactly one failure — its cursor stays put and its retry lands on the R52
    60 s backoff, not a second early.
    """
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["ETHUSDT"] = [
        _order(symbol="ETHUSDT", order_id=51, update_time_ms=_START_MS - 1_000),
    ]
    clock = _Clock()

    async def hanging_fetch(symbol: str, start_ms: int, end_ms: int) -> list[ForceOrder]:
        fetch.requests.append((symbol, start_ms, end_ms))
        if symbol == "BTCUSDT":
            await asyncio.Event().wait()  # a request that never comes back
        return list(fetch.orders.get(symbol, []))

    watcher = _make_watcher(
        hanging_fetch,
        recorder,
        symbols=("BTCUSDT", "ETHUSDT"),
        clock=clock,
        fetch_timeout_s=0.05,
    )

    await asyncio.wait_for(watcher.poll_once(), timeout=2)

    assert [symbol for symbol, _, _ in fetch.requests] == ["BTCUSDT", "ETHUSDT"]
    assert [event.order_id for event in recorder.events] == [51]
    assert recorder.warnings == []  # one failure is below degraded_after
    assert watcher.state()["cursors"] == {"BTCUSDT": _START_MS, "ETHUSDT": _START_MS}

    # Exactly one failure counted for the hung symbol: it is not retried before its
    # 60 s backoff...
    clock.advance(59)
    await asyncio.wait_for(watcher.poll_once(), timeout=2)
    assert [symbol for symbol, _, _ in fetch.requests] == ["BTCUSDT", "ETHUSDT"]

    # ... and is due again exactly on it.
    clock.advance(1)
    await asyncio.wait_for(watcher.poll_once(), timeout=2)
    assert [symbol for symbol, _, _ in fetch.requests].count("BTCUSDT") == 2
    assert [event.order_id for event in recorder.events] == [51]  # R52b


_ORDER_API = frozenset(
    {
        "submit",
        "submit_order",
        "cancel",
        "cancel_all",
        "close",
        "halt",
        "flatten",
        "modify_order",
        "on_order",
    },
)


class _OrderSpy:
    """The node surface the watcher must never use for orders (R49).

    The injected fetch and the three handlers are bound methods of this object,
    exactly as P3b will wire them, so the watcher does hold a live reference to it
    (``handler.__self__``). Any attribute the watcher reaches for that the class
    does not define lands in :meth:`__getattr__` and is recorded — including every
    order-path name — so the assertions below can actually fail. The order names
    are deliberately absent from the class, which keeps the ``dir()`` checks honest.
    """

    def __init__(self, recorder: _Recorder, fetch: _FakeFetch) -> None:
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_fetch", fetch)
        object.__setattr__(self, "order_accesses", [])
        object.__setattr__(self, "other_accesses", [])

    def __getattr__(self, name: str) -> object:
        # Reached only for names the class does not define.
        if name in _ORDER_API:
            object.__getattribute__(self, "order_accesses").append(name)
            raise AssertionError(f"the liquidation watcher reached the order path via {name!r}")
        object.__getattribute__(self, "other_accesses").append(name)
        raise AttributeError(name)

    async def fetch(self, symbol: str, start_ms: int, end_ms: int) -> list[ForceOrder]:
        return await object.__getattribute__(self, "_fetch")(symbol, start_ms, end_ms)

    async def on_liquidation(self, event: ForceOrder) -> None:
        await object.__getattribute__(self, "_recorder").on_liquidation(event)

    async def on_warning(self, message: str) -> None:
        await object.__getattribute__(self, "_recorder").on_warning(message)

    async def on_recovered(self, symbol: str, failures: int) -> None:
        await object.__getattribute__(self, "_recorder").on_recovered(symbol, failures)


async def test_watcher_has_no_strategy_guard_or_exec_reference() -> None:
    """R49: a full liquidation cycle reaches only the injected fetch and handlers.

    Everything the watcher is given (fetch, handlers, clock) is either the spy or
    a bound method of it, so a cycle that reached for ``submit_order``,
    ``cancel_all``, ``close`` or ``halt`` would be recorded and fail below.
    """
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [_order(order_id=11, update_time_ms=_START_MS - 1_000)]
    spy = _OrderSpy(recorder, fetch)
    clock = _Clock()
    watcher = LiquidationWatcher(
        spy.fetch,
        ["BTCUSDT"],
        on_liquidation=spy.on_liquidation,
        on_warning=spy.on_warning,
        on_recovered=spy.on_recovered,
        now_ms=clock,
        cursors=None,
        seen=None,
    )

    await watcher.poll_once()

    assert [event.order_id for event in recorder.events] == [11]
    assert spy.order_accesses == []
    assert spy.other_accesses == []
    # No stored attribute — callable or not — exposes an order path, and every
    # callable it holds is one of the injected ones.
    assert not (_ORDER_API & set(dir(watcher)))
    for name, value in vars(watcher).items():
        owners = (value, getattr(value, "__self__", None))
        for owner in owners:
            if owner is None or owner is watcher:
                continue
            assert not (_ORDER_API & set(dir(owner))), name
    injected = {spy.fetch, spy.on_liquidation, spy.on_warning, spy.on_recovered, clock}
    callables = {name for name, value in vars(watcher).items() if callable(value)}
    assert all(getattr(watcher, name) in injected for name in callables)


async def test_run_returns_immediately_when_stop_is_already_set() -> None:
    """R53: a stop event that is already set ends the loop before any request."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    watcher = _make_watcher(fetch, recorder)
    stop = asyncio.Event()
    stop.set()

    await asyncio.wait_for(watcher.run(stop), timeout=1)

    assert fetch.requests == []
    assert recorder.events == []


async def test_run_exits_on_stop_set_by_a_handler_without_sleeping_the_interval() -> None:
    """R53: run waits on the stop event, so a 60 s interval never has to elapse."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [_order(order_id=12, update_time_ms=_START_MS - 1_000)]
    stop = asyncio.Event()

    async def on_liquidation(event: ForceOrder) -> None:
        recorder.events.append(event)
        stop.set()

    watcher = _make_watcher(fetch, recorder, on_liquidation=on_liquidation)

    await asyncio.wait_for(watcher.run(stop), timeout=1)

    assert [event.order_id for event in recorder.events] == [12]
    assert len(fetch.requests) == 1


async def test_run_propagates_cancellation() -> None:
    """R53: cancelling the watch task propagates CancelledError out of run."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    started = asyncio.Event()

    async def hanging_fetch(symbol: str, start_ms: int, end_ms: int) -> list[ForceOrder]:
        fetch.requests.append((symbol, start_ms, end_ms))
        started.set()
        await asyncio.Event().wait()  # a request that never comes back
        return []

    watcher = _make_watcher(hanging_fetch, recorder)
    stop = asyncio.Event()
    task = asyncio.create_task(watcher.run(stop))
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(fetch.requests) == 1


_LOGGER_NAME = "vibe_quant.paper.liquidation_watch"
_SECRET_EXCEPTION_TEXT = (
    "GET https://testnet.binancefuture.com/fapi/v1/forceOrders"
    "?symbol=BTCUSDT&timestamp=1&signature=deadbeefcafe failed: 401 apiKey=SECRETKEY123"
)


def _logged_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The module's WARNING records, rendered, in log order (R1)."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _LOGGER_NAME and record.levelname == "WARNING"
    ]


async def test_failed_fetch_logs_one_warning_with_symbol_class_and_redacted_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1: a failed fetch logs one WARNING with the symbol, the exception class and
    the exception text — with the signed query string gone and no api key left.

    The exception text is the shape a Binance HTTP failure actually has: request URL
    with its ``?symbol=..&timestamp=..&signature=..`` query, then the key the client
    echoes. Neither the query string (so no ``signature=``) nor the key value may
    reach the log.
    """
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = RuntimeError(_SECRET_EXCEPTION_TEXT)
    watcher = _make_watcher(fetch, recorder)

    with caplog.at_level("WARNING", logger=_LOGGER_NAME):
        await watcher.poll_once()

    warnings = _logged_warnings(caplog)
    assert len(warnings) == 1  # one failure: below the R52 degraded threshold
    assert "BTCUSDT" in warnings[0]
    assert "RuntimeError" in warnings[0]
    assert "?<redacted>" in warnings[0]
    assert "deadbeefcafe" not in warnings[0]
    assert "SECRETKEY123" not in warnings[0]
    assert "signature=" not in warnings[0]


async def test_credential_mask_covers_equals_and_colon_forms_and_the_header_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1b (chief ruling): the mask covers ``name=value`` and ``name: value``, and the
    Binance request-header name ``X-MBX-APIKEY`` — no credential value reaches the log.
    """
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = RuntimeError(
        "auth failed: apiKey: COLONSECRET1, X-MBX-APIKEY=HEADERSECRET2, secret: COLONSECRET3",
    )
    watcher = _make_watcher(fetch, recorder)

    with caplog.at_level("WARNING", logger=_LOGGER_NAME):
        await watcher.poll_once()

    warnings = _logged_warnings(caplog)
    assert len(warnings) == 1
    assert "auth failed" in warnings[0]
    assert "COLONSECRET1" not in warnings[0]
    assert "HEADERSECRET2" not in warnings[0]
    assert "COLONSECRET3" not in warnings[0]


async def test_credential_mask_covers_a_quoted_mapping_repr(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1c (chief ruling): the quoted mapping form masks too -- ``{'apiKey': 'S'}``.

    This is the shape an httpx/requests failure dump has: name and value both quoted,
    so a mask that requires ``=``/``:`` directly after the name leaves the key in the
    log. Both the Binance header name and a bare ``apiKey`` are pinned.
    """
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = RuntimeError(
        "auth failed: {'X-MBX-APIKEY': 'DICTREPRSECRET', 'apiKey': 'DICTSECRET2'}",
    )
    watcher = _make_watcher(fetch, recorder)

    with caplog.at_level("WARNING", logger=_LOGGER_NAME):
        await watcher.poll_once()

    warnings = _logged_warnings(caplog)
    assert len(warnings) == 1
    assert "auth failed" in warnings[0]
    assert "DICTREPRSECRET" not in warnings[0]
    assert "DICTSECRET2" not in warnings[0]


async def test_failure_log_truncates_the_exception_text_to_300_chars(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1: the redacted exception text is cut to 300 characters."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = RuntimeError("x" * 1_000)
    watcher = _make_watcher(fetch, recorder)

    with caplog.at_level("WARNING", logger=_LOGGER_NAME):
        await watcher.poll_once()

    warnings = _logged_warnings(caplog)
    assert len(warnings) == 1
    assert "x" * 300 in warnings[0]
    assert "x" * 301 not in warnings[0]


async def test_every_failure_is_logged_and_the_degraded_alert_text_is_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1/R52: one WARNING per failure — not only at the degraded threshold — and the
    R52 degraded alert text is untouched by the new per-failure log."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = TimeoutError("binance timeout")
    clock = _Clock()
    watcher = _make_watcher(fetch, recorder, clock=clock)

    with caplog.at_level("WARNING", logger=_LOGGER_NAME):
        for _ in range(3):
            clock.advance(600)
            await watcher.poll_once()

    assert len(_logged_warnings(caplog)) == 3
    assert recorder.warnings == ["liquidation detector degraded: BTCUSDT TimeoutError"]


async def test_cancelled_fetch_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """R1: cancellation is not a failure to report — poll_once re-raises it, no WARNING."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = asyncio.CancelledError()
    watcher = _make_watcher(fetch, recorder)

    with caplog.at_level("WARNING", logger=_LOGGER_NAME), pytest.raises(asyncio.CancelledError):
        await watcher.poll_once()

    assert _logged_warnings(caplog) == []


@pytest.mark.parametrize(
    "text",
    [
        'headers={"token": "Bearer TOKENSECRET123"}',
        'request failed: {"apiKey": " SECRETKEY123 with spaces"}',
        "auth failed: Authorization: Bearer ZZLEAK9 more",
        "boom signature=SIGSECRET123",
    ],
    ids=["quoted-bearer-token", "quoted-leading-space-key", "authorization-header", "signature"],
)
async def test_no_secret_substring_reaches_the_formatted_log(
    caplog: pytest.LogCaptureFixture, text: str
) -> None:
    """R1e: from the first credential name on, the text is cut -- quoted or
    space-containing values cannot leak. Asserted on ``caplog.text`` (formatted)."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = RuntimeError(text)
    watcher = _make_watcher(fetch, recorder)

    with caplog.at_level("DEBUG", logger=_LOGGER_NAME):
        await watcher.poll_once()

    assert "BTCUSDT" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "SECRET" not in caplog.text
    assert "ZZLEAK9" not in caplog.text
    assert "<redacted>" in caplog.text


async def test_failing_degraded_alert_never_logs_the_fetch_exception_chain(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1f: the 3rd failure's on_warning raises OSError; the handler-failure log must
    carry no traceback, so the active fetch exception (signed URL, apiKey) stays out."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = RuntimeError(_SECRET_EXCEPTION_TEXT)
    clock = _Clock()

    async def disk_full(message: str) -> None:
        raise OSError("disk full")

    watcher = _make_watcher(fetch, recorder, clock=clock, on_warning=disk_full)

    with caplog.at_level("DEBUG", logger=_LOGGER_NAME):
        for _ in range(3):
            clock.advance(600)
            await watcher.poll_once()

    assert "alert handler failed" in caplog.text
    assert "OSError" in caplog.text
    assert "Traceback" not in caplog.text
    for secret in ("SECRETKEY123", "deadbeefcafe", "signature="):
        assert secret not in caplog.text


#: R1f leak markers: contain no credential name, so redaction cannot strip them
#: (a marker like ``BEARERSECRET`` would be cut at ``bearer`` and hide a leak).
_LEAK_EXCEPTION_TEXT = (
    "GET https://testnet.binancefuture.com/fapi/v1/forceOrders"
    "?symbol=BTCUSDT&signature=ZZLEAK9 failed: 401 apiKey=QQLEAK7"
)


class _ArmedClock(_Clock):
    """A clock that raises :data:`_LEAK_EXCEPTION_TEXT` on its next call once armed.

    Arming it from the fetch makes the failure bookkeeping raise *after* the fetch
    error is handled, so the exception escaping to :meth:`LiquidationWatcher.run`
    carries the secret-bearing fetch error as its ``__context__`` (the R1f chain).
    """

    def __init__(self) -> None:
        super().__init__()
        self.armed = False

    def __call__(self) -> int:
        if self.armed:
            self.armed = False
            raise RuntimeError(_LEAK_EXCEPTION_TEXT)
        return self.ms


async def test_run_cycle_failure_log_is_redacted_and_has_no_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1f: run()'s cycle-failure log never carries ``exc_info``.

    A failing fetch is handled per symbol inside poll_once, but an error in the
    failure bookkeeping that runs while that fetch error is still active escapes
    with the secret-bearing fetch error as its ``__context__``. run() must log only
    the redacted text of the escaping error: with ``logger.exception`` the traceback
    would print the raw message (signed query, key) and 'Traceback'.
    """
    recorder = _Recorder()
    clock = _ArmedClock()

    async def arming_fetch(symbol: str, start_ms: int, end_ms: int) -> list[ForceOrder]:
        clock.armed = True  # the next now_ms() call, in _note_failure, raises
        raise RuntimeError(_LEAK_EXCEPTION_TEXT)

    watcher = _make_watcher(arming_fetch, recorder, clock=clock)
    stop = asyncio.Event()
    asyncio.get_running_loop().call_soon(stop.set)

    with caplog.at_level("DEBUG", logger=_LOGGER_NAME):
        await asyncio.wait_for(watcher.run(stop), timeout=1)

    assert "liquidation watch cycle failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "Traceback" not in caplog.text
    assert "ZZLEAK9" not in caplog.text
    assert "QQLEAK7" not in caplog.text
    assert "signature=" not in caplog.text


# --------------------------------------------------- 0m3tf carried cursors


async def test_untraded_restored_cursor_is_carried_through_state() -> None:
    """0m3tf/W1: ``state()`` keeps a restored cursor for a symbol this session skips.

    That symbol is watched after the traded ones.
    """
    recorder = _Recorder()
    fetch = _FakeFetch()
    btc_cursor = _START_MS - 60_000
    watcher = _make_watcher(
        fetch,
        recorder,
        symbols=("ETHUSDT",),
        cursors={"BTCUSDT": btc_cursor, "ETHUSDT": _START_MS},
        seen={"BTCUSDT:1": btc_cursor},
    )

    assert watcher.state() == {
        "cursors": {"BTCUSDT": btc_cursor, "ETHUSDT": _START_MS},
        "seen": {"BTCUSDT:1": btc_cursor},
    }
    assert watcher._symbols == ["ETHUSDT", "BTCUSDT"]


async def test_carried_symbol_is_polled_after_the_traded_ones_every_cycle() -> None:
    """1cvw6/R1: a restored untraded symbol is polled after the traded ones, every cycle."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    sol_cursor = _START_MS - 120_000
    btc_cursor = _START_MS - 60_000
    clock = _Clock()
    watcher = _make_watcher(
        fetch,
        recorder,
        symbols=("ETHUSDT",),
        clock=clock,
        cursors={"SOLUSDT": sol_cursor, "BTCUSDT": btc_cursor, "ETHUSDT": _START_MS},
    )

    await watcher.poll_once()
    clock.advance(60)
    await watcher.poll_once()

    assert fetch.requests == [
        ("ETHUSDT", _START_MS - _OVERLAP_MS, _START_MS),
        ("BTCUSDT", btc_cursor - _OVERLAP_MS, _START_MS),
        ("SOLUSDT", sol_cursor - _OVERLAP_MS, _START_MS),
        ("ETHUSDT", _START_MS - _OVERLAP_MS, _START_MS + 60_000),
        ("BTCUSDT", _START_MS - _OVERLAP_MS, _START_MS + 60_000),
        ("SOLUSDT", _START_MS - _OVERLAP_MS, _START_MS + 60_000),
    ]
    assert watcher.state()["cursors"] == {
        "ETHUSDT": _START_MS + 60_000,
        "BTCUSDT": _START_MS + 60_000,
        "SOLUSDT": _START_MS + 60_000,
    }


async def test_carried_symbol_has_its_own_backoff_degraded_and_recovered() -> None:
    """1cvw6/R3: a restored untraded symbol backs off, degrades and recovers like a traded one."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    btc_cursor = _START_MS - 60_000
    clock = _Clock()
    watcher = _make_watcher(
        fetch,
        recorder,
        symbols=("ETHUSDT",),
        clock=clock,
        cursors={"BTCUSDT": btc_cursor, "ETHUSDT": _START_MS},
    )
    fetch.errors["BTCUSDT"] = TimeoutError("binance timeout")

    for _ in range(3):
        clock.advance(600)
        await watcher.poll_once()

    assert recorder.warnings == [
        "liquidation detector degraded: BTCUSDT TimeoutError (not traded this session)"
    ]
    assert recorder.recoveries == []
    assert watcher.state()["cursors"] == {
        "ETHUSDT": _START_MS + 1_800_000,
        "BTCUSDT": btc_cursor,
    }
    assert [symbol for symbol, _, _ in fetch.requests] == ["ETHUSDT", "BTCUSDT"] * 3

    del fetch.errors["BTCUSDT"]
    clock.advance(600)
    await watcher.poll_once()

    assert recorder.recoveries == [("BTCUSDT", 3)]
    assert watcher.state()["cursors"] == {
        "ETHUSDT": _START_MS + 2_400_000,
        "BTCUSDT": _START_MS + 2_400_000,
    }
    assert fetch.requests[-1] == (
        "BTCUSDT",
        btc_cursor - _OVERLAP_MS,
        _START_MS + 2_400_000,
    )


async def test_stale_carried_cursor_is_clamped_with_one_warning() -> None:
    """1cvw6/R4: a carried cursor older than the 7-day cap is clamped, one warning only."""
    recorder = _Recorder()
    fetch = _FakeFetch()
    clock = _Clock()
    stale = _START_MS - _EIGHT_DAYS_MS
    fetch.orders["BTCUSDT"] = [
        _order(order_id=1, update_time_ms=stale + 1_000),
        _order(order_id=2, update_time_ms=_START_MS - 3 * 86_400_000),
    ]
    watcher = _make_watcher(
        fetch,
        recorder,
        symbols=("ETHUSDT",),
        clock=clock,
        cursors={"BTCUSDT": stale, "ETHUSDT": _START_MS},
    )

    await watcher.poll_once()
    clock.advance(60)
    await watcher.poll_once()

    assert fetch.requests[1] == (
        "BTCUSDT",
        _START_MS - _LOOKBACK_CAP_MS + _CLAMP_SLACK_MS,
        _START_MS,
    )
    assert recorder.warnings == [
        "liquidation detector lookback clamped to 7 days: BTCUSDT (not traded this session)"
    ]
    assert [event.order_id for event in recorder.events] == [2]
    assert watcher.state()["cursors"] == {
        "ETHUSDT": _START_MS + 60_000,
        "BTCUSDT": _START_MS + 60_000,
    }


async def test_disjoint_chain_keeps_the_untraded_cursor_and_its_seen_key() -> None:
    """0m3tf/W3: B watches the carried BTC and delivers row 8; C does not re-deliver."""
    recorder_a = _Recorder()
    recorder_b = _Recorder()
    recorder_c = _Recorder()
    fetch = _FakeFetch()
    fetch.orders["BTCUSDT"] = [
        _order(order_id=7, update_time_ms=_START_MS - 5_000),
        _order(order_id=8, update_time_ms=_START_MS + 60_000),
    ]
    clock = _Clock()

    first = _make_watcher(fetch, recorder_a, symbols=("BTCUSDT",), clock=clock)
    await first.poll_once()
    assert [event.order_id for event in recorder_a.events] == [7]
    state_a = first.state()
    assert state_a["cursors"]["BTCUSDT"] == _START_MS

    clock.advance(60)
    second = _make_watcher(
        fetch,
        recorder_b,
        symbols=("ETHUSDT",),
        clock=clock,
        cursors=state_a["cursors"],
        seen=state_a["seen"],
    )
    await second.poll_once()
    state_b = second.state()
    assert [event.order_id for event in recorder_b.events] == [8]
    assert state_b["cursors"] == {
        "BTCUSDT": _START_MS + 60_000,
        "ETHUSDT": _START_MS + 60_000,
    }

    clock.advance(60)
    third = _make_watcher(
        fetch,
        recorder_c,
        symbols=("BTCUSDT",),
        clock=clock,
        cursors=state_b["cursors"],
        seen=state_b["seen"],
    )
    await third.poll_once()

    btc_requests = [request for request in fetch.requests if request[0] == "BTCUSDT"]
    assert btc_requests[-1][1] == _START_MS + 60_000 - _OVERLAP_MS
    assert recorder_c.events == []
    assert third.state()["cursors"] == {
        "BTCUSDT": _START_MS + 120_000,
        "ETHUSDT": _START_MS + 120_000,
    }


async def test_overlapping_chain_carries_every_symbol_without_duplicates() -> None:
    """0m3tf/W4: A(BTC,ETH) -> B(ETH,SOL) -> C(BTC,SOL) keeps all three cursors."""
    recorder_a = _Recorder()
    recorder_b = _Recorder()
    recorder_c = _Recorder()
    fetch = _FakeFetch()
    clock = _Clock()

    first = _make_watcher(fetch, recorder_a, symbols=("BTCUSDT", "ETHUSDT"), clock=clock)
    await first.poll_once()
    state_a = first.state()
    assert state_a["cursors"] == {"BTCUSDT": _START_MS, "ETHUSDT": _START_MS}

    clock.advance(60)
    second = _make_watcher(
        fetch,
        recorder_b,
        symbols=("ETHUSDT", "SOLUSDT"),
        clock=clock,
        cursors=state_a["cursors"],
        seen=state_a["seen"],
    )
    await second.poll_once()
    state_b = second.state()
    assert state_b["cursors"] == {
        "BTCUSDT": _START_MS + 60_000,
        "ETHUSDT": _START_MS + 60_000,
        "SOLUSDT": _START_MS + 60_000,
    }

    clock.advance(60)
    third = _make_watcher(
        fetch,
        recorder_c,
        symbols=("BTCUSDT", "SOLUSDT"),
        clock=clock,
        cursors=state_b["cursors"],
        seen=state_b["seen"],
    )
    await third.poll_once()

    assert third.state()["cursors"] == {
        "BTCUSDT": _START_MS + 120_000,
        "ETHUSDT": _START_MS + 120_000,
        "SOLUSDT": _START_MS + 120_000,
    }


async def test_carried_stale_cursor_does_not_pin_the_seen_prune_floor() -> None:
    """0m3tf/W5: the seen floor is clamped at ``now - lookback_cap`` (W1 ruling)."""
    day_ms = 86_400_000
    recorder = _Recorder()
    fetch = _FakeFetch()
    watcher = _make_watcher(
        fetch,
        recorder,
        symbols=("ETHUSDT",),
        clock=_Clock(),
        cursors={"BTCUSDT": _START_MS - 30 * day_ms, "ETHUSDT": _START_MS},
        seen={
            "BTCUSDT:1": _START_MS - 10 * day_ms,
            "BTCUSDT:2": _START_MS - day_ms,
            "ETHUSDT:3": _START_MS - 60_000,
        },
    )

    assert watcher.state() == {
        "cursors": {"BTCUSDT": _START_MS - 30 * day_ms, "ETHUSDT": _START_MS},
        "seen": {"BTCUSDT:2": _START_MS - day_ms, "ETHUSDT:3": _START_MS - 60_000},
    }


async def test_carried_symbol_later_traded_is_polled_and_not_duplicated() -> None:
    """0m3tf/W7: a carried cursor becomes a normal traded cursor, once."""
    recorder_a = _Recorder()
    fetch = _FakeFetch()
    clock = _Clock()
    btc_cursor = _START_MS - 60_000
    first = _make_watcher(
        fetch,
        recorder_a,
        symbols=("ETHUSDT",),
        clock=clock,
        cursors={"BTCUSDT": btc_cursor, "ETHUSDT": _START_MS},
    )
    state = first.state()

    recorder_b = _Recorder()
    second = _make_watcher(
        fetch,
        recorder_b,
        symbols=("BTCUSDT", "ETHUSDT"),
        clock=clock,
        cursors=state["cursors"],
        seen=state["seen"],
    )
    await second.poll_once()

    assert fetch.requests == [
        ("BTCUSDT", btc_cursor - _OVERLAP_MS, _START_MS),
        ("ETHUSDT", _START_MS - _OVERLAP_MS, _START_MS),
    ]
    assert second.state() == {"cursors": {"BTCUSDT": _START_MS, "ETHUSDT": _START_MS}, "seen": {}}


async def test_failed_alert_handler_log_text_is_redacted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1f: a raising alert handler is logged with its error text redacted.

    The handler's own error text has the same signed-URL/key shape the fetch error
    does, so it must go through the redactor. The existing 'disk full' test cannot
    catch a regression to raw ``str(exc)``: its message carries no secret.
    """
    recorder = _Recorder()
    fetch = _FakeFetch()
    fetch.errors["BTCUSDT"] = TimeoutError("binance timeout")
    clock = _Clock()

    async def leaky_warning(message: str) -> None:
        raise RuntimeError(_LEAK_EXCEPTION_TEXT)

    watcher = _make_watcher(fetch, recorder, clock=clock, on_warning=leaky_warning)

    with caplog.at_level("DEBUG", logger=_LOGGER_NAME):
        for _ in range(3):
            clock.advance(600)
            await watcher.poll_once()

    assert "alert handler failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "ZZLEAK9" not in caplog.text
    assert "QQLEAK7" not in caplog.text
    assert "signature=" not in caplog.text
