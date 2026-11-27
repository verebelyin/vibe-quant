"""Paper warm-up on the REAL NautilusTrader ``TradingNode`` (design.md R15, R17, R9 scheduling).

The exec client is a stub (no network); the strategy, risk engine, exec engine, data engine and
the paper ``TradingGuard`` are the live ones, and so is the REST seed path (only the HTTP fetch is
faked). A long position is adopted from the "venue" at connect:

- R15: the ATR-based stop of the adopted position is placed during ``on_start`` from the warm ATR
  (before any live bar); without the seed the ATR is cold and no stop could be placed.
- R9: the first live bar sits three bars after the seed: the node fetches the two missing bars off
  the loop, the strategy resumes on the next contiguous bar and enters.
- R17: that top-up fails (raises, or keeps answering short): the session is KEPT with entries off,
  the venue SL/TP stay open, nothing is cancelled, flattened or submitted, the node is not stopped.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderType, PositionSide
from nautilus_trader.model.objects import Price, Quantity

from tests.conftest import synthetic_klines
from tests.unit.paper_venue_harness import inject_matching_read
from tests.unit.test_paper_atr_restart_real_node import (
    _ADOPTED_QTY,
    _AdoptingExecClient,
    _AdoptingExecFactory,
    _BarDataFactory,
    _entries,
    _protection,
    _until,
    _Venue,
)
from tests.unit.test_paper_halt_restore import (
    _write_checkpoint,
    node_timing,  # noqa: F401  # autouse fixture
)
from tests.unit.test_paper_margin_refusal_real_node import _FastNodeConfig
from tests.unit.test_paper_warmup_node import _make, _source
from vibe_quant.data.catalog import get_bar_type
from vibe_quant.data.downloader import RestDownloadError
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.paper import node as node_mod
from vibe_quant.paper import warmup
from vibe_quant.paper.guard import GuardState
from vibe_quant.paper.node import NodeState

if TYPE_CHECKING:
    from pathlib import Path

_STEP_NS = 3_600_000_000_000  # 1h
_SOURCE = _source(
    "1h",
    {"ema": {"type": "EMA", "period": 2}, "atr": {"type": "ATR", "period": 14}},
    sl={"type": "atr_fixed", "atr_multiplier": 2.0, "indicator": "atr"},
)

_FAILED_TEXT = (
    "BTCUSDT 1h: warm-up top-up failed after 3 tries; entries OFF and signal exits PAUSED; "
    "venue SL/TP still active. Restarting paper closes open positions at market; "
    "to keep the position, let its SL/TP close it, then restart."
)


class _Cancels:
    cancelled: list[Any] = []  # noqa: RUF012 - reset per test


class _RecordingExecClient(_AdoptingExecClient):
    async def _cancel_order(self, command: Any) -> None:
        _Cancels.cancelled.append(command)

    async def _cancel_all_orders(self, command: Any) -> None:
        _Cancels.cancelled.append(command)


class _RecordingExecFactory(_AdoptingExecFactory):
    @staticmethod
    def create(  # type: ignore[override]
        loop: Any, name: str, config: Any, msgbus: Any, cache: Any, clock: Any
    ) -> Any:
        client = _RecordingExecClient(loop, msgbus, cache, clock)
        _AdoptingExecFactory.created = client
        return client


def _live_bar(strategy: Any, steps_after_seed: int) -> Bar:
    """The live bar ``steps_after_seed`` bars after the seed's last bar (same stamp the adapter uses)."""
    bar_type = get_bar_type("BTCUSDT", "1h")
    ts = strategy._warm_wm[bar_type] + steps_after_seed * _STEP_NS
    return Bar(
        bar_type,
        Price.from_str("40000.0"),
        Price.from_str("40025.0"),
        Price.from_str("39975.0"),
        Price.from_str("40000.0"),
        Quantity.from_str("10.000"),
        ts,
        ts,
    )


class _Rest:
    """The faked REST endpoint: seed rows always; gap rows per ``mode``."""

    def __init__(self) -> None:
        self.mode = "ok"  # ok | raise | short
        self.gap_calls = 0
        self.seeded = False

    def __call__(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Any]:
        if not self.seeded:  # the first call is the pre-node seed
            self.seeded = True
            return synthetic_klines(symbol, interval, start_ms, end_ms)
        self.gap_calls += 1
        if self.mode == "raise":
            raise RestDownloadError("down", [], start_ms)
        if self.mode == "short":
            return []
        return synthetic_klines(symbol, interval, start_ms, end_ms)


async def _run_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rest: _Rest
) -> tuple[Any, Any, Any, asyncio.Task[None]]:
    import nautilus_trader.adapters.binance.factories as binance_factories

    _Venue.submitted = []
    _Cancels.cancelled = []
    monkeypatch.setattr("vibe_quant.paper.warmup.fetch_closed_klines", rest)
    monkeypatch.setattr(warmup, "RETRY_SLEEP_S", 0.0)
    monkeypatch.setattr(binance_factories, "BinanceLiveExecClientFactory", _RecordingExecFactory)
    monkeypatch.setattr(binance_factories, "BinanceLiveDataClientFactory", _BarDataFactory)
    monkeypatch.setattr("nautilus_trader.live.config.TradingNodeConfig", _FastNodeConfig)
    # The stub exec client is not a Binance one (it never sends): no write gate to install.
    monkeypatch.setattr(
        "vibe_quant.paper.node.PaperTradingNode._install_write_gate", lambda self, node: None
    )

    async def _idle_watch() -> None:
        await asyncio.Event().wait()

    node = _make(tmp_path, _SOURCE, ["BTCUSDT"])
    # opspt P2: the adopted long is the bot's own from its previous run (its saved record).
    _write_checkpoint(
        tmp_path / "state.db",
        {"state": "running", "owned_net": {"BTCUSDT": _ADOPTED_QTY}, "own_orders": {}},
    )
    monkeypatch.setattr(node, "_liquidation_loop", _idle_watch)
    alerts: list[Any] = []
    real_record = node._journal.record

    def record(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("alert") is not None:
            alerts.append(kwargs["alert"])
        return real_record(*args, **kwargs)

    monkeypatch.setattr(node._journal, "record", record)
    await node._initialize()
    inject_matching_read(monkeypatch, node)  # 68m53: the venue answers the resolved config

    async def held() -> object:
        """positionRisk = what the session holds: the venue holds the adopted position. A flat
        answer latched a position drift when the venue pass ran after adoption (under load)."""
        net = node._session_net_positions(["BTCUSDT"])
        rows = {} if isinstance(net, str) else net
        return [
            {"symbol": s, "positionSide": "BOTH", "positionAmt": format(a, "f")}
            for s, a in rows.items()
        ]

    monkeypatch.setattr(node, "_build_position_read", lambda: held)
    strategy: Any = node._compiled_strategies[0]
    _Venue.adopt_for = strategy.id
    run_loop = asyncio.create_task(node._run_loop())
    return node, strategy, alerts, run_loop


async def _stop(node: Any, run_loop: asyncio.Task[None]) -> None:
    trading_node = node._trading_node
    node._shutdown_event.set()
    try:
        await asyncio.wait_for(run_loop, timeout=10.0)
    finally:
        if not run_loop.done():
            run_loop.cancel()
            await asyncio.gather(run_loop, return_exceptions=True)
        await asyncio.wait_for(node._shutdown(), timeout=5.0)
        retain_log_guard(trading_node)


async def test_adopted_position_gets_its_atr_stop_at_on_start_from_the_warm_atr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R15: restart with an adopted long and an ATR stop: the stop is placed before any live bar."""
    node, strategy, _alerts, run_loop = await _run_node(tmp_path, monkeypatch, _Rest())
    try:
        await _until(lambda: len(_protection()) >= 2, "SL and TP for the adopted position")
        stop = next(o for o in _protection() if o.order_type == OrderType.STOP_MARKET)
        atr = strategy.ind_atr.value
        assert strategy.ind_atr.initialized
        assert atr > 0.0  # synthetic seed: the warm ATR is real, not zero or NaN
        assert float(stop.trigger_price) == pytest.approx(40000.0 - 2.0 * atr, abs=0.06)
        assert float(stop.trigger_price) < 40000.0
        assert _entries() == []
    finally:
        await _stop(node, run_loop)


async def test_topup_lands_and_the_strategy_resumes_on_the_next_contiguous_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R9 (real scheduling): a first live bar 3 bars after the seed: 2 bars fetched off the loop,
    fed on it, no entry on the gap bar, an entry on the next contiguous bar."""
    rest = _Rest()
    node, strategy, alerts, run_loop = await _run_node(tmp_path, monkeypatch, rest)
    try:
        await _until(lambda: len(_protection()) >= 2, "SL and TP")
        stop = next(o for o in _protection() if o.order_type == OrderType.STOP_MARKET)
        trading_node = node._trading_node
        cache = trading_node.kernel.cache
        engine = trading_node.kernel.data_engine
        bar_type = get_bar_type("BTCUSDT", "1h")
        wm = strategy._warm_wm[bar_type]
        await _until(lambda: stop.venue_order_id is not None, "the stop to be accepted")
        # The venue pass lands before the SL fills: since opspt Q7 a positionRisk reading
        # taken before the fill and compared after it latches (no confirming re-read).
        guard: Any = node._guard
        await _until(lambda: guard.guard_state == GuardState.RUNNING, "the venue pass")
        _AdoptingExecFactory.created.fill_at(stop, "40000.0")
        await _until(lambda: not cache.positions_open(), "the SL to close the position")

        engine.process(_live_bar(strategy, 3))
        await _until(lambda: strategy._warm_wm[bar_type] == wm + 3 * _STEP_NS, "the top-up to land")
        assert rest.gap_calls == 1
        assert strategy._warm_pending == set()
        assert _entries() == []  # the gap bar itself never signals
        engine.process(_live_bar(strategy, 4))
        await _until(lambda: len(_entries()) == 1, "the entry on the next contiguous bar")
        assert node._warmup_entries_off is None
        assert alerts == []
    finally:
        await _stop(node, run_loop)


@pytest.mark.parametrize(("mode", "gap_calls"), [("raise", 3), ("short", 3)])
async def test_failed_topup_keeps_the_session_with_entries_off_and_touches_no_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, gap_calls: int
) -> None:
    """R17: the top-up cannot be completed (3 raising fetches, or 3 empty answers: the strategy
    asks for the same hole again each time, the 4th ask fails the session): the session
    stays RUNNING-or-PAUSED (never stopped), the venue SL/TP stay open, no order is submitted or
    cancelled after the failure, and a later contiguous bar sends no entry."""
    rest = _Rest()
    rest.mode = mode
    node, strategy, alerts, run_loop = await _run_node(tmp_path, monkeypatch, rest)
    try:
        await _until(lambda: len(_protection()) >= 2, "SL and TP")
        stop = next(o for o in _protection() if o.order_type == OrderType.STOP_MARKET)
        trading_node = node._trading_node
        cache = trading_node.kernel.cache
        engine = trading_node.kernel.data_engine
        protection_before = list(_protection())
        engine.process(_live_bar(strategy, 3))
        await _until(lambda: node._warmup_entries_off is not None, "the failed top-up")
        assert node._warmup_entries_off == _FAILED_TEXT
        assert rest.gap_calls == gap_calls
        assert [a.message for a in alerts] == [_FAILED_TEXT]
        assert alerts[0].critical is True
        assert node._guard.guard_state == GuardState.PAUSED
        assert node.status.state == NodeState.PAUSED
        assert node._trading_node is trading_node and not run_loop.done()  # never stopped
        submitted_at_failure = len(_Venue.submitted)
        # the venue protection is intact: same orders, still open, nothing cancelled
        assert _protection() == protection_before
        assert _Cancels.cancelled == []
        assert all(o.is_open or o.is_inflight for o in (stop,)), stop.status_string()
        assert [p.side for p in cache.positions_open()] == [PositionSide.LONG]

        # the position closes at its stop; further contiguous bars never enter (pending + paused)
        await _until(lambda: stop.venue_order_id is not None, "the stop to be accepted")
        _AdoptingExecFactory.created.fill_at(stop, "40000.0")
        await _until(lambda: not cache.positions_open(), "the SL to close the position")
        for steps in (5, 6):
            engine.process(_live_bar(strategy, steps))
            await asyncio.sleep(0.2)
        assert len(_Venue.submitted) == submitted_at_failure
        assert _entries() == []
    finally:
        await _stop(node, run_loop)


def test_node_module_has_no_cold_atr_refusal_left() -> None:
    assert not hasattr(node_mod, "_check_atr_warmup")
    assert not hasattr(node_mod, "_MAX_ATR_WARMUP_SECONDS")
