"""Paper warm-up at the node (design.md R2.2/R2.3, rows R3, R13-R15, R17, R22).

A paper start fetches the REST seed and replays it into each strategy BEFORE the
NautilusTrader node exists. A seed that cannot reach parity REFUSES the start through
``start()`` (status ERROR, critical alert, 0 node builds, 0 orders). A handover top-up that
fails after the node runs keeps the session with entries off: it must never stop the node
(``on_stop`` flattens positions in paper).

No exchange is contacted: the fetch is the autouse synthetic series of ``conftest.py`` or a
per-test fake of ``vibe_quant.paper.warmup.fetch_closed_klines``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.adapters.binance.common.enums import BinanceKlineInterval
from nautilus_trader.adapters.binance.common.schemas.market import BinanceCandlestick
from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesEnumParser
from nautilus_trader.core.datetime import millis_to_nanos
from nautilus_trader.model.identifiers import InstrumentId

from tests.unit.conftest import synthetic_klines
from vibe_quant.data.catalog import get_bar_type
from vibe_quant.data.downloader import RestDownloadError
from vibe_quant.db.state_manager import StateManager
from vibe_quant.paper import warmup
from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
)
from vibe_quant.paper.journal import FakeJournal
from vibe_quant.paper.node import NodeState, PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path


class _NodeBuilt(Exception):  # noqa: N818 - control-flow sentinel
    pass


def _source(
    timeframe: str, indicators: dict[str, dict[str, object]], sl: dict[str, object] | None = None
) -> dict[str, object]:
    first = next(iter(indicators))
    return {
        "name": "warmup_probe",
        "timeframe": timeframe,
        "indicators": indicators,
        "entry_conditions": {"long": [f"{first} > 0"]},
        "stop_loss": sl or {"type": "fixed_pct", "percent": 5.0},
        "take_profit": {"type": "fixed_pct", "percent": 10.0},
    }


def _make(
    tmp_path: Path,
    source: dict[str, object],
    symbols: list[str],
    *,
    with_run: bool = True,
) -> PaperTradingNode:
    state = StateManager(tmp_path / "state.db")
    sid = state.create_strategy(name=str(source["name"]), dsl_config=source)
    rid = None
    if with_run:
        rid = state.create_backtest_run(
            strategy_id=sid,
            run_mode="validation",
            symbols=symbols,
            timeframe=str(source["timeframe"]),
            start_date="2025-01-01",
            end_date="2025-02-01",
            parameters={},
        )
        state.update_backtest_run_status(rid, "completed")
    state.close()
    cfg = PaperTradingConfig(
        trader_id="PAPER-007",
        binance=BinanceTestnetConfig("key", "secret", testnet=True),
        strategy_id=sid,
        validation_run_id=rid,
        symbols=[] if with_run else symbols,
        db_path=tmp_path / "state.db",
        logs_path=tmp_path / "logs",
    )
    return PaperTradingNode(cfg)


class _Builds:
    """Counts ``_create_live_trading_node`` calls; the sentinel marks 'the seed passed'."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> Any:
        self.count += 1
        raise _NodeBuilt


def _wire(node: PaperTradingNode, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeJournal, _Builds]:
    journal = FakeJournal()
    node._journal = journal
    builds = _Builds()
    monkeypatch.setattr(node, "_create_live_trading_node", builds)
    monkeypatch.setattr(node, "_setup_signal_handlers", lambda: None)
    return journal, builds


def _fake_fetch(monkeypatch: pytest.MonkeyPatch, fn: Any) -> list[tuple[str, str]]:
    """Replace the REST fetch with ``fn(symbol, interval, start_ms, end_ms)``; the calls made."""
    calls: list[tuple[str, str]] = []

    def fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Any]:
        calls.append((symbol, interval))
        return list(fn(symbol, interval, start_ms, end_ms))

    monkeypatch.setattr("vibe_quant.paper.warmup.fetch_closed_klines", fetch)
    return calls


async def _refused(node: PaperTradingNode) -> str:
    with pytest.raises(ConfigurationError) as exc:
        await node.start()
    return str(exc.value)


def _critical_failed_alerts(journal: FakeJournal) -> list[tuple[Any, ...]]:
    return [c[1:] for c in journal.calls if c[0] == "alert"]


# ----------------------------------------------------------------------- R13 / refusals


async def test_fetch_failure_refuses_start_before_node_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R13: ETHUSDT warmed, the BTCUSDT 4h fetch raises: the start is refused, 0 node builds."""
    node = _make(
        tmp_path, _source("4h", {"ema": {"type": "EMA", "period": 20}}), ["ETHUSDT", "BTCUSDT"]
    )
    journal, builds = _wire(node, monkeypatch)

    def fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Any]:
        if symbol == "BTCUSDT":
            raise RestDownloadError("boom https://fapi.example/klines?secret=1", [], start_ms)
        return synthetic_klines(symbol, interval, start_ms, end_ms)

    calls = _fake_fetch(monkeypatch, fetch)
    text = await _refused(node)

    assert text == "paper warm-up refused: BTCUSDT 4h ema (EMA): fetch failed: RestDownloadError"
    assert "secret" not in text  # the class only, never the exception text
    assert calls == [("ETHUSDT", "4h"), ("BTCUSDT", "4h")]
    assert builds.count == 0
    assert node.status.state is NodeState.ERROR
    assert node.status.error_message == text
    assert _critical_failed_alerts(journal) == [("error", f"Paper node failed: {text}", True)]


@pytest.mark.parametrize(
    ("indicator", "reason", "fetched"),
    [
        (
            {"type": "OBV"},
            "paper warm-up refused: BTCUSDT 15m x (OBV): cumulative level never converges",
            0,
        ),
        (
            {"type": "VIDYA"},
            "paper warm-up refused: BTCUSDT 15m x (VIDYA): "
            "start-phase drift above 1e-9 in validation itself (vibe-quant-d42m5)",
            0,
        ),
        (
            {"type": "EMA", "period": 600},
            "paper warm-up refused: BTCUSDT 15m x (EMA): needs 9000 bars, seed holds 8000",
            1,
        ),
    ],
)
async def test_unclassifiable_or_too_long_indicator_refuses_the_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    indicator: dict[str, object],
    reason: str,
    fetched: int,
) -> None:
    """R3/R4: a kind that never converges, or a need above the 8,000 rows, refuses the START
    with the symbol, timeframe, indicator and reason; a classification refusal fetches nothing."""
    node = _make(tmp_path, _source("15m", {"x": indicator}), ["BTCUSDT"])
    journal, builds = _wire(node, monkeypatch)
    calls = _fake_fetch(monkeypatch, synthetic_klines)

    text = await _refused(node)

    assert text == reason
    assert len(calls) == fetched
    assert builds.count == 0
    assert node.status.state is NodeState.ERROR
    assert _critical_failed_alerts(journal) == [("error", f"Paper node failed: {text}", True)]


@pytest.mark.parametrize(
    ("period", "accepted"),
    [(200, True), (250, False)],  # EMA needs 15 x period: 3,000 passes, 3,750 > 3,000 refuses
)
async def test_young_listing_bound_is_checked_against_the_rows_it_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, period: int, accepted: bool
) -> None:
    """R14: only 3,000 REST rows exist; the bound is checked against 3,000 (the equality edge passes)."""
    node = _make(tmp_path, _source("15m", {"ema": {"type": "EMA", "period": period}}), ["BTCUSDT"])
    _journal, builds = _wire(node, monkeypatch)

    def young(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Any]:
        return synthetic_klines(symbol, interval, start_ms, end_ms)[-3_000:]

    _fake_fetch(monkeypatch, young)
    if accepted:
        with pytest.raises(_NodeBuilt):
            await node.start()
        assert builds.count == 1
        return
    text = await _refused(node)
    assert text == (
        "paper warm-up refused: BTCUSDT 15m ema (EMA): needs 3750 bars, seed holds 3000"
    )
    assert builds.count == 0


async def test_seed_gap_refuses_the_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R14: a missing bar (close_time step != bar length) refuses, naming where it is missing."""
    node = _make(tmp_path, _source("15m", {"ema": {"type": "EMA", "period": 20}}), ["BTCUSDT"])
    _journal, builds = _wire(node, monkeypatch)
    missing: list[int] = []

    def holed(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Any]:
        rows = synthetic_klines(symbol, interval, start_ms, end_ms)
        missing.append(rows[5_000][0])
        return rows[:5_000] + rows[5_001:]

    _fake_fetch(monkeypatch, holed)
    text = await _refused(node)

    iso = datetime.fromtimestamp(missing[0] / 1_000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert text == f"paper warm-up refused: BTCUSDT 15m ema (EMA): seed gap at {iso}"
    assert builds.count == 0


async def test_strategy_only_launch_warms_up_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R22: no validation run: the same 8,000-bar seed is installed (no start date needed)."""
    node = _make(
        tmp_path,
        _source("15m", {"ema": {"type": "EMA", "period": 20}}),
        ["BTCUSDT"],
        with_run=False,
    )
    _journal, builds = _wire(node, monkeypatch)
    calls = _fake_fetch(monkeypatch, synthetic_klines)

    with pytest.raises(_NodeBuilt):
        await node.start()

    assert calls == [("BTCUSDT", "15m")]
    assert builds.count == 1
    strategy: Any = node._compiled_strategies[0]
    bar_type = strategy.bar_type_15m
    assert strategy._warm_wm is not None
    last_close_ns = strategy._warm_wm[bar_type]
    assert (last_close_ns // 1_000_000) % 900_000 == 899_999  # a bar's last millisecond
    assert callable(strategy._warm_request_topup)


async def test_two_timeframes_are_seeded_per_bar_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A secondary-timeframe indicator gets its own 8,000-row seed, keyed by its bar type."""
    source = _source(
        "15m",
        {"ema": {"type": "EMA", "period": 20}, "slow": {"type": "EMA", "period": 20, "timeframe": "4h"}},
    )
    source["additional_timeframes"] = ["4h"]
    node = _make(tmp_path, source, ["BTCUSDT"])
    _journal, _builds = _wire(node, monkeypatch)
    calls = _fake_fetch(monkeypatch, synthetic_klines)

    with pytest.raises(_NodeBuilt):
        await node.start()

    assert sorted(calls) == [("BTCUSDT", "15m"), ("BTCUSDT", "4h")]
    strategy: Any = node._compiled_strategies[0]
    assert set(strategy._warm_wm) == {strategy.bar_type_15m, strategy.bar_type_4h}


# ------------------------------------------------------------------- seed == live bars


def test_seed_bar_stamps_match_what_the_live_adapter_delivers() -> None:
    """The live websocket bar carries ts_event = kline close_time (ms -> ns); the seed bar of the
    same kline carries the identical ts_event, so the strategy's watermark compares like with like,
    and the next live bar sits exactly one bar length later."""
    step_ms = 900_000
    rows = synthetic_klines("BTCUSDT", "15m", 1_700_000_100_000, 1_700_000_100_000 + 3 * step_ms)
    bar_type = get_bar_type("BTCUSDT", "15m")
    seed = [warmup.kline_to_bar(bar_type, row) for row in rows]

    def live(row: tuple[Any, ...]) -> Any:
        kline = BinanceCandlestick(
            t=row[0], T=row[6], s="BTCUSDT", i=BinanceKlineInterval.MINUTE_15, f=0, L=0,
            o=row[1], c=row[4], h=row[2], l=row[3], v=row[5], n=1, x=True, q="0", V="0", Q="0", B="0",
        )  # fmt: skip
        return kline.parse_to_binance_bar(
            InstrumentId.from_str("BTCUSDT-PERP.BINANCE"), BinanceFuturesEnumParser(), ts_init=1
        )

    live_bars = [live(row) for row in rows]
    assert [b.bar_type for b in live_bars] == [b.bar_type for b in seed]
    assert [b.ts_event for b in live_bars] == [b.ts_event for b in seed]
    # millis_to_nanos goes through a double: NOT close_time * 1_000_000 (64 ns off here)
    assert seed[0].ts_event != rows[0][6] * 1_000_000
    assert seed[1].ts_event - seed[0].ts_event == step_ms * 1_000_000
    assert [str(b.close) for b in live_bars] == [str(b.close) for b in seed]


def test_interval_of_bar_type_and_closed_boundary() -> None:
    assert [warmup.interval_of(get_bar_type("BTCUSDT", tf)) for tf in ("1m", "15m", "4h", "1d")] == [
        "1m",
        "15m",
        "4h",
        "1d",
    ]
    boundary = warmup.last_closed_boundary_ms(1_700_000_123_456, "15m")
    assert boundary == 1_699_999_200_000 + 900_000 * 1 or boundary % 900_000 == 0
    assert 0 <= 1_700_000_123_456 - boundary < 900_000
    assert warmup.last_closed_boundary_ms(900_000, "15m") == 900_000  # on the boundary: unchanged
    assert warmup.last_closed_boundary_ms(899_999, "15m") == 0


# ---------------------------------------------------------------- top-up bounds (R17)


class _Strategy:
    """Records what the scheduler feeds back."""

    def __init__(self) -> None:
        self.fed: list[tuple[Any, list[Any]]] = []

    def _warm_feed_topup(self, bar_type: Any, bars: list[Any]) -> None:
        self.fed.append((bar_type, bars))


async def _settle(scheduler: warmup.TopUpScheduler) -> None:
    await asyncio.gather(*list(scheduler.tasks), return_exceptions=True)


async def test_a_fetch_that_keeps_raising_is_attempted_three_times_then_fails_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(warmup, "RETRY_SLEEP_S", 0.0)
    attempts: list[int] = []
    failures: list[tuple[str, str]] = []

    def fetch(symbol: str, bar_type: Any, edge: int, ts: int) -> list[Any]:
        attempts.append(1)
        raise RestDownloadError("x", [], 0)

    scheduler = warmup.TopUpScheduler(lambda s, i: failures.append((s, i)), fetch=fetch)
    strategy = _Strategy()
    bar_type = get_bar_type("BTCUSDT", "1m")
    scheduler.request(strategy, "BTCUSDT", bar_type, 60_000_000_000, 240_000_000_000)
    await _settle(scheduler)

    assert len(attempts) == 3
    assert failures == [("BTCUSDT", "1m")]
    assert strategy.fed == []


async def test_a_short_response_is_re_requested_three_times_then_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint answers but never fills the hole: the strategy keeps asking for the SAME
    hole; the 3rd answer is the last (the 4th ask fails the session without a 4th fetch)."""
    fetched: list[int] = []
    failures: list[tuple[str, str]] = []

    def fetch(symbol: str, bar_type: Any, edge: int, ts: int) -> list[Any]:
        fetched.append(edge)
        return []

    scheduler = warmup.TopUpScheduler(lambda s, i: failures.append((s, i)), fetch=fetch)
    strategy = _Strategy()
    bar_type = get_bar_type("BTCUSDT", "1m")
    for _ in range(3):
        scheduler.request(strategy, "BTCUSDT", bar_type, 60_000_000_000, 240_000_000_000)
        await _settle(scheduler)
        assert failures == []
    scheduler.request(strategy, "BTCUSDT", bar_type, 60_000_000_000, 240_000_000_000)
    await _settle(scheduler)

    assert fetched == [60_000_000_000] * 3
    assert len(strategy.fed) == 3
    assert failures == [("BTCUSDT", "1m")]
    # after the failure nothing is fetched or fed for that strategy again
    scheduler.request(strategy, "BTCUSDT", bar_type, 120_000_000_000, 240_000_000_000)
    await _settle(scheduler)
    assert len(fetched) == 3
    assert failures == [("BTCUSDT", "1m")]


async def test_progress_on_a_hole_is_not_counted_against_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-request from a NEW watermark (the earlier bars were fed) is a different hole."""
    failures: list[tuple[str, str]] = []
    scheduler = warmup.TopUpScheduler(
        lambda s, i: failures.append((s, i)), fetch=lambda s, b, e, t: []
    )
    strategy = _Strategy()
    bar_type = get_bar_type("BTCUSDT", "1m")
    for edge in range(1, 8):
        scheduler.request(strategy, "BTCUSDT", bar_type, edge * 60_000_000_000, 900_000_000_000)
        await _settle(scheduler)
    assert failures == []
    assert len(strategy.fed) == 7


def test_gap_fetch_keeps_only_bars_strictly_between_the_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """The REST page may carry the watermark bar or the queued bar: neither is fed twice."""
    rows = synthetic_klines("BTCUSDT", "15m", 1_700_000_100_000, 1_700_000_100_000 + 6 * 900_000)
    asked: list[tuple[int, int]] = []

    def fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Any]:
        asked.append((start_ms, end_ms))
        return rows

    monkeypatch.setattr("vibe_quant.paper.warmup.fetch_closed_klines", fetch)
    bar_type = get_bar_type("BTCUSDT", "15m")
    edge = millis_to_nanos(rows[1][6])
    ts = millis_to_nanos(rows[4][6])
    bars = warmup.fetch_gap("BTCUSDT", bar_type, edge, ts)

    assert [b.ts_event for b in bars] == [millis_to_nanos(rows[2][6]), millis_to_nanos(rows[3][6])]
    assert asked == [(rows[1][6] + 1, rows[4][6])]


# ------------------------------------------------------------------ equality edges


@pytest.mark.parametrize(
    ("rows", "ok"),
    [(3_001, True), (3_000, True), (2_999, False)],  # need == rows passes, one fewer refuses
)
def test_need_equal_to_the_rows_held_passes_and_one_fewer_refuses(rows: int, ok: bool) -> None:
    needs = [warmup.WarmupNeed("ema", "EMA", "15m", bars=3_000)]
    if ok:
        warmup.check_warmup("BTCUSDT", needs, {"15m": rows})
        return
    with pytest.raises(ConfigurationError) as exc:
        warmup.check_warmup("BTCUSDT", needs, {"15m": rows})
    assert str(exc.value) == (
        "paper warm-up refused: BTCUSDT 15m ema (EMA): needs 3000 bars, seed holds 2999"
    )


@pytest.mark.parametrize(("cap", "ok"), [(8_000, True), (8_001, False)])
def test_compute_fn_buffer_cap_must_fit_the_seed(cap: int, ok: bool) -> None:
    """The buffer's steady-state size (``_pta_buffer_cap``) is compared too; 0 = unbounded passes
    this comparison (a cumulative kind is refused by its class instead)."""
    needs = [warmup.WarmupNeed("cci", "CCI", "15m", bars=40)]
    rows = {"15m": 8_000}
    if ok:
        warmup.check_warmup("BTCUSDT", needs, rows, {"15m": cap})
        warmup.check_warmup("BTCUSDT", needs, rows, {"15m": 0})
        return
    with pytest.raises(ConfigurationError, match=r"needs 8001 bars \(buffer cap\), seed holds 8000"):
        warmup.check_warmup("BTCUSDT", needs, rows, {"15m": cap})


def test_fetch_seed_pages_8000_closed_bars_through_the_real_rest_client() -> None:
    """Unpatched ``fetch_closed_klines`` over the suite's fake klines endpoint: 6 pages of 1500,
    exactly 8,000 contiguous closed bars, the forming bar dropped, oldest first."""
    end_ms = warmup.last_closed_boundary_ms(1_794_993_300_000, "15m") + 450_000
    boundary = warmup.last_closed_boundary_ms(end_ms, "15m")
    bar_type = get_bar_type("BTCUSDT", "15m")
    bars = warmup.fetch_seed("BTCUSDT", bar_type, boundary)
    assert len(bars) == warmup.SEED_BARS == 8_000
    assert bars[-1].ts_event == millis_to_nanos(boundary - 1)  # the last CLOSED bar
    step_ns = 900_000_000_000
    assert all(b.ts_event - a.ts_event == step_ns for a, b in zip(bars, bars[1:], strict=False))


async def test_price_only_strategy_fetch_failure_refuses_with_a_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A strategy with no indicators still seeds its primary bar type; a failed fetch is a
    refusal naming symbol and timeframe (no indicator to name), never an IndexError."""
    source: dict[str, object] = {
        "name": "warmup_probe",
        "timeframe": "15m",
        "indicators": {},
        "entry_conditions": {"long": ["close > 0"]},
        "stop_loss": {"type": "fixed_pct", "percent": 5.0},
        "take_profit": {"type": "fixed_pct", "percent": 10.0},
    }
    node = _make(tmp_path, source, ["BTCUSDT"])
    journal, builds = _wire(node, monkeypatch)

    def fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Any]:
        raise RestDownloadError("boom", [], start_ms)

    _fake_fetch(monkeypatch, fetch)
    text = await _refused(node)

    assert text == "paper warm-up refused: BTCUSDT 15m (price only): fetch failed: RestDownloadError"
    assert builds.count == 0
    assert _critical_failed_alerts(journal) == [("error", f"Paper node failed: {text}", True)]
