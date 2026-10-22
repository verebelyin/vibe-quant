"""Validation tier x liquidation overlay: extract_trades / runner / event log.

Positions are duck-typed NT stubs (real NT enums), marks are synthetic 1m bars. The shared
helper ``vibe_quant.liquidation.apply.apply_liquidation`` is replaced by a shim that runs the real
``liquidation_overlay`` on those marks (only the catalog mark loading is skipped).
Expectations are hand-computed from the Binance one-way formula
``LP = (WB + cum - side*q*EP) / (q*MMR - side*q)`` (BTC tier 1: MMR .004, cum 0).
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
import pytest
from nautilus_trader.model.enums import OrderSide, OrderType

from tests.unit.fake_engine import fake_engine
from vibe_quant.launch_settings import PARSE_ORDER, LaunchSettings
from vibe_quant.liquidation import liquidation_overlay
from vibe_quant.liquidation.tiers import load_tier_tables
from vibe_quant.logging.events import Event, EventType, LiquidationEvent, create_event
from vibe_quant.metrics import BarSeries
from vibe_quant.validation.extraction import date_to_ns, extract_trades
from vibe_quant.validation.results import ValidationResult
from vibe_quant.validation.runner import (
    ValidationRunner,
    ValidationRunnerError,
    _parse_launch_fields,
)

if TYPE_CHECKING:
    from pathlib import Path

MIN = 60_000_000_000
_DAY0 = date_to_ns("2024-01-01")
assert _DAY0 is not None
BASE = _DAY0 + 100 * MIN
# the run ends where the synthetic 60 one-minute bars end (01:40 + 60 min): LAST covers the
# final execution bar, as in a real run
START, END = "2024-01-01", "2024-01-01T02:40:00"
BTC = "BTCUSDT-PERP.BINANCE"
ETH = "ETHUSDT-PERP.BINANCE"
TABLES = load_tier_tables()
LP_ISOLATED = (4000.0 - 40000.0) / (0.004 - 1.0)  # 36144.57831325301
START_BAL = 10_000.0


def bar(i: int) -> int:
    return BASE + i * MIN


@dataclass
class Fill:
    ts_event: int
    order_side: Any
    last_qty: float
    last_px: float
    commission: float = 0.0
    order_type: Any = OrderType.MARKET


@dataclass
class Pos:
    id: str
    instrument_id: str
    events: list[Fill] = field(default_factory=list)
    is_closed: bool = True

    @property
    def is_open(self) -> bool:
        return not self.is_closed

    @property
    def ts_opened(self) -> int:
        return self.events[0].ts_event

    @property
    def ts_closed(self) -> int:
        return self.events[-1].ts_event if self.is_closed else 0

    @property
    def entry(self) -> Any:
        return self.events[0].order_side

    @property
    def peak_qty(self) -> float:
        return self.events[0].last_qty

    @property
    def avg_px_open(self) -> float:
        return self.events[0].last_px

    @property
    def avg_px_close(self) -> float:
        return self.events[-1].last_px if self.is_closed else 0.0

    @property
    def signed_qty(self) -> float:
        sign = 1.0 if self.entry == OrderSide.BUY else -1.0
        return 0.0 if self.is_closed else sign * self.peak_qty

    @property
    def realized_pnl(self) -> float:
        # NT's own number: price PnL of the closing fills net of ALL commissions
        sign = 1.0 if self.entry == OrderSide.BUY else -1.0
        gross = 0.0
        if self.is_closed:
            gross = sign * self.peak_qty * (self.avg_px_close - self.avg_px_open)
        return gross - sum(f.commission for f in self.events)

    def commissions(self) -> list[float]:
        return [f.commission for f in self.events]


def liq_note(res: ValidationResult) -> dict[str, Any]:
    return cast("dict[str, Any]", res.notes["liquidation"])


def key_of(p: Pos) -> str:
    return f"{p.id}@{p.ts_opened}"


def long_btc(*, sell_bar: int | None = 30, entry_bar: int = 2, pid: str = "P-1") -> Pos:
    """1 BTC long from 40000 (fee 20); the strategy's own exit (fee 20.5) at 41000."""
    fills = [Fill(bar(entry_bar), OrderSide.BUY, 1.0, 40000.0, 20.0)]
    if sell_bar is not None:
        fills.append(Fill(bar(sell_bar), OrderSide.SELL, 1.0, 41000.0, 20.5))
    return Pos(pid, BTC, fills, is_closed=sell_bar is not None)


class Marks(NamedTuple):
    ts: np.ndarray
    low: np.ndarray
    high: np.ndarray
    close: np.ndarray
    open: np.ndarray


def marks(lows: list[float], highs: list[float] | None = None) -> Marks:
    n = len(lows)
    low = np.asarray(lows, dtype=np.float64)
    high = np.full(n, 41500.0) if highs is None else np.asarray(highs, dtype=np.float64)
    close = low.copy()
    open_ = np.concatenate((close[:1], close[:-1]))
    return Marks(BASE + np.arange(n, dtype=np.int64) * MIN, low, high, close, open_)


def crash_marks(*, to: float, at: int = 10, n: int = 60) -> Marks:
    """40000 until bar ``at``, then ``to`` for the rest (open of bar ``at`` = 40000)."""
    return marks([40000.0] * at + [to] * (n - at))


def exec_series(mk: Marks) -> BarSeries:
    """1m LAST execution bars = the mark bars, stamped like catalog bars (close - 1 ms)."""
    first = mk.open[:1]
    return BarSeries(
        ts=np.concatenate((mk.ts[:1] - 1_000_000, mk.ts + MIN - 1_000_000)),
        open=np.concatenate((first, mk.open)),
        high=np.concatenate((first, mk.high)),
        low=np.concatenate((first, mk.low)),
        close=np.concatenate((first, mk.close)),
    )


class _ExecBarType:
    """A 1m LAST ``BarType``: its instrument plus the string the cache is keyed by."""

    def __init__(self, instrument_id: str) -> None:
        self.instrument_id = instrument_id

    def __str__(self) -> str:
        return f"{self.instrument_id}-1-MINUTE-LAST-EXTERNAL"


def engine_for(positions: list[Any], series: dict[str, Marks]) -> Any:
    bars_by_type: dict[str, list[SimpleNamespace]] = {}
    for instrument_id, mk in series.items():
        ex = exec_series(mk)
        bar_type = _ExecBarType(instrument_id)
        bars_by_type[str(bar_type)] = [
            SimpleNamespace(
                bar_type=bar_type,
                ts_init=int(t),
                volume=0.0,
                open=float(o),
                high=float(h),
                low=float(lo),
                close=float(c),
            )
            for t, o, h, lo, c in zip(ex.ts, ex.open, ex.high, ex.low, ex.close, strict=True)
        ]
    return fake_engine(positions, bars_by_type=bars_by_type)


VENUE = SimpleNamespace(
    default_leverage=Decimal("10"),
    # engine slippage on: no post-fill SPEC slippage, so costs are exactly the fees
    fill_config=SimpleNamespace(impact_coefficient=0.1, prob_slippage=1.0),
)


@pytest.fixture
def apply_calls() -> list[dict[str, Any]]:
    return []


def install_apply(
    monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]], mk: dict[str, Marks]
) -> None:
    """Replace the run-level helper by the real overlay on the synthetic marks."""

    def apply_liquidation(
        nt_positions: Any,
        *,
        mode: str,
        leverage: float,
        starting_balance: float,
        catalog_path: Any,
        start_ns: int,
        end_ns: int,
        execution_bar_ns: int,
        funding_by_position: Any,
        cash_by_position: Any,
        tables: Any = None,
    ) -> Any:
        calls.append(
            {
                "positions": list(nt_positions),
                "mode": mode,
                "leverage": leverage,
                "starting_balance": starting_balance,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "execution_bar_ns": execution_bar_ns,
                "funding": funding_by_position,
                "cash": cash_by_position,
            }
        )
        return liquidation_overlay(
            list(nt_positions),
            {iid.split("-")[0]: series for iid, series in mk.items()},  # real helper: by symbol
            mode=mode,  # type: ignore[arg-type]
            starting_balance=starting_balance,
            leverage=leverage,
            tables=TABLES if tables is None else tables,
            funding_by_position=funding_by_position,
            cash_by_position=cash_by_position,
            execution_bar_ns=execution_bar_ns,
        )

    module = types.ModuleType("vibe_quant.liquidation.apply")
    module.apply_liquidation = apply_liquidation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vibe_quant.liquidation.apply", module)


def run(
    positions: list[Pos],
    series: dict[str, Marks],
    *,
    mode: str | None = None,
) -> ValidationResult:
    result = ValidationResult(starting_balance=START_BAL)
    kw: dict[str, Any] = {} if mode is None else {"liquidation_mode": mode}
    extract_trades(
        result,
        engine_for(positions, series),  # type: ignore[arg-type]
        VENUE,  # type: ignore[arg-type]
        run_start_date=START,
        run_end_date=END,
        execution_timeframe="1m",
        catalog_path="unused-by-the-shim",
        **kw,
    )
    return result


# ------------------------------------------------------------------ 1. mode off


def test_mode_off_matches_default_and_never_touches_the_liquidation_package(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    mk = {BTC: crash_marks(to=30000.0)}  # would liquidate if the overlay ran
    install_apply(monkeypatch, apply_calls, mk)
    default = run([long_btc()], mk)
    off = run([long_btc()], mk, mode="off")
    # the off path must not even import the package: a blocked module still works
    monkeypatch.setitem(sys.modules, "vibe_quant.liquidation.apply", None)
    blocked = run([long_btc()], mk, mode="off")

    assert apply_calls == []
    for res in (off, blocked):
        assert res.trades == default.trades
        assert res.total_return == default.total_return
        assert res.max_drawdown == default.max_drawdown
        assert res.profit_factor == default.profit_factor
        assert "liquidation" not in res.notes
        assert res.liquidation_events == []
    # untouched NT path: the strategy's own exit at 41000, net of both fees
    (trade,) = default.trades
    assert trade.exit_reason == "signal" and trade.exit_price == 41000.0
    assert trade.net_pnl == pytest.approx(1000.0 - 40.5)


def test_mode_on_without_hits_changes_nothing_but_the_note(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    mk = {BTC: marks([40000.0] * 60)}
    install_apply(monkeypatch, apply_calls, mk)
    off = run([long_btc()], mk)
    on = run([long_btc()], mk, mode="isolated")

    assert len(apply_calls) == 1
    assert on.trades == off.trades
    assert on.total_return == off.total_return
    assert on.max_drawdown == off.max_drawdown
    assert on.profit_factor == off.profit_factor
    assert liq_note(on)["count"] == 0
    assert on.liquidation_events == []


# --------------------------------------------- 2. an isolated liquidated trade record


def test_isolated_liquidation_trade_uses_the_venue_fill_and_the_account_loss(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    mk = {BTC: crash_marks(to=36000.0)}  # bar 10 opens 40000, trades through the LP
    install_apply(monkeypatch, apply_calls, mk)
    res = run([long_btc()], mk, mode="isolated")

    (trade,) = res.trades
    assert trade.exit_reason == "liquidation"
    assert trade.exit_price == pytest.approx(LP_ISOLATED)  # the venue fill, not the 41000 exit
    assert trade.exit_time is not None and trade.exit_time.endswith("+00:00")
    assert trade.exit_time == "2024-01-01T01:50:00+00:00"  # bar(10) open: BASE is 01:40 + 10m
    assert trade.entry_price == 40000.0 and trade.quantity == 1.0
    # account loss of the position = its initial margin 4000 + the entry fee 20 (hand-computed);
    # the strategy's later 41000 exit and its fee do not exist
    assert trade.net_pnl == pytest.approx(-4000.0 - 20.0)
    assert trade.entry_fee == pytest.approx(20.0)
    assert trade.exit_fee == pytest.approx(4000.0 - (40000.0 - LP_ISOLATED))  # clearance fee
    assert res.total_trades == 1 and res.losing_trades == 1 and res.winning_trades == 0
    assert res.total_return == pytest.approx(-4020.0 / START_BAL)
    assert res.total_return == pytest.approx(sum(t.net_pnl for t in res.trades) / START_BAL)
    assert res.profit_factor == 0.0
    assert res.total_fees == pytest.approx(20.0 + trade.exit_fee)
    call = apply_calls[0]
    assert call["mode"] == "isolated" and call["leverage"] == 10.0
    assert call["execution_bar_ns"] == MIN and call["starting_balance"] == START_BAL
    assert (call["start_ns"], call["end_ns"]) == (date_to_ns(START), date_to_ns(END))


def test_gap_liquidation_exit_price_is_the_venue_fill_not_the_ledger_close(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    """Bar 10 gaps to 35000, below the bankruptcy price 36000: the venue fills at the open
    35000 and the insurance fund bears 1000; the ledger's synthetic close sits at 36000."""
    lows = [40000.0] * 10 + [35000.0] * 50
    mk = {BTC: marks(lows)}
    mk = {BTC: mk[BTC]._replace(open=np.where(mk[BTC].ts >= bar(10), 35000.0, 40000.0))}
    install_apply(monkeypatch, apply_calls, mk)
    res = run([long_btc()], mk, mode="isolated")

    (trade,) = res.trades
    assert trade.exit_reason == "liquidation"
    assert trade.exit_price == 35000.0
    assert trade.net_pnl == pytest.approx(-4000.0 - 20.0)  # loss floored at the margin
    note = liq_note(res)
    assert note["total_insurance_deficit"] == pytest.approx(1000.0)


# ------------------------------------------------- 3. cross bust drops a later position


def test_cross_bust_drops_the_later_position_and_equity_stays_non_negative(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    mk = {BTC: crash_marks(to=30000.0)}
    install_apply(monkeypatch, apply_calls, mk)
    later = Pos("P-2", BTC, [Fill(bar(45), OrderSide.BUY, 1.0, 30000.0, 15.0),
                             Fill(bar(50), OrderSide.SELL, 1.0, 30500.0, 15.0)])
    res = run([long_btc(), later], mk, mode="cross")

    assert apply_calls[0]["mode"] == "cross"
    (trade,) = res.trades  # P-2 opened after the bust: no trade, no fees
    assert trade.symbol == BTC and trade.exit_reason == "liquidation"
    wallet_lp = (START_BAL - 20.0 - 40000.0) / (0.004 - 1.0)  # cross LP with the entry fee paid
    assert trade.exit_price == pytest.approx(wallet_lp)
    assert trade.net_pnl == pytest.approx(-START_BAL)  # forfeit rule: the whole wallet
    assert res.total_trades == 1 and res.total_return == pytest.approx(-1.0)
    assert res.total_return >= -1.0 - 1e-9
    assert res.max_drawdown == pytest.approx(1.0) and res.max_drawdown <= 1.0
    assert res.total_fees == pytest.approx(20.0 + trade.exit_fee)  # P-2's 30 never counted
    note = liq_note(res)
    assert note["mode"] == "cross" and note["dropped"] == 1 and note["count"] == 1


# ------------------------------------------------------------ 4. notes / multi-symbol


def test_notes_carry_the_explicit_loss_sums_for_a_multi_symbol_run(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    eth_pos = Pos("P-9", ETH, [Fill(bar(3), OrderSide.SELL, 10.0, 2000.0, 5.0),
                               Fill(bar(20), OrderSide.BUY, 10.0, 1980.0, 5.0)])
    eth_marks = marks([2000.0] * 60, [2010.0] * 60)
    mk = {BTC: crash_marks(to=36000.0), ETH: eth_marks}
    install_apply(monkeypatch, apply_calls, mk)
    off = run([long_btc(), eth_pos], mk)
    on = run([long_btc(), eth_pos], mk, mode="isolated")

    assert on.total_trades == 2 and len(on.trades) == 2
    eth_off = next(t for t in off.trades if t.symbol == ETH)
    eth_on = next(t for t in on.trades if t.symbol == ETH)
    assert eth_on == eth_off  # the unaffected symbol keeps its record
    note = liq_note(on)
    assert note["count"] == 1 and note["dropped"] == 0
    assert note["pct_trades"] == pytest.approx(50.0)
    assert note["total_forfeited"] == pytest.approx(4000.0 - (40000.0 - LP_ISOLATED))
    assert note["total_realized_pnl"] == pytest.approx(LP_ISOLATED - 40000.0)
    assert note["total_insurance_deficit"] == 0.0
    assert note["leverage"] == 10.0 and note["bust_ts"] is None
    # liquidation counts toward the headline: BTC -4020 and the ETH short's +200 - fees 10
    assert on.total_return == pytest.approx((-4020.0 + (200.0 - 10.0)) / START_BAL)
    assert "liquidation" not in off.notes


def test_open_position_that_is_liquidated_is_not_marked_at_the_end(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    mk = {BTC: crash_marks(to=36000.0)}
    install_apply(monkeypatch, apply_calls, mk)
    res = run([long_btc(sell_bar=None)], mk, mode="isolated")

    (trade,) = res.trades
    assert trade.exit_reason == "liquidation"  # not "end_of_data" at the last 36000 close
    assert trade.exit_price == pytest.approx(LP_ISOLATED)
    assert trade.net_pnl == pytest.approx(-4020.0)


def test_unknown_mode_and_missing_window_fail_closed() -> None:
    with pytest.raises(ValueError, match="liquidation_mode"):
        run([long_btc()], {BTC: marks([40000.0] * 60)}, mode="margin")
    result = ValidationResult(starting_balance=START_BAL)
    mk = {BTC: marks([40000.0] * 60)}
    with pytest.raises(ValueError, match="run window"):
        extract_trades(
            result,
            engine_for([long_btc()], mk),  # type: ignore[arg-type]
            VENUE,  # type: ignore[arg-type]
            execution_timeframe="1m",
            catalog_path="x",
            liquidation_mode="isolated",
        )


# ---------------------------------------------------------- 5. LiquidationEvent / log


def test_event_log_gets_one_liquidation_event_per_hit_and_old_events_still_parse(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]], tmp_path: Path
) -> None:
    from vibe_quant.logging.writer import EventWriter

    mk = {BTC: crash_marks(to=36000.0)}
    install_apply(monkeypatch, apply_calls, mk)
    res = run([long_btc()], mk, mode="isolated")

    runner = ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")
    try:
        with EventWriter(run_id="7", base_path=tmp_path / "ev") as writer:
            runner._write_liquidation_events(writer, 7, "strat", res)
        lines = [
            json.loads(x) for x in (tmp_path / "ev" / "7.jsonl").read_text().splitlines() if x
        ]
    finally:
        runner.close()
    liq = [e for e in lines if e["event"] == "LIQUIDATION"]
    assert len(liq) == 1
    data = liq[0]["data"]
    assert data["position_id"] == key_of(long_btc())
    assert data["symbol"] == "BTCUSDT" and data["mode"] == "isolated"
    assert data["liquidation_price"] == pytest.approx(LP_ISOLATED)
    assert data["quantity"] == 1.0
    assert data["loss"] == pytest.approx(4000.0)
    assert data["ts_ns"] == bar(10)
    assert data["mark_price"] == 36000.0  # the trigger bar's low
    assert data["clearance_fee"] == pytest.approx(4000.0 - (40000.0 - LP_ISOLATED))
    assert data["margin_balance"] > 0 and data["maintenance_margin"] > 0
    assert Event.from_dict(liq[0]).event_type == EventType.LIQUIDATION

    old = create_event(
        EventType.LIQUIDATION,
        "run1",
        "strat",
        {
            "position_id": "P-1",
            "symbol": "BTCUSDT",
            "liquidation_price": 40000.0,
            "quantity": 0.1,
            "loss": 420.0,
        },
    )
    assert isinstance(old, LiquidationEvent)
    assert old.ts_ns is None and old.mode is None
    assert set(old.data) == {"position_id", "symbol", "liquidation_price", "quantity", "loss"}
    assert Event.from_dict(old.to_dict()).data == old.data


# --------------------------------------------------- 6. run parameter plumbing


def _runner(tmp_path: Path) -> ValidationRunner:
    return ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")


def test_liquidation_mode_is_not_a_strategy_param_and_is_resolved_from_launch_params(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    try:
        cfg: dict[str, object] = {"parameters": {"rsi_period": 14, "liquidation_mode": "cross"}}
        assert runner._build_strategy_params(cfg) == {"rsi_period": 14}
        assert _pre_launch(cfg).liquidation.mode == "cross"
        assert _pre_launch({"parameters": {}}).liquidation.mode == "off"
        assert _pre_launch({}).liquidation.mode == "off"
        for bad in ("margin", "", None, 1, True, ["isolated"]):
            with pytest.raises(ValidationRunnerError, match="liquidation_mode"):
                _pre_launch({"parameters": {"liquidation_mode": bad}})
        # E11/C14: mode on with leverage 0 used to fall back to 10x silently; now it fails.
        with pytest.raises(ValidationRunnerError, match="leverage must be a whole number in"):
            _pre_launch({"parameters": {"liquidation_mode": "isolated", "leverage": 0}})
        # C14: the leverage is validated in EVERY mode (off included) through the one rule.
        for bad_lev in (0, 2.5, True, "10"):
            with pytest.raises(ValidationRunnerError, match=r"\[1, 125\]"):
                _pre_launch({"parameters": {"leverage": bad_lev}})
        assert _pre_launch({"parameters": {"leverage": 3}}).liquidation.leverage == 3.0
    finally:
        runner.close()


def _pre_launch(run_config: dict[str, object]) -> LaunchSettings:
    """The pre-detail launch settings ``run`` parses from ``run_config`` (rabmo.3 S5)."""
    params = run_config.get("parameters")
    return LaunchSettings.from_fields(
        _parse_launch_fields(
            params if isinstance(params, dict) else {},
            PARSE_ORDER["validation_pre_detail"],
        )
    )


def test_validation_venue_uses_the_run_leverage_when_liquidation_is_on(tmp_path: Path) -> None:
    """E12: the venue's ``default_leverage`` equals the run leverage for a mode-on run."""
    runner = _runner(tmp_path)
    try:
        cfg: dict[str, object] = {
            "parameters": {
                "liquidation_mode": "isolated",
                "leverage": 20,
                "initial_balance": 10000,
            }
        }
        venue, _launch = runner._create_venue_config(cfg, None, timeframe="1h")
        assert venue.default_leverage == Decimal("20")
    finally:
        runner.close()


def test_validation_venue_uses_the_off_leverage_and_rejects_bad_values(tmp_path: Path) -> None:
    """C14/R2: the venue follows ``leverage`` in every mode; a bad value fails the run.

    R2: a stored integer 10 stays ``Decimal("10")`` (== today's bytes for NautilusTrader).
    """
    runner = _runner(tmp_path)
    try:
        off_three, _launch = runner._create_venue_config(
            {"parameters": {"leverage": 3, "initial_balance": 10000}}, None, timeframe="1h"
        )
        assert off_three.default_leverage == Decimal("3")
        stored_ten, _launch = runner._create_venue_config(
            {"parameters": {"leverage": 10, "initial_balance": 10000}}, None, timeframe="1h"
        )
        assert stored_ten.default_leverage == Decimal("10")
        for bad in (0, 2.5, True, "10", None):
            with pytest.raises(ValidationRunnerError, match=r"\[1, 125\]"):
                runner._create_venue_config(
                    {"parameters": {"leverage": bad}}, None, timeframe="1h"
                )
    finally:
        runner.close()


# ---------------------------------------------- funding, plumbing and the import rule


class _FakeFunding:
    """5 USDT paid at fixed settlement bars, counted up to and including the exit."""

    def __init__(self, bars: tuple[int, ...]) -> None:
        self.bars = bars

    def accrue(
        self,
        instrument_id: str,
        direction: str,
        entry_notional: float,
        entry_ns: int,
        exit_ns: int | None,
    ) -> Any:
        from vibe_quant.validation.funding import FundingAccrual

        accrual = FundingAccrual()
        if exit_ns is None:
            return accrual
        accrual.payments = [(bar(i), 5.0) for i in self.bars if entry_ns < bar(i) <= exit_ns]
        accrual.total = sum(amount for _, amount in accrual.payments)
        return accrual


def test_liquidated_trade_pays_funding_only_up_to_the_liquidation(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    mk = {BTC: crash_marks(to=36000.0)}  # liquidated at bar 10; the strategy's exit is bar 30
    install_apply(monkeypatch, apply_calls, mk)
    result = ValidationResult(starting_balance=START_BAL)
    funding = _FakeFunding((5, 8, 20, 40))
    extract_trades(
        result,
        engine_for([long_btc()], mk),  # type: ignore[arg-type]
        VENUE,  # type: ignore[arg-type]
        funding_calculator=funding,  # type: ignore[arg-type]
        run_start_date=START,
        run_end_date=END,
        execution_timeframe="1m",
        catalog_path="x",
        liquidation_mode="isolated",
    )

    # the overlay was fed the untruncated payments (ledger sign: negative = paid) ...
    assert apply_calls[0]["funding"] == {
        key_of(long_btc()): [(bar(5), -5.0), (bar(8), -5.0), (bar(20), -5.0)]
    }
    # ... the trade keeps the two settlements before the liquidation (bar 10)
    (trade,) = result.trades
    assert trade.exit_reason == "liquidation"
    assert trade.funding_fees == 10.0 and result.total_funding == 10.0
    # funding already reduced the isolated margin (hit loss 3990); fee 20 + funding 10 on top:
    # the account loses exactly the initial margin plus the entry fee
    assert trade.net_pnl == pytest.approx(-4000.0 - 20.0)
    assert result.total_return == pytest.approx(-4020.0 / START_BAL)


def test_liquidated_run_keeps_the_ledger_in_step_with_the_trades(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    """Drawdown comes from the truncated ledger: it reaches the account loss, no more."""
    mk = {BTC: crash_marks(to=36000.0)}
    install_apply(monkeypatch, apply_calls, mk)
    res = run([long_btc()], mk, mode="isolated")
    # equity 10000 -> 10000 - 20 fee - 4000 margin = 5980 at the liquidation
    assert res.max_drawdown == pytest.approx(4020.0 / START_BAL)


def test_extract_results_forwards_the_mode_and_only_a_recomputed_run_skips_the_engine_moments(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    import vibe_quant.validation.extraction as ex

    seen: dict[str, Any] = {}
    real = ex.extract_trades

    def spy(*args: Any, **kwargs: Any) -> bool:
        seen.update(kwargs)
        return bool(real(*args, **kwargs))

    monkeypatch.setattr(ex, "extract_trades", spy)
    mk = {BTC: crash_marks(to=36000.0)}
    install_apply(monkeypatch, apply_calls, mk)
    bt = SimpleNamespace(
        elapsed_time=1.0, total_positions=1, stats_pnls={}, stats_returns={}
    )
    venue = SimpleNamespace(
        starting_balance_usdt=START_BAL,
        default_leverage=VENUE.default_leverage,
        fill_config=VENUE.fill_config,
    )
    moments: list[bool] = []
    monkeypatch.setattr(ex, "validation_engine_moments", lambda r, e: moments.append(True))

    res = ex.extract_results(
        1, "s", bt, engine_for([long_btc()], mk), venue,  # type: ignore[arg-type]
        run_start_date=START, run_end_date=END, execution_timeframe="1m",
        catalog_path="x", liquidation_mode="isolated",
    )
    assert seen["liquidation_mode"] == "isolated"
    assert res.trades[0].exit_reason == "liquidation"
    assert moments == []  # recomputed from the ledgers, not from the engine's positions

    ex.extract_results(
        1, "s", bt, engine_for([long_btc()], mk), venue,  # type: ignore[arg-type]
        run_start_date=START, run_end_date=END, execution_timeframe="1m",
    )
    assert moments == [True]  # mode off: the unchanged engine path


def test_mode_off_does_not_import_the_liquidation_package() -> None:
    import subprocess

    code = (
        "import sys; import vibe_quant.validation.extraction, vibe_quant.validation.runner; "
        "bad = sorted(m for m in sys.modules "
        "if m == 'vibe_quant.liquidation' or m.startswith('vibe_quant.liquidation.')); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_clearance_fee_counts_as_a_cost_in_fees_gross_and_fee_drag(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    """Chief ruling: total_fees = commissions + clearance; gross is cost-free; net unchanged."""
    from vibe_quant.metrics import fee_drag

    # a big ETH short winner (+6000 price pnl, fees 10) keeps the run's gross PnL positive
    eth_pos = Pos("P-9", ETH, [Fill(bar(3), OrderSide.SELL, 300.0, 2000.0, 5.0),
                               Fill(bar(20), OrderSide.BUY, 300.0, 1980.0, 5.0)])
    mk = {BTC: crash_marks(to=36000.0), ETH: marks([2000.0] * 60, [2010.0] * 60)}
    install_apply(monkeypatch, apply_calls, mk)
    res = run([long_btc(), eth_pos], mk, mode="isolated")

    clearance = 4000.0 - (40000.0 - LP_ISOLATED)  # 144.578...
    assert liq_note(res)["total_forfeited"] == pytest.approx(clearance)
    # commissions of both trades (BTC entry 20, ETH 5 + 5) PLUS the clearance fee
    assert res.total_fees == pytest.approx(20.0 + 10.0 + clearance)
    net = sum(t.net_pnl for t in res.trades)
    assert net == pytest.approx(-4020.0 + 6000.0 - 10.0)  # unchanged by the fee rule
    assert res.total_return == pytest.approx(net / START_BAL)
    # cost-free PnL = net + fees (incl. clearance) + funding: BTC price loss + ETH price gain
    assert res.gross_pnl == pytest.approx(net + res.total_fees + res.total_funding)
    assert res.gross_pnl == pytest.approx((LP_ISOLATED - 40000.0) + 6000.0)
    assert sum(t.gross_pnl for t in res.trades) == pytest.approx(res.gross_pnl)
    drag = fee_drag(res.total_fees, res.gross_pnl)
    assert drag == pytest.approx((30.0 + clearance) / ((LP_ISOLATED - 40000.0) + 6000.0))
    assert drag is not None and drag > 30.0 / res.gross_pnl  # clearance moves the drag


# ================================================================== fix round 2 (review)
# Hits come from the RAW positions (independent of any LAST mark), entry slippage survives a
# liquidation, the overlay's execution grid is the run's own, the walk-forward aggregate keeps
# the cost-free PnL, and the survivor's end-of-run exit fee is a valuation cost, not a trigger.


def spy_marking(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the position ids handed to ``mark_open_positions`` (one list per call)."""
    import vibe_quant.validation.extraction as ex

    real = ex.mark_open_positions
    seen: list[list[str]] = []

    def spy(engine: Any, positions: Any, **kwargs: Any) -> Any:
        listed = list(positions)
        seen.append([str(p.id) for p in listed])
        return real(engine, listed, **kwargs)

    monkeypatch.setattr(ex, "mark_open_positions", spy)
    return seen


def no_last_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    """The catalog and the engine cache hold no LAST execution bars at all."""
    import vibe_quant.validation.extraction as ex

    monkeypatch.setattr(ex, "load_catalog_bars", lambda *a, **kw: None)
    monkeypatch.setattr(ex, "cache_bars", lambda *a, **kw: None)


def ledger_equity_delta(ledgers: dict[str, list[Any]]) -> float:
    """Terminal realized equity change of the overlay ledgers (replayed like pnl_path)."""
    from vibe_quant.metrics import LedgerBook

    total = 0.0
    for events in ledgers.values():
        book = LedgerBook()
        for e in events:
            book.apply(e)
        total += book.realized
    return total


def spy_ledgers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, list[Any]]]:
    """Capture the ledgers handed to the mark-to-market drawdown."""
    import vibe_quant.validation.extraction as ex

    real = ex.mark_to_market_drawdown
    seen: list[dict[str, list[Any]]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("ledgers") is not None:
            seen.append(kwargs["ledgers"])
        return real(*args, **kwargs)

    monkeypatch.setattr(ex, "mark_to_market_drawdown", spy)
    return seen


def test_open_liquidated_position_survives_a_missing_last_bar(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    mk = {BTC: crash_marks(to=36000.0)}
    install_apply(monkeypatch, apply_calls, mk)
    no_last_bars(monkeypatch)
    marked = spy_marking(monkeypatch)
    res = run([long_btc(sell_bar=None)], mk, mode="isolated")

    assert len(apply_calls) == 1  # the scan ran on the raw position
    (trade,) = res.trades
    assert trade.exit_reason == "liquidation" and trade.net_pnl == pytest.approx(-4020.0)
    assert res.total_trades == 1 and liq_note(res)["count"] == 1
    assert res.total_return == pytest.approx(-0.402)
    assert res.max_drawdown >= 0.402 - 1e-9
    assert len(res.liquidation_events) == 1
    assert [ids for ids in marked if ids] == []  # a liquidated position is never marked


def test_open_and_closed_hits_are_both_booked_without_any_last_bar(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    eth_marks = marks([2000.0] * 12 + [1800.0] * 48, [2010.0] * 60)  # LP 1807.2: hit at bar 12
    mk = {BTC: crash_marks(to=36000.0), ETH: eth_marks}
    install_apply(monkeypatch, apply_calls, mk)
    no_last_bars(monkeypatch)
    marked = spy_marking(monkeypatch)
    closed = long_btc(sell_bar=30, entry_bar=2, pid="P-1")
    eth_open = Pos("P-2", ETH, [Fill(bar(4), OrderSide.BUY, 10.0, 2000.0, 5.0)], is_closed=False)
    res = run([closed, eth_open], mk, mode="isolated")

    assert liq_note(res)["count"] == 2
    assert res.total_trades == 2 and {t.exit_reason for t in res.trades} == {"liquidation"}
    # BTC: margin 4000 + fee 20; ETH: margin 2000 + fee 5
    assert res.total_return == pytest.approx(-6025.0 / START_BAL)
    assert res.total_return == pytest.approx(sum(t.net_pnl for t in res.trades) / START_BAL)
    assert res.max_drawdown >= 6025.0 / START_BAL - 1e-9
    assert [ids for ids in marked if ids] == []


def test_dropped_and_liquidated_open_positions_are_never_marked(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    mk = {BTC: crash_marks(to=30000.0)}
    install_apply(monkeypatch, apply_calls, mk)
    no_last_bars(monkeypatch)
    marked = spy_marking(monkeypatch)
    later = Pos("P-2", BTC, [Fill(bar(45), OrderSide.BUY, 1.0, 30000.0, 15.0)], is_closed=False)
    res = run([long_btc(sell_bar=None), later], mk, mode="cross")

    note = liq_note(res)
    assert note["count"] == 1 and note["dropped"] == 1
    (trade,) = res.trades  # the dropped position leaves no trade, the hit one stays
    assert trade.exit_reason == "liquidation" and trade.net_pnl == pytest.approx(-START_BAL)
    assert [ids for ids in marked if ids] == []


def test_unpriceable_survivor_fails_closed_while_the_mode_is_on_and_is_skipped_when_off(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    eth_open = Pos("P-ETH", ETH, [Fill(bar(3), OrderSide.BUY, 1.0, 2000.0, 1.0)], is_closed=False)
    mk = {BTC: crash_marks(to=36000.0), ETH: marks([2000.0] * 60, [2010.0] * 60)}
    install_apply(monkeypatch, apply_calls, mk)
    no_last_bars(monkeypatch)

    # mode on: the hit BTC position is booked, the ETH survivor has no LAST bar -> fail closed
    with pytest.raises(ValueError, match=r"ETHUSDT-PERP\.BINANCE.*cannot be valued"):
        run([long_btc(sell_bar=None), eth_open], mk, mode="isolated")
    # ... also without any hit
    with pytest.raises(ValueError, match="cannot be valued"):
        run([eth_open], {ETH: marks([2000.0] * 60, [2010.0] * 60)}, mode="isolated")
    # mode off keeps the old behaviour: warn and leave the position out
    off = run([eth_open], mk)
    assert off.trades == [] and off.total_trades == 0


def test_entry_slippage_survives_a_liquidation_and_is_booked_at_the_entry(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    import vibe_quant.validation.extraction as ex

    class Slip:
        def __init__(self, **kw: Any) -> None:
            pass

        def estimate_cost(self, entry_price: float, **kw: Any) -> float:
            return 7.0 if entry_price == 40000.0 else 9.0  # entry leg 7, exit leg 9

    mk = {BTC: crash_marks(to=36000.0)}
    install_apply(monkeypatch, apply_calls, mk)
    monkeypatch.setattr(ex, "SlippageEstimator", Slip)
    ledgers = spy_ledgers(monkeypatch)
    venue = SimpleNamespace(
        default_leverage=VENUE.default_leverage,
        fill_config=SimpleNamespace(impact_coefficient=0.1, prob_slippage=0.0),
    )
    res = ValidationResult(starting_balance=START_BAL)
    p = long_btc()
    ex.extract_trades(
        res, engine_for([p], mk), venue,  # type: ignore[arg-type]
        run_start_date=START, run_end_date=END, execution_timeframe="1m",
        catalog_path="x", liquidation_mode="isolated",
    )

    # the overlay gets the entry leg at the entry and the exit leg at the (later) exit
    assert apply_calls[0]["cash"] == {key_of(p): [(bar(2), -7.0), (bar(30), -9.0)]}
    (trade,) = res.trades
    assert trade.exit_reason == "liquidation"
    assert trade.slippage_cost == pytest.approx(7.0)  # the strategy's exit never happened
    assert res.total_slippage == pytest.approx(7.0)
    assert trade.net_pnl == pytest.approx(-4027.0)
    assert res.total_return == pytest.approx(-0.4027)
    assert res.max_drawdown == pytest.approx(0.4027)
    assert ledger_equity_delta(ledgers[0]) == pytest.approx(-4027.0)  # ledger == trades
    assert res.gross_pnl == pytest.approx(
        sum(t.net_pnl for t in res.trades) + res.total_fees + res.total_slippage
    )


@pytest.mark.parametrize(
    ("timeframe", "bar_ns"),
    [("1m", 60_000_000_000), ("5s", 5_000_000_000), ("4h", 14_400_000_000_000)],
)
def test_overlay_execution_grid_is_the_runs_own_timeframe(
    monkeypatch: pytest.MonkeyPatch,
    apply_calls: list[dict[str, Any]],
    timeframe: str,
    bar_ns: int,
) -> None:
    mk = {BTC: crash_marks(to=36000.0)}
    install_apply(monkeypatch, apply_calls, mk)
    res = ValidationResult(starting_balance=START_BAL)
    import vibe_quant.validation.extraction as ex

    ex.extract_trades(
        res, engine_for([long_btc()], mk), VENUE,  # type: ignore[arg-type]
        run_start_date=START, run_end_date=END, execution_timeframe=timeframe,
        catalog_path="x", liquidation_mode="isolated",
    )
    assert apply_calls[0]["execution_bar_ns"] == bar_ns


def test_resting_entry_and_hit_inside_one_coarse_execution_bar_is_rejected(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    import vibe_quant.validation.extraction as ex

    mk = {BTC: crash_marks(to=36000.0)}  # entry 01:42 (resting), hit 01:50: one 4h bar
    install_apply(monkeypatch, apply_calls, mk)
    resting = long_btc()
    resting.events[0].order_type = OrderType.LIMIT

    def go(timeframe: str | None) -> ValidationResult:
        res = ValidationResult(starting_balance=START_BAL)
        ex.extract_trades(
            res, engine_for([resting], mk), VENUE,  # type: ignore[arg-type]
            run_start_date=START, run_end_date=END, execution_timeframe=timeframe,
            catalog_path="x", liquidation_mode="isolated",
        )
        return res

    assert go("1m").trades[0].exit_reason == "liquidation"  # distinct 1m bars: replayable
    with pytest.raises(ValueError, match="resting fill precedes"):
        go("4h")
    with pytest.raises(ValueError, match="execution timeframe"):
        go(None)  # no grid to value the liquidation on: fail closed


def test_walk_forward_aggregate_keeps_the_cost_free_pnl_and_the_liquidation_totals(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    from vibe_quant.metrics import fee_drag

    eth_pos = Pos("P-9", ETH, [Fill(bar(3), OrderSide.SELL, 300.0, 2000.0, 5.0),
                               Fill(bar(20), OrderSide.BUY, 300.0, 1980.0, 5.0)])
    mk = {BTC: crash_marks(to=36000.0), ETH: marks([2000.0] * 60, [2010.0] * 60)}
    install_apply(monkeypatch, apply_calls, mk)
    window = run([long_btc(), eth_pos], mk, mode="isolated")
    assert window.gross_pnl > 0

    agg = ValidationRunner._aggregate_walk_forward_results(
        run_id=1, strategy_name="s", window_results=[window, window]
    )
    assert agg.gross_pnl == pytest.approx(2.0 * window.gross_pnl)
    assert agg.gross_pnl == pytest.approx(
        sum(t.net_pnl for t in agg.trades) + agg.total_fees + agg.total_slippage + agg.total_funding
    )
    assert fee_drag(agg.total_fees, agg.gross_pnl) == pytest.approx(
        fee_drag(window.total_fees, window.gross_pnl)
    )
    note = liq_note(agg)
    wnote = liq_note(window)
    assert note["count"] == 2 and note["mode"] == "isolated" and note["leverage"] == 10.0
    assert note["total_forfeited"] == pytest.approx(2.0 * wnote["total_forfeited"])
    assert note["total_realized_pnl"] == pytest.approx(2.0 * wnote["total_realized_pnl"])
    assert note["dropped"] == 0
    assert note["pct_trades"] == pytest.approx(100.0 * 2 / agg.total_trades)
    assert len(agg.liquidation_events) == 2
    # no window used the overlay: no note
    plain = ValidationRunner._aggregate_walk_forward_results(
        run_id=1, strategy_name="s", window_results=[run([long_btc()], mk)]
    )
    assert "liquidation" not in plain.notes
    only = run([long_btc()], mk)
    assert plain.gross_pnl == pytest.approx(only.gross_pnl) and plain.gross_pnl != 0.0


def test_walk_forward_rejects_a_bad_liquidation_mode_before_any_window(tmp_path: Path) -> None:
    from vibe_quant.db import StateManager

    db = tmp_path / "wf.db"
    state = StateManager(db)
    dsl = {
        "name": "wf_strategy", "description": "t", "version": 1, "timeframe": "5m",
        "additional_timeframes": [],
        "indicators": {"rsi": {"type": "RSI", "period": 14, "source": "close"}},
        "entry_conditions": {"long": ["rsi < 30"], "short": ["rsi > 70"]},
        "exit_conditions": {"long": ["rsi > 50"], "short": ["rsi < 50"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
    sid = state.create_strategy(name="wf_strategy", dsl_config=dsl, description="t")
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="validation", symbols=["BTCUSDT"], timeframe="5m",
        start_date="2025-01-01", end_date="2025-01-31",
        parameters={"liquidation_mode": "bogus"}, latency_preset="retail",
    )
    state.close()
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    started: list[int] = []
    runner._run_backtest = lambda **kw: started.append(1)  # type: ignore[method-assign,assignment]
    try:
        with pytest.raises(ValidationRunnerError, match="liquidation_mode"):
            runner.run_walk_forward(run_id, train_days=7, test_days=7, step_days=7)
    finally:
        runner.close()
    assert started == []  # rejected before the first window's backtest
    state = StateManager(db)
    try:
        row = state.get_backtest_run(run_id)
    finally:
        state.close()
    assert row is not None and row["status"] == "failed"


def test_hit_plus_open_survivor_ledger_equity_equals_the_sum_of_trade_net_pnl(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    """The survivor's exit fee (ETH 2000 x 0.0005 = 1.0) is in the trade AND in the ledger."""
    eth_open = Pos("P-ETH", ETH, [Fill(bar(3), OrderSide.BUY, 1.0, 2000.0, 1.0)], is_closed=False)
    mk = {BTC: crash_marks(to=36000.0), ETH: marks([2000.0] * 60, [2010.0] * 60)}
    install_apply(monkeypatch, apply_calls, mk)
    ledgers = spy_ledgers(monkeypatch)
    res = run([long_btc(), eth_open], mk, mode="isolated")

    eth_trade = next(t for t in res.trades if t.symbol == ETH)
    assert eth_trade.exit_reason == "end_of_data"
    assert eth_trade.net_pnl == pytest.approx(-1.0 - 1.0)  # entry fee + estimated exit fee
    net = sum(t.net_pnl for t in res.trades)
    assert net == pytest.approx(-4020.0 - 2.0)
    assert ledger_equity_delta(ledgers[0]) == pytest.approx(net)
    assert res.total_return == pytest.approx(net / START_BAL)
    assert res.total_fees == pytest.approx(20.0 + (4000.0 - (40000.0 - LP_ISOLATED)) + 2.0)


def test_return_moments_are_recomputed_from_the_truncated_trades(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    mk = {BTC: crash_marks(to=36000.0), ETH: marks([2000.0] * 60, [2010.0] * 60)}
    install_apply(monkeypatch, apply_calls, mk)
    pos = [long_btc()]
    for i, px in enumerate([2010.0, 2020.0, 1990.0]):
        pos.append(
            Pos(
                f"E-{i}",
                ETH,
                [Fill(bar(3 + i * 2), OrderSide.BUY, 1.0, 2000.0, 0.5),
                 Fill(bar(4 + i * 2), OrderSide.SELL, 1.0, px, 0.5)],
            )
        )
    res = run(pos, mk, mode="isolated")
    # hand-checked with the sample-skew / bias-corrected-kurtosis formulas on the four
    # per-trade returns (liquidated -0.1 ... and three small ETH ones)
    assert res.skewness == pytest.approx(-1.9156, abs=1e-4)
    assert res.kurtosis == pytest.approx(6.7034, abs=1e-4)


def test_survivor_exit_fee_is_a_valuation_cost_not_a_liquidation_trigger(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    """Cross wallet 250 after the entry fee; LP 39909.6 sits under the 39912 low. The ~20
    end-of-run exit fee would lift the LP to ~39929 -- it must NOT liquidate the position."""
    mk = {BTC: marks([39912.0] * 60)}
    install_apply(monkeypatch, apply_calls, mk)
    result = ValidationResult(starting_balance=270.0)
    import vibe_quant.validation.extraction as ex

    ex.extract_trades(
        result, engine_for([long_btc(sell_bar=None)], mk), VENUE,  # type: ignore[arg-type]
        run_start_date=START, run_end_date=END, execution_timeframe="1m",
        catalog_path="x", liquidation_mode="cross",
    )

    assert liq_note(result)["count"] == 0 and result.liquidation_events == []
    (trade,) = result.trades
    exit_fee = 39912.0 * 0.0005
    assert trade.exit_reason == "end_of_data"
    expected_net = -20.0 + (39912.0 - 40000.0) - exit_fee
    assert trade.net_pnl == pytest.approx(expected_net)
    assert result.total_fees == pytest.approx(20.0 + exit_fee)
    # (the stub engine carries no NT stats: total_return holds only the mark-to-market part)
    assert result.total_return == pytest.approx((-88.0 - exit_fee) / 270.0)
    # the scan decided on the fee-free ledger: the fee is booked in the first scan's own
    # call (no survivor cash handed to the overlay)
    assert all(not call["cash"] for call in apply_calls)


# ----------------------------------------------- fix round 3: one scan, one horizon, gross

def comparable(res: ValidationResult) -> dict[str, Any]:
    """Every reported field except the liquidation provenance."""
    from dataclasses import asdict

    out = asdict(res)
    out.pop("notes")
    out.pop("liquidation_events")
    return out


def test_gross_pnl_is_cost_free_in_every_mode(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    mk = {BTC: marks([40000.0] * 60)}
    install_apply(monkeypatch, apply_calls, mk)
    off = run([long_btc()], mk)
    net = sum(t.net_pnl for t in off.trades)
    # price pnl 1000 gross of both fees (20 + 20.5): net + fees + slippage + funding
    assert off.gross_pnl == pytest.approx(net + off.total_fees + off.total_slippage)
    assert off.gross_pnl == pytest.approx(1000.0)


def test_zero_hit_mode_on_equals_off_in_every_field_with_a_survivor_and_funding(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    import vibe_quant.validation.extraction as ex

    eth_open = Pos("P-ETH", ETH, [Fill(bar(3), OrderSide.BUY, 1.0, 2000.0, 1.0)], is_closed=False)
    mk = {BTC: marks([40000.0] * 60), ETH: marks([2000.0] * 60, [2010.0] * 60)}
    install_apply(monkeypatch, apply_calls, mk)
    funding = _FakeFunding((5, 30, 59))

    def go(mode: str) -> ValidationResult:
        res = ValidationResult(starting_balance=START_BAL)
        ex.extract_trades(
            res, engine_for([long_btc(), eth_open], mk), VENUE,  # type: ignore[arg-type]
            funding_calculator=funding,  # type: ignore[arg-type]
            run_start_date=START, run_end_date=END, execution_timeframe="1m",
            catalog_path="x", liquidation_mode=mode,
        )
        return res

    off, on = go("off"), go("isolated")
    assert liq_note(on)["count"] == 0
    assert comparable(on) == comparable(off)
    assert off.total_funding > 0 and off.gross_pnl != 0.0


class _AtFunding:
    """5 USDT paid at explicit settlement timestamps in (entry, exit]."""

    def __init__(self, timestamps: tuple[int, ...]) -> None:
        self.timestamps = timestamps

    def accrue(
        self, instrument_id: str, direction: str, entry_notional: float, entry_ns: int,
        exit_ns: int | None,
    ) -> Any:
        from vibe_quant.validation.funding import FundingAccrual

        accrual = FundingAccrual()
        if exit_ns is None:
            return accrual
        accrual.payments = [(t, 5.0) for t in self.timestamps if entry_ns < t <= exit_ns]
        accrual.total = sum(amount for _, amount in accrual.payments)
        return accrual


def test_survivor_scan_funding_equals_mode_off_funding_including_the_final_bar(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    """Settlements at bar 30, inside the final bar (bar 59) and 0.5 ms before the run end:
    the last one lies after the survivor's mark (end - 1 ms) so mode off never charges it.
    Net, ledger and the funding fed to the scan agree with the survivor's real funding."""
    import vibe_quant.validation.extraction as ex

    end_ns = date_to_ns(END)
    assert end_ns is not None
    funding = _AtFunding((bar(30), bar(59), end_ns - 500_000))
    eth_open = Pos("P-ETH", ETH, [Fill(bar(3), OrderSide.BUY, 1.0, 2000.0, 1.0)], is_closed=False)
    mk = {BTC: crash_marks(to=36000.0), ETH: marks([2000.0] * 60, [2010.0] * 60)}
    install_apply(monkeypatch, apply_calls, mk)
    ledgers = spy_ledgers(monkeypatch)
    res = ValidationResult(starting_balance=START_BAL)
    ex.extract_trades(
        res, engine_for([long_btc(), eth_open], mk), VENUE,  # type: ignore[arg-type]
        funding_calculator=funding,  # type: ignore[arg-type]
        run_start_date=START, run_end_date=END, execution_timeframe="1m",
        catalog_path="x", liquidation_mode="isolated",
    )

    eth_trade = next(t for t in res.trades if t.symbol == ETH)
    assert eth_trade.funding_fees == 10.0  # bar 30 and bar 59, not the one after the mark
    assert apply_calls[0]["funding"][key_of(eth_open)] == [(bar(30), -5.0), (bar(59), -5.0)]
    btc_trade = next(t for t in res.trades if t.symbol == BTC)
    assert btc_trade.exit_reason == "liquidation"
    net = sum(t.net_pnl for t in res.trades)
    assert net == pytest.approx(-4020.0 + (-1.0 - 1.0 - 10.0))  # BTC margin+fee; ETH fees+funding
    assert ledger_equity_delta(ledgers[0]) == pytest.approx(net)  # residual 0
    assert res.total_funding == pytest.approx(10.0 + btc_trade.funding_fees)
    assert res.gross_pnl == pytest.approx(net + res.total_fees + res.total_slippage + res.total_funding)


def test_survivor_whose_last_ends_before_the_final_bar_fails_closed_naming_the_ts(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    import vibe_quant.validation.extraction as ex

    eth_open = Pos("P-ETH", ETH, [Fill(bar(3), OrderSide.BUY, 1.0, 2000.0, 1.0)], is_closed=False)
    mk = {BTC: crash_marks(to=36000.0), ETH: marks([2000.0] * 60, [2010.0] * 60)}
    install_apply(monkeypatch, apply_calls, mk)
    last_end = bar(60) - 1_000_000  # last LAST bar's ts: 02:39:59.999
    late_end = "2024-01-02"  # the run goes on to 24:00: LAST stops 21 h short of the run end
    end_ns = date_to_ns(late_end)

    def go(positions: list[Pos]) -> ValidationResult:
        res = ValidationResult(starting_balance=START_BAL)
        ex.extract_trades(
            res, engine_for(positions, mk), VENUE,  # type: ignore[arg-type]
            run_start_date=START, run_end_date=late_end, execution_timeframe="1m",
            catalog_path="x", liquidation_mode="isolated",
        )
        return res

    for positions in ([eth_open], [long_btc(), eth_open]):  # without and with a BTC hit
        with pytest.raises(ValueError) as err:
            go(positions)
        msg = str(err.value)
        assert "ETHUSDT-PERP.BINANCE" in msg and "cannot be valued" in msg
        assert str(last_end) in msg and str(end_ns) in msg
    # a stale-LAST position that the scan itself liquidates needs no valuation (R2)
    res = go([long_btc(sell_bar=None)])
    (trade,) = res.trades
    assert trade.exit_reason == "liquidation" and trade.net_pnl == pytest.approx(-4020.0)
    # mode off marks the same stale bar as before (no new check)
    off = ValidationResult(starting_balance=START_BAL)
    ex.extract_trades(
        off, engine_for([eth_open], mk), VENUE,  # type: ignore[arg-type]
        run_start_date=START, run_end_date=late_end, execution_timeframe="1m", catalog_path="x",
    )
    assert off.trades[0].exit_reason == "end_of_data"


# ------------------------------------------- fix round 4: hit funding truncation, no look-ahead


def real_apply_with_marks(
    monkeypatch: pytest.MonkeyPatch, series: Marks | dict[str, Marks]
) -> list[dict[str, Any]]:
    """Use the REAL shared ``apply_liquidation``; only the catalog MARK load is replaced
    (one series for every symbol, or one per instrument id).

    Returns the funding handed to each scan (``funding_by_position``).
    """
    import vibe_quant.liquidation.apply as ap

    by_symbol = (
        {iid.split("-")[0]: s for iid, s in series.items()} if isinstance(series, dict) else None
    )
    monkeypatch.setattr(
        ap,
        "load_mark_series",
        lambda _path, symbol: series if by_symbol is None else by_symbol[symbol],
    )
    scans: list[dict[str, Any]] = []
    real = ap.apply_liquidation

    def capture(*args: Any, **kwargs: Any) -> Any:
        scans.append(kwargs["funding_by_position"])
        return real(*args, **kwargs)

    monkeypatch.setattr(ap, "apply_liquidation", capture)
    return scans


def test_a_crash_after_the_run_end_cannot_liquidate_a_position_real_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibe_quant.validation.extraction as ex

    # MARK is safe through the run (ends 02:40) and crashes at 02:45
    mk = {BTC: crash_marks(to=36000.0, at=65, n=70)}
    real_apply_with_marks(monkeypatch, mk[BTC])

    def go(mode: str) -> ValidationResult:
        res = ValidationResult(starting_balance=START_BAL, total_return=-20.0 / START_BAL)
        ex.extract_trades(
            res, engine_for([long_btc(sell_bar=None)], mk), VENUE,  # type: ignore[arg-type]
            run_start_date="2024-01-01T01:40:00", run_end_date=END,
            execution_timeframe="1m", catalog_path="x", liquidation_mode=mode,
        )
        return res

    off, on = go("off"), go("isolated")
    assert off.trades[0].net_pnl == pytest.approx(-40.0)  # entry fee + estimated exit fee
    assert liq_note(on)["count"] == 0 and on.liquidation_events == []
    assert on.trades[0].net_pnl == pytest.approx(-40.0) and on.trades[0].exit_reason == "end_of_data"
    assert on.total_return == off.total_return


def test_hit_after_the_valuation_horizon_reuses_the_scan_funding_truncated_at_the_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1d execution: LAST ends 02:39:59.999 (the horizon, inside the final 1d bar), a 5 USDT
    settlement at 02:45 is beyond it, the MARK crash hits at 02:50 -- still the same 1d bar.
    The scan never saw the 02:45 settlement, so neither may the trade (and the ledger)."""
    import vibe_quant.validation.extraction as ex

    crash = crash_marks(to=36000.0, at=170, n=1440)
    day0 = date_to_ns(START)
    assert day0 is not None
    full_day = Marks(
        day0 + np.arange(1440, dtype=np.int64) * MIN, crash.low, crash.high, crash.close, crash.open
    )
    scans = real_apply_with_marks(monkeypatch, full_day)
    last = exec_series(marks([40000.0] * 60))  # LAST stops at 02:39:59.999
    monkeypatch.setattr(ex, "load_catalog_bars", lambda *a, **kw: last)
    ledgers = spy_ledgers(monkeypatch)
    res = ValidationResult(starting_balance=START_BAL)
    p = long_btc(sell_bar=None)
    ex.extract_trades(
        res, engine_for([p], {BTC: full_day}), VENUE,  # type: ignore[arg-type]
        funding_calculator=_AtFunding((bar(65),)),  # type: ignore[arg-type]
        run_start_date=START, run_end_date="2024-01-02", execution_timeframe="1d",
        catalog_path="x", liquidation_mode="isolated",
    )

    assert res.liquidation_events[0]["ts"] == bar(70)  # 02:50, after the horizon and the 02:45 payment
    assert bar(70) > int(last.ts[-1])
    assert scans[0] == {}  # nothing funded inside the horizon
    (trade,) = res.trades
    assert trade.exit_reason == "liquidation" and trade.funding_fees == 0.0
    assert res.total_funding == 0.0
    net = sum(t.net_pnl for t in res.trades)
    assert net == pytest.approx(-4020.0)
    assert ledger_equity_delta(ledgers[0]) == pytest.approx(net)  # residual 0


def test_hit_funding_is_the_scan_accrual_cut_at_the_hit_with_its_fallback_count(
    monkeypatch: pytest.MonkeyPatch, apply_calls: list[dict[str, Any]]
) -> None:
    """Settlements at bar 5 (fallback rate), bar 8 and bar 20; liquidated at bar 10: the trade
    keeps the first two and exactly one fallback settlement; bar 20 is gone."""
    from vibe_quant.validation.funding import FundingAccrual

    class Calc:
        def accrue(
            self, instrument_id: str, direction: str, entry_notional: float, entry_ns: int,
            exit_ns: int | None,
        ) -> Any:
            acc = FundingAccrual()
            if exit_ns is None:
                return acc
            acc.payments = [(bar(i), 5.0) for i in (5, 8, 20) if entry_ns < bar(i) <= exit_ns]
            acc.total = sum(a for _, a in acc.payments)
            acc.fallback_settlements = len([1 for i in (5, 20) if entry_ns < bar(i) <= exit_ns])
            acc.fallback_timestamps = [bar(i) for i in (5, 20) if entry_ns < bar(i) <= exit_ns]
            return acc

    import vibe_quant.validation.extraction as ex

    mk = {BTC: crash_marks(to=36000.0)}
    install_apply(monkeypatch, apply_calls, mk)
    res = ValidationResult(starting_balance=START_BAL)
    ex.extract_trades(
        res, engine_for([long_btc(sell_bar=None)], mk), VENUE,  # type: ignore[arg-type]
        funding_calculator=Calc(),  # type: ignore[arg-type]
        run_start_date=START, run_end_date=END, execution_timeframe="1m",
        catalog_path="x", liquidation_mode="isolated",
    )
    (trade,) = res.trades
    assert trade.funding_fees == 10.0 and res.total_funding == 10.0
    assert res.funding_fallback_settlements == 1  # bar 5 yes, the truncated bar 20 no


# ------------------------------- fix round 5: real funding provenance, half-open run end

DAY = 1440 * MIN
DAY0 = BASE - 100 * MIN  # 2024-01-01 00:00 (an int for the type checker)


def test_real_funding_calculator_keeps_the_fallback_settlement_before_the_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REAL FundingCalculator on an archive with a hole: 1 BTC long from 40000 at 01:42,
    1d execution (horizon 23:59:59.999), no archived rate at 08:00 or 16:00 (fallback 0.0001 x
    40000 = 4.0 each), MARK crash liquidates at 08:10. The trade keeps the 08:00 settlement
    AND its fallback provenance; 16:00 (after the hit) is gone with its fallback."""
    import vibe_quant.validation.extraction as ex
    from vibe_quant.data.archive import RawDataArchive
    from vibe_quant.validation.funding import FundingCalculator

    archive_path = tmp_path / "archive.db"
    archive = RawDataArchive(archive_path)
    try:  # archived rates only at the day's edges: 08:00 and 16:00 are an interior hole
        archive.insert_funding_rates(
            "BTCUSDT",
            [(DAY0 // 1_000_000, 0.0003, None), ((DAY0 + DAY) // 1_000_000, 0.0003, None)],
            "test",
        )
    finally:
        archive.close()
    calculator = FundingCalculator(archive_path=archive_path)
    assert calculator.default_rate == 0.0001

    crash = crash_marks(to=36000.0, at=490, n=1440)  # 08:10
    full_day = Marks(
        DAY0 + np.arange(1440, dtype=np.int64) * MIN, crash.low, crash.high, crash.close,
        crash.open,
    )
    scans = real_apply_with_marks(monkeypatch, full_day)
    last = BarSeries(  # the day's one 1d LAST bar, stamped like the catalog (close - 1 ms)
        ts=np.array([DAY0 + DAY - 1_000_000], dtype=np.int64),
        open=np.array([40000.0]), high=np.array([40000.0]), low=np.array([36000.0]),
        close=np.array([36000.0]),
    )
    monkeypatch.setattr(ex, "load_catalog_bars", lambda *a, **kw: last)
    ledgers = spy_ledgers(monkeypatch)
    p = long_btc(sell_bar=None)  # entry bar(2) = 01:42
    res = ValidationResult(starting_balance=START_BAL)
    ex.extract_trades(
        res, engine_for([p], {BTC: full_day}), VENUE,  # type: ignore[arg-type]
        funding_calculator=calculator,
        run_start_date=START, run_end_date="2024-01-02", execution_timeframe="1d",
        catalog_path="x", liquidation_mode="isolated",
    )

    h08, h16 = DAY0 + 8 * 60 * MIN, DAY0 + 16 * 60 * MIN
    # the scan was funded to the horizon with both fallback settlements
    assert scans[0] == {key_of(p): [(h08, -4.0), (h16, -4.0)]}
    assert res.liquidation_events[0]["ts"] == DAY0 + 490 * MIN
    (trade,) = res.trades
    assert trade.exit_reason == "liquidation"
    assert trade.funding_fees == 4.0 and res.total_funding == 4.0
    assert res.funding_fallback_settlements == 1  # 08:00 kept; 16:00 cut with the hit
    net = sum(t.net_pnl for t in res.trades)
    assert net == pytest.approx(-4020.0)
    assert ledger_equity_delta(ledgers[0]) == pytest.approx(net)  # residual 0


@pytest.mark.parametrize(
    ("crash_bar", "hits", "net", "ret"),
    [(59, 1, -4020.0, -0.402), (60, 0, -40.0, -0.004)],
    ids=["final-minute-hits", "bar-at-run-end-does-not"],
)
def test_validation_mark_window_is_half_open_at_the_run_end_real_apply(
    monkeypatch: pytest.MonkeyPatch, crash_bar: int, hits: int, net: float, ret: float
) -> None:
    """Run 01:40..02:40, 1 BTC long @40000 (fee 20) still open. A MARK crash in the bar opening
    02:39 (the run's final minute) liquidates; the same crash in the bar opening at exactly
    02:40 = end_ns lies outside [start, end) and changes nothing (ON == OFF)."""
    import vibe_quant.validation.extraction as ex

    mk = {BTC: crash_marks(to=36000.0, at=crash_bar, n=70)}
    real_apply_with_marks(monkeypatch, mk[BTC])

    def go(mode: str) -> ValidationResult:
        # NT's PnL% for the entry fee; mode off adds the estimated exit fee
        res = ValidationResult(starting_balance=START_BAL, total_return=-20.0 / START_BAL)
        ex.extract_trades(
            res, engine_for([long_btc(sell_bar=None)], mk), VENUE,  # type: ignore[arg-type]
            run_start_date="2024-01-01T01:40:00", run_end_date=END,
            execution_timeframe="1m", catalog_path="x", liquidation_mode=mode,
        )
        return res

    on = go("isolated")
    assert liq_note(on)["count"] == hits
    assert [e["ts"] for e in on.liquidation_events] == ([bar(59)] if hits else [])
    (trade,) = on.trades
    assert trade.net_pnl == pytest.approx(net)
    assert on.total_return == pytest.approx(ret)
    if not hits:
        assert comparable(on) == comparable(go("off"))


def _end_fill_positions(exit_ns: int) -> list[Pos]:
    """Cross, balance 4020: 1 BTC long @40000 (fee 20) stays open at MARK 36150 (margin 150 >
    MM 144.6); an ETH 0.001 @2000 round trip exits at ``exit_ns`` paying commission 20, which
    takes the margin to 130 < 144.6."""
    eth = Pos(
        "P-ETH",
        ETH,
        [
            Fill(bar(3), OrderSide.BUY, 0.001, 2000.0, 0.0),
            Fill(exit_ns, OrderSide.SELL, 0.001, 2000.0, 20.0),
        ],
    )
    return [long_btc(sell_bar=None), eth]


def _end_fill_marks() -> dict[str, Marks]:
    return {
        BTC: marks([36150.0] * 60, [36150.0] * 60),
        ETH: marks([2000.0] * 60, [2000.0] * 60),
    }


END_FILL_BALANCE = 4020.0


def test_a_raw_fill_at_the_run_end_fails_closed_instead_of_liquidating_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R6b: NT processes an input stamped exactly end_ns, so a fill there can exist. Its
    commission must never decide a liquidation at end_ns: the shared apply refuses it."""
    import vibe_quant.validation.extraction as ex

    end_ns = date_to_ns(END)
    assert end_ns is not None
    mk = _end_fill_marks()
    real_apply_with_marks(monkeypatch, mk)

    def go(exit_ns: int) -> ValidationResult:
        res = ValidationResult(starting_balance=END_FILL_BALANCE)
        ex.extract_trades(
            res, engine_for(_end_fill_positions(exit_ns), mk), VENUE,  # type: ignore[arg-type]
            run_start_date="2024-01-01T01:40:00", run_end_date=END,
            execution_timeframe="1m", catalog_path="x", liquidation_mode="cross",
        )
        return res

    with pytest.raises(ValueError, match=str(end_ns)):
        go(end_ns)
    # control: the same fill inside the window (the last catalog stamp, end - 1 ms) is real
    # exposure and liquidates the account there
    inside = go(end_ns - 1_000_000)
    assert [e["ts"] for e in inside.liquidation_events] == [end_ns - 1_000_000]


def _cross_liquidation_run(db: Path, *, start: str, end: str) -> int:
    """A 1m BTC+ETH validation run launched with ``liquidation_mode="cross"``."""
    from vibe_quant.db import StateManager

    state = StateManager(db)
    try:
        dsl = {
            "name": "r6b_strategy", "description": "t", "version": 1, "timeframe": "1m",
            "additional_timeframes": [],
            "indicators": {"rsi": {"type": "RSI", "period": 14, "source": "close"}},
            "entry_conditions": {"long": ["rsi < 30"], "short": ["rsi > 70"]},
            "exit_conditions": {"long": ["rsi > 50"], "short": ["rsi < 50"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 4.0},
        }
        sid = state.create_strategy(name="r6b_strategy", dsl_config=dsl, description="t")
        return state.create_backtest_run(
            strategy_id=sid, run_mode="validation", symbols=["BTCUSDT", "ETHUSDT"],
            timeframe="1m", start_date=start, end_date=end,
            parameters={"liquidation_mode": "cross"}, latency_preset="retail",
        )
    finally:
        state.close()


def _offline_runner(monkeypatch: pytest.MonkeyPatch, db: Path, tmp_path: Path) -> ValidationRunner:
    """A runner whose detail-data resolution and window clamp need no catalog."""
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    monkeypatch.setattr(runner, "_resolve_detail_timeframe", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_clamp_run_window", lambda cfg, tf: (cfg, None))
    return runner


def _extract_end_fill_window(
    runner: ValidationRunner,
    kw: dict[str, Any],
    positions: list[Pos],
    mk: dict[str, Marks],
    *,
    catalog_path: Path,
) -> ValidationResult:
    """Stand-in for ``_run_backtest``: the NT run is replaced, the extraction is the runner's
    own (real shared apply), on the 01:40..02:40 window with the cross balance 4020."""
    venue = SimpleNamespace(
        starting_balance_usdt=END_FILL_BALANCE,
        default_leverage=VENUE.default_leverage,
        fill_config=VENUE.fill_config,
    )
    bt = SimpleNamespace(
        elapsed_time=1.0, total_positions=len(positions), stats_pnls={}, stats_returns={}
    )
    return runner._extract_results(
        run_id=kw["run_id"], strategy_name=kw["strategy_name"], bt_result=bt,  # type: ignore[arg-type]
        engine=engine_for(positions, mk),  # type: ignore[arg-type]
        venue_config=venue,  # type: ignore[arg-type]
        primary_timeframe="1m", run_start_date="2024-01-01T01:40:00", run_end_date=END,
        execution_timeframe="1m", catalog_path=catalog_path,
        liquidation_mode=kw["launch"].liquidation.mode,
    )


def test_a_raw_fill_at_the_run_end_surfaces_as_a_failed_validation_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The R6b refusal reaches the user as a FAILED run whose error message names the run end,
    through the runner's own extraction path (only the NT engine run is replaced)."""
    from vibe_quant.db import StateManager

    end_ns = date_to_ns(END)
    assert end_ns is not None
    mk = _end_fill_marks()
    real_apply_with_marks(monkeypatch, mk)

    db = tmp_path / "r6b.db"
    run_id = _cross_liquidation_run(db, start="2024-01-01", end="2024-01-01")
    runner = _offline_runner(monkeypatch, db, tmp_path)

    def fake_backtest(**kw: Any) -> ValidationResult:  # the NT run, then the real extraction
        return _extract_end_fill_window(
            runner, kw, _end_fill_positions(end_ns), mk, catalog_path=tmp_path
        )

    monkeypatch.setattr(runner, "_run_backtest", fake_backtest)
    try:
        with pytest.raises(ValidationRunnerError, match=rf"ValueError: .*{end_ns}"):
            runner.run(run_id)
    finally:
        runner.close()
    state = StateManager(db)
    try:
        row = state.get_backtest_run(run_id)
    finally:
        state.close()
    assert row is not None and row["status"] == "failed"
    assert str(row["error_message"]).startswith("ValueError: ") and str(end_ns) in str(
        row["error_message"]
    )


def test_a_position_without_a_valuation_horizon_is_funded_only_before_the_run_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 1m run ending at midnight (an 8h settlement boundary = end_ns): the open long's LAST
    stops at 02:39:59.999, so it has no final-bar horizon and is funded to just short of the
    run end. The real FundingCalculator charges 08:00 and 16:00 (fallback 4.0 each) but never
    the settlement AT end_ns, which the shared apply would refuse. The 01:50 crash liquidates
    it, so it needs no valuation and the run completes."""
    import vibe_quant.validation.extraction as ex
    from vibe_quant.validation.funding import FundingCalculator

    crash = crash_marks(to=36000.0, at=110, n=1440)  # 01:50
    full_day = Marks(
        DAY0 + np.arange(1440, dtype=np.int64) * MIN, crash.low, crash.high, crash.close,
        crash.open,
    )
    scans = real_apply_with_marks(monkeypatch, full_day)
    last = exec_series(marks([40000.0] * 60))  # LAST stops at 02:39:59.999
    monkeypatch.setattr(ex, "load_catalog_bars", lambda *a, **kw: last)
    p = long_btc(sell_bar=None)
    res = ValidationResult(starting_balance=START_BAL)
    ex.extract_trades(
        res, engine_for([p], {BTC: full_day}), VENUE,  # type: ignore[arg-type]
        funding_calculator=FundingCalculator(archive_path=tmp_path / "no-archive.db"),
        run_start_date=START, run_end_date="2024-01-02", execution_timeframe="1m",
        catalog_path="x", liquidation_mode="isolated",
    )

    assert scans[0] == {key_of(p): [(DAY0 + 8 * 60 * MIN, -4.0), (DAY0 + 16 * 60 * MIN, -4.0)]}
    (trade,) = res.trades
    assert trade.exit_reason == "liquidation" and trade.funding_fees == 0.0
    assert trade.net_pnl == pytest.approx(-4020.0)


def test_a_walk_forward_window_refused_by_r6b_fails_the_whole_run_without_an_aggregate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Walk-forward, 3 windows: the 2nd one's extraction hits the R6b refusal (a raw fill at
    exactly its end). The run must end FAILED with that message persisted, and no aggregate of
    the other windows may be stored as if the run had completed."""
    from datetime import date

    from vibe_quant.db import StateManager

    end_ns = date_to_ns(END)
    assert end_ns is not None
    mk = _end_fill_marks()
    real_apply_with_marks(monkeypatch, mk)

    db = tmp_path / "wf_r6b.db"
    run_id = _cross_liquidation_run(db, start="2024-01-01", end="2024-01-29")
    runner = _offline_runner(monkeypatch, db, tmp_path)
    windows: list[str] = []

    def fake_backtest(**kw: Any) -> ValidationResult:
        windows.append(str(kw["run_config"]["start_date"]))
        # the 2nd window is the R6b one; the others are a clean BTC round trip (no hit:
        # margin 150 > MM 144.6)
        positions = _end_fill_positions(end_ns) if len(windows) == 2 else [long_btc()]
        return _extract_end_fill_window(runner, kw, positions, mk, catalog_path=tmp_path)

    monkeypatch.setattr(runner, "_run_backtest", fake_backtest)
    assert len(
        runner._build_walk_forward_windows(
            range_start=date(2024, 1, 1), range_end=date(2024, 1, 29),
            train_days=7, test_days=7, step_days=7,
        )
    ) == 3
    try:
        with pytest.raises(ValidationRunnerError, match=rf"ValueError: .*{end_ns}"):
            runner.run_walk_forward(run_id, train_days=7, test_days=7, step_days=7)
    finally:
        runner.close()

    assert len(windows) == 2  # stopped at the refused window
    state = StateManager(db)
    try:
        row = state.get_backtest_run(run_id)
        stored = state.get_backtest_result(run_id)
    finally:
        state.close()
    assert row is not None and row["status"] == "failed"
    assert str(row["error_message"]).startswith("ValueError: ") and str(end_ns) in str(
        row["error_message"]
    )
    assert stored is None  # no aggregate of the surviving windows
