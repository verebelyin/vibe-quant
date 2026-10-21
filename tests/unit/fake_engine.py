"""Shared fake NT engine for the tier-consistency tests.

The real screening (``NTScreeningRunner._extract_metrics``) and validation
(``extract_results``) extractors read positions, fills and execution bars
through NautilusTrader duck types. This module builds those objects without an
engine, so one scenario can be pinned through both tiers on the same inputs.

Nothing here constructs an NT engine: only the real ``OrderSide``/``OrderType``
enums are used, because the extractors compare against them.

``fake_engine`` serves ``engine.kernel.cache``:
``positions()`` returns every fake position (``position_snapshots()`` stays
empty), ``bars(bar_type=None)`` returns the bars of one bar type or every bar,
and ``orders()`` returns ``orders``. ``trader`` exists only when ``strategies``
is not ``None`` (an engine without a trader is a supported case).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from nautilus_trader.model.enums import OrderSide, OrderType

from vibe_quant.fees import FeeSchedule


@dataclass
class FakeFill:
    """One NT fill: the attributes either extractor reads off a position event."""

    ts_event: int
    order_side: Any
    last_qty: float
    last_px: float
    commission: float = 0.0
    order_type: Any = OrderType.MARKET


@dataclass
class FakePosition:
    """NT position stub: every attribute the screening/validation extractors read.

    ``commissions()`` returns ``commission_list`` (the screening path sums the
    absolute values, validation sums them signed then takes the absolute total:
    a rebate leg therefore differs between tiers, as the real ones do).
    """

    id: str
    instrument_id: str
    ts_opened: int
    ts_closed: int
    events: list[FakeFill]
    realized_pnl: float
    is_closed: bool = True
    entry: Any = OrderSide.BUY
    avg_px_open: float = 0.0
    avg_px_close: float = 0.0
    peak_qty: float = 0.0
    signed_qty: float = 0.0
    commission_list: list[float] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return not self.is_closed

    def commissions(self) -> list[float]:
        return self.commission_list

    @classmethod
    def closed(
        cls,
        pid: str,
        instrument_id: str,
        *,
        entry_ns: int,
        exit_ns: int,
        entry_price: float,
        exit_price: float,
        qty: float = 1.0,
        entry_side: Any = OrderSide.BUY,
        entry_fee: float = 0.0,
        exit_fee: float = 0.0,
    ) -> FakePosition:
        """A closed round trip whose realized PnL is NT's: price PnL minus commissions."""
        sign = 1.0 if entry_side == OrderSide.BUY else -1.0
        exit_side = OrderSide.SELL if entry_side == OrderSide.BUY else OrderSide.BUY
        return cls(
            id=pid,
            instrument_id=instrument_id,
            ts_opened=entry_ns,
            ts_closed=exit_ns,
            events=[
                FakeFill(entry_ns, entry_side, qty, entry_price, entry_fee),
                FakeFill(exit_ns, exit_side, qty, exit_price, exit_fee),
            ],
            realized_pnl=sign * qty * (exit_price - entry_price) - (entry_fee + exit_fee),
            entry=entry_side,
            avg_px_open=entry_price,
            avg_px_close=exit_price,
            peak_qty=qty,
            signed_qty=0.0,
            commission_list=[entry_fee, exit_fee],
        )

    @classmethod
    def open(
        cls,
        pid: str,
        instrument_id: str,
        *,
        entry_ns: int,
        entry_price: float,
        qty: float = 1.0,
        entry_side: Any = OrderSide.BUY,
        entry_fee: float = 0.0,
    ) -> FakePosition:
        """A position still open at the run end (NT reports ``-entry_fee`` as realized)."""
        sign = 1.0 if entry_side == OrderSide.BUY else -1.0
        return cls(
            id=pid,
            instrument_id=instrument_id,
            ts_opened=entry_ns,
            ts_closed=0,
            events=[FakeFill(entry_ns, entry_side, qty, entry_price, entry_fee)],
            realized_pnl=-entry_fee,
            is_closed=False,
            entry=entry_side,
            avg_px_open=entry_price,
            avg_px_close=0.0,
            peak_qty=qty,
            signed_qty=sign * qty,
            commission_list=[entry_fee],
        )


def fake_engine(
    positions: list[FakePosition],
    bars_by_type: dict[str, list[Any]] | None = None,
    orders: Any = (),
    strategies: list[Any] | None = None,
) -> Any:
    """NT engine stub serving ``kernel.cache`` (and ``cache``) to both extractors.

    Args:
        positions: every position the run touched (closed and open).
        bars_by_type: bar-type string -> bars (objects with ``bar_type``,
            ``ts_init``, ``open``, ``high``, ``low``, ``close``, ``volume``).
            ``None`` -> no bars: the mark-to-market path is unavailable.
        orders: what ``cache.orders()`` returns (unused by mode-off extraction).
        strategies: when not ``None``, ``engine.trader.strategies()`` returns it;
            ``None`` leaves the engine without a ``trader`` attribute.
    """
    groups = {} if bars_by_type is None else bars_by_type
    everything = [bar for group in groups.values() for bar in group]

    def cache_bars(bar_type: Any = None) -> list[Any]:
        if bar_type is None:
            return list(everything)
        return list(groups.get(str(bar_type), []))

    cache = SimpleNamespace(
        positions=lambda: list(positions),
        position_snapshots=list,
        bars=cache_bars,
        orders=lambda: list(orders),
    )
    engine = SimpleNamespace(kernel=SimpleNamespace(cache=cache), cache=cache)
    if strategies is not None:
        engine.trader = SimpleNamespace(strategies=lambda: list(strategies))
    return engine


def fake_bt_result(
    total_positions: int = 0,
    stats_pnls: dict[str, dict[str, float]] | None = None,
    stats_returns: dict[str, float] | None = None,
    elapsed_time: float = 0.0,
) -> Any:
    """NT ``BacktestResult`` stub with deterministic (deliberately non-zero) stats."""
    if stats_pnls is None:
        stats_pnls = {
            "USDT": {
                "PnL% (total)": 1.5,
                "Sharpe Ratio (252 days)": 0.75,
                "Sortino Ratio (252 days)": 1.1,
                "Max Drawdown": 0.25,
                "Win Rate": 0.5,
                "Avg Winner": 3.0,
                "Avg Loser": -2.0,
            }
        }
    if stats_returns is None:
        stats_returns = {
            "Sharpe Ratio (252 days)": 0.8,
            "Sortino Ratio (252 days)": 1.2,
            "Max Drawdown": 0.3,
            "Win Rate": 0.4,
        }
    return SimpleNamespace(
        total_positions=total_positions,
        total_orders=0,
        total_events=0,
        elapsed_time=elapsed_time,
        stats_pnls=stats_pnls,
        stats_returns=stats_returns,
    )


def fake_venue(
    slippage: float = 0.0,
    balance: float = 1000.0,
    leverage: int = 10,
    fee_schedule: Any = None,
) -> Any:
    """Validation ``VenueConfig`` stub.

    ``slippage`` is the venue's ``prob_slippage``: ``0.0`` leaves the post-fill
    SPEC estimator charged by :class:`_SlippageModel` active, a positive value
    makes the engine model slippage so the extraction charges none.
    """
    schedule = FeeSchedule() if fee_schedule is None else fee_schedule
    return SimpleNamespace(
        starting_balance_usdt=balance,
        default_leverage=leverage,
        fee_schedule=schedule,
        taker_fee=float(schedule.taker_rate),
        fill_config=SimpleNamespace(prob_slippage=slippage, impact_coefficient=0.1),
    )
