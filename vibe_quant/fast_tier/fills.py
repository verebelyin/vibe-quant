"""Event-jumping replay of the screening venue for one 1m strategy on one instrument.

What it reproduces (NautilusTrader 1.231 ``OrderMatchingEngine`` with the screening venue:
``bar_execution``, adaptive high/low ordering, L1 book, ``FillModel(prob_fill_on_limit=0.8,
prob_slippage=0.5, random_seed=SCREENING_FILL_SEED)``, NETTING, reduce-only on; and the
generated strategy with ``execution_delay_probability=1.0`` and no outbox):

* Bar ``t`` reaches the venue before the strategy: its trade ticks are the open (only if
  it differs from the last price), then high and low (high first iff
  ``|H-O| < |L-O|``, each only if it moves the last price past itself), then the close
  (only if it differs). Tick sizes are ``compute_bar_quarter_sizes`` of the bar volume.
* A signal on ready bar ``e`` defers one bar: bar ``e+1``'s ``on_bar`` dispatches it and
  returns (no new signal is evaluated on a dispatch bar). An entry is sized by the
  generated ``_calculate_position_size`` against the realized balance, filled by market at
  the book price (bar ``e+1``'s close); its SL (stop-market) and TP (limit, maker) are
  live from bar ``e+2``. A deferred exit is a no-op when an SL/TP filled during the
  dispatch bar's ticks.
* A stop fills at its trigger, or at the tick price when the OPEN tick triggers it; a
  limit fills at its price, and a tick exactly at the limit fills it only on a successful
  ``0.8 >= random()`` draw. Every fill then draws once more and slips one tick against
  the order side when ``0.5 >= random()``. One position per instrument, no reversal;
  under ``both`` the long entry is checked first.
* At run end an open position is closed by market at the last close (``on_stop``).

Anything outside that model raises :class:`FastTierUnsupported`: a fill larger than the
tick's size (NT's IOC partial fill), an open tick gapping through a take-profit, an
indicator that is not ready on a dispatch bar.
"""

from __future__ import annotations

import bisect
import random
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np

from vibe_quant.fast_tier.errors import FastTierUnsupported

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.model.position import Position

    from vibe_quant.fast_tier.signals import Signals


def bar_quarter_sizes(volume_raw: int, min_size_raw: int) -> tuple[int, int]:
    """``nautilus_trader.model.data.compute_bar_quarter_sizes`` (quarter, close) raws."""
    quarter = volume_raw // 4
    quarter = (quarter // min_size_raw) * min_size_raw
    if quarter < min_size_raw:
        quarter = min_size_raw
    three = quarter * 3
    if three >= volume_raw:
        close = min_size_raw
    else:
        close = ((volume_raw - three) // min_size_raw) * min_size_raw
        if close < min_size_raw:
            close = min_size_raw
    return quarter, close


def bar_ticks(
    o: int, h: int, lo: int, c: int, last: int | None, quarter: int, close_size: int
) -> list[tuple[int, str, int]]:
    """The trade ticks NT's matching engine processes for one bar, in order.

    (price raw, 'O'|'H'|'L'|'C', size raw). ``last`` is the last trade price before the
    bar (None before the first bar). Open only if it differs from ``last``; high first iff
    ``|H-O| < |L-O|``; high/low only if they move past the running last price; close only
    if it differs from it (``_process_trade_ticks_from_bar``).
    """
    out: list[tuple[int, str, int]] = []
    if last is None or o != last:
        out.append((o, "O", quarter))
        last = o
    for kind in ("H", "L") if abs(h - o) < abs(lo - o) else ("L", "H"):
        if kind == "H" and h > last:
            out.append((h, "H", quarter))
            last = h
        elif kind == "L" and lo < last:
            out.append((lo, "L", quarter))
            last = lo
    if c != last:
        out.append((c, "C", close_size))
    return out


class Replay:
    """One replay; :meth:`run` returns the positions in creation order."""

    def __init__(
        self,
        *,
        strategy: Any,
        cache: Any,
        instrument: Instrument,
        bars: list[Bar],
        signals: Signals,
        starting_balance: float,
        fill_seed: int,
        prob_slippage: float,
        prob_fill_on_limit: float,
        end_ts: int,
    ) -> None:
        from nautilus_trader.model.objects import Money

        self.strategy = strategy
        self.cache = cache
        self.inst = instrument
        self.bars = bars
        self.sig = signals
        self.n = len(bars)
        self.rng = random.Random(fill_seed)
        self.p_slip = prob_slippage
        self.p_limit = prob_fill_on_limit
        self.end_ts = end_ts
        self.draws = 0
        self.ccy = instrument.quote_currency
        self.start_money = Money(starting_balance, self.ccy)
        self._closed_balance = self.start_money
        self.inc_raw = instrument.price_increment.raw
        self.min_size_raw = instrument.size_increment.raw
        self.prec = instrument.price_precision
        self.o = [b.open.raw for b in bars]
        self.h = [b.high.raw for b in bars]
        self.lo = [b.low.raw for b in bars]
        self.c = [b.close.raw for b in bars]
        self.v = [b.volume.raw for b in bars]
        self.high_f = np.fromiter((float(b.high) for b in bars), dtype=np.float64, count=self.n)
        self.low_f = np.fromiter((float(b.low) for b in bars), dtype=np.float64, count=self.n)
        self.positions: list[Position] = []
        self._fill_no = 0
        self._ids = _Ids(strategy, instrument)

    # -- venue pieces ---------------------------------------------------------------

    def _draw(self) -> float:
        self.draws += 1
        return self.rng.random()

    def _event(self, probability: float) -> bool:
        """``FillModel._event_success``."""
        if probability == 0:
            return False
        if probability == 1:
            return True
        return probability >= self._draw()

    def _ticks(self, t: int) -> list[tuple[int, str, int]]:
        """Processed trade ticks of bar ``t`` (see :func:`bar_ticks`)."""
        quarter, close_size = bar_quarter_sizes(self.v[t], self.min_size_raw)
        last = self.c[t - 1] if t > 0 else None
        return bar_ticks(self.o[t], self.h[t], self.lo[t], self.c[t], last, quarter, close_size)

    def _book_size(self, t: int) -> int:
        """Size of the book level after bar ``t``: its last processed tick's (or an earlier bar's)."""
        for k in range(t, -1, -1):
            ticks = self._ticks(k)
            if ticks:
                return ticks[-1][2]
        return 0

    def _balance(self) -> Any:
        """Realized balance: start + every position's realized PnL (Money is fixed-point)."""
        if self.positions and self.positions[-1].is_open:
            return self._closed_balance + self.positions[-1].realized_pnl
        return self._closed_balance

    def _fill(
        self, t: int, *, buy: bool, qty: Any, base_raw: int, order_type: str, maker: bool, size_raw: int
    ) -> None:
        """Apply one fill (slip draw, commission) to the open position or a new one."""
        from nautilus_trader.model.objects import Money, Price
        from nautilus_trader.model.position import Position

        if qty.raw > size_raw:
            raise FastTierUnsupported(
                f"fill of {qty} exceeds the tick size at bar {t} (NT would fill partially)"
            )
        px_raw = base_raw
        if self._event(self.p_slip):
            px_raw += self.inc_raw if buy else -self.inc_raw
        px = Price.from_raw(px_raw, self.prec)
        notional = self.inst.notional_value(qty, px)
        rate = self.inst.maker_fee if maker else self.inst.taker_fee
        commission = Money(notional.as_decimal() * rate, self.ccy)
        ts = int(self.bars[t].ts_init) if t < self.n else self.end_ts
        fill = self._ids.fill(
            buy=buy, order_type=order_type, qty=qty, px=px, commission=commission,
            maker=maker, ts=ts,
        )
        self._fill_no += 1
        if self.positions and self.positions[-1].is_open:
            pos = self.positions[-1]
            pos.apply(fill)
            if pos.is_closed:
                self._closed_balance = self._closed_balance + pos.realized_pnl
        else:
            self.positions.append(Position(self.inst, fill))

    # -- strategy pieces ------------------------------------------------------------

    def _entry_qty(self, d: int, is_long: bool) -> Any:
        """The generated sizing at dispatch bar ``d``, then the margin limit (None = skipped)."""
        s = self.strategy
        self.cache.balance = self._balance()
        s._last_close = float(self.bars[d].close)
        qty = s._calculate_position_size(self.bars[d], is_long)
        if qty is None:
            return None
        limit = getattr(s.config, "max_margin_pct", None)
        if limit is not None:
            from vibe_quant.margin_limit import MarginItem, check_entry_margin

            notional = (
                abs(qty.as_decimal())
                * Decimal(str(s._last_close))
                * self.inst.multiplier.as_decimal()
            )
            decision = check_entry_margin(
                open_items=[],
                pending_items=[],
                new_item=MarginItem(notional, s.config.margin_leverage, str(self.inst.id)),
                equity=self.cache.balance.as_decimal(),
                max_margin_pct=limit,
            )
            if not decision.allowed:
                return None
        return qty

    def _protection(self, is_long: bool) -> tuple[int | None, int | None]:
        """SL trigger and TP limit raws from the entry fill (generated price helpers)."""
        s = self.strategy
        entry = float(self.positions[-1].avg_px_open)
        sl = s._calculate_sl_price(entry, is_long)
        tp = s._calculate_tp_price(entry, is_long)
        sl_raw = self.inst.make_price(sl).raw if sl is not None and sl > 0 else None
        tp_raw = self.inst.make_price(tp).raw if tp is not None and tp > 0 else None
        return sl_raw, tp_raw

    def _protective_fill(self, t: int, is_long: bool, sl: int | None, tp: int | None) -> bool:
        """Run bar ``t``'s ticks against the SL/TP; True when one filled (position closed)."""
        qty = self.positions[-1].quantity
        for price, kind, size in self._ticks(t):
            if sl is not None and (price <= sl if is_long else price >= sl):
                base = price if kind == "O" else sl
                self._fill(t, buy=not is_long, qty=qty, base_raw=base, order_type="STOP_MARKET",
                           maker=False, size_raw=size)
                return True
            if tp is not None and (price >= tp if is_long else price <= tp):
                if price != tp:
                    if kind == "O":
                        raise FastTierUnsupported(f"open tick gaps through the take-profit at bar {t}")
                elif not self._event(self.p_limit):
                    continue
                self._fill(t, buy=not is_long, qty=qty, base_raw=tp, order_type="LIMIT",
                           maker=True, size_raw=size)
                return True
        return False

    def _first_protective_fill(
        self, start: int, stop: int, is_long: bool, sl: int | None, tp: int | None
    ) -> int | None:
        """First bar in [start, stop] whose ticks fill the SL or TP (draws in bar order)."""
        if start > stop:
            return None
        lo = self.low_f[start : stop + 1]
        hi = self.high_f[start : stop + 1]
        mask = np.zeros(stop + 1 - start, dtype=bool)
        sl_f = None if sl is None else sl / 10 ** _RAW_SCALE
        tp_f = None if tp is None else tp / 10 ** _RAW_SCALE
        if is_long:
            if sl_f is not None:
                mask |= lo <= sl_f * (1 + 1e-9)
            if tp_f is not None:
                mask |= hi >= tp_f * (1 - 1e-9)
        else:
            if sl_f is not None:
                mask |= hi >= sl_f * (1 - 1e-9)
            if tp_f is not None:
                mask |= lo <= tp_f * (1 + 1e-9)
        for off in np.flatnonzero(mask):
            t = start + int(off)
            if self._protective_fill(t, is_long, sl, tp):
                return t
        return None

    # -- the run ----------------------------------------------------------------------

    def run(self) -> list[Position]:
        sig = self.sig
        n = self.n
        entry_idx = np.flatnonzero(sig.long_entry | sig.short_entry).tolist()
        long_exit_idx = np.flatnonzero(sig.long_exit).tolist()
        short_exit_idx = np.flatnonzero(sig.short_exit).tolist()
        i = 0
        while True:
            e = _next(entry_idx, i)
            if e is None or e + 1 >= n:
                break
            is_long = bool(sig.long_entry[e])
            d = e + 1
            if not sig.ready[d]:
                raise FastTierUnsupported(f"indicators not ready on dispatch bar {d}")
            qty = self._entry_qty(d, is_long)
            if qty is None:
                i = d + 1
                continue
            self._fill(d, buy=is_long, qty=qty, base_raw=self.c[d], order_type="MARKET",
                       maker=False, size_raw=self._book_size(d))
            sl, tp = self._protection(is_long)
            a = d + 1
            k = _next(long_exit_idx if is_long else short_exit_idx, a)
            j = self._first_protective_fill(a, n - 1 if k is None else k, is_long, sl, tp)
            if j is not None:
                i = j  # flat at bar j's on_bar: entries are evaluated there
                continue
            if k is None or k + 1 >= n:
                break  # still open at the end
            x = k + 1
            if not self._protective_fill(x, is_long, sl, tp):
                qty_open = self.positions[-1].quantity
                self._fill(x, buy=not is_long, qty=qty_open, base_raw=self.c[x],
                           order_type="MARKET", maker=False, size_raw=self._book_size(x))
            i = x + 1
        if self.positions and self.positions[-1].is_open:
            pos = self.positions[-1]
            self._fill(n, buy=not pos.is_long, qty=pos.quantity, base_raw=self.c[n - 1],
                       order_type="MARKET", maker=False, size_raw=self._book_size(n - 1))
        return self.positions


# Price.raw scale of this NT build (FIXED_PRECISION): 16 high-precision, 9 standard.
def _raw_scale() -> int:
    from nautilus_trader.model.objects import Price

    raw = Price.from_str("1").raw
    return len(str(raw)) - 1


_RAW_SCALE = _raw_scale()


def _next(indices: list[int], start: int) -> int | None:
    pos = bisect.bisect_left(indices, start)
    return indices[pos] if pos < len(indices) else None


class _Ids:
    """Identifiers and OrderFilled construction for the replayed fills."""

    def __init__(self, strategy: Any, instrument: Instrument) -> None:
        from nautilus_trader.model.identifiers import (
            AccountId,
            PositionId,
            StrategyId,
            TraderId,
        )

        self.trader = TraderId("BACKTESTER-001")
        self.strategy = StrategyId(f"{type(strategy).__mro__[1].__name__}-000")
        self.account = AccountId("BINANCE-001")
        self.instrument = instrument
        self.position = PositionId(f"{instrument.id}-{self.strategy}")
        self.count = 0

    def fill(
        self, *, buy: bool, order_type: str, qty: Any, px: Any, commission: Any, maker: bool, ts: int
    ) -> Any:
        from nautilus_trader.core.uuid import UUID4
        from nautilus_trader.model.enums import LiquiditySide, OrderSide, OrderType
        from nautilus_trader.model.events import OrderFilled
        from nautilus_trader.model.identifiers import ClientOrderId, TradeId, VenueOrderId

        self.count += 1
        return OrderFilled(
            trader_id=self.trader,
            strategy_id=self.strategy,
            instrument_id=self.instrument.id,
            client_order_id=ClientOrderId(f"O-FAST-{self.count}"),
            venue_order_id=VenueOrderId(f"V-FAST-{self.count}"),
            account_id=self.account,
            trade_id=TradeId(f"T-FAST-{self.count}"),
            position_id=self.position,
            order_side=OrderSide.BUY if buy else OrderSide.SELL,
            order_type=getattr(OrderType, order_type),
            last_qty=qty,
            last_px=px,
            currency=self.instrument.quote_currency,
            commission=commission,
            liquidity_side=LiquiditySide.MAKER if maker else LiquiditySide.TAKER,
            event_id=UUID4(),
            ts_event=ts,
            ts_init=ts,
        )
