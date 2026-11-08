"""TradingView-style broker emulator for a compiled vibe-quant strategy.

The emulator NEVER computes a signal. The entry/exit booleans, the sizing, the stop/target
price helpers and the signal-level reads all come from the compiled strategy's own generated
code (:func:`vibe_quant.fast_tier.signals.signals_for_runner`: ``compute_signals`` over the
strategy timeframe merged with every additional one, recording the readings a level makes).
Only BROKER rules live here: positions, reversal, signal-level snapshots and refresh, stops
and targets on detail bars, commissions, slippage, funding and the clock that releases an order.

Two fill modes, one machine:

``engine``  the screening engine's own release rules, so the emulator's trade list equals the
            engine's (R60). By the loaded timeframes (``screening_fill_mode``):

            * a 1m strategy defers every action one bar (``execution_delay_probability``=1) and
              fills at that bar's close, protection live at once;
            * otherwise commands wait in the strategy's outbox and leave on the strategy's own
              next datum -- the next fill tick (the 1m closes at the open and the open+1m of
              each boundary of the finest loaded timeframe) or 1m detail bar -- strictly after
              the queue instant; protection queued by a fill leaves on the NEXT datum;
            * the venue replays the finest loaded timeframe's bars as trade ticks (open only if
              it moved, high/low in the engine's order, close) and fills with the seeded
              screening fill model (one slip draw per fill, one limit draw per exact touch).

``tv``      TradingView's broker: a signal at a bar close fills at the next bar's OPEN, protection
            is live straight after the fill (the same bar's later ticks can hit it), no slippage,
            limits fill on touch. Signals come from the same code with TradingView's merge order
            (a higher timeframe turns over on the day's final intraday bar for every primary).

Both modes charge the engine's own fee schedule and funding: positions are real NautilusTrader
``Position`` objects fed to the same ``CostLedger`` the engine installs, funding window (entry,
exit], and sizing reads the costed balance.

Stop and target on the same bar: the bar is replayed as the engine's trade ticks
(:func:`vibe_quant.fast_tier.fills.bar_ticks`); at each tick the stop is tested before the
target, so the first leg the tick sequence reaches wins and a tick through both takes the stop.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from vibe_quant.fast_tier.fills import _Ids, bar_quarter_sizes, bar_ticks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nautilus_trader.model.objects import Money, Quantity
    from nautilus_trader.model.position import Position

    from vibe_quant.fast_tier.signals import RunSignals
    from vibe_quant.screening.nt_runner import NTScreeningRunner

Mode = Literal["tv", "engine"]
Release = Literal["defer", "tick", "bar", "next_open"]

#: tick-kind letters of :func:`bar_ticks`; ``F`` = a fill tick (a gap fills at the tick)
_GAPS = ("O", "F")
_BIG = 10**15


class EmulatorUnsupported(ValueError):
    """A rule the emulator does not model (it never replays a different rule silently)."""


@dataclass(frozen=True)
class Fill:
    """One executed order: ``action`` is entry | exit | stop | target."""

    ts: int
    side: str
    action: str
    qty: str
    price: str
    commission: str


@dataclass(frozen=True)
class Level:
    """One protective order that reached the venue (kind ``STOP_MARKET`` | ``LIMIT``)."""

    ts: int
    kind: str
    price: str
    qty: str


@dataclass
class EmulatorResult:
    """Everything a comparison needs; ``positions`` are NT ``Position`` objects in open order."""

    fills: list[Fill] = field(default_factory=list)
    levels: list[Level] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)
    ledger: Any = None
    starting_balance: float = 0.0
    margin_refusals: int = 0
    reverse_dropped: int = 0
    refresh_invalid: int = 0
    sizing_skips: dict[str, int] = field(default_factory=dict)

    def costed_equity(self) -> list[float]:
        """Costed equity after each CLOSED position: start + realized PnL - modeled costs paid."""
        return costed_equity_series(self.positions, self.ledger, self.starting_balance)


def costed_equity_series(positions: Sequence[Any], ledger: Any, starting_balance: float) -> list[float]:
    """Costed equity after each closed position, in open order (one rule for the engine and the emulator)."""
    out: list[float] = []
    realized = 0.0
    costs = 0.0
    for pos in sorted(positions, key=lambda p: int(p.ts_opened)):
        if not pos.is_closed:
            continue
        realized += float(pos.realized_pnl)
        costs += ledger.finalized.get(_cost_key(pos), 0.0)  # the on_stop close is never finalized
        out.append(starting_balance + realized - costs)
    return out


def _cost_key(pos: Any) -> str:
    from vibe_quant.modeled_costs import cost_key

    return str(cost_key(pos))


@dataclass
class _Cmd:
    """A venue command: queued in the outbox or executed at once."""

    ts: int
    kind: str  # entry | exit | sl | tp | cancel_all | cancel
    side: str | None = None
    qty: Any = None
    price_raw: int | None = None
    oid: int = 0
    ids: tuple[int, ...] = ()


@dataclass
class _Rest:
    """A resting protective order (reduce-only stop-market ``sl`` or limit ``tp``)."""

    oid: int
    kind: str
    side: str
    price_raw: int
    qty: Any


@dataclass
class _Intent:
    """A reversal waiting for its exit fill (the generated ``ReversalIntent``)."""

    side: str
    qty: Any
    exit_id: int
    created_ts: int
    levels: Any


class Emulator:
    """One replay of a :class:`RunSignals` through a broker; :meth:`run` returns the result."""

    def __init__(
        self,
        run: RunSignals,
        *,
        mode: Mode,
        release: Release,
        exec_tf: str,
        cost_model: Any,
        starting_balance: float,
        fill_seed: int = 42,
        prob_slippage: float = 0.5,
        prob_fill_on_limit: float = 0.8,
        fill_ticks: Sequence[tuple[int, int]] = (),
        dsl: dict[str, Any] | None = None,
    ) -> None:
        from vibe_quant.modeled_costs import CostLedger

        self.run_ = run
        self.mode = mode
        self.release = release
        self.exec_tf = exec_tf
        self.strat = run.strategy
        self.cache = run.cache
        self.inst = run.instrument
        self.sig = run.signals
        self.primary_tf = run.timeframe
        self.pbars = run.bars[run.timeframe]
        self.fill_ticks = list(fill_ticks)
        self.rng = random.Random(fill_seed)
        self.p_slip = prob_slippage if mode == "engine" else 0.0
        self.p_limit = prob_fill_on_limit if mode == "engine" else 1.0
        self.ccy = self.inst.quote_currency
        from nautilus_trader.model.objects import Money

        self.start_balance = float(starting_balance)
        self._closed_balance: Money = Money(starting_balance, self.ccy)
        self.inc_raw = self.inst.price_increment.raw
        self.min_size_raw = self.inst.size_increment.raw
        self.prec = self.inst.price_precision
        self.fill_tick_size = self.inst.make_qty(1_000_000).raw
        self.ids = _Ids(self.strat, self.inst)
        self.res = EmulatorResult(starting_balance=self.start_balance)
        self.positions = self.res.positions
        self.cache.positions = self.positions
        self.ledger = CostLedger(cost_model, self.cache)
        self.strat._cost_ledger = self.ledger
        self.res.ledger = self.ledger
        if self.ledger.model.refine:
            raise EmulatorUnsupported(
                "funding exit refinement (execution bar longer than the funding interval) is not modeled"
            )
        cfg = self.strat.config
        for prefix in ("stop_loss", "stop_loss_long", "stop_loss_short"):
            if getattr(cfg, f"{prefix}_type", None) == "atr_trailing":
                raise EmulatorUnsupported("atr_trailing stops are not modeled")
        if getattr(self.strat, "_post_only", False):
            raise EmulatorUnsupported("post_only entries are not modeled")
        if hasattr(self.strat, "_bars_in_trade"):
            raise EmulatorUnsupported("max_bars_in_trade time exits are not modeled")
        self.reverse = hasattr(self.strat, "_begin_reverse")
        self.sl_levels = bool(getattr(self.strat, "_signal_levels_on", False))
        self.defer = release == "defer"
        self.dsl = dsl or {}
        tfc = self.dsl.get("time_filters") or {}
        self.session_gate = bool(tfc.get("allowed_sessions") or tfc.get("blocked_days"))
        self.funding_gate = bool((tfc.get("avoid_around_funding") or {}).get("enabled"))
        self.refresh_sides = self._refresh_sides()
        # venue / strategy state
        self.last: int | None = None
        self.last_size = 0
        self.now = 0
        self.resting: list[_Rest] = []
        self.outbox: list[_Cmd] = []
        self.pos_open = False
        self.pos_side: str | None = None
        self.pending: str | None = None
        self.intent: _Intent | None = None
        self._next_id = 0
        self.rows = self.sig.events.primary_rows() if self.sig.events is not None else None
        self._cur_row = -1
        self.strat._get_indicator_value = self._ind_value

    # ----------------------------------------------------------------------------------
    # recorded indicator readings
    # ----------------------------------------------------------------------------------

    def _ind_value(self, name: str) -> float:
        events = self.sig.events
        if events is None or name not in events.values:
            raise EmulatorUnsupported(f"indicator reading {name!r} was not recorded")
        return float(events.values[name][self._cur_row])

    def _refresh_sides(self) -> dict[str, bool]:
        """Sides whose same-side entry signal refreshes signal-level protection (compiler rule)."""
        if not self.dsl.get("refresh_on_signal") or not self.sl_levels:
            return {}
        out: dict[str, bool] = {}
        for is_long, side in ((True, "BUY"), (False, "SELL")):
            sl_t = self.strat._sl_config(is_long)[0]
            tp_t = self.strat._tp_config(is_long)[0]
            if "signal_level" in (sl_t, tp_t):
                out[side] = is_long
        return out

    def _oid(self) -> int:
        self._next_id += 1
        return self._next_id

    # ----------------------------------------------------------------------------------
    # costed balance and sizing (the generated code, at the instant the engine calls it)
    # ----------------------------------------------------------------------------------

    def _balance(self) -> Money:
        if self.positions and self.positions[-1].is_open:
            return self._closed_balance + self.positions[-1].realized_pnl
        return self._closed_balance

    def _size(self, k: int, is_long: bool) -> Quantity | None:
        s = self.strat
        bar = self.pbars[k]
        self.cache.balance = self._balance()
        s._last_close = float(bar.close)
        if self.sig.sizing_atr is not None:
            value = float(self.sig.sizing_atr[k])
            s._fast_sizing_atr = None if value != value else value
        return s._calculate_position_size(bar, is_long)  # type: ignore[no-any-return]

    def _margin_ok(self, qty: Quantity, price: float) -> bool:
        s = self.strat
        limit = getattr(s.config, "max_margin_pct", None)
        if limit is None:
            return True
        from vibe_quant.margin_limit import MarginItem, check_entry_margin

        mult = self.inst.multiplier.as_decimal()
        self.cache.balance = self._balance()
        open_items = [
            MarginItem(
                abs(p.quantity.as_decimal()) * Decimal(str(float(p.avg_px_open))) * mult,
                s.config.margin_leverage,
                str(self.inst.id),
            )
            for p in self.cache.positions_open(venue=self.inst.id.venue)
        ]
        paid = self.ledger.paid_through(self.now, self.cache.positions_open(venue=self.inst.id.venue))
        decision = check_entry_margin(
            open_items=open_items,
            pending_items=[],
            new_item=MarginItem(
                abs(qty.as_decimal()) * Decimal(str(price)) * mult, s.config.margin_leverage, str(self.inst.id)
            ),
            equity=self.cache.balance.as_decimal() - Decimal(str(paid)),
            max_margin_pct=limit,
        )
        if not decision.allowed:
            self.res.margin_refusals += 1
        return bool(decision.allowed)

    # ----------------------------------------------------------------------------------
    # fills
    # ----------------------------------------------------------------------------------

    def _draw(self) -> float:
        return self.rng.random()

    def _event(self, probability: float) -> bool:
        """``FillModel._event_success``."""
        if probability == 0:
            return False
        if probability == 1:
            return True
        return probability >= self._draw()

    def _fill(
        self, *, buy: bool, qty: Any, base_raw: int, order_type: str, maker: bool, action: str, oid: int,
        finalize: bool = True,
    ) -> None:
        """Apply one fill: slip draw, commission, position update, then the strategy's event handlers."""
        from nautilus_trader.model.objects import Money, Price
        from nautilus_trader.model.position import Position

        if self.mode == "engine" and qty.raw > self.last_size:
            raise EmulatorUnsupported(f"fill of {qty} exceeds the tick size (NT would fill partially)")
        px_raw = base_raw
        if self._event(self.p_slip):
            px_raw += self.inc_raw if buy else -self.inc_raw
        px = Price.from_raw(px_raw, self.prec)
        notional = self.inst.notional_value(qty, px)
        rate = self.inst.maker_fee if maker else self.inst.taker_fee
        commission = Money(notional.as_decimal() * rate, self.ccy)
        event = self.ids.fill(
            buy=buy, order_type=order_type, qty=qty, px=px, commission=commission, maker=maker, ts=self.now
        )
        self.res.fills.append(
            Fill(self.now, "BUY" if buy else "SELL", action, str(qty), str(px), str(commission))
        )
        if self.positions and self.positions[-1].is_open:
            pos = self.positions[-1]
            pos.apply(event)
            if pos.is_closed:
                self._closed_balance = self._closed_balance + pos.realized_pnl
                if finalize:
                    self.ledger.on_position_closed(pos)
                    self._on_closed(oid)
        else:
            self.positions.append(Position(self.inst, event))
            self._on_opened()

    def _on_opened(self) -> None:
        """PositionOpened: hand over the level snapshot, then arm the protection."""
        pos = self.positions[-1]
        self.pos_open = True
        self.pos_side = "BUY" if pos.is_long else "SELL"
        s = self.strat
        if self.sl_levels:
            s._levels, s._pending_levels = s._pending_levels, None
        entry = float(pos.avg_px_open)
        is_long = bool(pos.is_long)
        qty = pos.quantity
        sl = s._calculate_sl_price(entry, is_long)
        if sl is not None and sl <= 0:
            sl = None
        tp = s._calculate_tp_price(entry, is_long)
        if tp is not None and tp <= 0:
            tp = None
        close_side = "SELL" if is_long else "BUY"
        if sl is not None:
            self._send(_Cmd(self.now, "sl", close_side, qty, self.inst.make_price(sl).raw, self._oid()), protection=True)
        if tp is not None:
            self._send(_Cmd(self.now, "tp", close_side, qty, self.inst.make_price(tp).raw, self._oid()), protection=True)

    def _on_closed(self, closing_oid: int) -> None:
        """PositionClosed: the venue drops the position's reduce-only orders; the strategy reacts."""
        self.pos_open = False
        self.pos_side = None
        self.resting.clear()
        s = self.strat
        if self.sl_levels:
            s._levels = None
        if self.reverse and self._reverse_on_closed(closing_oid):
            return
        self._send(_Cmd(self.now, "cancel_all"), protection=True)

    # ----------------------------------------------------------------------------------
    # commands: outbox / immediate execution
    # ----------------------------------------------------------------------------------

    def _send(self, cmd: _Cmd, *, protection: bool = False) -> None:
        """``_send_command``: queue (ts = now) or execute now.

        Engine ``defer`` sends everything at once; ``tv`` queues entries/exits for the next open
        and arms protection and cancels at once; ``tick``/``bar`` queue everything.
        """
        immediate = self.defer or (self.mode == "tv" and (protection or cmd.kind in ("cancel", "cancel_all")))
        if immediate:
            if self._cmd_ok(cmd, None):
                self._exec(cmd)
            return
        self.outbox.append(cmd)

    def _cmd_ok(self, cmd: _Cmd, price: float | None) -> bool:
        """The margin gate of a market ENTRY at its venue-call instant (``_entry_command_ok``)."""
        if cmd.kind != "entry":
            return True
        px = self.strat._last_close if price is None else price
        return self._margin_ok(cmd.qty, px)

    def _flush(self, ts: int, price: float) -> None:
        """``_flush_outbox``: send the commands queued strictly before ``ts``, in order."""
        if not self.outbox:
            return
        ready = [c for c in self.outbox if c.ts < ts]
        if not ready:
            return
        self.outbox = [c for c in self.outbox if c.ts >= ts]
        for cmd in ready:
            if self._cmd_ok(cmd, price):
                self._exec(cmd)

    def _exec(self, cmd: _Cmd) -> None:
        kind = cmd.kind
        if kind == "cancel_all":
            self.resting.clear()
        elif kind == "cancel":
            self.resting = [r for r in self.resting if r.oid not in cmd.ids]
        elif kind in ("sl", "tp"):
            if not self.pos_open:
                return  # a reduce-only order with nothing to reduce is denied
            self.resting.append(_Rest(cmd.oid, kind, str(cmd.side), int(cmd.price_raw or 0), cmd.qty))
            from nautilus_trader.model.objects import Price

            self.res.levels.append(
                Level(
                    self.now,
                    "STOP_MARKET" if kind == "sl" else "LIMIT",
                    str(Price.from_raw(int(cmd.price_raw or 0), self.prec)),
                    str(cmd.qty),
                )
            )
            if kind == "sl" and self.last is not None and (
                self.last <= int(cmd.price_raw or 0) if cmd.side == "SELL" else self.last >= int(cmd.price_raw or 0)
            ):
                # a stop already triggered by the last price fills at once, at that price
                self._fill(
                    buy=cmd.side == "BUY", qty=self.positions[-1].quantity, base_raw=self.last,
                    order_type="STOP_MARKET", maker=False, action="stop", oid=cmd.oid,
                )
            if kind == "tp" and self.pos_open and self.last is not None and (
                self.last >= int(cmd.price_raw or 0) if cmd.side == "SELL" else self.last <= int(cmd.price_raw or 0)
            ):
                # a limit already marketable at submission fills at once as a taker at the book
                self._fill(
                    buy=cmd.side == "BUY", qty=self.positions[-1].quantity, base_raw=self.last,
                    order_type="LIMIT", maker=False, action="target", oid=cmd.oid,
                )
        elif kind == "exit":
            if not self.pos_open:
                return  # reduce-only market with no position: denied, no fill, no draw
            pos = self.positions[-1]
            self._fill(
                buy=not pos.is_long, qty=pos.quantity, base_raw=self._book(), order_type="MARKET",
                maker=False, action="exit", oid=cmd.oid,
            )
        elif kind == "entry":
            self._fill(
                buy=cmd.side == "BUY", qty=cmd.qty, base_raw=self._book(), order_type="MARKET",
                maker=False, action="entry", oid=cmd.oid,
            )

    def _book(self) -> int:
        if self.last is None:
            raise EmulatorUnsupported("a market order before any price")
        return self.last

    # ----------------------------------------------------------------------------------
    # venue: ticks against the resting protection
    # ----------------------------------------------------------------------------------

    def _venue_ticks(self, ticks: list[tuple[int, str, int]]) -> None:
        for price, kind, size in ticks:
            self.last = price
            self.last_size = size
            if self.resting and self.pos_open:
                self._match(price, kind)

    def _match(self, price: int, kind: str) -> None:
        pos = self.positions[-1]
        is_long = bool(pos.is_long)
        sl = next((r for r in self.resting if r.kind == "sl"), None)
        tp = next((r for r in self.resting if r.kind == "tp"), None)
        if sl is not None and (price <= sl.price_raw if is_long else price >= sl.price_raw):
            base = price if kind in _GAPS else sl.price_raw
            self._fill(
                buy=not is_long, qty=pos.quantity, base_raw=base, order_type="STOP_MARKET",
                maker=False, action="stop", oid=sl.oid,
            )
            return
        if tp is not None and (price >= tp.price_raw if is_long else price <= tp.price_raw):
            base = tp.price_raw
            if price != tp.price_raw:
                if kind == "O":
                    if self.mode != "tv":
                        raise EmulatorUnsupported("an open tick gaps through the take-profit")
                    base = price  # TradingView fills a gapped limit at the bar's open
            elif not self._event(self.p_limit):
                return
            self._fill(
                buy=not is_long, qty=pos.quantity, base_raw=base, order_type="LIMIT",
                maker=True, action="target", oid=tp.oid,
            )

    def _bar_ticks(self, bar: Any) -> list[tuple[int, str, int]]:
        quarter, close_size = bar_quarter_sizes(bar.volume.raw, self.min_size_raw)
        return bar_ticks(bar.open.raw, bar.high.raw, bar.low.raw, bar.close.raw, self.last, quarter, close_size)

    # ----------------------------------------------------------------------------------
    # strategy: the generated on_bar, driven by the recorded signal booleans
    # ----------------------------------------------------------------------------------

    def _entries_allowed(self, bar: Any) -> bool:
        """The compiler's entry gate: session / blocked-day filter and funding avoidance, at bar close."""
        if not (self.session_gate or self.funding_gate):
            return True
        s = self.strat
        close_ns = s._bar_close_ns(bar)
        ok = bool(s._check_time_filters(close_ns)) if self.session_gate else True
        return ok and not (self.funding_gate and s._is_near_funding_time(close_ns))

    def _delay(self, action: str) -> bool:
        """``_maybe_delay_validation_action`` at probability 1.0 (screening's 1m mode)."""
        if not self.defer:
            return False
        if self.pending is not None:
            return False
        self.pending = action
        return True

    def _primary_bar(self, k: int) -> None:
        bar = self.pbars[k]
        sig = self.sig
        self.strat._last_close = float(bar.close)
        if not sig.ready[k]:
            return
        if self.reverse:
            self._expire_reversal(bar)
        if self.defer and self.pending is not None:
            self._dispatch(k)
            return
        allowed = self._entries_allowed(bar)
        was_open = self.pos_open
        long_sig, short_sig = bool(sig.long_entry[k]), bool(sig.short_entry[k])
        if not self.pos_open and allowed:
            if long_sig:
                if not self._delay("long_entry"):
                    self._submit_entry(k, True)
            elif short_sig and not self._delay("short_entry"):
                self._submit_entry(k, False)
        elif self.reverse and self.pos_open and allowed:
            if self.pos_side == "BUY" and short_sig:
                if not self._delay("short_entry"):
                    self._submit_entry(k, False)
            elif self.pos_side == "SELL" and long_sig and not self._delay("long_entry"):
                self._submit_entry(k, True)
        if (
            self.refresh_sides
            and was_open
            and self.pos_open
            and allowed
            and (not self.reverse or self.intent is None)
            and self.pending is None
        ):
            for held, is_long in self.refresh_sides.items():
                if self.pos_side != held:
                    continue
                if (long_sig if is_long else short_sig) and not self._delay("long_refresh" if is_long else "short_refresh"):
                    self._refresh_levels(k, is_long)
                break
        if self.pos_open and (not self.reverse or (self.intent is None and self.pending is None)):
            exit_sig = bool(sig.long_exit[k]) if self.pos_side == "BUY" else bool(sig.short_exit[k])
            if exit_sig and not self._delay("exit"):
                self._submit_exit()

    def _dispatch(self, k: int) -> None:
        action, self.pending = self.pending, None
        if action == "long_entry":
            self._submit_entry(k, True)
        elif action == "short_entry":
            self._submit_entry(k, False)
        elif action == "exit":
            self._submit_exit()
        elif action == "long_refresh":
            self._refresh_levels(k, True)
        elif action == "short_refresh":
            self._refresh_levels(k, False)

    def _has_pending_entry(self) -> bool:
        return any(c.kind == "entry" for c in self.outbox)

    def _submit_entry(self, k: int, is_long: bool) -> None:
        side = "BUY" if is_long else "SELL"
        if self.pos_open:
            if self.reverse:
                self._begin_reverse(k, side)
            return
        if self._has_pending_entry():
            return
        qty = self._size(k, is_long)
        if qty is None:
            return
        self._send(_Cmd(self.now, "entry", side, qty, None, self._oid()))

    def _submit_exit(self) -> None:
        if not self.pos_open:
            return
        self._send(_Cmd(self.now, "cancel_all"))
        pos = self.positions[-1]
        self._send(_Cmd(self.now, "exit", "SELL" if pos.is_long else "BUY", pos.quantity, None, self._oid()))

    def _drop_queued_protection(self) -> None:
        self.outbox = [c for c in self.outbox if c.kind not in ("sl", "tp")]

    def _live_protection_ids(self) -> tuple[int, ...]:
        queued = {i for c in self.outbox if c.kind == "cancel" for i in c.ids}
        return tuple(r.oid for r in self.resting if r.oid not in queued)

    def _begin_reverse(self, k: int, new_side: str) -> None:
        if self.intent is not None or self.pos_side == new_side:
            return
        qty = self._size(k, new_side == "BUY")
        if qty is None:
            return
        if not self.pos_open:
            if self.sl_levels:
                self.strat._pending_levels = None
            return
        self._drop_queued_protection()
        for rid in self._live_protection_ids():
            self._send(_Cmd(self.now, "cancel", ids=(rid,)))
        exit_id = self._oid()
        levels = self.strat._pending_levels if self.sl_levels else None
        self.intent = _Intent(new_side, qty, exit_id, int(self.pbars[k].ts_init), levels)
        if self.sl_levels:
            self.strat._pending_levels = None
        pos = self.positions[-1]
        self._send(_Cmd(self.now, "exit", "SELL" if pos.is_long else "BUY", pos.quantity, None, exit_id))

    def _reverse_on_closed(self, closing_oid: int) -> bool:
        intent, self.intent = self.intent, None
        if intent is None:
            return False
        if closing_oid != intent.exit_id:
            self.res.reverse_dropped += 1
            return False
        avg_close = float(self.positions[-1].avg_px_close)
        if not self._margin_ok(intent.qty, avg_close):
            return True
        if self.sl_levels:
            self.strat._pending_levels = intent.levels
        self._exec(_Cmd(self.now, "entry", intent.side, intent.qty, None, self._oid()))
        return True

    def _expire_reversal(self, bar: Any) -> None:
        intent = self.intent
        if intent is None or int(bar.ts_init) <= intent.created_ts:
            return
        self.intent = None
        self.res.reverse_dropped += 1
        if self.pos_open:
            raise EmulatorUnsupported("a reversal exit left the position open (protection re-arm is not modeled)")

    def _refresh_levels(self, k: int, is_long: bool) -> None:
        """``_refresh_levels``: valid new levels replace the protection; invalid ones keep it."""
        if not self.pos_open or (self.pos_side == "BUY") != is_long:
            return
        s = self.strat
        levels = s._read_levels(self.pbars[k], is_long)
        if levels is None:
            self.res.refresh_invalid += 1
            return
        s._levels = levels
        self._drop_queued_protection()
        for rid in self._live_protection_ids():
            self._send(_Cmd(self.now, "cancel", ids=(rid,)))
        pos = self.positions[-1]
        entry = float(pos.avg_px_open)
        sl = s._calculate_sl_price(entry, is_long)
        if sl is not None and sl <= 0:
            sl = None
        tp = s._calculate_tp_price(entry, is_long)
        if tp is not None and tp <= 0:
            tp = None
        close_side = "SELL" if is_long else "BUY"
        if sl is not None:
            self._send(_Cmd(self.now, "sl", close_side, pos.quantity, self.inst.make_price(sl).raw, self._oid()))
        if tp is not None:
            self._send(_Cmd(self.now, "tp", close_side, pos.quantity, self.inst.make_price(tp).raw, self._oid()))

    # ----------------------------------------------------------------------------------
    # the run
    # ----------------------------------------------------------------------------------

    def _items(self) -> Any:
        """Data items in the engine's order: ``ts_init``, ties by ``sorted(timeframes)``, ticks last."""
        order = {tf: i for i, tf in enumerate(sorted(self.run_.bars))}

        def bars(tf: str) -> Any:
            i = order[tf]
            return ((int(b.ts_init), i, "bar", tf, n) for n, b in enumerate(self.run_.bars[tf]))

        streams = [bars(tf) for tf in self.run_.bars]
        if self.fill_ticks:
            big = len(order)
            streams.append((ts, big, "tick", "", price) for ts, price in self.fill_ticks)
        return heapq.merge(*streams, key=lambda e: (e[0], e[1]))

    def run(self) -> EmulatorResult:
        events = self.sig.events
        last_ts = 0
        for ts, _o, kind, tf, ref in self._items():
            last_ts = ts
            self.now = ts
            if events is not None:
                self._cur_row = events.row_at(ts)
            if kind == "tick":
                self.last = int(ref)
                self.last_size = self.fill_tick_size
                if self.resting and self.pos_open:
                    self._match(int(ref), "F")
                if self.release == "tick":
                    self._flush(ts, self._px(int(ref)))
                continue
            bar = self.run_.bars[tf][ref]
            is_exec = tf == self.exec_tf
            is_primary = tf == self.primary_tf
            if is_exec:
                self._venue_bar(bar)
            if is_primary:
                if self.rows is not None:
                    self._cur_row = int(self.rows[ref])
                self._primary_bar(ref)
            if self.release == "bar" and tf == "1m":
                self._flush(ts, float(bar.close))
        self.ledger.stopped(last_ts)
        if self.defer and self.positions and self.positions[-1].is_open:
            pos = self.positions[-1]
            self.now = last_ts
            self._fill(
                buy=not pos.is_long, qty=pos.quantity, base_raw=self._book(), order_type="MARKET",
                maker=False, action="exit", oid=self._oid(), finalize=False,
            )
        self.res.sizing_skips = dict(getattr(self.strat, "_sizing_skips", {}) or {})
        return self.res

    def _px(self, raw: int) -> float:
        from nautilus_trader.model.objects import Price

        return float(Price.from_raw(raw, self.prec))

    def _venue_bar(self, bar: Any) -> None:
        """The matching engine processes an execution bar; ``tv`` first releases at its OPEN."""
        ticks = self._bar_ticks(bar)
        if self.release != "next_open":
            self._venue_ticks(ticks)
            return
        # TradingView: the bar opens at ``ts_event`` knowing only what closed before it. The open
        # tick (a gap through resting protection), the order released at the open and every
        # fill of the bar read the indicator state of that instant -- never the bar's own close.
        at_close, row = self.now, self._cur_row
        open_ns = int(bar.ts_event)
        self.now = open_ns
        if self.sig.events is not None:
            self._cur_row = self.sig.events.row_at(open_ns)
        # the open is the release instant even when it did not move the last price
        if ticks and ticks[0][1] == "O":
            self._venue_ticks(ticks[:1])
            ticks = ticks[1:]
        self._flush(int(bar.ts_init), float(bar.open))
        self.now = at_close
        self._venue_ticks(ticks)
        self._cur_row = row


def fill_ticks_for(runner: NTScreeningRunner, symbol: str, exec_tf: str) -> list[tuple[int, int]]:
    """(``ts_init`` ns, price raw) of the screening fill ticks of ``exec_tf``.

    The 1m closes of the bars opening at a boundary of ``exec_tf`` and one minute after it
    (``vibe_quant.data.fill_ticks``), each stamped at that 1m bar's ``ts_init``.
    """
    from vibe_quant.data.fill_ticks import _BAR_INIT_OFFSET_NS, _MINUTE_NS, _TF_NS
    from vibe_quant.fast_tier.signals import load_bars, load_instrument

    catalog = str(runner._resolved_catalog_path.resolve())
    instrument_id = f"{symbol}-PERP.BINANCE"
    instrument = load_instrument(catalog, instrument_id)
    step = _TF_NS[exec_tf]
    bars = load_bars(
        catalog, f"{instrument_id}-1-MINUTE-LAST-EXTERNAL", runner._start_date, runner._end_date
    )
    out: list[tuple[int, int]] = []
    for bar in bars:
        rem = (int(bar.ts_init) - _BAR_INIT_OFFSET_NS) % step
        if rem in (0, _MINUTE_NS):
            out.append((int(bar.ts_init), instrument.make_price(float(bar.close)).raw))
    return out


def emulate(runner: NTScreeningRunner, symbol: str, mode: Mode = "engine") -> EmulatorResult:
    """Replay ``runner``'s compiled strategy on ``symbol`` through the broker of ``mode``.

    ``engine`` follows ``screening_fill_mode`` for the DSL's loaded timeframes and the seeded
    screening fill model, so its trades equal the screening engine's. ``tv`` fills at the next
    bar's open with TradingView's timeframe merge order (see the module doc).
    """
    from vibe_quant.fast_tier.signals import signals_for_runner
    from vibe_quant.screening.nt_runner import screening_fill_mode
    from vibe_quant.validation.extraction import finest_timeframe
    from vibe_quant.validation.fill_model import ScreeningFillModelConfig
    from vibe_quant.validation.venue import SCREENING_FILL_SEED, create_venue_config_for_screening

    if mode not in ("tv", "engine"):
        raise ValueError(f"mode must be 'tv' or 'engine', got {mode!r}")
    run = signals_for_runner(runner, symbol, record_levels=True, coarse_first=mode == "tv")
    loaded = runner._all_timeframes
    exec_tf = finest_timeframe({run.timeframe, *loaded}) or run.timeframe
    venue = create_venue_config_for_screening(starting_balance_usdt=runner._starting_balance)
    fill_cfg = venue.fill_config
    if not isinstance(fill_cfg, ScreeningFillModelConfig):
        raise EmulatorUnsupported(f"screening fill config {fill_cfg!r}")
    ticks: list[tuple[int, int]] = []
    release: Release
    if mode == "tv":
        release = "next_open"
    else:
        fill_mode = screening_fill_mode(runner._timeframe, loaded)
        if fill_mode.command_release == "":
            release = "defer"
        elif fill_mode.command_release == "bar":
            release = "bar"
        else:
            release = "tick"
            ticks = fill_ticks_for(runner, symbol, exec_tf)
    return Emulator(
        run,
        mode=mode,
        release=release,
        exec_tf=exec_tf,
        cost_model=runner._cost_model(),
        starting_balance=float(venue.starting_balance_usdt),
        fill_seed=SCREENING_FILL_SEED,
        prob_slippage=fill_cfg.prob_slippage,
        prob_fill_on_limit=fill_cfg.prob_fill_on_limit,
        fill_ticks=ticks,
        dsl=runner._dsl_dict,
    ).run()
