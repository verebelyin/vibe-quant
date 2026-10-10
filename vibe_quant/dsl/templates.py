"""Static code templates for the DSL-to-NautilusTrader compiler.

Contains generated method bodies that are the same for every compiled strategy.
Dynamic/indicator-dependent generation stays in :mod:`compiler`.

The generated runtime is shared by screening/discovery, validation AND
paper/live trading, so the order-management rules here are money-path code
(vibe-quant-e70tl.4):

* sizing never falls back to a fixed quantity -- unsizeable entries are skipped
  and logged; equity is read in the instrument's settlement currency;
  quantities are rounded DOWN to the size increment;
* no new entry while an entry order is open/in flight or a position is open;
* every exit / SL / TP order is ``reduce_only`` (an exit racing a stop fill can
  never open a reverse position);
* SL/TP are re-sized to the full position on ``PositionChanged`` (partial fills);
* ``on_start`` adopts an already-open position and re-arms missing SL/TP
  (restart recovery);
* a trailing stop never loosens, including the first update after entry.

Every venue command goes through ``_send_command`` (vibe-quant-yul7u.9): with
``command_release`` set (backtests only) it is queued in a per-instrument
outbox and sent on this strategy's OWN next datum -- NT's venue-wide latency
release let another symbol's data event release it at a stale book.
"""

from __future__ import annotations

import textwrap


def _lines(src: str) -> tuple[str, ...]:
    """Dedent a template block into the tuple-of-lines form the compiler emits."""
    return tuple(textwrap.dedent(src).strip("\n").splitlines())


# ---------------------------------------------------------------------------
# Event handling (static – no DSL-dependent logic)
# ---------------------------------------------------------------------------

ON_EVENT_LINES: tuple[str, ...] = _lines(
    '''
    def on_event(self, event) -> None:
        """Handle strategy events for position tracking and SL/TP management."""
        if isinstance(event, PositionOpened):
            if event.instrument_id == self.instrument_id:
                self._sync_position_state()
                # Submit SL/TP using actual fill price from opened position
                pos = self.cache.position(event.position_id)
                if pos is not None:
                    entry_price = float(pos.avg_px_open)
                    entry_side = OrderSide.BUY if pos.side == PositionSide.LONG else OrderSide.SELL
                    self._submit_sl_tp_orders(entry_price, entry_side, pos.quantity)
        elif isinstance(event, PositionChanged):
            if event.instrument_id == self.instrument_id:
                self._sync_position_state()
                # Partial fills: keep SL/TP sized to the whole open position
                pos = self.cache.position(event.position_id)
                if pos is not None and pos.is_open:
                    self._sync_protective_orders(pos)
        elif isinstance(event, PositionClosed):
            if event.instrument_id == self.instrument_id:
                self._position_open = False
                self._position_side = None
                self._trailing_best_sl = None
                self._send_command(self.cancel_all_orders, self.instrument_id)
        elif isinstance(event, OrderFilled):
            if event.instrument_id == self.instrument_id:
                self._sync_position_state()
    '''
)

ON_STOP_LINES: tuple[str, ...] = _lines(
    '''
    def on_stop(self) -> None:
        """Strategy shutdown: cancel orders and close positions.

        Outbox mode sends nothing: end-of-run commands stay unsent, like latency
        never releasing end-of-run closes; open positions are marked at the last
        price by the metrics layer.
        """
        if getattr(self.config, 'command_release', ''):
            return
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
    '''
)

ON_RESET_LINES: tuple[str, ...] = _lines(
    '''
    def on_reset(self) -> None:
        """Reset strategy state between backtest runs."""
        self._position_open = False
        self._position_side = None
        self._pending_validation_action = None
        self._trailing_best_sl = None
        self._rearm_protection = False
        self._outbox = []
    '''
)

# Appended to the generated on_start (after indicator registration).
ON_START_RECOVERY_LINES: tuple[str, ...] = _lines(
    '''
    # Restart recovery: adopt an already-open position (reconciled into the
    # cache) and make sure it carries SL/TP; ATR-based levels wait for warmup.
    self._sync_position_state()
    if self._position_open and not self._ensure_protection():
        self._rearm_protection = True
    '''
)

# Inserted in the generated on_start right after the bar subscriptions.
ON_START_OUTBOX_LINES: tuple[str, ...] = _lines(
    '''
    # Command outbox: queued commands are released by this instrument's OWN
    # next trade tick / detail bar, never by another symbol's data event.
    _release = getattr(self.config, 'command_release', '')
    if _release and getattr(self.config, 'execution_delay_probability', 0.0) > 0:
        # The outbox IS the execution delay; a one-bar defer on top double-delays.
        raise ValueError("command_release needs execution_delay_probability=0")
    if _release == "trade_tick":
        self.subscribe_trade_ticks(self.instrument_id)
    elif _release == "bar":
        _bar_type = getattr(self.config, 'command_release_bar_type', '')
        if not _bar_type:
            raise ValueError("command_release='bar' needs command_release_bar_type")
        self._command_release_bar_type = BarType.from_str(_bar_type)
        if self._command_release_bar_type.instrument_id != self.instrument_id:
            raise ValueError(
                f"command_release_bar_type {self._command_release_bar_type} is not "
                f"for {self.instrument_id}"
            )
        self.subscribe_bars(self._command_release_bar_type)
    elif _release:
        raise ValueError(f"Unknown command_release {_release!r}")
    '''
)

ON_TRADE_TICK_LINES: tuple[str, ...] = _lines(
    '''
    def on_trade_tick(self, tick: TradeTick) -> None:
        """Command outbox: this instrument's own trade tick releases queued commands."""
        if tick.instrument_id == self.instrument_id:
            self._flush_outbox(tick.ts_init)
    '''
)

# ---------------------------------------------------------------------------
# compute_fn (pandas) indicator buffers (static; emitted only when needed)
# ---------------------------------------------------------------------------

PTA_FEED_LINES: tuple[str, ...] = (
    "def _feed_pta_buffer(self, tf: str, bar: Bar) -> None:",
    '    """Append a bar to its timeframe buffer, trim, recompute that timeframe.',
    "",
    "    PtaBuffer trims to the cap (with 25% slack so the trim amortizes):",
    "    recomputing indicators over full history every bar is O(n^2) across a",
    "    backtest and dominates 1m-data runtime.",
    '    """',
    "    self._pta_bufs[tf].append(",
    "        float(bar.open), float(bar.high), float(bar.low), float(bar.close), float(bar.volume)",
    "    )",
    "    self._update_pta_indicators(tf)",
)

# ---------------------------------------------------------------------------
# Order submission and position management (static)
# ---------------------------------------------------------------------------

ORDER_METHODS_LINES: tuple[str, ...] = _lines(
    '''
    def _has_pending_entry(self) -> bool:
        """True while an entry (non-reduce-only) order of this strategy is open, in
        flight or queued in the command outbox."""
        for order in self.cache.orders_open(instrument_id=self.instrument_id, strategy_id=self.id):
            if not order.is_reduce_only:
                return True
        for order in self.cache.orders_inflight(instrument_id=self.instrument_id, strategy_id=self.id):
            if not order.is_reduce_only:
                return True
        for _ts, fn, args in self._outbox:
            if fn == self.submit_order and not args[0].is_reduce_only:
                return True
        return False

    def _send_command(self, fn, *args) -> None:
        """Send a venue command (submit/cancel) now, or queue it in the outbox.

        command_release "" (live, paper, 1m strategies) sends immediately.
        Otherwise (queue ts, fn, args) waits for _flush_outbox on this
        instrument's own next datum.
        """
        if getattr(self.config, 'command_release', ''):
            self._outbox.append((self.clock.timestamp_ns(), fn, args))
            return
        fn(*args)

    def _flush_outbox(self, ts: int) -> None:
        """Send queued commands with queue ts < ts, in order (strict: a datum at
        the queue ts itself never releases)."""
        if not self._outbox:
            return
        ready = [cmd for cmd in self._outbox if cmd[0] < ts]
        if not ready:
            return
        self._outbox = [cmd for cmd in self._outbox if cmd[0] >= ts]
        for _ts, fn, args in ready:
            fn(*args)

    def _submit_long_entry(self, bar: Bar) -> None:
        """Submit a long market entry (SL/TP follow on PositionOpened)."""
        if self._position_open or self._has_pending_entry():
            return

        qty = self._calculate_position_size(bar, is_long=True)
        if qty is None:
            return
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=qty,
            time_in_force=TimeInForce.IOC,
        )
        self._send_command(self.submit_order, order)
        # SL/TP orders are submitted from on_event(PositionOpened) after fill

    def _submit_short_entry(self, bar: Bar) -> None:
        """Submit a short market entry (SL/TP follow on PositionOpened)."""
        if self._position_open or self._has_pending_entry():
            return

        qty = self._calculate_position_size(bar, is_long=False)
        if qty is None:
            return
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=qty,
            time_in_force=TimeInForce.IOC,
        )
        self._send_command(self.submit_order, order)
        # SL/TP orders are submitted from on_event(PositionOpened) after fill

    def _submit_exit(self, bar: Bar) -> None:
        """Close the open position with a reduce-only market order."""
        if not self._position_open:
            return

        # Cancel existing SL/TP orders
        self._send_command(self.cancel_all_orders, self.instrument_id)

        # Determine exit side and get actual position quantity from cache
        exit_side = OrderSide.SELL if self._position_side == OrderSide.BUY else OrderSide.BUY
        quantity = None
        positions = self.cache.positions_open(venue=self.instrument_id.venue)
        for pos in positions:
            if pos.instrument_id == self.instrument_id and pos.is_open:
                quantity = pos.quantity
                break

        if quantity is None:
            return

        # reduce_only: if a stop fills first, this order is rejected instead of
        # opening a reverse position.
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=exit_side,
            quantity=quantity,
            time_in_force=TimeInForce.IOC,
            reduce_only=True,
        )
        self._send_command(self.submit_order, order)

    def _sl_config(self, is_long: bool) -> tuple[str, str]:
        """(stop-loss type, config field prefix), per-direction override first."""
        dir_suffix = 'long' if is_long else 'short'
        sl_type = getattr(self.config, f'stop_loss_{dir_suffix}_type', None)
        if sl_type is not None:
            return sl_type, f'stop_loss_{dir_suffix}'
        return self.config.stop_loss_type, 'stop_loss'

    def _tp_config(self, is_long: bool) -> tuple[str, str]:
        """(take-profit type, config field prefix), per-direction override first."""
        dir_suffix = 'long' if is_long else 'short'
        tp_type = getattr(self.config, f'take_profit_{dir_suffix}_type', None)
        if tp_type is not None:
            return tp_type, f'take_profit_{dir_suffix}'
        return self.config.take_profit_type, 'take_profit'

    def _submit_sl_tp_orders(self, entry_price: float, side: OrderSide, qty: Quantity) -> None:
        """Submit reduce-only stop-loss and take-profit orders for a position."""
        is_long = side == OrderSide.BUY

        # Calculate and submit stop-loss order
        sl_price = self._calculate_sl_price(entry_price, is_long)
        if self._sl_config(is_long)[0] == "atr_trailing":
            best = self._trailing_best_sl
            if best is not None:
                # Re-arming (partial fill / restart) never loosens the trail
                sl_price = best if sl_price is None else (max(sl_price, best) if is_long else min(sl_price, best))
            if sl_price is not None and sl_price > 0:
                # Seed the trail with the submitted level: the first update
                # after entry may only tighten it.
                self._trailing_best_sl = sl_price
        if sl_price is not None and sl_price <= 0:
            sl_price = None  # skip invalid negative/zero price
        if sl_price is not None:
            sl_side = OrderSide.SELL if is_long else OrderSide.BUY
            sl_order = self.order_factory.stop_market(
                instrument_id=self.instrument_id,
                order_side=sl_side,
                quantity=qty,
                trigger_price=self.instrument.make_price(sl_price),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
            self._send_command(self.submit_order, sl_order)

        # Calculate and submit take-profit order
        tp_price = self._calculate_tp_price(entry_price, is_long)
        if tp_price is not None and tp_price <= 0:
            tp_price = None  # skip invalid negative/zero price
        if tp_price is not None:
            tp_side = OrderSide.SELL if is_long else OrderSide.BUY
            tp_order = self.order_factory.limit(
                instrument_id=self.instrument_id,
                order_side=tp_side,
                quantity=qty,
                price=self.instrument.make_price(tp_price),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
            self._send_command(self.submit_order, tp_order)

    def _protective_orders(self) -> list:
        """Live reduce-only SL (stop-market) / TP (limit) orders of this strategy."""
        seen = set()
        out = []
        for order in (
            self.cache.orders_open(instrument_id=self.instrument_id, strategy_id=self.id)
            + self.cache.orders_inflight(instrument_id=self.instrument_id, strategy_id=self.id)
        ):
            if order.client_order_id in seen or order.is_pending_cancel:
                continue
            seen.add(order.client_order_id)
            if order.is_reduce_only and order.order_type in (OrderType.STOP_MARKET, OrderType.LIMIT):
                out.append(order)
        return out

    def _sync_protective_orders(self, pos) -> None:
        """Size SL/TP to the full open position: re-arm if none, else cancel/replace
        any order whose remaining quantity differs (prices are kept)."""
        orders = self._protective_orders()
        if not orders:
            entry_side = OrderSide.BUY if pos.side == PositionSide.LONG else OrderSide.SELL
            self._submit_sl_tp_orders(float(pos.avg_px_open), entry_side, pos.quantity)
            return
        for order in orders:
            if order.leaves_qty == pos.quantity:
                continue
            self._send_command(self.cancel_order, order)
            if order.order_type == OrderType.STOP_MARKET:
                replacement = self.order_factory.stop_market(
                    instrument_id=self.instrument_id,
                    order_side=order.side,
                    quantity=pos.quantity,
                    trigger_price=order.trigger_price,
                    time_in_force=TimeInForce.GTC,
                    reduce_only=True,
                )
            else:
                replacement = self.order_factory.limit(
                    instrument_id=self.instrument_id,
                    order_side=order.side,
                    quantity=pos.quantity,
                    price=order.price,
                    time_in_force=TimeInForce.GTC,
                    reduce_only=True,
                )
            self._send_command(self.submit_order, replacement)

    def _ensure_protection(self) -> bool:
        """Restart recovery: give an adopted open position its SL/TP.

        Returns False (caller retries on the first ready bar) while ATR-based
        levels cannot be computed yet.
        """
        self._rearm_protection = False
        pos = None
        for p in self.cache.positions_open(venue=self.instrument_id.venue):
            if p.instrument_id == self.instrument_id and p.is_open:
                pos = p
                break
        if pos is None:
            return True
        is_long = pos.side == PositionSide.LONG
        existing = self._protective_orders()
        for order in existing:
            if order.order_type == OrderType.STOP_MARKET and self._sl_config(is_long)[0] == "atr_trailing":
                self._trailing_best_sl = float(order.trigger_price)
        if existing:
            self._sync_protective_orders(pos)
            return True
        uses_atr = (
            self._sl_config(is_long)[0] in ("atr_fixed", "atr_trailing")
            or self._tp_config(is_long)[0] in ("atr_fixed", "risk_reward")
        )
        if uses_atr and not self._indicators_ready():
            self.log.warning(
                f"Recovered open {pos.side} position {pos.quantity} has no SL/TP; "
                "re-arming once indicators are ready"
            )
            return False
        self.log.warning(f"Recovered open {pos.side} position {pos.quantity} has no SL/TP; re-arming")
        self._sync_protective_orders(pos)
        return True

    def _calculate_sl_price(self, entry_price: float, is_long: bool) -> float | None:
        """Calculate stop-loss price based on config type (per-direction aware)."""
        sl_type, prefix = self._sl_config(is_long)
        if sl_type == "fixed_pct":
            pct = getattr(self.config, f'{prefix}_percent')
            if is_long:
                return entry_price * (1 - pct / 100)
            else:
                return entry_price * (1 + pct / 100)
        elif sl_type in ("atr_fixed", "atr_trailing"):
            sl_ind = getattr(self.config, f'{prefix}_indicator', None)
            if sl_ind is None:
                return None
            atr_value = self._get_indicator_value(sl_ind)
            multiplier = getattr(self.config, f'{prefix}_atr_multiplier')
            if is_long:
                return entry_price - atr_value * multiplier
            else:
                return entry_price + atr_value * multiplier
        return None

    def _update_trailing_stop(self, bar: Bar) -> None:
        """Move the trailing stop only in the favorable direction (never loosens)."""
        if not self._position_open:
            return
        is_long = self._position_side == OrderSide.BUY
        sl_type, prefix = self._sl_config(is_long)
        if sl_type != "atr_trailing":
            return
        sl_ind = getattr(self.config, f'{prefix}_indicator', None)
        if sl_ind is None:
            return
        atr_value = self._get_indicator_value(sl_ind)
        multiplier = getattr(self.config, f'{prefix}_atr_multiplier')
        current_price = float(bar.close.as_double())
        best = self._trailing_best_sl
        if is_long:
            new_sl = current_price - atr_value * multiplier
            if best is not None and new_sl <= best:
                return  # SL hasn't improved
        else:
            new_sl = current_price + atr_value * multiplier
            if best is not None and new_sl >= best:
                return  # SL hasn't improved
        if new_sl <= 0:
            return  # skip invalid negative/zero price
        self._trailing_best_sl = new_sl
        # Cancel existing SL orders and resubmit at new price
        for order in self.cache.orders_open(venue=self.instrument_id.venue):
            if (order.instrument_id == self.instrument_id
                    and order.is_reduce_only and order.order_type == OrderType.STOP_MARKET):
                self._send_command(self.cancel_order, order)
        positions = self.cache.positions_open(venue=self.instrument_id.venue)
        for pos in positions:
            if pos.instrument_id == self.instrument_id and pos.is_open:
                sl_side = OrderSide.SELL if is_long else OrderSide.BUY
                sl_order = self.order_factory.stop_market(
                    instrument_id=self.instrument_id,
                    order_side=sl_side,
                    quantity=pos.quantity,
                    trigger_price=self.instrument.make_price(self._trailing_best_sl),
                    time_in_force=TimeInForce.GTC,
                    reduce_only=True,
                )
                self._send_command(self.submit_order, sl_order)
                break

    def _calculate_tp_price(self, entry_price: float, is_long: bool) -> float | None:
        """Calculate take-profit price based on config type (per-direction aware)."""
        tp_type, prefix = self._tp_config(is_long)
        if tp_type == "fixed_pct":
            pct = getattr(self.config, f'{prefix}_percent')
            if is_long:
                return entry_price * (1 + pct / 100)
            else:
                return entry_price * (1 - pct / 100)
        elif tp_type == "atr_fixed":
            tp_ind = getattr(self.config, f'{prefix}_indicator', None)
            if tp_ind is None:
                return None
            atr_value = self._get_indicator_value(tp_ind)
            multiplier = getattr(self.config, f'{prefix}_atr_multiplier')
            if is_long:
                return entry_price + atr_value * multiplier
            else:
                return entry_price - atr_value * multiplier
        elif tp_type == "risk_reward":
            sl_price = self._calculate_sl_price(entry_price, is_long)
            if sl_price is not None:
                sl_distance = abs(entry_price - sl_price)
                ratio = getattr(self.config, f'{prefix}_risk_reward')
                if is_long:
                    return entry_price + sl_distance * ratio
                else:
                    return entry_price - sl_distance * ratio
        return None

    def _calculate_position_size(self, bar: Bar, is_long: bool = True) -> Quantity | None:
        """Risk-based size, rounded DOWN to the size increment.

        Returns None -- entry skipped and logged, never a fixed fallback
        quantity -- when the account/equity/price is unusable or the size
        cannot satisfy the instrument minimums within max_position_pct.
        """
        account = self.cache.account_for_venue(self.instrument_id.venue)
        if account is None:
            self.log.warning("Sizing: no account for venue yet -- entry skipped")
            return None

        # Equity in the settlement currency (USDT for USDT-M perps), never
        # whichever wallet currency happens to be listed first.
        ccy = self.instrument.get_settlement_currency()
        balance = account.balance_total(ccy)
        equity = float(balance) if balance is not None else 0.0
        if equity <= 0:
            self.log.warning(f"Sizing: no positive {ccy} equity ({equity}) -- entry skipped")
            return None

        price = float(bar.close)
        if price <= 0:
            self.log.warning(f"Sizing: invalid price {price} -- entry skipped")
            return None

        # Fixed fractional sizing: risk_per_trade % of equity
        risk_pct = getattr(self.config, 'risk_per_trade', 0.02)
        risk_amount = equity * risk_pct

        # Use stop loss distance if available, otherwise use 2% of price
        sl_price = self._calculate_sl_price(price, is_long)
        if sl_price is not None and sl_price > 0:
            stop_distance = abs(price - sl_price)
        else:
            stop_distance = price * 0.02  # 2% default stop distance

        if stop_distance <= 0:
            stop_distance = price * 0.02

        # Size = risk_amount / stop_distance, capped at max_position_pct of equity
        raw_size = risk_amount / stop_distance
        max_pos_pct = getattr(self.config, 'max_position_pct', 0.5)
        max_size = (equity * max_pos_pct) / price
        final_size = min(raw_size, max_size)
        if final_size <= 0:
            self.log.warning(f"Sizing: non-positive size {final_size} -- entry skipped")
            return None

        # Instrument minimum: clamp up only while it stays within max_position_pct
        min_qty = self.instrument.min_quantity
        if min_qty is not None and final_size < float(min_qty):
            if float(min_qty) > max_size:
                self.log.warning(
                    f"Sizing: min quantity {min_qty} exceeds max_position_pct cap "
                    f"{max_size:.8f} -- entry skipped"
                )
                return None
            final_size = float(min_qty)

        try:
            qty = self.instrument.make_qty(final_size, round_down=True)
        except ValueError:
            self.log.warning(f"Sizing: {final_size} rounds below the size increment -- entry skipped")
            return None
        min_notional = self.instrument.min_notional
        if min_notional is not None and float(qty) * price < float(min_notional):
            self.log.warning(
                f"Sizing: notional {float(qty) * price:.2f} below minimum {min_notional} -- entry skipped"
            )
            return None
        return qty

    def _sync_position_state(self) -> None:
        """Sync position tracking state from cache."""
        positions = self.cache.positions_open(venue=self.instrument_id.venue)
        for pos in positions:
            if pos.instrument_id == self.instrument_id and pos.is_open:
                self._position_open = True
                self._position_side = OrderSide.BUY if pos.side == PositionSide.LONG else OrderSide.SELL
                return
        self._position_open = False
        self._position_side = None

    def _maybe_delay_validation_action(self, action: str) -> bool:
        """Defer an entry/exit by one bar so it fills at the next bar close, not the
        signal bar's close. Screening/discovery set execution_delay_probability=1.0
        (always defer -> no same-bar look-ahead); sub-5m validation sets 0.3 for
        partial degradation. prob >= 1.0 skips the RNG to stay deterministic."""
        delay_prob = getattr(self.config, 'execution_delay_probability', 0.0)
        if delay_prob <= 0.0:
            return False
        if self._pending_validation_action is not None:
            return False
        if delay_prob < 1.0 and self._delay_rng.random() >= delay_prob:
            return False
        self._pending_validation_action = action
        return True

    def _dispatch_pending_validation_action(self, bar: Bar) -> bool:
        """Execute a delayed validation action on the next primary bar."""
        action = self._pending_validation_action
        if action is None:
            return False
        self._pending_validation_action = None
        if action == 'long_entry':
            self._submit_long_entry(bar)
        elif action == 'short_entry':
            self._submit_short_entry(bar)
        elif action == 'exit':
            self._submit_exit(bar)
        return True
    '''
)
