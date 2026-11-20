"""Every exit a compiled DSL strategy sends carries the explicit ``reduce_only`` flag.

ro-vc-1 F3: while the venue check blocks (or the node stops) the paper guard admits an
order only on its explicit ``reduce_only`` flag, no longer on a reduction inferred from
NT's local net. This pins the premise: each order that would reduce the strategy's net
at the moment it reaches the risk engine (signal exit, fixed SL, TP, trailing-stop
replacement, time exit, reversal exit, the ``on_stop`` close) is flagged.

Direct BacktestEngine runs of the module the paper node compiles (``compile_to_module``,
no ``command_release``: paper strips it) on flat synthetic 5m bars; every SubmitOrder is
recorded at the ``RiskEngine.execute`` endpoint (where the paper guard sits) with the
portfolio's net position at that instant.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, NamedTuple

import pytest
from nautilus_trader.execution.messages import SubmitOrder, SubmitOrderList
from nautilus_trader.model.enums import OrderSide

from tests.unit.test_command_outbox import _M5, _bar, _compiled, _engine
from vibe_quant.paper.guard import _RISK_ENDPOINT, _order_reduces

_BTC = "BTCUSDT-PERP.BINANCE"


class _Sent(NamedTuple):
    order: Any
    net: Decimal  # the instrument's net position when the order reached the risk engine

    @property
    def reduces(self) -> bool:
        return _order_reduces(self.order.side, self.order.quantity.as_decimal(), self.net)


def _dsl(name: str, **overrides: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": name,
        "timeframe": "5m",
        "indicators": {"atr": {"type": "ATR", "period": 2}},
        "entry_conditions": {"long": ["close > 50"]},
        "stop_loss": {"type": "fixed_pct", "percent": 20.0},
        "take_profit": {"type": "fixed_pct", "percent": 40.0},
    }
    d.update(overrides)
    return d


def _run(dsl: dict[str, Any], closes: list[float]) -> list[_Sent]:
    """Run ``dsl`` on BTC 5m bars at ``closes``; every order submitted, in order."""
    strategy_cls, config_cls = _compiled(dsl)
    engine = _engine()
    sent: list[_Sent] = []
    risk = engine.kernel.risk_engine
    msgbus = engine.kernel.msgbus

    def record(command: Any) -> None:
        if isinstance(command, SubmitOrder):
            orders = [command.order]
        elif isinstance(command, SubmitOrderList):
            orders = list(command.order_list.orders)
        else:
            orders = []
        for order in orders:
            net = Decimal(str(engine.portfolio.net_position(order.instrument_id)))
            sent.append(_Sent(order, net))
        risk.execute(command)

    msgbus.deregister(_RISK_ENDPOINT, risk.execute)
    msgbus.register(_RISK_ENDPOINT, record)
    engine.add_data([_bar(_BTC, "5-MINUTE", _M5, i, c) for i, c in enumerate(closes)])
    engine.add_strategy(strategy_cls(config_cls(instrument_id=_BTC)))
    try:
        engine.run()
    finally:
        engine.dispose()
    return sent


def _filled_closer(sent: list[_Sent]) -> list[tuple[str, str, int]]:
    """(type, side, 5m bar index it was submitted on) of each reducing order that filled;
    the ``on_stop`` close is submitted on index ``len(closes)`` (after the last bar)."""
    return [
        (s.order.order_type.name, s.order.side.name, s.order.ts_init // _M5)
        for s in sent
        if s.reduces and s.order.is_closed and s.order.filled_qty
    ]


@pytest.mark.parametrize(
    ("dsl", "closes", "closer"),
    [
        pytest.param(
            _dsl("ro_signal", exit_conditions={"long": ["close < 90"]}),
            [100.0] * 4 + [85.0] * 3,
            ("MARKET", "SELL", 5),
            id="signal_exit",
        ),
        pytest.param(
            _dsl("ro_sl"),
            [100.0] * 4 + [70.0] * 3,
            ("STOP_MARKET", "SELL", 2),
            id="fixed_stop_loss",
        ),
        pytest.param(
            _dsl("ro_tp"),
            [100.0] * 4 + [150.0] * 3,
            ("LIMIT", "SELL", 2),
            id="take_profit",
        ),
        pytest.param(
            _dsl(
                "ro_trail",
                stop_loss={"type": "atr_trailing", "indicator": "atr", "atr_multiplier": 1.0},
                take_profit={"type": "fixed_pct", "percent": 50.0},
            ),
            [100.0, 102.0, 104.0, 106.0, 110.0, 114.0, 118.0, 122.0, 90.0, 90.0],
            ("STOP_MARKET", "SELL", 8),
            id="trailing_stop",
        ),
        pytest.param(
            _dsl("ro_time", max_bars_in_trade=2),
            [100.0] * 8,
            ("MARKET", "SELL", 4),
            id="time_exit",
        ),
        pytest.param(
            _dsl(
                "ro_reverse",
                entry_conditions={"long": ["close > 90"], "short": ["close < 90"]},
                position_mode="reverse",
            ),
            [100.0] * 4 + [85.0] * 4,
            ("MARKET", "SELL", 5),
            id="reversal_exit",
        ),
        pytest.param(
            _dsl("ro_on_stop", stop_loss={"type": "fixed_pct", "percent": 40.0}),
            [100.0] * 5,
            ("MARKET", "SELL", 5),
            id="on_stop_close",
        ),
    ],
)
def test_every_exit_order_is_reduce_only(
    dsl: dict[str, Any], closes: list[float], closer: tuple[str, str, int]
) -> None:
    sent = _run(dsl, closes)
    exits = [s for s in sent if s.reduces]
    assert closer in _filled_closer(sent), [
        (s.order.order_type.name, s.order.side.name, str(s.net), s.order.status.name) for s in sent
    ]
    unflagged = [
        (s.order.order_type.name, s.order.side.name, str(s.net))
        for s in exits
        if not s.order.is_reduce_only
    ]
    assert unflagged == [], "an exit without reduce_only is denied while the venue blocks"
    # Entries open from flat (never ride the inferred-reduction exemption).
    assert all(s.net == 0 for s in sent if not s.order.is_reduce_only)


def test_trailing_stop_replacements_are_flagged() -> None:
    """The trailing stop is moved by cancel + replace: each replacement is reduce-only."""
    dsl = _dsl(
        "ro_trail_moves",
        stop_loss={"type": "atr_trailing", "indicator": "atr", "atr_multiplier": 1.0},
        take_profit={"type": "fixed_pct", "percent": 50.0},
    )
    sent = _run(dsl, [100.0, 102.0, 104.0, 106.0, 110.0, 114.0, 118.0, 122.0, 90.0, 90.0])
    stops = [s for s in sent if s.order.order_type.name == "STOP_MARKET"]
    assert len(stops) >= 3, "the initial stop plus at least two trailing replacements"
    assert all(s.order.is_reduce_only and s.order.side == OrderSide.SELL for s in stops)
