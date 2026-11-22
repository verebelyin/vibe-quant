"""Drive the paper node's venue config check (68m53) to a matching pass in tests.

Since 68m53 a paper node starts with new entries blocked (``VENUE_CHECK_PENDING``)
until the venue's leverage/margin type is verified and published into the guard's
account. Tests about other subjects get there two ways:

* :func:`venue_pass` applies one matching symbolConfig answer through the node's real
  apply path (classifier, publish map, publish + read-back, lift, retell). For unit
  guards whose portfolio has no NT account, a recording account stands in.
* :func:`inject_matching_read` answers the node's real verifier loop (``start()`` /
  ``_run_loop`` with a running guard actor) through the ``_build_venue_read`` seam, and
  the positionRisk read (T4, ``_build_position_read``) with no position and the
  open-order reads (opspt P2, ``_build_open_orders_read``) with no order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.paper.venue_config import classify_symbol_configs

if TYPE_CHECKING:
    from decimal import Decimal

    from nautilus_trader.model.data import Bar, BarType

    from vibe_quant.paper.node import PaperTradingNode


@pytest.fixture()
def synthetic_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed every strategy with flat synthetic bars instead of public Binance klines."""
    from nautilus_trader.core.datetime import millis_to_nanos
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.objects import Price, Quantity

    from vibe_quant.paper import warmup

    price = Price.from_str("100.00")
    volume = Quantity.from_str("1.000")

    def seed(_symbol: str, bar_type: BarType, end_ms: int) -> list[Bar]:
        step = warmup.interval_ms(warmup.interval_of(bar_type))
        stamps = [
            millis_to_nanos(close_ms + step - 1)
            for close_ms in range(end_ms - warmup.SEED_BARS * step, end_ms, step)
        ]
        return [Bar(bar_type, price, price, price, price, volume, ts, ts) for ts in stamps]

    monkeypatch.setattr(warmup, "fetch_seed", seed)


class LeverageAccount:
    """Stands in for NT's ``MarginAccount`` leverage map (set, then read back)."""

    def __init__(self) -> None:
        self.leverages: dict[Any, Decimal] = {}
        self.sets: list[tuple[str, Decimal]] = []

    def set_leverage(self, instrument_id: Any, leverage: Decimal) -> None:
        self.sets.append((str(instrument_id), leverage))
        self.leverages[instrument_id] = leverage

    def leverage(self, instrument_id: Any) -> Decimal | None:
        return self.leverages.get(instrument_id)


def matching_body(node: PaperTradingNode) -> list[dict[str, object]]:
    """The symbolConfig answer that matches the node's resolved config exactly."""
    resolved = node.resolved_config
    assert resolved is not None
    return [
        {"symbol": symbol, "leverage": resolved.leverage, "marginType": resolved.margin_type}
        for symbol in resolved.symbols
    ]


def venue_pass(node: PaperTradingNode, monkeypatch: pytest.MonkeyPatch) -> LeverageAccount | None:
    """Apply one matching venue answer through the node's real apply path.

    Returns the stand-in account when the guard's portfolio had none (unit guards).
    """
    from nautilus_trader.model.identifiers import Venue

    import vibe_quant.paper.node as node_module
    from vibe_quant.paper.venue_config import publish_leverage

    guard = node.guard
    assert guard is not None
    stand_in: LeverageAccount | None = None
    portfolio = guard.portfolio
    account = None if portfolio is None else portfolio.account(Venue(guard.config.venue))
    if not hasattr(account, "set_leverage"):
        # A unit guard has no NT account: the real publish writes to a stand-in.
        stand_in = LeverageAccount()

        def publish(_account: object, iids: Any, leverage: int) -> str | None:
            return publish_leverage(stand_in, iids, leverage)

        monkeypatch.setattr(node_module, "publish_leverage", publish)
    resolved = node.resolved_config
    assert resolved is not None
    verdict = classify_symbol_configs(
        matching_body(node), resolved.symbols, resolved.leverage, resolved.margin_type
    )
    assert node._venue_apply(verdict, None) is True, "the matching answer did not pass"
    return stand_in


def inject_matching_read(monkeypatch: pytest.MonkeyPatch, node: PaperTradingNode) -> list[int]:
    """Answer the real verifier loop with the matching symbolConfig body.

    The actor poll and retry delays are shrunk (outcomes unchanged); the 60 s wait after
    a pass never ends, so the session stays on its first pass. Returns the list that
    grows by one per read.
    """
    import asyncio

    from vibe_quant.paper.node import VENUE_CHECK_INTERVAL_S

    reads: list[int] = []

    async def read() -> object:
        reads.append(1)
        return matching_body(node)

    async def short_wait(seconds: float) -> None:
        if seconds >= VENUE_CHECK_INTERVAL_S:
            await asyncio.Event().wait()
        await asyncio.sleep(0.01)

    async def flat() -> object:
        return []  # positionRisk (T4): no venue position

    async def no_orders() -> tuple[object, object]:
        return [], []  # openOrders + openAlgoOrders (opspt P2): no open order

    monkeypatch.setattr(node, "_build_venue_read", lambda: read)
    monkeypatch.setattr(node, "_build_position_read", lambda: flat)
    monkeypatch.setattr(node, "_build_open_orders_read", lambda: no_orders)
    monkeypatch.setattr(node, "_venue_wait", short_wait)
    return reads


def pass_at_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every node arms the check (entries paused on PENDING) and the matching answer
    lands at once, before the start-up command drain (the venue answered first)."""
    from vibe_quant.paper.node import PaperTradingNode

    real_arm = PaperTradingNode._arm_venue_check

    def arm_then_pass(self: PaperTradingNode) -> None:
        real_arm(self)
        venue_pass(self, monkeypatch)

    monkeypatch.setattr(PaperTradingNode, "_arm_venue_check", arm_then_pass)
