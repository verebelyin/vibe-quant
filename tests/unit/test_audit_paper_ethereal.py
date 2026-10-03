"""Ethereal audit fixes: unknown order status after acceptance, bar precision."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx
import pytest
from eth_account import Account

from vibe_quant.ethereal.execution_client import (
    EtherealConfig,
    EtherealExecutionClient,
    EtherealOrder,
    OrderSide,
    OrderStatus,
    OrderType,
    parse_order_status,
)
from vibe_quant.ethereal.ingestion import get_ethereal_bar_type, klines_to_bars

if TYPE_CHECKING:
    from pathlib import Path


async def test_unknown_status_after_acceptance_does_not_raise() -> None:
    """Audit repro: venue accepted ({"status": "NEW"}) but place_order raised ->
    caller retry -> duplicate order. Now: one POST, result returned."""
    posted: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request.read())
        return httpx.Response(200, json={"orderId": "abc", "status": "NEW"})

    client = EtherealExecutionClient(
        EtherealConfig(private_key=Account.create().key.hex(), testnet=True)
    )

    async def _client() -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url="https://x", transport=httpx.MockTransport(handler))

    client._get_client = _client  # type: ignore[method-assign]
    order = EtherealOrder(
        symbol="BTC-USD", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0.01")
    )
    result = await client.place_order(order)
    assert result.order_id == "abc"
    assert result.status == OrderStatus.OPEN
    assert len(posted) == 1
    assert json.loads(posted[0])["order"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NEW", OrderStatus.OPEN),
        ("open", OrderStatus.OPEN),
        ("Partially-Filled", OrderStatus.PARTIALLY_FILLED),
        ("CANCELED", OrderStatus.CANCELLED),
        ("rejected", OrderStatus.REJECTED),
        ("SOMETHING_NEW", OrderStatus.UNKNOWN),
        (None, OrderStatus.UNKNOWN),
    ],
)
def test_parse_order_status(raw: object, expected: OrderStatus) -> None:
    assert parse_order_status(raw) == expected


def test_bars_use_instrument_precision(tmp_path: Path) -> None:
    """ETHUSD price precision is 2: '2500.0' must become 2500.00 on every bar."""
    rows = [
        {
            "open_time": 1704067200000,
            "open": 2500.0,
            "high": 2550.25,
            "low": 2480.0,
            "close": 2530.5,
            "volume": 100.0,
            "close_time": 1704067259999,
        },
        {
            "open_time": 1704067260000,
            "open": 2530.5,
            "high": 2560.125,
            "low": 2520.0,
            "close": 2545.0,
            "volume": 150.12345,
            "close_time": 1704067319999,
        },
    ]
    bars = klines_to_bars(rows, get_ethereal_bar_type("ETHUSD", "1m"))
    assert {b.open.precision for b in bars} == {2}
    assert {b.high.precision for b in bars} == {2}
    assert {b.volume.precision for b in bars} == {4}
    assert str(bars[0].open) == "2500.00"
    assert str(bars[1].high) == "2560.12"  # half-even at the 3rd decimal
    assert str(bars[1].volume) == "150.1234"
