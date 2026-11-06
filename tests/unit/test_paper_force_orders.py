"""forceOrders decode and fetch.

Contract rows R46 (request shape), R47 (payload fields, auto-close mapping,
emission order), R48 (zero executedQty rows) and R51 (a full page fails the
cycle). ``vibe_quant.paper.force_orders`` is the read-only decode/fetch half of
the exchange-liquidation detector: it signs one GET /fapi/v1/forceOrders and
never submits or cancels anything. The poll loop and node wiring are P3a2/P3b.

Every test drives a fake ``sign_request``; no network is touched.
"""

from __future__ import annotations

import json
import time

import pytest
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType, BinanceEnvironment
from nautilus_trader.adapters.binance.common.urls import get_http_base_url
from nautilus_trader.core.nautilus_pyo3 import HttpMethod

from vibe_quant.paper.force_orders import ForceOrder, IncompleteForceOrders, build_binance_fetch

_LIQUIDATION_CLIENT_ORDER_ID = "autoclose-1596107620040000020"


class _FakeClient:
    """Stands in for NT's ``BinanceHttpClient``: records the call, replays rows.

    It exposes ``sign_request`` only — there is no submit or cancel surface for
    the fetch to reach, which is what keeps the query read-only.
    """

    def __init__(
        self,
        rows: object,
        record: dict[str, object],
        error: Exception | None = None,
    ) -> None:
        self._rows = rows
        self._record = record
        self._error = error
        self.base_url = ""

    async def sign_request(
        self,
        http_method: object,
        url_path: str,
        payload: dict[str, str] | None = None,
        ratelimiter_keys: list[str] | None = None,
    ) -> bytes:
        self._record["http_method"] = http_method
        self._record["url_path"] = url_path
        self._record["payload"] = dict(payload or {})
        self._record["ratelimiter_keys"] = ratelimiter_keys
        if self._error is not None:
            raise self._error
        return json.dumps(self._rows).encode()


def _patch_factory(
    monkeypatch: pytest.MonkeyPatch,
    rows: object,
    error: Exception | None = None,
) -> dict[str, object]:
    """Patch NT's cached-client factory and return the captured call record."""
    record: dict[str, object] = {}
    client = _FakeClient(rows, record, error)
    record["client"] = client

    def fake_factory(*, clock, account_type, api_key, api_secret, environment):
        record["clock"] = clock
        record["account_type"] = account_type
        record["environment"] = environment
        record["api_key"] = api_key
        record["api_secret"] = api_secret
        client.base_url = get_http_base_url(account_type, environment, False)
        return client

    monkeypatch.setattr(
        "nautilus_trader.adapters.binance.factories.get_cached_binance_http_client",
        fake_factory,
    )
    return record


def _row(
    *,
    order_id: int,
    update_time: int,
    executed_qty: str = "0.002",
    client_order_id: str = _LIQUIDATION_CLIENT_ORDER_ID,
) -> dict[str, object]:
    """A forceOrders row shaped like the Binance USD-M response."""
    return {
        "symbol": "BTCUSDT",
        "orderId": order_id,
        "clientOrderId": client_order_id,
        "side": "SELL",
        "status": "FILLED",
        "avgPrice": "10913.5200",
        "executedQty": executed_qty,
        "updateTime": update_time,
        "autoCloseType": "LIQUIDATION",
    }


# --- Request shape and venue -------------------------------------------------


async def test_request_shape_is_exactly_the_contract_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One signed GET with symbol/startTime/endTime/limit/recvWindow/timestamp."""
    record = _patch_factory(monkeypatch, [])
    fetch = build_binance_fetch("key", "secret", "USDT_FUTURES", testnet=True)

    before_ms = int(time.time() * 1000)
    await fetch("BTCUSDT", 1_699_000_000_000, 1_700_000_000_000)
    after_ms = int(time.time() * 1000)

    payload = record["payload"]
    assert record["http_method"] is HttpMethod.GET
    assert record["url_path"] == "/fapi/v1/forceOrders"
    assert set(payload) == {
        "symbol",
        "startTime",
        "endTime",
        "limit",
        "recvWindow",
        "timestamp",
    }
    assert payload["symbol"] == "BTCUSDT"
    assert payload["startTime"] == "1699000000000"
    assert payload["endTime"] == "1700000000000"
    assert payload["limit"] == "100"
    assert payload["recvWindow"] == "5000"
    assert before_ms <= int(payload["timestamp"]) <= after_ms
    assert "autoCloseType" not in payload


@pytest.mark.parametrize(
    ("testnet", "expected_base_url"),
    [
        (True, "https://demo-fapi.binance.com"),
        (False, "https://fapi.binance.com"),
    ],
)
async def test_account_type_and_testnet_pick_the_venue(
    monkeypatch: pytest.MonkeyPatch,
    testnet: bool,
    expected_base_url: str,
) -> None:
    """Testnet is demo-fapi.binance.com; live is fapi.binance.com."""
    record = _patch_factory(monkeypatch, [])
    fetch = build_binance_fetch("key", "secret", "USDT_FUTURES", testnet=testnet)

    await fetch("BTCUSDT", 1, 2)

    assert record["account_type"] is BinanceAccountType.USDT_FUTURES
    assert record["environment"] is (
        BinanceEnvironment.TESTNET if testnet else BinanceEnvironment.LIVE
    )
    assert record["api_key"] == "key"
    assert record["api_secret"] == "secret"
    assert record["client"].base_url == expected_base_url


# --- Decoded payload fields --------------------------------------------------


def test_from_binance_decodes_every_contract_field() -> None:
    """The value object carries exactly the contract fields."""
    order = ForceOrder.from_binance(_row(order_id=6071832819, update_time=1_697_600_000_000))

    assert order == ForceOrder(
        symbol="BTCUSDT",
        order_id=6071832819,
        side="SELL",
        auto_close_type="LIQUIDATION",
        avg_price=10913.52,
        executed_qty=0.002,
        update_time_ms=1_697_600_000_000,
        ts_ns=1_697_600_000_000_000_000,
        status="FILLED",
        client_order_id=_LIQUIDATION_CLIENT_ORDER_ID,
    )
    assert isinstance(order.ts_ns, int)
    assert order.ts_ns == order.update_time_ms * 1_000_000


@pytest.mark.parametrize(
    ("client_order_id", "expected"),
    [
        (_LIQUIDATION_CLIENT_ORDER_ID, "LIQUIDATION"),
        ("autoclose-0", "LIQUIDATION"),
        ("adl_autoclose", "ADL"),
        ("adl_autoclose-123", "ADL"),
        ("web_9ac2e1f4", "OTHER"),
        ("", "OTHER"),
    ],
)
def test_auto_close_type_is_classified_from_the_client_order_id(
    client_order_id: str,
    expected: str,
) -> None:
    """'autoclose-*' -> LIQUIDATION, 'adl_autoclose*' -> ADL, else OTHER."""
    order = ForceOrder.from_binance(
        _row(order_id=1, update_time=1_000, client_order_id=client_order_id)
    )

    assert order.auto_close_type == expected


async def test_rows_are_returned_in_update_time_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """UpdateTime order decides the emission order, not the API's order."""
    rows = [
        _row(order_id=3, update_time=3_000),
        _row(order_id=1, update_time=1_000),
        _row(order_id=2, update_time=2_000),
    ]
    _patch_factory(monkeypatch, rows)
    fetch = build_binance_fetch("key", "secret", "USDT_FUTURES", testnet=True)

    orders = await fetch("BTCUSDT", 1, 4_000)

    assert [order.order_id for order in orders] == [1, 2, 3]


# --- Rows that never executed ------------------------------------------------


async def test_zero_executed_qty_row_is_returned_not_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fetch returns an unexecuted row; its executed_qty == 0 is the flag.

    The fetch decides nothing about skips — the watcher (P3a2) skips these rows
    and never marks them seen (R48). Filtering here would hide the row instead.
    """
    rows = [
        _row(order_id=1, update_time=1_000),
        _row(order_id=2, update_time=2_000, executed_qty="0"),
    ]
    _patch_factory(monkeypatch, rows)
    fetch = build_binance_fetch("key", "secret", "USDT_FUTURES", testnet=True)

    orders = await fetch("BTCUSDT", 1, 3_000)

    assert [order.order_id for order in orders] == [1, 2]
    assert orders[1].executed_qty == 0.0


# --- A full page fails the cycle ---------------------------------------------


async def test_full_page_raises_incomplete_force_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 100-row page is refused (there is no paging); the cycle fails."""
    rows = [_row(order_id=i, update_time=1_700_000_000_000 + i) for i in range(100)]
    _patch_factory(monkeypatch, rows)
    fetch = build_binance_fetch("key", "secret", "USDT_FUTURES", testnet=True)

    with pytest.raises(IncompleteForceOrders):
        await fetch("BTCUSDT", 1_699_000_000_000, 1_700_000_000_000)


async def test_page_below_the_limit_is_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under 100 rows the answer is complete and is returned as-is."""
    rows = [_row(order_id=i, update_time=1_000 + i) for i in range(99)]
    _patch_factory(monkeypatch, rows)
    fetch = build_binance_fetch("key", "secret", "USDT_FUTURES", testnet=True)

    orders = await fetch("BTCUSDT", 1, 2_000)

    assert len(orders) == 99


async def test_full_page_is_rejected_before_any_row_is_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The page check runs first, so a full page fails even with a bad row."""
    rows = [_row(order_id=i, update_time=1_000 + i) for i in range(99)]
    rows.append({"symbol": "BTCUSDT"})  # malformed: no avgPrice/updateTime
    _patch_factory(monkeypatch, rows)
    fetch = build_binance_fetch("key", "secret", "USDT_FUTURES", testnet=True)

    with pytest.raises(IncompleteForceOrders):
        await fetch("BTCUSDT", 1, 2_000)


async def test_non_list_response_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 200 with an error object (not an array) fails that symbol's fetch."""
    _patch_factory(monkeypatch, {"code": -1121, "msg": "Invalid symbol."})
    fetch = build_binance_fetch("key", "secret", "USDT_FUTURES", testnet=True)

    with pytest.raises(ValueError, match="not a list"):
        await fetch("BTCUSDT", 1, 2)


# --- malformed rows: ValueError names the field -----------------------------------


def test_missing_field_raises_value_error_naming_the_field() -> None:
    """A row missing avgPrice fails closed with the field named."""
    row = _row(order_id=1, update_time=1_000)
    del row["avgPrice"]

    with pytest.raises(ValueError, match="'avgPrice'"):
        ForceOrder.from_binance(row)


def test_non_numeric_price_raises_value_error_naming_the_field() -> None:
    """A non-numeric price fails closed with the field named."""
    row = _row(order_id=1, update_time=1_000)
    row["avgPrice"] = "not-a-number"

    with pytest.raises(ValueError, match="'avgPrice'"):
        ForceOrder.from_binance(row)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("orderId", "abc"),
        ("updateTime", None),
        ("updateTime", "not-a-time"),
        ("executedQty", []),
        ("symbol", 5),
        ("status", None),
    ],
)
def test_malformed_field_raises_value_error_naming_the_field(
    field: str,
    bad_value: object,
) -> None:
    """A null or wrongly typed field fails closed and names the offending field."""
    row = _row(order_id=1, update_time=1_000)
    row[field] = bad_value

    with pytest.raises(ValueError, match=f"'{field}'"):
        ForceOrder.from_binance(row)


# --- The numeric domain of every numeric field ------------------------------


@pytest.mark.parametrize(
    "bad_price",
    ["nan", "NaN", "inf", "-inf", "Infinity", float("nan"), float("inf"), 10**400],
)
def test_non_finite_or_overflowing_avg_price_fails_closed(bad_price: object) -> None:
    """Every numeric field must parse to a FINITE float; overflow is a ValueError.

    A NaN or infinite price would poison every downstream comparison, and an int
    too large for a float raises OverflowError — the ruler requires ValueError
    naming the field for both.
    """
    row = _row(order_id=1, update_time=1_000)
    row["avgPrice"] = bad_price

    with pytest.raises(ValueError, match="'avgPrice'"):
        ForceOrder.from_binance(row)


@pytest.mark.parametrize(
    "bad_qty",
    ["nan", "inf", "-inf", float("nan"), float("-inf"), 10**400],
)
def test_non_finite_or_overflowing_executed_qty_fails_closed(bad_qty: object) -> None:
    """ExecutedQty shares the finite-float domain and fails closed."""
    row = _row(order_id=1, update_time=1_000)
    row["executedQty"] = bad_qty

    with pytest.raises(ValueError, match="'executedQty'"):
        ForceOrder.from_binance(row)


@pytest.mark.parametrize("negative_qty", ["-0.002", "-1", -0.002, -1])
def test_negative_executed_qty_fails_closed(negative_qty: object) -> None:
    """A negative executedQty is not a real fill quantity and fails closed."""
    row = _row(order_id=1, update_time=1_000)
    row["executedQty"] = negative_qty

    with pytest.raises(ValueError, match="'executedQty'"):
        ForceOrder.from_binance(row)


# --- S2: sign_request failures are surfaced, never swallowed -----------------------


class _SignRequestFailure(RuntimeError):
    """Stands in for NT's HTTP layer failing (4xx/5xx, decode or transport)."""


async def test_sign_request_failure_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact sign_request exception must reach the caller (identity, not a copy).

    A fetch that swallowed or rewrapped the error would leave the P3a2 watcher's
    per-symbol failure counters (R52) blind to a real HTTP failure.
    """
    error = _SignRequestFailure("forceOrders HTTP 500")
    _patch_factory(monkeypatch, [], error=error)
    fetch = build_binance_fetch("key", "secret", "USDT_FUTURES", testnet=True)

    with pytest.raises(_SignRequestFailure) as excinfo:
        await fetch("BTCUSDT", 1, 2)

    assert excinfo.value is error
