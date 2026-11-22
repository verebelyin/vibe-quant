"""Venue leverage/margin check: read, classify and publish (bead vibe-quant-68m53, T1).

NautilusTrader sets each symbol's leverage and margin type at connect on a best-effort
basis and reads ``/fapi/v1/symbolConfig`` once; the user can change both later in the
Binance app. The trading guard prices every entry from the leverage cached on NT's
``MarginAccount``. This module is the pure half of the paper node's venue check:

* :func:`build_symbol_config_read` binds credentials to a read-only signed
  ``GET /fapi/v1/symbolConfig`` (USD-M only);
* :func:`classify_symbol_configs` compares each session symbol's venue leverage and
  margin type with the resolved run config, per symbol and per field (F3, R12);
* :func:`build_publish_map` turns the strategy configs' instrument ids (strings in
  production) into typed ``InstrumentId``s (R1);
* :func:`publish_leverage` writes the verified leverage onto the exact NT account the
  guard reads, then reads it back (R1);
* :func:`build_position_risk_read` and :func:`parse_positions` (T4) read the account's
  open positions (``GET /fapi/v3/positionRisk``) for the unknown-position check;
* :func:`build_open_orders_read` and :func:`parse_open_orders` (opspt P2) read every open
  order itself (``GET /fapi/v1/openOrders`` and ``/fapi/v1/openAlgoOrders``): NT's cache
  silently misses algo orders when its own read fails, so it cannot prove no hand order
  rests on a bot coin.

T4 scope (R10): the position check detects nonzero exposure on symbols outside this
session's configured set, and a venue amount that disagrees with the session's
reconciled NT position. Where venue and NT report the same amount, the position's
origin is unknown (including exposure NT adopted at start-up); the local account lock
is per API key and cannot attribute a position.

Fail closed: any value this module cannot read with certainty is "unverified" (never
a match, never a confirmed mismatch). Only a valid reading that differs from the
resolved value is a mismatch. No NT import at module level (like ``force_orders``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nautilus_trader.model.identifiers import InstrumentId

logger = logging.getLogger(__name__)

#: Binance USD-M per-symbol leverage/margin config endpoint (request weight 5).
SYMBOL_CONFIG_PATH: Final = "/fapi/v1/symbolConfig"

#: The only account type whose endpoints live under ``/fapi/`` (F10).
_USDT_FUTURES: Final = "USDT_FUTURES"

#: Binance USD-M leverage bounds; 0 is a valid (differing) reading (V5b).
_MAX_LEVERAGE: Final = 125
_LEVERAGE_DIGITS: Final = re.compile(r"[0-9]{1,3}")

_MARGIN_TYPES: Final = frozenset({"ISOLATED", "CROSSED"})

#: Binance USD-M position information endpoint (v3, request weight 5).
POSITION_RISK_PATH: Final = "/fapi/v3/positionRisk"

_POSITION_SIDES: Final = frozenset({"BOTH", "LONG", "SHORT"})

#: ``positionAmt``: a signed plain decimal in ASCII digits (no exponent, NaN or Inf).
_AMOUNT: Final = re.compile(r"-?[0-9]{1,20}(\.[0-9]{1,20})?")

#: Binance USD-M open regular orders (request weight 1 per symbol, 40 without ``symbol``).
OPEN_ORDERS_PATH: Final = "/fapi/v1/openOrders"
#: Binance USD-M open algo (conditional: stop, take-profit) orders (weight 1 with
#: ``symbol``, 40 without: github.com/binance/binance-connector-python, trade_api.py
#: ``current_all_algo_open_orders``).
OPEN_ALGO_ORDERS_PATH: Final = "/fapi/v1/openAlgoOrders"

#: HTTP statuses Binance answers when it rate-limits (429) or bans (418) the IP.
RATE_LIMIT_STATUSES: Final = frozenset({418, 429})

#: ``read() -> decoded JSON body`` of one signed symbolConfig (or positionRisk) request.
SymbolConfigRead = Callable[[], Awaitable[object]]
#: ``read() -> (openOrders body, openAlgoOrders body)``, each read per session symbol; a
#: failed endpoint's slot holds its exception (the other endpoint is still read).
OpenOrdersRead = Callable[[], Awaitable[tuple[object, object]]]

#: Telegram rejects a message over 4,096 characters (after entity parsing); the venue
#: detail in an alert is cut to this, leaving room for the prefix and the fixed text.
ALERT_DETAIL_MAX_CHARS: Final = 3_000
#: Room kept for the `` … (+N more; full text in status)`` suffix.
_ALERT_SUFFIX_ROOM: Final = 48


def bounded_alert_detail(text: str, limit: int = ALERT_DETAIL_MAX_CHARS) -> str:
    """``text`` as is when it fits ``limit``; else a sample of whole items and the count
    left out (cut at the last ``"; "`` or ``", "``). The full text stays in status and
    the event."""
    if len(text) <= limit:
        return text
    head = text[: limit - _ALERT_SUFFIX_ROOM]
    cut, sep = max((head.rfind(sep), sep) for sep in ("; ", ", "))
    if cut <= 0:
        return f"{head} … (cut; full text in status)"
    return f"{text[:cut]} … (+{text[cut:].count(sep)} more; full text in status)"


@dataclass(frozen=True)
class ConfigVerdict:
    """The comparison of the venue's symbolConfig answer with the resolved config.

    Attributes:
        observed: Per session symbol with exactly one row: ``(leverage, margin)``,
            each ``None`` when that field is unverified.
        mismatches: Per symbol, one string naming every valid field that differs.
        unverified: Per symbol, why it (or one of its fields) could not be read.
            A symbol may appear in both ``mismatches`` and ``unverified`` (R12).
    """

    observed: dict[str, tuple[int | None, str | None]]
    mismatches: dict[str, str]
    unverified: dict[str, str]


def build_symbol_config_read(
    api_key: str,
    api_secret: str,
    account_type: str,
    testnet: bool,
) -> SymbolConfigRead:
    """Bind credentials to a read-only signed ``GET /fapi/v1/symbolConfig``.

    Each call signs one request with ``recvWindow='5000'`` and a fresh ``timestamp``
    and returns the JSON-decoded body (a decode error propagates as a failed read).

    Raises:
        ValueError: If ``account_type`` is not ``USDT_FUTURES`` (F10): the endpoint
            exists for USD-M futures only.
    """
    return _build_signed_read(SYMBOL_CONFIG_PATH, api_key, api_secret, account_type, testnet)


def build_position_risk_read(
    api_key: str,
    api_secret: str,
    account_type: str,
    testnet: bool,
) -> SymbolConfigRead:
    """Bind credentials to a read-only signed ``GET /fapi/v3/positionRisk`` (all symbols).

    Same request shape and failure rules as :func:`build_symbol_config_read`.
    """
    return _build_signed_read(POSITION_RISK_PATH, api_key, api_secret, account_type, testnet)


def build_open_orders_read(
    api_key: str,
    api_secret: str,
    account_type: str,
    testnet: bool,
    symbols: Iterable[str],
) -> OpenOrdersRead:
    """Bind credentials to the read-only signed ``GET /fapi/v1/openOrders`` and
    ``GET /fapi/v1/openAlgoOrders``, one request per ``symbols`` entry (weight 1 each),
    read concurrently.

    Same request shape and failure rules as :func:`build_symbol_config_read`.
    """
    client, clock = _signed_client(api_key, api_secret, account_type, testnet)
    return open_orders_read(client, clock, symbols)


@dataclass(frozen=True)
class RateLimit:
    """Binance's HTTP 429 (rate limit) or 418 (IP ban) answer.

    ``retry_after_s`` is the ``Retry-After`` header in seconds (None = absent/unusable).
    """

    status: int
    retry_after_s: float | None


def rate_limit_of(error: BaseException) -> RateLimit | None:
    """The :class:`RateLimit` ``error`` is (NT's ``BinanceError``: ``status`` + ``headers``),
    else None. Never raises."""
    status = getattr(error, "status", None)
    if isinstance(status, bool) or status not in RATE_LIMIT_STATUSES:
        return None
    retry_after: float | None = None
    headers = getattr(error, "headers", None)
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if isinstance(key, str) and key.lower() == "retry-after":
                try:
                    seconds = float(value)
                except (TypeError, ValueError):
                    break
                retry_after = seconds if math.isfinite(seconds) and seconds > 0 else None
                break
    return RateLimit(int(status), retry_after)


class OpenOrdersReadError(Exception):
    """An endpoint's per-symbol reads of which some failed: the rows the others returned
    stay in ``rows`` (a foreign id among them still holds); ``cause`` is the failure."""

    def __init__(self, cause: BaseException, rows: list[object]) -> None:
        super().__init__(type(cause).__name__)
        self.cause = cause
        self.rows = rows


def open_orders_read(client: Any, clock: Any, symbols: Iterable[str]) -> OpenOrdersRead:
    """Both open-order reads through ``client`` (NT's ``BinanceHttpClient``), per symbol."""
    names = sorted(set(symbols))

    async def endpoint(path: str) -> object:
        reads = [_signed_read(client, clock, path, symbol)() for symbol in names]
        results = await asyncio.gather(*reads, return_exceptions=True)
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                raise result
        rows: list[object] = []
        failures = [r for r in results if isinstance(r, BaseException)]
        stray = next((r for r in results if not isinstance(r, (list, BaseException))), None)
        for result in results:
            if isinstance(result, list):
                rows.extend(result)
        if failures:
            # A rate-limit answer is the failure to report (it drives the back-off).
            limited = next((f for f in failures if rate_limit_of(f) is not None), failures[0])
            return OpenOrdersReadError(limited, rows)
        return rows if stray is None else stray

    async def read() -> tuple[object, object]:
        first, second = await asyncio.gather(
            endpoint(OPEN_ORDERS_PATH), endpoint(OPEN_ALGO_ORDERS_PATH)
        )
        return first, second

    return read


def _build_signed_read(
    path: str,
    api_key: str,
    api_secret: str,
    account_type: str,
    testnet: bool,
) -> SymbolConfigRead:
    client, clock = _signed_client(api_key, api_secret, account_type, testnet)
    return _signed_read(client, clock, path)


def _signed_client(
    api_key: str,
    api_secret: str,
    account_type: str,
    testnet: bool,
) -> tuple[Any, Any]:
    """NT's cached Binance USD-M HTTP client for these credentials, and a clock."""
    if account_type != _USDT_FUTURES:
        raise ValueError(
            f"venue config check supports account_type {_USDT_FUTURES} only, got {account_type!r}"
        )
    from nautilus_trader.adapters.binance.common.enums import BinanceAccountType, BinanceEnvironment
    from nautilus_trader.adapters.binance.factories import get_cached_binance_http_client
    from nautilus_trader.common.component import LiveClock

    clock = LiveClock()
    client = get_cached_binance_http_client(
        clock=clock,
        account_type=BinanceAccountType.USDT_FUTURES,
        api_key=api_key,
        api_secret=api_secret,
        environment=BinanceEnvironment.TESTNET if testnet else BinanceEnvironment.LIVE,
    )
    return client, clock


def _signed_read(client: Any, clock: Any, path: str, symbol: str | None = None) -> SymbolConfigRead:
    """One signed ``GET path`` (for ``symbol`` when given) with a fresh ``timestamp``:
    the JSON-decoded body."""
    from nautilus_trader.core.nautilus_pyo3 import HttpMethod

    async def read() -> object:
        payload = {"recvWindow": "5000", "timestamp": str(clock.timestamp_ms())}
        if symbol is not None:
            payload["symbol"] = symbol
        body = await client.sign_request(HttpMethod.GET, path, payload)
        decoded: object = json.loads(body)
        return decoded

    return read


#: Longest echo of an input in a diagnostic; longer text is cut and its length named.
_ECHO_LIMIT = 80


def _bounded(text: str) -> str:
    """``text`` up to ``_ECHO_LIMIT`` characters, else cut with its full length."""
    if len(text) <= _ECHO_LIMIT:
        return text
    return f"{text[:_ECHO_LIMIT]}...({len(text)} chars)"


def _safe_repr(value: object) -> str:
    """Bounded ``repr`` that never raises for a JSON value (an int beyond the
    str-conversion digit limit raises ValueError)."""
    try:
        return _bounded(repr(value))
    except ValueError:
        return f"<unprintable {type(value).__name__}>"


def _parse_leverage(value: object) -> int | None:
    """An ``int`` (not ``bool``) or a 1-3 ASCII-digit string, within 0..125; else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and _LEVERAGE_DIGITS.fullmatch(value):
        number = int(value)
    else:
        return None
    if 0 <= number <= _MAX_LEVERAGE:
        return number
    return None


def _parse_margin(value: object) -> str | None:
    """ISOLATED or CROSSED (ASCII, any case; ``CROSS`` means CROSSED); else None."""
    if not isinstance(value, str) or not value.isascii():
        return None
    margin = value.upper()
    if margin == "CROSS":
        margin = "CROSSED"
    return margin if margin in _MARGIN_TYPES else None


def classify_symbol_configs(
    body: object,
    symbols: Iterable[str],
    leverage: int,
    margin_type: str,
) -> ConfigVerdict:
    """Compare each session symbol's venue config with ``leverage``/``margin_type``.

    Per symbol (F3): 0 rows -> unverified ``"missing from symbolConfig"``; 2+ rows ->
    unverified ``"N rows"``. On a unique row each field is classified on its own
    (R12): a valid field that differs is a mismatch even when its sibling is
    unverified. Rows that are not objects or carry a non-string ``symbol`` are
    ignored, so a garbled session row reads as missing. Never raises for any
    JSON-decoded ``body``.
    """
    session = list(dict.fromkeys(symbols))
    observed: dict[str, tuple[int | None, str | None]] = {}
    mismatches: dict[str, str] = {}
    unverified: dict[str, str] = {}
    if not isinstance(body, list):
        why = f"answer is not a list: {type(body).__name__}"
        return ConfigVerdict(observed, mismatches, dict.fromkeys(session, why))

    rows: dict[str, list[dict[object, object]]] = {symbol: [] for symbol in session}
    for row in body:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        if isinstance(symbol, str) and symbol in rows:
            rows[symbol].append(row)

    for symbol in session:
        found = rows[symbol]
        if not found:
            unverified[symbol] = "missing from symbolConfig"
            continue
        if len(found) > 1:
            unverified[symbol] = f"{len(found)} rows"
            continue
        row = found[0]
        raw_leverage = row.get("leverage")
        raw_margin = row.get("marginType")
        venue_leverage = _parse_leverage(raw_leverage)
        venue_margin = _parse_margin(raw_margin)
        observed[symbol] = (venue_leverage, venue_margin)

        differs: list[str] = []
        unread: list[str] = []
        if venue_leverage is None:
            unread.append(
                f"leverage is not a whole number in 0..{_MAX_LEVERAGE}: {_safe_repr(raw_leverage)}"
            )
        elif venue_leverage != leverage:
            differs.append(f"venue leverage {venue_leverage}x != resolved {leverage}x")
        if venue_margin is None:
            unread.append(f"marginType is not ISOLATED/CROSSED: {_safe_repr(raw_margin)}")
        elif venue_margin != margin_type:
            differs.append(f"venue margin {venue_margin} != resolved {margin_type}")
        if differs:
            mismatches[symbol] = f"{symbol}: " + "; ".join(differs)
        if unread:
            unverified[symbol] = "; ".join(unread)

    return ConfigVerdict(observed, mismatches, unverified)


def build_publish_map(
    symbols: Iterable[str],
    configured_ids: Mapping[str, object],
    venue: str,
) -> dict[str, InstrumentId] | str:
    """Map each resolved symbol to the typed ``InstrumentId`` its strategy trades (R1).

    ``configured_ids`` values may be strings (the compiled strategy config declares
    ``instrument_id: str``) or ``InstrumentId``s. Each must parse, be on ``venue`` and
    name ``symbol`` (``<SYMBOL>-...``). Returns the map, or one diagnostic string
    naming every symbol without a valid id (``"<SYM>: no valid instrument id (...)"``).
    """
    from nautilus_trader.model.identifiers import InstrumentId

    mapped: dict[str, InstrumentId] = {}
    problems: list[str] = []
    for symbol in dict.fromkeys(symbols):
        raw = configured_ids.get(symbol)
        if raw is None:
            problems.append(f"{symbol}: no valid instrument id (not configured)")
            continue
        text = str(raw)
        try:
            iid = InstrumentId.from_str(text)
        except ValueError:
            problems.append(f"{symbol}: no valid instrument id (unparsable {_safe_repr(text)})")
            continue
        if iid.venue.value != venue:
            problems.append(
                f"{symbol}: no valid instrument id ({_bounded(str(iid))} is not on {venue})"
            )
            continue
        if iid.symbol.value.split("-")[0] != symbol:
            problems.append(
                f"{symbol}: no valid instrument id "
                f"({_bounded(str(iid))} is not a {symbol} contract)"
            )
            continue
        mapped[symbol] = iid
    if problems:
        return "; ".join(problems)
    return mapped


def publish_leverage(
    account: object,
    iids: Mapping[str, InstrumentId],
    leverage: int,
) -> str | None:
    """Set ``leverage`` on ``account`` for every instrument and read it back (R1).

    ``account`` is the NT ``MarginAccount`` the trading guard reads
    (``portfolio.account(venue)``); NT's setter mutates that cached object
    synchronously. Returns None when every read-back equals ``leverage``, else the
    first diagnostic (exception class only, never its text).
    """
    from decimal import Decimal

    if account is None:
        return "no venue account in the NT cache"
    target = Decimal(leverage)
    for symbol, iid in iids.items():
        try:
            account.set_leverage(iid, target)  # type: ignore[attr-defined]
            seen = account.leverage(iid)  # type: ignore[attr-defined]
        except Exception as exc:
            # Class only: NT's message may carry account data. The caller keeps entries
            # blocked on the returned diagnostic.
            logger.warning("venue leverage publish failed for %s (%s)", symbol, type(exc).__name__)
            return f"{symbol}: leverage publish failed ({type(exc).__name__})"
        if seen != target:
            return f"{symbol}: guard reads leverage {seen} after publishing {leverage}"
    return None


@dataclass(frozen=True)
class VenuePosition:
    """One nonzero venue position (side as the venue sent it: BOTH, LONG or SHORT)."""

    symbol: str
    side: str
    amount: Decimal

    def describe(self) -> str:
        """``SOLUSDT 3.5 (not traded by this session)`` or a hedge-mode row."""
        if self.side == "BOTH":
            return f"{self.symbol} {self.amount} (not traded by this session)"
        return f"{self.symbol} {self.side} {self.amount} (hedge-mode row)"


@dataclass(frozen=True)
class PositionReading:
    """The venue's positionRisk answer, read against this session's symbols (T4).

    Attributes:
        amounts: Per session symbol read with certainty: its one-way (BOTH) amount;
            a symbol the venue omits is flat (0).
        unknown: Nonzero positions this session does not trade: a BOTH row on
            another symbol, or any LONG/SHORT row (NT refuses hedge mode).
        unverified: Per symbol (``"answer"`` for the whole body) why it could not be
            read. A session symbol listed here has no entry in ``amounts``.
    """

    amounts: dict[str, Decimal]
    unknown: list[VenuePosition]
    unverified: dict[str, str]
    #: Both open-order reads of the same cycle (opspt P2); None = not read: the check
    #: cannot pass on it (fail closed).
    orders: OpenOrders | None = None


def _parse_amount(value: object) -> Decimal | None:
    """A plain ASCII decimal string (Binance sends ``positionAmt`` as ``"-0.010"``);
    else None (NaN, Inf, exponents, numbers, other digits: fail closed)."""
    if not isinstance(value, str) or _AMOUNT.fullmatch(value) is None:
        return None
    return Decimal(value)


def parse_positions(body: object, symbols: Iterable[str]) -> PositionReading:
    """Read a positionRisk answer against the session's ``symbols`` (U3, U4).

    Fail closed: a body that is not a list, or a row that cannot be attributed to a
    symbol (not an object, no string symbol), leaves every session symbol unverified.
    A row with a bad ``positionSide`` or ``positionAmt`` leaves its symbol
    unverified; 2+ BOTH rows for one symbol too. ``"-0.000"`` is flat. Never raises
    for any JSON-decoded ``body``.
    """
    session = list(dict.fromkeys(symbols))
    unverified: dict[str, str] = {}
    if not isinstance(body, list):
        why = f"answer is not a list: {type(body).__name__}"
        return PositionReading({}, [], {"answer": why, **dict.fromkeys(session, why)})

    unknown: list[VenuePosition] = []
    both: dict[str, list[Decimal]] = {}
    for index, row in enumerate(body):
        symbol = row.get("symbol") if isinstance(row, dict) else None
        if not isinstance(symbol, str) or not symbol:
            unverified["answer"] = f"row {index} names no symbol: {_safe_repr(row)}"
            continue
        side = row.get("positionSide")
        if not isinstance(side, str) or side not in _POSITION_SIDES:
            unverified[symbol] = f"positionSide is not BOTH/LONG/SHORT: {_safe_repr(side)}"
            continue
        raw = row.get("positionAmt")
        amount = _parse_amount(raw)
        if amount is None:
            unverified[symbol] = f"{side} positionAmt is not a finite decimal: {_safe_repr(raw)}"
            continue
        if side == "BOTH":
            both.setdefault(symbol, []).append(amount)
        elif amount != 0:
            unknown.append(VenuePosition(symbol, side, amount))

    amounts: dict[str, Decimal] = {}
    for symbol, found in both.items():
        if symbol in unverified:
            continue
        if len(found) > 1:
            unverified[symbol] = f"{len(found)} BOTH rows"
        elif symbol in session:
            amounts[symbol] = found[0]
        elif found[0] != 0:
            unknown.append(VenuePosition(symbol, "BOTH", found[0]))
    if "answer" in unverified:
        # An unattributable row may be any session symbol's: none is read with certainty.
        for symbol in session:
            unverified.setdefault(symbol, unverified["answer"])
        amounts.clear()
    for symbol in session:
        if symbol not in unverified:
            amounts.setdefault(symbol, Decimal(0))
    unknown.sort(key=lambda p: (p.symbol, p.side))
    return PositionReading(amounts, unknown, unverified)


@dataclass(frozen=True)
class RestingOrder:
    """One open order Binance reports on a session symbol.

    ``quantity`` is ``"whole position"`` for a closePosition order (it has none);
    ``side`` and ``order_type`` are display text (``"?"`` when unreadable).
    """

    symbol: str
    client_id: str
    side: str
    quantity: str
    order_type: str


@dataclass(frozen=True)
class OpenOrders:
    """Both open-order reads against the session's symbols (opspt P2, ruling #8022).

    Attributes:
        orders: Every readable open order on a session symbol.
        unverified: Why the reads do not show every open order on the session's
            symbols with certainty (a failed endpoint, a body that is not a list, a row
            that names no symbol or, on a session symbol, no client id); None = complete.
        rate_limit: Binance answered a read with HTTP 429/418 (the node backs off).
        unreadable: Session symbols with a row the bot cannot read (``""``: a row that
            names no symbol), for the user's hand action (#8044 R-W3).
    """

    orders: list[RestingOrder]
    unverified: str | None
    rate_limit: RateLimit | None = None
    unreadable: tuple[str, ...] = ()


#: Per endpoint: the client id, quantity and order type fields of its rows.
_ORDER_FIELDS: Final = {
    "openOrders": ("clientOrderId", "origQty", "type"),
    "openAlgoOrders": ("clientAlgoId", "quantity", "orderType"),
}


def _text(value: object) -> str:
    return _bounded(value) if isinstance(value, str) and value else "?"


def _read_failure(name: str, error: BaseException) -> str:
    """Class only: a fetch error can carry the signed URL."""
    limit = rate_limit_of(error)
    if limit is not None:
        return f"{name} rate-limited by Binance (HTTP {limit.status})"
    return f"{name} read failed ({type(error).__name__})"


def parse_open_orders(regular: object, algo: object, symbols: Iterable[str]) -> OpenOrders:
    """Read both open-order answers (or the exception a failed endpoint raised) against
    the session's ``symbols``. Never raises for JSON-decoded bodies.

    Fail closed: a failed endpoint, a body that is not a list, a row that names no
    symbol, or a session symbol's row with no client id leaves the reading unverified
    (rows read before and after it still count: a foreign id among them holds).
    """
    session = set(symbols)
    orders: list[RestingOrder] = []
    problems: list[str] = []
    unreadable: list[str] = []
    limit: RateLimit | None = None
    for name, body in (("openOrders", regular), ("openAlgoOrders", algo)):
        id_key, qty_key, type_key = _ORDER_FIELDS[name]
        if isinstance(body, OpenOrdersReadError):
            # Some symbols read, some not: the rows read still count.
            problems.append(_read_failure(name, body.cause))
            limit = limit or rate_limit_of(body.cause)
            body = body.rows
        elif isinstance(body, BaseException):
            problems.append(_read_failure(name, body))
            limit = limit or rate_limit_of(body)
            continue
        if not isinstance(body, list):
            problems.append(f"{name} answer is not a list: {type(body).__name__}")
            continue
        for index, row in enumerate(body):
            symbol = row.get("symbol") if isinstance(row, dict) else None
            if not isinstance(symbol, str) or not symbol:
                problems.append(f"{name} row {index} names no symbol: {_safe_repr(row)}")
                unreadable.append("")
                continue
            if symbol not in session:
                continue
            cid = row.get(id_key)
            if not isinstance(cid, str) or not cid:
                problems.append(f"{name} {symbol} row {index} has no {id_key}: {_safe_repr(cid)}")
                unreadable.append(symbol)
                continue
            whole = row.get("closePosition") is True
            quantity = "whole position" if whole else _text(row.get(qty_key))
            orders.append(
                RestingOrder(
                    symbol, cid, _text(row.get("side")), quantity, _text(row.get(type_key))
                )
            )
    return OpenOrders(orders, "; ".join(problems) or None, limit, tuple(dict.fromkeys(unreadable)))


__all__ = [
    "OPEN_ALGO_ORDERS_PATH",
    "OPEN_ORDERS_PATH",
    "POSITION_RISK_PATH",
    "SYMBOL_CONFIG_PATH",
    "ConfigVerdict",
    "OpenOrders",
    "OpenOrdersRead",
    "OpenOrdersReadError",
    "PositionReading",
    "RateLimit",
    "RestingOrder",
    "SymbolConfigRead",
    "VenuePosition",
    "build_open_orders_read",
    "build_position_risk_read",
    "build_publish_map",
    "build_symbol_config_read",
    "classify_symbol_configs",
    "open_orders_read",
    "parse_open_orders",
    "parse_positions",
    "publish_leverage",
    "rate_limit_of",
]
