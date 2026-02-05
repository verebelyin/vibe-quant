"""Ethereal DEX data client for real-time market data.

Connects to Ethereal WebSocket API via Socket.IO protocol to receive:
- BookDepth: Order book updates
- MarketPrice: Last trade price
- FundingRate: Funding rate updates (hourly)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

import socketio

logger = logging.getLogger(__name__)


class EtherealChannel(StrEnum):
    """Ethereal WebSocket subscription channels."""

    BOOK_DEPTH = "BookDepth"
    MARKET_PRICE = "MarketPrice"
    FUNDING_RATE = "FundingRate"


# Ethereal WebSocket endpoints
ETHEREAL_TESTNET_WS = "wss://ws.etherealtest.net"
ETHEREAL_MAINNET_WS = "wss://ws.ethereal.trade"

# Symbol mappings: Ethereal format -> normalized format
SYMBOL_MAP = {
    "BTC-USD": "BTCUSD",
    "ETH-USD": "ETHUSD",
    "SOL-USD": "SOLUSD",
}

# Reverse mapping
SYMBOL_MAP_REVERSE = {v: k for k, v in SYMBOL_MAP.items()}


@dataclass
class BookDepthUpdate:
    """Normalized order book depth update.

    Attributes:
        symbol: Normalized symbol (e.g., BTCUSD).
        timestamp: Update timestamp.
        bids: List of (price, size) tuples.
        asks: List of (price, size) tuples.
    """

    symbol: str
    timestamp: datetime
    bids: list[tuple[Decimal, Decimal]]
    asks: list[tuple[Decimal, Decimal]]


@dataclass
class MarketPriceUpdate:
    """Normalized market price (last trade) update.

    Attributes:
        symbol: Normalized symbol.
        timestamp: Update timestamp.
        price: Last trade price.
        size: Last trade size.
    """

    symbol: str
    timestamp: datetime
    price: Decimal
    size: Decimal


@dataclass
class FundingRateUpdate:
    """Normalized funding rate update.

    Attributes:
        symbol: Normalized symbol.
        timestamp: Update timestamp.
        funding_rate: Current funding rate.
        mark_price: Current mark price.
        next_funding_time: Next funding settlement time.
    """

    symbol: str
    timestamp: datetime
    funding_rate: Decimal
    mark_price: Decimal
    next_funding_time: datetime | None


# Callback type aliases
BookDepthCallback = Callable[[BookDepthUpdate], None]
MarketPriceCallback = Callable[[MarketPriceUpdate], None]
FundingRateCallback = Callable[[FundingRateUpdate], None]


@dataclass
class SubscriptionState:
    """Internal state for a symbol subscription."""

    symbol: str
    channels: set[EtherealChannel] = field(default_factory=set)
    book_depth_callback: BookDepthCallback | None = None
    market_price_callback: MarketPriceCallback | None = None
    funding_rate_callback: FundingRateCallback | None = None


class EtherealDataClient:
    """Async data client for Ethereal DEX WebSocket API.

    Connects via Socket.IO protocol to receive real-time market data.
    Supports auto-reconnect on connection loss.

    Example:
        async with EtherealDataClient(testnet=True) as client:
            await client.subscribe(
                "BTCUSD",
                channels=[EtherealChannel.MARKET_PRICE],
                on_market_price=handle_price,
            )
            await asyncio.sleep(60)  # Receive updates for 60 seconds
    """

    def __init__(
        self,
        *,
        testnet: bool = True,
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
    ) -> None:
        """Initialize Ethereal data client.

        Args:
            testnet: Use testnet endpoint if True, mainnet otherwise.
            auto_reconnect: Enable auto-reconnect on disconnect.
            reconnect_delay: Delay between reconnect attempts in seconds.
            max_reconnect_attempts: Max reconnect attempts (0 for unlimited).
        """
        self._testnet = testnet
        self._ws_url = ETHEREAL_TESTNET_WS if testnet else ETHEREAL_MAINNET_WS
        self._auto_reconnect = auto_reconnect
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_attempts = max_reconnect_attempts

        self._sio: socketio.AsyncClient | None = None
        self._connected = False
        self._subscriptions: dict[str, SubscriptionState] = {}
        self._reconnect_count = 0
        self._shutdown_event = asyncio.Event()
        self._reconnect_task: asyncio.Task[None] | None = None

    @property
    def connected(self) -> bool:
        """Return True if connected to WebSocket."""
        return self._connected

    @property
    def testnet(self) -> bool:
        """Return True if using testnet endpoint."""
        return self._testnet

    async def __aenter__(self) -> EtherealDataClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to Ethereal WebSocket.

        Raises:
            ConnectionError: If connection fails.
        """
        if self._connected:
            return

        self._sio = socketio.AsyncClient(
            reconnection=False,  # We handle reconnection ourselves
            logger=False,
            engineio_logger=False,
        )

        self._register_handlers()

        try:
            logger.info("Connecting to Ethereal WebSocket: %s", self._ws_url)
            await self._sio.connect(
                self._ws_url,
                transports=["websocket"],
                wait_timeout=30,
            )
            self._connected = True
            self._reconnect_count = 0
            logger.info("Connected to Ethereal WebSocket")
        except Exception as e:
            logger.error("Failed to connect to Ethereal: %s", e)
            raise ConnectionError(f"Failed to connect to Ethereal: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from Ethereal WebSocket."""
        self._shutdown_event.set()

        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconnect_task
            self._reconnect_task = None

        if self._sio is not None and self._connected:
            try:
                await self._sio.disconnect()
            except Exception as e:
                logger.warning("Error during disconnect: %s", e)

        self._connected = False
        self._subscriptions.clear()
        logger.info("Disconnected from Ethereal WebSocket")

    def _register_handlers(self) -> None:
        """Register Socket.IO event handlers."""
        if self._sio is None:
            return

        @self._sio.on("connect")  # type: ignore[untyped-decorator]
        async def on_connect() -> None:
            logger.info("Socket.IO connected")
            # Resubscribe after reconnection
            await self._resubscribe_all()

        @self._sio.on("disconnect")  # type: ignore[untyped-decorator]
        async def on_disconnect() -> None:
            logger.warning("Socket.IO disconnected")
            self._connected = False
            if self._auto_reconnect and not self._shutdown_event.is_set():
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())

        @self._sio.on("BookDepth")  # type: ignore[untyped-decorator]
        async def on_book_depth(data: dict[str, Any]) -> None:
            await self._handle_book_depth(data)

        @self._sio.on("MarketPrice")  # type: ignore[untyped-decorator]
        async def on_market_price(data: dict[str, Any]) -> None:
            await self._handle_market_price(data)

        @self._sio.on("FundingRate")  # type: ignore[untyped-decorator]
        async def on_funding_rate(data: dict[str, Any]) -> None:
            await self._handle_funding_rate(data)

        @self._sio.on("error")  # type: ignore[untyped-decorator]
        async def on_error(data: dict[str, Any]) -> None:
            logger.error("Ethereal WebSocket error: %s", data)

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        while not self._shutdown_event.is_set():
            if (
                self._max_reconnect_attempts > 0
                and self._reconnect_count >= self._max_reconnect_attempts
            ):
                logger.error(
                    "Max reconnect attempts (%d) reached",
                    self._max_reconnect_attempts,
                )
                return

            self._reconnect_count += 1
            delay = self._reconnect_delay * (2 ** min(self._reconnect_count - 1, 5))
            logger.info(
                "Reconnecting in %.1f seconds (attempt %d)",
                delay,
                self._reconnect_count,
            )

            try:
                await asyncio.sleep(delay)
                if self._shutdown_event.is_set():
                    return

                await self.connect()
                return  # Success
            except ConnectionError:
                logger.warning("Reconnection attempt %d failed", self._reconnect_count)
                continue

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all previous subscriptions after reconnect."""
        for state in self._subscriptions.values():
            try:
                await self._send_subscribe(state.symbol, state.channels)
            except Exception as e:
                logger.error(
                    "Failed to resubscribe to %s: %s",
                    state.symbol,
                    e,
                )

    async def subscribe(
        self,
        symbol: str,
        channels: list[EtherealChannel],
        *,
        on_book_depth: BookDepthCallback | None = None,
        on_market_price: MarketPriceCallback | None = None,
        on_funding_rate: FundingRateCallback | None = None,
    ) -> None:
        """Subscribe to market data for a symbol.

        Args:
            symbol: Normalized symbol (e.g., BTCUSD).
            channels: List of channels to subscribe to.
            on_book_depth: Callback for book depth updates.
            on_market_price: Callback for market price updates.
            on_funding_rate: Callback for funding rate updates.

        Raises:
            ValueError: If symbol is not supported.
            ConnectionError: If not connected.
        """
        if not self._connected:
            raise ConnectionError("Not connected to Ethereal WebSocket")

        if symbol not in SYMBOL_MAP_REVERSE:
            raise ValueError(f"Unsupported symbol: {symbol}")

        # Create or update subscription state
        if symbol not in self._subscriptions:
            self._subscriptions[symbol] = SubscriptionState(symbol=symbol)

        state = self._subscriptions[symbol]
        state.channels.update(channels)

        if on_book_depth:
            state.book_depth_callback = on_book_depth
        if on_market_price:
            state.market_price_callback = on_market_price
        if on_funding_rate:
            state.funding_rate_callback = on_funding_rate

        await self._send_subscribe(symbol, set(channels))

    async def _send_subscribe(
        self,
        symbol: str,
        channels: set[EtherealChannel],
    ) -> None:
        """Send subscription request to server."""
        if self._sio is None:
            return

        ethereal_symbol = SYMBOL_MAP_REVERSE[symbol]

        for channel in channels:
            await self._sio.emit(
                "subscribe",
                {"channel": channel.value, "symbol": ethereal_symbol},
            )
            logger.debug("Subscribed to %s:%s", channel.value, ethereal_symbol)

    async def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from all channels for a symbol.

        Args:
            symbol: Normalized symbol to unsubscribe.
        """
        if symbol not in self._subscriptions:
            return

        state = self._subscriptions.pop(symbol)

        if self._sio is None or not self._connected:
            return

        ethereal_symbol = SYMBOL_MAP_REVERSE.get(symbol)
        if ethereal_symbol is None:
            return

        for channel in state.channels:
            try:
                await self._sio.emit(
                    "unsubscribe",
                    {"channel": channel.value, "symbol": ethereal_symbol},
                )
                logger.debug("Unsubscribed from %s:%s", channel.value, ethereal_symbol)
            except Exception as e:
                logger.warning("Error unsubscribing from %s: %s", channel.value, e)

    async def _handle_book_depth(self, data: dict[str, Any]) -> None:
        """Handle BookDepth message."""
        try:
            ethereal_symbol = data.get("symbol", "")
            symbol = SYMBOL_MAP.get(ethereal_symbol)

            if symbol is None or symbol not in self._subscriptions:
                return

            state = self._subscriptions[symbol]
            if state.book_depth_callback is None:
                return

            update = self._normalize_book_depth(data, symbol)
            state.book_depth_callback(update)
        except Exception as e:
            logger.error("Error handling BookDepth: %s", e)

    async def _handle_market_price(self, data: dict[str, Any]) -> None:
        """Handle MarketPrice message."""
        try:
            ethereal_symbol = data.get("symbol", "")
            symbol = SYMBOL_MAP.get(ethereal_symbol)

            if symbol is None or symbol not in self._subscriptions:
                return

            state = self._subscriptions[symbol]
            if state.market_price_callback is None:
                return

            update = self._normalize_market_price(data, symbol)
            state.market_price_callback(update)
        except Exception as e:
            logger.error("Error handling MarketPrice: %s", e)

    async def _handle_funding_rate(self, data: dict[str, Any]) -> None:
        """Handle FundingRate message."""
        try:
            ethereal_symbol = data.get("symbol", "")
            symbol = SYMBOL_MAP.get(ethereal_symbol)

            if symbol is None or symbol not in self._subscriptions:
                return

            state = self._subscriptions[symbol]
            if state.funding_rate_callback is None:
                return

            update = self._normalize_funding_rate(data, symbol)
            state.funding_rate_callback(update)
        except Exception as e:
            logger.error("Error handling FundingRate: %s", e)

    def _normalize_book_depth(
        self,
        data: dict[str, Any],
        symbol: str,
    ) -> BookDepthUpdate:
        """Normalize BookDepth message to BookDepthUpdate.

        Expected Ethereal format:
        {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,
            "bids": [["45000.0", "1.5"], ...],
            "asks": [["45100.0", "2.0"], ...]
        }
        """
        timestamp_ms = data.get("timestamp", 0)
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

        bids = [
            (Decimal(str(b[0])), Decimal(str(b[1])))
            for b in data.get("bids", [])
        ]
        asks = [
            (Decimal(str(a[0])), Decimal(str(a[1])))
            for a in data.get("asks", [])
        ]

        return BookDepthUpdate(
            symbol=symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
        )

    def _normalize_market_price(
        self,
        data: dict[str, Any],
        symbol: str,
    ) -> MarketPriceUpdate:
        """Normalize MarketPrice message to MarketPriceUpdate.

        Expected Ethereal format:
        {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,
            "price": "45050.5",
            "size": "0.1"
        }
        """
        timestamp_ms = data.get("timestamp", 0)
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

        return MarketPriceUpdate(
            symbol=symbol,
            timestamp=timestamp,
            price=Decimal(str(data.get("price", "0"))),
            size=Decimal(str(data.get("size", "0"))),
        )

    def _normalize_funding_rate(
        self,
        data: dict[str, Any],
        symbol: str,
    ) -> FundingRateUpdate:
        """Normalize FundingRate message to FundingRateUpdate.

        Expected Ethereal format:
        {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,
            "fundingRate": "0.0001",
            "markPrice": "45050.5",
            "nextFundingTime": 1706749200000
        }
        """
        timestamp_ms = data.get("timestamp", 0)
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

        next_funding_ms = data.get("nextFundingTime")
        next_funding_time = None
        if next_funding_ms:
            next_funding_time = datetime.fromtimestamp(next_funding_ms / 1000, tz=UTC)

        return FundingRateUpdate(
            symbol=symbol,
            timestamp=timestamp,
            funding_rate=Decimal(str(data.get("fundingRate", "0"))),
            mark_price=Decimal(str(data.get("markPrice", "0"))),
            next_funding_time=next_funding_time,
        )


def normalize_symbol(ethereal_symbol: str) -> str | None:
    """Convert Ethereal symbol format to normalized format.

    Args:
        ethereal_symbol: Ethereal format symbol (e.g., "BTC-USD").

    Returns:
        Normalized symbol (e.g., "BTCUSD") or None if not found.
    """
    return SYMBOL_MAP.get(ethereal_symbol)


def to_ethereal_symbol(normalized_symbol: str) -> str | None:
    """Convert normalized symbol to Ethereal format.

    Args:
        normalized_symbol: Normalized symbol (e.g., "BTCUSD").

    Returns:
        Ethereal format symbol (e.g., "BTC-USD") or None if not found.
    """
    return SYMBOL_MAP_REVERSE.get(normalized_symbol)
