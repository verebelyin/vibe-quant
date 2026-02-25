"""Tests for Ethereal data client."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe_quant.ethereal.data_client import (
    ETHEREAL_MAINNET_WS,
    ETHEREAL_TESTNET_WS,
    SYMBOL_MAP,
    SYMBOL_MAP_REVERSE,
    BookDepthUpdate,
    EtherealChannel,
    EtherealDataClient,
    FundingRateUpdate,
    MarketPriceUpdate,
    normalize_symbol,
    to_ethereal_symbol,
)


class TestSymbolMappings:
    """Tests for symbol mapping functions."""

    def test_symbol_map_contains_expected_symbols(self) -> None:
        """SYMBOL_MAP contains expected Ethereal -> normalized mappings."""
        assert "BTC-USD" in SYMBOL_MAP
        assert "ETH-USD" in SYMBOL_MAP
        assert "SOL-USD" in SYMBOL_MAP
        assert SYMBOL_MAP["BTC-USD"] == "BTCUSD"
        assert SYMBOL_MAP["ETH-USD"] == "ETHUSD"
        assert SYMBOL_MAP["SOL-USD"] == "SOLUSD"

    def test_symbol_map_reverse_contains_expected_symbols(self) -> None:
        """SYMBOL_MAP_REVERSE contains expected normalized -> Ethereal mappings."""
        assert "BTCUSD" in SYMBOL_MAP_REVERSE
        assert "ETHUSD" in SYMBOL_MAP_REVERSE
        assert "SOLUSD" in SYMBOL_MAP_REVERSE
        assert SYMBOL_MAP_REVERSE["BTCUSD"] == "BTC-USD"

    def test_normalize_symbol_valid(self) -> None:
        """normalize_symbol converts Ethereal format to normalized."""
        assert normalize_symbol("BTC-USD") == "BTCUSD"
        assert normalize_symbol("ETH-USD") == "ETHUSD"

    def test_normalize_symbol_invalid(self) -> None:
        """normalize_symbol returns None for unknown symbols."""
        assert normalize_symbol("UNKNOWN-SYMBOL") is None

    def test_to_ethereal_symbol_valid(self) -> None:
        """to_ethereal_symbol converts normalized to Ethereal format."""
        assert to_ethereal_symbol("BTCUSD") == "BTC-USD"
        assert to_ethereal_symbol("ETHUSD") == "ETH-USD"

    def test_to_ethereal_symbol_invalid(self) -> None:
        """to_ethereal_symbol returns None for unknown symbols."""
        assert to_ethereal_symbol("INVALID") is None


class TestEtherealChannel:
    """Tests for EtherealChannel enum."""

    def test_channel_values(self) -> None:
        """Channel enum has expected values."""
        assert EtherealChannel.BOOK_DEPTH.value == "BookDepth"
        assert EtherealChannel.MARKET_PRICE.value == "MarketPrice"
        assert EtherealChannel.FUNDING_RATE.value == "FundingRate"


class TestWebSocketEndpoints:
    """Tests for WebSocket endpoint constants."""

    def test_testnet_endpoint(self) -> None:
        """Testnet endpoint is correct."""
        assert ETHEREAL_TESTNET_WS == "wss://ws.etherealtest.net"

    def test_mainnet_endpoint(self) -> None:
        """Mainnet endpoint is correct."""
        assert ETHEREAL_MAINNET_WS == "wss://ws.ethereal.trade"


class TestEtherealDataClientInit:
    """Tests for EtherealDataClient initialization."""

    def test_default_testnet(self) -> None:
        """Default configuration uses testnet."""
        client = EtherealDataClient()
        assert client.testnet is True
        assert client._ws_url == ETHEREAL_TESTNET_WS

    def test_mainnet_config(self) -> None:
        """Can configure for mainnet."""
        client = EtherealDataClient(testnet=False)
        assert client.testnet is False
        assert client._ws_url == ETHEREAL_MAINNET_WS

    def test_default_reconnect_settings(self) -> None:
        """Default reconnect settings are sensible."""
        client = EtherealDataClient()
        assert client._auto_reconnect is True
        assert client._reconnect_delay == 5.0
        assert client._max_reconnect_attempts == 10

    def test_custom_reconnect_settings(self) -> None:
        """Can customize reconnect settings."""
        client = EtherealDataClient(
            auto_reconnect=False,
            reconnect_delay=10.0,
            max_reconnect_attempts=5,
        )
        assert client._auto_reconnect is False
        assert client._reconnect_delay == 10.0
        assert client._max_reconnect_attempts == 5

    def test_initial_state(self) -> None:
        """Client starts disconnected with empty subscriptions."""
        client = EtherealDataClient()
        assert client.connected is False
        assert len(client._subscriptions) == 0


class TestBookDepthUpdate:
    """Tests for BookDepthUpdate dataclass."""

    def test_create_book_depth_update(self) -> None:
        """Can create BookDepthUpdate."""
        ts = datetime.now(UTC)
        bids = [(Decimal("45000"), Decimal("1.5"))]
        asks = [(Decimal("45100"), Decimal("2.0"))]

        update = BookDepthUpdate(
            symbol="BTCUSD",
            timestamp=ts,
            bids=bids,
            asks=asks,
        )

        assert update.symbol == "BTCUSD"
        assert update.timestamp == ts
        assert update.bids == bids
        assert update.asks == asks


class TestMarketPriceUpdate:
    """Tests for MarketPriceUpdate dataclass."""

    def test_create_market_price_update(self) -> None:
        """Can create MarketPriceUpdate."""
        ts = datetime.now(UTC)

        update = MarketPriceUpdate(
            symbol="BTCUSD",
            timestamp=ts,
            price=Decimal("45050.5"),
            size=Decimal("0.1"),
        )

        assert update.symbol == "BTCUSD"
        assert update.timestamp == ts
        assert update.price == Decimal("45050.5")
        assert update.size == Decimal("0.1")


class TestFundingRateUpdate:
    """Tests for FundingRateUpdate dataclass."""

    def test_create_funding_rate_update(self) -> None:
        """Can create FundingRateUpdate."""
        ts = datetime.now(UTC)
        next_funding = datetime.now(UTC)

        update = FundingRateUpdate(
            symbol="BTCUSD",
            timestamp=ts,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("45050.5"),
            next_funding_time=next_funding,
        )

        assert update.symbol == "BTCUSD"
        assert update.timestamp == ts
        assert update.funding_rate == Decimal("0.0001")
        assert update.mark_price == Decimal("45050.5")
        assert update.next_funding_time == next_funding

    def test_funding_rate_with_none_next_time(self) -> None:
        """FundingRateUpdate allows None next_funding_time."""
        ts = datetime.now(UTC)

        update = FundingRateUpdate(
            symbol="BTCUSD",
            timestamp=ts,
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("45050.5"),
            next_funding_time=None,
        )

        assert update.next_funding_time is None


class TestDataNormalization:
    """Tests for data normalization methods."""

    @pytest.fixture
    def client(self) -> EtherealDataClient:
        """Create client for testing normalization."""
        return EtherealDataClient()

    def test_normalize_book_depth(self, client: EtherealDataClient) -> None:
        """BookDepth message is normalized correctly."""
        data: dict[str, Any] = {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,  # 2024-02-01 00:00:00 UTC
            "bids": [["45000.0", "1.5"], ["44900.0", "2.0"]],
            "asks": [["45100.0", "0.5"], ["45200.0", "1.0"]],
        }

        update = client._normalize_book_depth(data, "BTCUSD")

        assert update.symbol == "BTCUSD"
        assert update.timestamp == datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)
        assert len(update.bids) == 2
        assert update.bids[0] == (Decimal("45000.0"), Decimal("1.5"))
        assert len(update.asks) == 2
        assert update.asks[0] == (Decimal("45100.0"), Decimal("0.5"))

    def test_normalize_book_depth_empty_book(self, client: EtherealDataClient) -> None:
        """BookDepth with empty bids/asks is handled."""
        data: dict[str, Any] = {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,
            "bids": [],
            "asks": [],
        }

        update = client._normalize_book_depth(data, "BTCUSD")

        assert update.bids == []
        assert update.asks == []

    def test_normalize_market_price(self, client: EtherealDataClient) -> None:
        """MarketPrice message is normalized correctly."""
        data: dict[str, Any] = {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,
            "price": "45050.5",
            "size": "0.1",
        }

        update = client._normalize_market_price(data, "BTCUSD")

        assert update.symbol == "BTCUSD"
        assert update.timestamp == datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)
        assert update.price == Decimal("45050.5")
        assert update.size == Decimal("0.1")

    def test_normalize_market_price_missing_fields(
        self,
        client: EtherealDataClient,
    ) -> None:
        """MarketPrice with missing fields uses defaults."""
        data: dict[str, Any] = {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,
        }

        update = client._normalize_market_price(data, "BTCUSD")

        assert update.price == Decimal("0")
        assert update.size == Decimal("0")

    def test_normalize_funding_rate(self, client: EtherealDataClient) -> None:
        """FundingRate message is normalized correctly."""
        data: dict[str, Any] = {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,
            "fundingRate": "0.0001",
            "markPrice": "45050.5",
            "nextFundingTime": 1706749200000,  # +1 hour
        }

        update = client._normalize_funding_rate(data, "BTCUSD")

        assert update.symbol == "BTCUSD"
        assert update.timestamp == datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC)
        assert update.funding_rate == Decimal("0.0001")
        assert update.mark_price == Decimal("45050.5")
        assert update.next_funding_time == datetime(2024, 2, 1, 1, 0, 0, tzinfo=UTC)

    def test_normalize_funding_rate_no_next_time(
        self,
        client: EtherealDataClient,
    ) -> None:
        """FundingRate without nextFundingTime sets None."""
        data: dict[str, Any] = {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,
            "fundingRate": "0.0001",
            "markPrice": "45050.5",
        }

        update = client._normalize_funding_rate(data, "BTCUSD")

        assert update.next_funding_time is None


class TestSubscriptionHandling:
    """Tests for subscription handling."""

    @pytest.fixture
    def connected_client(self) -> EtherealDataClient:
        """Create mock-connected client."""
        client = EtherealDataClient()
        client._connected = True
        client._sio = MagicMock()
        client._sio.emit = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_subscribe_requires_connection(self) -> None:
        """Subscribe raises error when not connected."""
        client = EtherealDataClient()

        with pytest.raises(ConnectionError, match="Not connected"):
            await client.subscribe("BTCUSD", [EtherealChannel.MARKET_PRICE])

    @pytest.mark.asyncio
    async def test_subscribe_invalid_symbol(
        self,
        connected_client: EtherealDataClient,
    ) -> None:
        """Subscribe raises error for invalid symbol."""
        with pytest.raises(ValueError, match="Unsupported symbol"):
            await connected_client.subscribe("INVALID", [EtherealChannel.MARKET_PRICE])

    @pytest.mark.asyncio
    async def test_subscribe_creates_state(
        self,
        connected_client: EtherealDataClient,
    ) -> None:
        """Subscribe creates subscription state."""
        await connected_client.subscribe(
            "BTCUSD",
            [EtherealChannel.MARKET_PRICE],
        )

        assert "BTCUSD" in connected_client._subscriptions
        state = connected_client._subscriptions["BTCUSD"]
        assert EtherealChannel.MARKET_PRICE in state.channels

    @pytest.mark.asyncio
    async def test_subscribe_stores_callbacks(
        self,
        connected_client: EtherealDataClient,
    ) -> None:
        """Subscribe stores provided callbacks."""
        price_callback = MagicMock()
        book_callback = MagicMock()

        await connected_client.subscribe(
            "BTCUSD",
            [EtherealChannel.MARKET_PRICE, EtherealChannel.BOOK_DEPTH],
            on_market_price=price_callback,
            on_book_depth=book_callback,
        )

        state = connected_client._subscriptions["BTCUSD"]
        assert state.market_price_callback is price_callback
        assert state.book_depth_callback is book_callback

    @pytest.mark.asyncio
    async def test_subscribe_sends_emit(
        self,
        connected_client: EtherealDataClient,
    ) -> None:
        """Subscribe emits subscription message to server."""
        await connected_client.subscribe(
            "BTCUSD",
            [EtherealChannel.MARKET_PRICE],
        )

        connected_client._sio.emit.assert_called_once_with(
            "subscribe",
            {"channel": "MarketPrice", "symbol": "BTC-USD"},
        )

    @pytest.mark.asyncio
    async def test_subscribe_multiple_channels(
        self,
        connected_client: EtherealDataClient,
    ) -> None:
        """Subscribe emits for each channel."""
        await connected_client.subscribe(
            "BTCUSD",
            [EtherealChannel.MARKET_PRICE, EtherealChannel.BOOK_DEPTH],
        )

        assert connected_client._sio.emit.call_count == 2

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_state(
        self,
        connected_client: EtherealDataClient,
    ) -> None:
        """Unsubscribe removes subscription state."""
        await connected_client.subscribe(
            "BTCUSD",
            [EtherealChannel.MARKET_PRICE],
        )
        assert "BTCUSD" in connected_client._subscriptions

        await connected_client.unsubscribe("BTCUSD")
        assert "BTCUSD" not in connected_client._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_symbol(
        self,
        connected_client: EtherealDataClient,
    ) -> None:
        """Unsubscribe for nonexistent symbol does nothing."""
        # Should not raise
        await connected_client.unsubscribe("BTCUSD")


class TestMessageHandling:
    """Tests for WebSocket message handling."""

    @pytest.fixture
    def client_with_subscription(self) -> EtherealDataClient:
        """Create client with BTCUSD subscription."""
        client = EtherealDataClient()
        client._connected = True
        from vibe_quant.ethereal.data_client import SubscriptionState

        client._subscriptions["BTCUSD"] = SubscriptionState(symbol="BTCUSD")
        return client

    @pytest.mark.asyncio
    async def test_handle_book_depth_calls_callback(
        self,
        client_with_subscription: EtherealDataClient,
    ) -> None:
        """BookDepth handler calls registered callback."""
        callback = MagicMock()
        client_with_subscription._subscriptions["BTCUSD"].book_depth_callback = callback
        client_with_subscription._subscriptions["BTCUSD"].channels.add(EtherealChannel.BOOK_DEPTH)

        data = {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,
            "bids": [["45000.0", "1.0"]],
            "asks": [["45100.0", "1.0"]],
        }

        await client_with_subscription._handle_book_depth(data)

        callback.assert_called_once()
        update = callback.call_args[0][0]
        assert isinstance(update, BookDepthUpdate)
        assert update.symbol == "BTCUSD"

    @pytest.mark.asyncio
    async def test_handle_book_depth_ignores_unknown_symbol(
        self,
        client_with_subscription: EtherealDataClient,
    ) -> None:
        """BookDepth handler ignores unknown symbols."""
        callback = MagicMock()
        client_with_subscription._subscriptions["BTCUSD"].book_depth_callback = callback

        data = {
            "symbol": "UNKNOWN-USD",
            "timestamp": 1706745600000,
            "bids": [],
            "asks": [],
        }

        await client_with_subscription._handle_book_depth(data)
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_market_price_calls_callback(
        self,
        client_with_subscription: EtherealDataClient,
    ) -> None:
        """MarketPrice handler calls registered callback."""
        callback = MagicMock()
        client_with_subscription._subscriptions["BTCUSD"].market_price_callback = callback
        client_with_subscription._subscriptions["BTCUSD"].channels.add(EtherealChannel.MARKET_PRICE)

        data = {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,
            "price": "45050.5",
            "size": "0.1",
        }

        await client_with_subscription._handle_market_price(data)

        callback.assert_called_once()
        update = callback.call_args[0][0]
        assert isinstance(update, MarketPriceUpdate)
        assert update.price == Decimal("45050.5")

    @pytest.mark.asyncio
    async def test_handle_funding_rate_calls_callback(
        self,
        client_with_subscription: EtherealDataClient,
    ) -> None:
        """FundingRate handler calls registered callback."""
        callback = MagicMock()
        client_with_subscription._subscriptions["BTCUSD"].funding_rate_callback = callback
        client_with_subscription._subscriptions["BTCUSD"].channels.add(EtherealChannel.FUNDING_RATE)

        data = {
            "symbol": "BTC-USD",
            "timestamp": 1706745600000,
            "fundingRate": "0.0001",
            "markPrice": "45050.5",
            "nextFundingTime": 1706749200000,
        }

        await client_with_subscription._handle_funding_rate(data)

        callback.assert_called_once()
        update = callback.call_args[0][0]
        assert isinstance(update, FundingRateUpdate)
        assert update.funding_rate == Decimal("0.0001")


def _create_mock_sio() -> MagicMock:
    """Create a mock Socket.IO client with proper method signatures."""
    mock_sio = MagicMock()
    mock_sio.connect = AsyncMock()
    mock_sio.disconnect = AsyncMock()
    mock_sio.emit = AsyncMock()
    # .on() must be a regular method that returns a decorator (identity function)
    mock_sio.on = MagicMock(side_effect=lambda event: lambda fn: fn)
    return mock_sio


class TestConnectionLifecycle:
    """Tests for connection lifecycle management."""

    @pytest.mark.asyncio
    async def test_connect_creates_socket(self) -> None:
        """Connect creates Socket.IO client."""
        with patch("vibe_quant.ethereal.data_client.socketio.AsyncClient") as mock_sio_class:
            mock_sio = _create_mock_sio()
            mock_sio_class.return_value = mock_sio

            client = EtherealDataClient()
            await client.connect()

            mock_sio.connect.assert_called_once()
            assert client.connected is True

    @pytest.mark.asyncio
    async def test_connect_uses_websocket_transport(self) -> None:
        """Connect uses websocket transport."""
        with patch("vibe_quant.ethereal.data_client.socketio.AsyncClient") as mock_sio_class:
            mock_sio = _create_mock_sio()
            mock_sio_class.return_value = mock_sio

            client = EtherealDataClient()
            await client.connect()

            call_kwargs = mock_sio.connect.call_args[1]
            assert call_kwargs["transports"] == ["websocket"]

    @pytest.mark.asyncio
    async def test_connect_raises_on_failure(self) -> None:
        """Connect raises ConnectionError on failure."""
        with patch("vibe_quant.ethereal.data_client.socketio.AsyncClient") as mock_sio_class:
            mock_sio = _create_mock_sio()
            mock_sio.connect = AsyncMock(side_effect=Exception("Connection refused"))
            mock_sio_class.return_value = mock_sio

            client = EtherealDataClient()

            with pytest.raises(ConnectionError, match="Failed to connect"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self) -> None:
        """Disconnect clears subscriptions and connection state."""
        with patch("vibe_quant.ethereal.data_client.socketio.AsyncClient") as mock_sio_class:
            mock_sio = _create_mock_sio()
            mock_sio_class.return_value = mock_sio

            client = EtherealDataClient()
            await client.connect()
            await client.disconnect()

            assert client.connected is False
            assert len(client._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Async context manager connects and disconnects."""
        with patch("vibe_quant.ethereal.data_client.socketio.AsyncClient") as mock_sio_class:
            mock_sio = _create_mock_sio()
            mock_sio_class.return_value = mock_sio

            async with EtherealDataClient() as client:
                assert client.connected is True

            mock_sio.disconnect.assert_called()


class TestReconnection:
    """Tests for auto-reconnection behavior."""

    @pytest.mark.asyncio
    async def test_reconnect_resets_count_on_success(self) -> None:
        """Successful connection resets reconnect count."""
        with patch("vibe_quant.ethereal.data_client.socketio.AsyncClient") as mock_sio_class:
            mock_sio = _create_mock_sio()
            mock_sio_class.return_value = mock_sio

            client = EtherealDataClient()
            client._reconnect_count = 5
            await client.connect()

            assert client._reconnect_count == 0

    @pytest.mark.asyncio
    async def test_max_reconnect_attempts_respected(self) -> None:
        """Reconnect loop stops after max attempts."""
        client = EtherealDataClient(
            max_reconnect_attempts=2,
            reconnect_delay=0.01,
        )

        with patch.object(
            client,
            "connect",
            side_effect=ConnectionError("Failed"),
        ):
            client._reconnect_count = 2  # Already at max
            await client._reconnect_loop()

            # Should return without attempting reconnect
            assert client._reconnect_count == 2
