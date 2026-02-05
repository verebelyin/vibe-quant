"""Tests for Ethereal execution client."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from eth_account import Account

from vibe_quant.ethereal.execution_client import (
    EIP712_ORDER_TYPES,
    ETHEREAL_TESTNET_CHAIN_ID,
    ConfigurationError,
    EIP712OrderData,
    EtherealAPIError,
    EtherealConfig,
    EtherealExecutionClient,
    EtherealOrder,
    Fill,
    OrderSide,
    OrderStatus,
    OrderType,
    generate_nonce,
    sign_order,
)

# Test private key (DO NOT USE IN PRODUCTION)
# This is a well-known test key from eth-account docs
TEST_PRIVATE_KEY = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
TEST_WALLET_ADDRESS = Account.from_key(TEST_PRIVATE_KEY).address


class TestEtherealConfig:
    """Tests for EtherealConfig."""

    def test_create_directly(self) -> None:
        """Can create config directly."""
        config = EtherealConfig(
            private_key=TEST_PRIVATE_KEY,
            testnet=True,
        )

        assert config.private_key == TEST_PRIVATE_KEY
        assert config.testnet is True

    def test_from_env_missing_key(self) -> None:
        """Raises error if private key missing."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ConfigurationError, match="ETHEREAL_PRIVATE_KEY"),
        ):
            EtherealConfig.from_env()

    def test_from_env_success(self) -> None:
        """Creates config from env vars."""
        env = {"ETHEREAL_PRIVATE_KEY": TEST_PRIVATE_KEY}
        with patch.dict(os.environ, env, clear=True):
            config = EtherealConfig.from_env()

        assert config.private_key == TEST_PRIVATE_KEY

    def test_api_base_testnet(self) -> None:
        """Returns testnet API base."""
        config = EtherealConfig(private_key=TEST_PRIVATE_KEY, testnet=True)
        assert "etherealtest" in config.api_base

    def test_api_base_mainnet(self) -> None:
        """Returns mainnet API base."""
        config = EtherealConfig(private_key=TEST_PRIVATE_KEY, testnet=False)
        assert "ethereal.trade" in config.api_base

    def test_chain_id_testnet(self) -> None:
        """Returns testnet chain ID."""
        config = EtherealConfig(private_key=TEST_PRIVATE_KEY, testnet=True)
        assert config.chain_id == ETHEREAL_TESTNET_CHAIN_ID

    def test_chain_id_mainnet(self) -> None:
        """Returns mainnet chain ID."""
        config = EtherealConfig(private_key=TEST_PRIVATE_KEY, testnet=False)
        assert config.chain_id == 1


class TestEtherealOrder:
    """Tests for EtherealOrder."""

    def test_create_market_order(self) -> None:
        """Can create market order."""
        order = EtherealOrder(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order.symbol == "BTC-USD"
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("0.1")
        assert order.price is None

    def test_create_limit_order(self) -> None:
        """Can create limit order."""
        order = EtherealOrder(
            symbol="ETH-USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("2000.50"),
        )

        assert order.price == Decimal("2000.50")

    def test_limit_order_requires_price(self) -> None:
        """Limit order without price raises error."""
        with pytest.raises(ValueError, match="Limit orders require a price"):
            EtherealOrder(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
            )

    def test_quantity_must_be_positive(self) -> None:
        """Zero or negative quantity raises error."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            EtherealOrder(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0"),
            )


class TestEIP712Signing:
    """Tests for EIP-712 signature generation."""

    def test_generate_nonce_format(self) -> None:
        """Nonce is a positive integer based on time."""
        nonce = generate_nonce()
        assert isinstance(nonce, int)
        assert nonce > 0
        # Should be roughly current time in milliseconds
        assert nonce > 1_000_000_000_000  # After year 2001

    def test_eip712_order_data_to_message(self) -> None:
        """Order data converts to message dict."""
        order_data = EIP712OrderData(
            symbol="BTC-USD",
            side=1,
            order_type="market",
            quantity="0.1",
            price="0",
            nonce=12345,
            expiry=1700000000,
            reduce_only=False,
        )

        msg = order_data.to_message_data()

        assert msg["symbol"] == "BTC-USD"
        assert msg["side"] == 1
        assert msg["orderType"] == "market"
        assert msg["quantity"] == "0.1"
        assert msg["nonce"] == 12345

    def test_sign_order_produces_signature(self) -> None:
        """sign_order produces valid hex signature."""
        order_data = EIP712OrderData(
            symbol="BTC-USD",
            side=1,
            order_type="market",
            quantity="0.1",
            price="0",
            nonce=12345,
            expiry=1700000000,
            reduce_only=False,
        )

        signature = sign_order(
            order_data=order_data,
            private_key=TEST_PRIVATE_KEY,
            chain_id=ETHEREAL_TESTNET_CHAIN_ID,
            verifying_contract="0x0000000000000000000000000000000000000000",
        )

        # Signature should be hex string
        assert len(signature) == 130  # 65 bytes = 130 hex chars
        assert all(c in "0123456789abcdef" for c in signature)

    def test_sign_order_deterministic(self) -> None:
        """Same input produces same signature."""
        order_data = EIP712OrderData(
            symbol="BTC-USD",
            side=1,
            order_type="limit",
            quantity="1.0",
            price="50000.00",
            nonce=99999,
            expiry=1700000000,
            reduce_only=True,
        )

        sig1 = sign_order(
            order_data=order_data,
            private_key=TEST_PRIVATE_KEY,
            chain_id=ETHEREAL_TESTNET_CHAIN_ID,
            verifying_contract="0x0000000000000000000000000000000000000000",
        )

        sig2 = sign_order(
            order_data=order_data,
            private_key=TEST_PRIVATE_KEY,
            chain_id=ETHEREAL_TESTNET_CHAIN_ID,
            verifying_contract="0x0000000000000000000000000000000000000000",
        )

        assert sig1 == sig2

    def test_different_orders_different_signatures(self) -> None:
        """Different order data produces different signatures."""
        order1 = EIP712OrderData(
            symbol="BTC-USD",
            side=1,
            order_type="market",
            quantity="0.1",
            price="0",
            nonce=12345,
            expiry=1700000000,
            reduce_only=False,
        )

        order2 = EIP712OrderData(
            symbol="ETH-USD",  # Different symbol
            side=1,
            order_type="market",
            quantity="0.1",
            price="0",
            nonce=12345,
            expiry=1700000000,
            reduce_only=False,
        )

        sig1 = sign_order(
            order_data=order1,
            private_key=TEST_PRIVATE_KEY,
            chain_id=ETHEREAL_TESTNET_CHAIN_ID,
            verifying_contract="0x0000000000000000000000000000000000000000",
        )

        sig2 = sign_order(
            order_data=order2,
            private_key=TEST_PRIVATE_KEY,
            chain_id=ETHEREAL_TESTNET_CHAIN_ID,
            verifying_contract="0x0000000000000000000000000000000000000000",
        )

        assert sig1 != sig2

    def test_eip712_types_structure(self) -> None:
        """EIP-712 types have correct structure."""
        assert "Order" in EIP712_ORDER_TYPES

        order_fields = {f["name"] for f in EIP712_ORDER_TYPES["Order"]}
        assert "symbol" in order_fields
        assert "side" in order_fields
        assert "quantity" in order_fields
        assert "nonce" in order_fields


class TestEtherealExecutionClient:
    """Tests for EtherealExecutionClient."""

    @pytest.fixture
    def config(self) -> EtherealConfig:
        """Create test config."""
        return EtherealConfig(private_key=TEST_PRIVATE_KEY, testnet=True)

    @pytest.fixture
    def client(self, config: EtherealConfig) -> EtherealExecutionClient:
        """Create test client."""
        return EtherealExecutionClient(config=config)

    def test_init(self, config: EtherealConfig) -> None:
        """Client initializes with config."""
        client = EtherealExecutionClient(config=config)
        assert client.config == config

    def test_from_env(self) -> None:
        """Creates client from env vars."""
        env = {"ETHEREAL_PRIVATE_KEY": TEST_PRIVATE_KEY}
        with patch.dict(os.environ, env, clear=True):
            client = EtherealExecutionClient.from_env()

        assert client.config.private_key == TEST_PRIVATE_KEY

    def test_wallet_address(self, client: EtherealExecutionClient) -> None:
        """Derives wallet address from private key."""
        assert client.wallet_address == TEST_WALLET_ADDRESS

    def test_prepare_order_data(self, client: EtherealExecutionClient) -> None:
        """Prepares order data for signing."""
        order = EtherealOrder(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
        )

        order_data = client._prepare_order_data(order)

        assert order_data.symbol == "BTC-USD"
        assert order_data.side == 1
        assert order_data.quantity == "0.5"
        assert order_data.nonce > 0

    @pytest.mark.asyncio
    async def test_place_order_success(self, client: EtherealExecutionClient) -> None:
        """Successfully places order."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "orderId": "order-123",
            "status": "pending",
        }

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        order = EtherealOrder(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            client_order_id="my-order-1",
        )

        result = await client.place_order(order)

        assert result.order_id == "order-123"
        assert result.client_order_id == "my-order-1"
        assert result.status == OrderStatus.PENDING

        # Verify POST was called
        mock_http.post.assert_called_once()
        call_args = mock_http.post.call_args
        assert "/orders" in call_args.args[0]
        payload = call_args.kwargs["json"]
        assert "signature" in payload
        assert payload["wallet"] == TEST_WALLET_ADDRESS

    @pytest.mark.asyncio
    async def test_place_order_api_error(self, client: EtherealExecutionClient) -> None:
        """Raises EtherealAPIError on HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Invalid order"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=mock_response
        )

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        order = EtherealOrder(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        with pytest.raises(EtherealAPIError, match="Order placement failed"):
            await client.place_order(order)

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, client: EtherealExecutionClient) -> None:
        """Successfully cancels order."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.cancel_order("order-123")

        assert result is True
        mock_http.request.assert_called_once()
        call_args = mock_http.request.call_args
        assert call_args.args[0] == "DELETE"
        assert "/orders/order-123" in call_args.args[1]

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, client: EtherealExecutionClient) -> None:
        """Returns False if order not found."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.cancel_order("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_orders(self, client: EtherealExecutionClient) -> None:
        """Gets list of orders."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "orders": [
                {"orderId": "order-1", "symbol": "BTC-USD", "status": "open"},
                {"orderId": "order-2", "symbol": "ETH-USD", "status": "filled"},
            ]
        }

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        orders = await client.get_orders()

        assert len(orders) == 2
        assert orders[0]["orderId"] == "order-1"

    @pytest.mark.asyncio
    async def test_get_orders_with_filter(self, client: EtherealExecutionClient) -> None:
        """Gets orders with symbol filter."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"orders": []}

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        await client.get_orders(symbol="BTC-USD", status=OrderStatus.OPEN)

        call_args = mock_http.get.call_args
        params = call_args.kwargs["params"]
        assert params["symbol"] == "BTC-USD"
        assert params["status"] == "open"

    @pytest.mark.asyncio
    async def test_get_fills(self, client: EtherealExecutionClient) -> None:
        """Gets fills for a symbol."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "fills": [
                {
                    "fillId": "fill-1",
                    "orderId": "order-1",
                    "symbol": "BTC-USD",
                    "side": 1,
                    "quantity": "0.1",
                    "price": "50000.00",
                    "fee": "0.15",
                    "timestamp": 1700000000000,
                }
            ]
        }

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        start_time = datetime(2023, 11, 1, tzinfo=UTC)
        fills = await client.get_fills("BTC-USD", start_time)

        assert len(fills) == 1
        assert fills[0].fill_id == "fill-1"
        assert fills[0].symbol == "BTC-USD"
        assert fills[0].side == OrderSide.BUY
        assert fills[0].quantity == Decimal("0.1")
        assert fills[0].price == Decimal("50000.00")
        assert fills[0].fee == Decimal("0.15")

    @pytest.mark.asyncio
    async def test_get_fills_with_end_time(self, client: EtherealExecutionClient) -> None:
        """Gets fills with end time filter."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"fills": []}

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.is_closed = False
        client._client = mock_http

        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2023, 11, 15, tzinfo=UTC)
        await client.get_fills("BTC-USD", start, end)

        call_args = mock_http.get.call_args
        params = call_args.kwargs["params"]
        assert "endTime" in params

    @pytest.mark.asyncio
    async def test_close(self, client: EtherealExecutionClient) -> None:
        """Closes HTTP client."""
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.aclose = AsyncMock()
        client._client = mock_http

        await client.close()

        mock_http.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_already_closed(self, client: EtherealExecutionClient) -> None:
        """Close handles already closed client."""
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = True
        client._client = mock_http

        await client.close()

        mock_http.aclose.assert_not_called()


class TestOrderSideEnum:
    """Tests for OrderSide enum."""

    def test_buy_value(self) -> None:
        """BUY has value 1."""
        assert int(OrderSide.BUY) == 1

    def test_sell_value(self) -> None:
        """SELL has value 2."""
        assert int(OrderSide.SELL) == 2


class TestOrderTypeEnum:
    """Tests for OrderType enum."""

    def test_market_value(self) -> None:
        """MARKET has string value."""
        assert OrderType.MARKET.value == "market"

    def test_limit_value(self) -> None:
        """LIMIT has string value."""
        assert OrderType.LIMIT.value == "limit"


class TestOrderStatusEnum:
    """Tests for OrderStatus enum."""

    def test_all_statuses_defined(self) -> None:
        """All expected statuses exist."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"


class TestFill:
    """Tests for Fill dataclass."""

    def test_create_fill(self) -> None:
        """Can create Fill."""
        fill = Fill(
            fill_id="fill-123",
            order_id="order-456",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.5"),
            price=Decimal("50000.00"),
            fee=Decimal("0.75"),
            timestamp=datetime(2023, 11, 15, 12, 0, 0),
        )

        assert fill.fill_id == "fill-123"
        assert fill.order_id == "order-456"
        assert fill.symbol == "BTC-USD"
        assert fill.side == OrderSide.BUY
        assert fill.quantity == Decimal("0.5")
        assert fill.price == Decimal("50000.00")
        assert fill.fee == Decimal("0.75")
