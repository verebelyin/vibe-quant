"""Ethereal DEX execution client with EIP-712 signing.

Implements order placement, cancellation, and fill retrieval via REST API.
Uses EIP-712 typed data signatures for non-custodial order submission.
"""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

import httpx
from eth_account import Account
from eth_account.messages import encode_typed_data

if TYPE_CHECKING:
    from collections.abc import Mapping

# Environment variable for private key
ENV_ETHEREAL_PRIVATE_KEY = "ETHEREAL_PRIVATE_KEY"

# Ethereal API endpoints
ETHEREAL_TESTNET_API = "https://api.etherealtest.net"
ETHEREAL_MAINNET_API = "https://api.ethereal.trade"

# Testnet chain ID
ETHEREAL_TESTNET_CHAIN_ID = 11155111  # Sepolia
ETHEREAL_MAINNET_CHAIN_ID = 1  # Ethereum mainnet

# EIP-712 domain separator for Ethereal
ETHEREAL_EIP712_DOMAIN_NAME = "Ethereal"
ETHEREAL_EIP712_DOMAIN_VERSION = "1"


class ConfigurationError(Exception):
    """Error in Ethereal configuration."""

    pass


class EtherealAPIError(Exception):
    """Error from Ethereal API."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OrderSide(IntEnum):
    """Order side enum matching Ethereal API."""

    BUY = 1
    SELL = 2


class OrderType(StrEnum):
    """Order type enum."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    """Order status enum."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class EtherealOrder:
    """Order to submit to Ethereal.

    Attributes:
        symbol: Trading pair (e.g., "BTC-USD").
        side: Buy or sell.
        order_type: Market or limit.
        quantity: Order quantity.
        price: Limit price (required for limit orders).
        reduce_only: If True, only reduces position.
        client_order_id: Optional client-assigned ID.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None
    reduce_only: bool = False
    client_order_id: str | None = None

    def __post_init__(self) -> None:
        """Validate order parameters."""
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")


@dataclass(frozen=True)
class Fill:
    """Execution fill from Ethereal.

    Attributes:
        fill_id: Unique fill identifier.
        order_id: Parent order ID.
        symbol: Trading pair.
        side: Buy or sell.
        quantity: Filled quantity.
        price: Execution price.
        fee: Fee paid.
        timestamp: Fill timestamp.
    """

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    timestamp: datetime


@dataclass
class EtherealConfig:
    """Configuration for Ethereal execution client.

    Attributes:
        private_key: Wallet private key for signing.
        testnet: If True, use testnet API (default).
        verifying_contract: EIP-712 verifying contract address.
    """

    private_key: str
    testnet: bool = True
    verifying_contract: str = "0x0000000000000000000000000000000000000000"

    @classmethod
    def from_env(cls, testnet: bool = True) -> EtherealConfig:
        """Create config from environment variables.

        Args:
            testnet: Use testnet API.

        Returns:
            EtherealConfig instance.

        Raises:
            ConfigurationError: If required env vars missing.
        """
        private_key = os.getenv(ENV_ETHEREAL_PRIVATE_KEY)

        if not private_key:
            raise ConfigurationError(
                f"Missing {ENV_ETHEREAL_PRIVATE_KEY} environment variable"
            )

        return cls(private_key=private_key, testnet=testnet)

    @property
    def api_base(self) -> str:
        """Get API base URL."""
        return ETHEREAL_TESTNET_API if self.testnet else ETHEREAL_MAINNET_API

    @property
    def chain_id(self) -> int:
        """Get chain ID."""
        return ETHEREAL_TESTNET_CHAIN_ID if self.testnet else ETHEREAL_MAINNET_CHAIN_ID


@dataclass
class OrderResult:
    """Result of order placement.

    Attributes:
        order_id: Server-assigned order ID.
        client_order_id: Client-assigned ID (if provided).
        status: Order status.
    """

    order_id: str
    client_order_id: str | None
    status: OrderStatus


@dataclass
class EIP712OrderData:
    """EIP-712 typed data for order signing.

    Attributes:
        symbol: Trading pair.
        side: Order side (1=buy, 2=sell).
        order_type: Order type.
        quantity: Order quantity as string.
        price: Order price as string.
        nonce: Unique nonce for replay protection.
        expiry: Order expiration timestamp.
        reduce_only: Reduce-only flag.
    """

    symbol: str
    side: int
    order_type: str
    quantity: str
    price: str
    nonce: int
    expiry: int
    reduce_only: bool

    def to_message_data(self) -> dict[str, object]:
        """Convert to EIP-712 message data dict."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "orderType": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "nonce": self.nonce,
            "expiry": self.expiry,
            "reduceOnly": self.reduce_only,
        }


# EIP-712 type definitions for Ethereal orders
EIP712_ORDER_TYPES: dict[str, list[dict[str, str]]] = {
    "Order": [
        {"name": "symbol", "type": "string"},
        {"name": "side", "type": "uint8"},
        {"name": "orderType", "type": "string"},
        {"name": "quantity", "type": "string"},
        {"name": "price", "type": "string"},
        {"name": "nonce", "type": "uint256"},
        {"name": "expiry", "type": "uint256"},
        {"name": "reduceOnly", "type": "bool"},
    ],
}


def generate_nonce() -> int:
    """Generate unique nonce using timestamp + random bits."""
    ts = time.time_ns() // 1_000_000  # ms precision
    random_bits = int.from_bytes(secrets.token_bytes(4))
    return (ts << 32) | random_bits


def sign_order(
    order_data: EIP712OrderData,
    private_key: str,
    chain_id: int,
    verifying_contract: str,
) -> str:
    """Sign order using EIP-712 typed data.

    Args:
        order_data: Order data to sign.
        private_key: Wallet private key.
        chain_id: Network chain ID.
        verifying_contract: Ethereal contract address.

    Returns:
        Hex-encoded signature.
    """
    domain_data = {
        "name": ETHEREAL_EIP712_DOMAIN_NAME,
        "version": ETHEREAL_EIP712_DOMAIN_VERSION,
        "chainId": chain_id,
        "verifyingContract": verifying_contract,
    }

    full_message = {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            **EIP712_ORDER_TYPES,
        },
        "primaryType": "Order",
        "domain": domain_data,
        "message": order_data.to_message_data(),
    }

    signable = encode_typed_data(full_message=full_message)
    signed = Account.sign_message(signable, private_key=private_key)
    return str(signed.signature.hex())


@dataclass
class EtherealExecutionClient:
    """Execution client for Ethereal DEX.

    Handles order placement, cancellation, and fill retrieval
    with EIP-712 signatures for non-custodial operation.

    Attributes:
        config: Client configuration.
    """

    config: EtherealConfig
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _nonce_counter: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_env(cls, testnet: bool = True) -> EtherealExecutionClient:
        """Create client from environment variables.

        Args:
            testnet: Use testnet API.

        Returns:
            EtherealExecutionClient instance.
        """
        return cls(EtherealConfig.from_env(testnet=testnet))

    @property
    def wallet_address(self) -> str:
        """Get wallet address derived from private key."""
        account = Account.from_key(self.config.private_key)
        return str(account.address)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base,
                timeout=30.0,
            )
        return self._client

    def _prepare_order_data(
        self,
        order: EtherealOrder,
        expiry: int | None = None,
    ) -> EIP712OrderData:
        """Prepare order data for signing.

        Args:
            order: Order to prepare.
            expiry: Optional expiration timestamp.

        Returns:
            EIP712OrderData ready for signing.
        """
        self._nonce_counter += 1
        nonce = generate_nonce() + self._nonce_counter

        # Default expiry: 1 hour from now
        if expiry is None:
            expiry = int(time.time()) + 3600

        price_str = str(order.price) if order.price else "0"

        return EIP712OrderData(
            symbol=order.symbol,
            side=int(order.side),
            order_type=order.order_type.value,
            quantity=str(order.quantity),
            price=price_str,
            nonce=nonce,
            expiry=expiry,
            reduce_only=order.reduce_only,
        )

    async def place_order(self, order: EtherealOrder) -> OrderResult:
        """Place order on Ethereal.

        Args:
            order: Order to place.

        Returns:
            OrderResult with order ID and status.

        Raises:
            EtherealAPIError: If order placement fails.
        """
        order_data = self._prepare_order_data(order)
        signature = sign_order(
            order_data=order_data,
            private_key=self.config.private_key,
            chain_id=self.config.chain_id,
            verifying_contract=self.config.verifying_contract,
        )

        payload: dict[str, object] = {
            "order": order_data.to_message_data(),
            "signature": f"0x{signature}",
            "wallet": self.wallet_address,
        }

        if order.client_order_id:
            payload["clientOrderId"] = order.client_order_id

        client = await self._get_client()

        try:
            response = await client.post("/orders", json=payload)
            response.raise_for_status()
            data: Mapping[str, object] = response.json()

            return OrderResult(
                order_id=str(data["orderId"]),
                client_order_id=order.client_order_id,
                status=OrderStatus(str(data.get("status", "pending"))),
            )
        except httpx.HTTPStatusError as e:
            msg = f"Order placement failed: {e.response.text}"
            raise EtherealAPIError(msg, e.response.status_code) from e
        except httpx.RequestError as e:
            raise EtherealAPIError(f"Request failed: {e}") from e

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancelled successfully.

        Raises:
            EtherealAPIError: If cancellation fails.
        """
        # Sign cancellation request
        cancel_nonce = generate_nonce()
        cancel_message = {
            "action": "cancel",
            "orderId": order_id,
            "nonce": cancel_nonce,
        }

        # Simple message signing for cancel (not full EIP-712)
        account = Account.from_key(self.config.private_key)
        signed = account.sign_message(
            encode_typed_data(
                full_message={
                    "types": {
                        "EIP712Domain": [
                            {"name": "name", "type": "string"},
                            {"name": "version", "type": "string"},
                            {"name": "chainId", "type": "uint256"},
                        ],
                        "Cancel": [
                            {"name": "action", "type": "string"},
                            {"name": "orderId", "type": "string"},
                            {"name": "nonce", "type": "uint256"},
                        ],
                    },
                    "primaryType": "Cancel",
                    "domain": {
                        "name": ETHEREAL_EIP712_DOMAIN_NAME,
                        "version": ETHEREAL_EIP712_DOMAIN_VERSION,
                        "chainId": self.config.chain_id,
                    },
                    "message": cancel_message,
                }
            )
        )

        payload = {
            "orderId": order_id,
            "nonce": cancel_nonce,
            "signature": f"0x{signed.signature.hex()}",
            "wallet": self.wallet_address,
        }

        client = await self._get_client()

        try:
            response = await client.request("DELETE", f"/orders/{order_id}", json=payload)
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False  # Order not found (may be already filled/cancelled)
            msg = f"Cancel failed: {e.response.text}"
            raise EtherealAPIError(msg, e.response.status_code) from e
        except httpx.RequestError as e:
            raise EtherealAPIError(f"Request failed: {e}") from e

    async def get_orders(
        self,
        symbol: str | None = None,
        status: OrderStatus | None = None,
    ) -> list[dict[str, object]]:
        """Get orders, optionally filtered.

        Args:
            symbol: Filter by symbol.
            status: Filter by status.

        Returns:
            List of order dicts.

        Raises:
            EtherealAPIError: If request fails.
        """
        params: dict[str, str] = {"wallet": self.wallet_address}
        if symbol:
            params["symbol"] = symbol
        if status:
            params["status"] = status.value

        client = await self._get_client()

        try:
            response = await client.get("/orders", params=params)
            response.raise_for_status()
            data = response.json()
            return list(data.get("orders", []))
        except httpx.HTTPStatusError as e:
            msg = f"Get orders failed: {e.response.text}"
            raise EtherealAPIError(msg, e.response.status_code) from e
        except httpx.RequestError as e:
            raise EtherealAPIError(f"Request failed: {e}") from e

    async def get_fills(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> list[Fill]:
        """Get fills for a symbol in time range.

        Args:
            symbol: Trading pair.
            start_time: Start of time range.
            end_time: End of time range (default: now).

        Returns:
            List of Fill objects.

        Raises:
            EtherealAPIError: If request fails.
        """
        params: dict[str, str | int] = {
            "wallet": self.wallet_address,
            "symbol": symbol,
            "startTime": int(start_time.timestamp() * 1000),
        }
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        client = await self._get_client()

        try:
            response = await client.get("/fills", params=params)
            response.raise_for_status()
            data = response.json()

            fills: list[Fill] = []
            for f in data.get("fills", []):
                fills.append(
                    Fill(
                        fill_id=str(f["fillId"]),
                        order_id=str(f["orderId"]),
                        symbol=str(f["symbol"]),
                        side=OrderSide(int(f["side"])),
                        quantity=Decimal(str(f["quantity"])),
                        price=Decimal(str(f["price"])),
                        fee=Decimal(str(f.get("fee", "0"))),
                        timestamp=datetime.fromtimestamp(int(f["timestamp"]) / 1000, tz=UTC),
                    )
                )
            return fills
        except httpx.HTTPStatusError as e:
            msg = f"Get fills failed: {e.response.text}"
            raise EtherealAPIError(msg, e.response.status_code) from e
        except httpx.RequestError as e:
            raise EtherealAPIError(f"Request failed: {e}") from e

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
