"""Ethereal DEX instrument definitions for CryptoPerpetual contracts.

Defines BTCUSD, ETHUSD, SOLUSD perpetual contracts with Ethereal-specific
parameters: leverage limits, fee structure, and 1-hour funding interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

# Ethereal venue identifier
ETHEREAL_VENUE = Venue("ETHEREAL")

# USDe settlement currency (yield-bearing stablecoin)
USDE = Currency.from_str("USDE")

# Funding interval: 1 hour = 3600 seconds (vs Binance 8 hours)
ETHEREAL_FUNDING_INTERVAL_SECS = 3600


@dataclass(frozen=True)
class EtherealInstrumentConfig:
    """Configuration for an Ethereal perpetual instrument.

    Attributes:
        symbol: Trading symbol (e.g., "BTCUSD").
        base_currency: Base currency code (e.g., "BTC").
        max_leverage: Maximum allowed leverage (e.g., 20 for 20x).
        price_precision: Number of decimal places for price.
        size_precision: Number of decimal places for size.
        price_increment: Minimum price tick size.
        size_increment: Minimum size increment (lot size).
        maker_fee: Maker fee rate (0% for Ethereal).
        taker_fee: Taker fee rate (0.03% for Ethereal).
    """

    symbol: str
    base_currency: str
    max_leverage: int
    price_precision: int
    size_precision: int
    price_increment: str
    size_increment: str
    maker_fee: Decimal = Decimal("0.0000")  # 0%
    taker_fee: Decimal = Decimal("0.0003")  # 0.03%


# Instrument configurations for Ethereal perpetuals
ETHEREAL_INSTRUMENT_CONFIGS: dict[str, EtherealInstrumentConfig] = {
    "BTCUSD": EtherealInstrumentConfig(
        symbol="BTCUSD",
        base_currency="BTC",
        max_leverage=20,
        price_precision=1,
        size_precision=4,
        price_increment="0.1",
        size_increment="0.0001",
    ),
    "ETHUSD": EtherealInstrumentConfig(
        symbol="ETHUSD",
        base_currency="ETH",
        max_leverage=20,
        price_precision=2,
        size_precision=4,
        price_increment="0.01",
        size_increment="0.0001",
    ),
    "SOLUSD": EtherealInstrumentConfig(
        symbol="SOLUSD",
        base_currency="SOL",
        max_leverage=10,
        price_precision=3,
        size_precision=2,
        price_increment="0.001",
        size_increment="0.01",
    ),
}


def create_ethereal_instrument(symbol: str) -> CryptoPerpetual:
    """Create NautilusTrader CryptoPerpetual for an Ethereal symbol.

    Args:
        symbol: Trading symbol (e.g., "BTCUSD", "ETHUSD", "SOLUSD").

    Returns:
        CryptoPerpetual instrument configured for Ethereal.

    Raises:
        KeyError: If symbol is not a valid Ethereal instrument.
    """
    config = ETHEREAL_INSTRUMENT_CONFIGS[symbol]

    # Calculate margin requirements from leverage
    # margin_init = 1 / max_leverage
    margin_init = Decimal(1) / Decimal(config.max_leverage)
    # margin_maint ~= 50% of initial margin
    margin_maint = margin_init / Decimal(2)

    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol(f"{symbol}-PERP"), ETHEREAL_VENUE),
        raw_symbol=Symbol(symbol),
        base_currency=USDE,  # Settlement in USDe
        quote_currency=USDE,
        settlement_currency=USDE,
        is_inverse=False,
        price_precision=config.price_precision,
        size_precision=config.size_precision,
        price_increment=Price.from_str(config.price_increment),
        size_increment=Quantity.from_str(config.size_increment),
        max_quantity=None,
        min_quantity=None,
        max_notional=None,
        min_notional=Money(1, USDE),
        max_price=None,
        min_price=None,
        margin_init=margin_init,
        margin_maint=margin_maint,
        maker_fee=config.maker_fee,
        taker_fee=config.taker_fee,
        ts_event=0,
        ts_init=0,
    )


def get_ethereal_instrument(symbol: str) -> CryptoPerpetual:
    """Get Ethereal instrument by symbol.

    Convenience alias for create_ethereal_instrument.

    Args:
        symbol: Trading symbol (e.g., "BTCUSD").

    Returns:
        CryptoPerpetual instrument.
    """
    return create_ethereal_instrument(symbol)


def get_all_ethereal_instruments() -> dict[str, CryptoPerpetual]:
    """Get all Ethereal instruments.

    Returns:
        Dictionary mapping symbol to CryptoPerpetual.
    """
    return {sym: create_ethereal_instrument(sym) for sym in ETHEREAL_INSTRUMENT_CONFIGS}


def get_max_leverage(symbol: str) -> int:
    """Get maximum leverage for an Ethereal symbol.

    Args:
        symbol: Trading symbol.

    Returns:
        Maximum leverage (e.g., 20 for BTC/ETH, 10 for SOL).

    Raises:
        KeyError: If symbol not found.
    """
    return ETHEREAL_INSTRUMENT_CONFIGS[symbol].max_leverage


def get_ethereal_symbols() -> list[str]:
    """Get list of all Ethereal trading symbols.

    Returns:
        List of symbols ["BTCUSD", "ETHUSD", "SOLUSD"].
    """
    return list(ETHEREAL_INSTRUMENT_CONFIGS.keys())
