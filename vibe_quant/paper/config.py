"""Configuration for paper trading TradingNode.

Provides configuration dataclasses and factory functions for setting up
NautilusTrader TradingNode with Binance testnet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Environment variable names
ENV_BINANCE_API_KEY = "BINANCE_API_KEY"
ENV_BINANCE_API_SECRET = "BINANCE_API_SECRET"


class ConfigurationError(Exception):
    """Error in paper trading configuration."""

    pass


@dataclass(frozen=True)
class BinanceTestnetConfig:
    """Configuration for Binance testnet connection.

    Attributes:
        api_key: Binance API key (from env var or direct).
        api_secret: Binance API secret (from env var or direct).
        testnet: If True, use Binance testnet. Defaults to True.
        account_type: Account type (USDT_FUTURES for perpetuals).
    """

    api_key: str
    api_secret: str
    testnet: bool = True
    account_type: str = "USDT_FUTURES"

    @classmethod
    def from_env(cls, testnet: bool = True) -> BinanceTestnetConfig:
        """Create config from environment variables.

        Args:
            testnet: If True, use Binance testnet.

        Returns:
            BinanceTestnetConfig instance.

        Raises:
            ConfigurationError: If required env vars are missing.
        """
        api_key = os.getenv(ENV_BINANCE_API_KEY)
        api_secret = os.getenv(ENV_BINANCE_API_SECRET)

        if not api_key:
            raise ConfigurationError(
                f"Missing {ENV_BINANCE_API_KEY} environment variable"
            )
        if not api_secret:
            raise ConfigurationError(
                f"Missing {ENV_BINANCE_API_SECRET} environment variable"
            )

        return cls(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )


@dataclass
class SizingModuleConfig:
    """Configuration for position sizing module.

    Attributes:
        method: Sizing method (fixed_fractional, kelly, atr).
        max_leverage: Maximum leverage to use.
        max_position_pct: Maximum position size as % of equity.
        risk_per_trade: Risk per trade as % of equity.
        kelly_fraction: Fractional Kelly to use (if kelly method).
        atr_multiplier: ATR multiplier for stop distance (if atr method).
    """

    method: str = "fixed_fractional"
    max_leverage: Decimal = field(default_factory=lambda: Decimal("20"))
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))
    risk_per_trade: Decimal = field(default_factory=lambda: Decimal("0.02"))
    kelly_fraction: Decimal = field(default_factory=lambda: Decimal("0.5"))
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))


@dataclass
class RiskModuleConfig:
    """Configuration for risk management module.

    Attributes:
        max_drawdown_pct: Maximum drawdown before halting (0.15 = 15%).
        max_daily_loss_pct: Maximum daily loss before halting.
        max_consecutive_losses: Maximum consecutive losses before halting.
        max_position_count: Maximum concurrent positions.
        max_portfolio_drawdown_pct: Maximum portfolio-level drawdown.
        max_total_exposure_pct: Maximum total exposure as % of equity.
        max_single_instrument_pct: Maximum exposure to single instrument.
    """

    max_drawdown_pct: Decimal = field(default_factory=lambda: Decimal("0.15"))
    max_daily_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))
    max_consecutive_losses: int = 10
    max_position_count: int = 5
    max_portfolio_drawdown_pct: Decimal = field(default_factory=lambda: Decimal("0.20"))
    max_total_exposure_pct: Decimal = field(default_factory=lambda: Decimal("0.50"))
    max_single_instrument_pct: Decimal = field(default_factory=lambda: Decimal("0.30"))


@dataclass
class PaperTradingConfig:
    """Complete configuration for paper trading node.

    Attributes:
        trader_id: Unique identifier for this trader instance.
        binance: Binance testnet configuration.
        symbols: List of symbols to subscribe to.
        strategy_id: Strategy ID from database to deploy.
        sizing: Position sizing configuration.
        risk: Risk management configuration.
        db_path: Path to SQLite state database.
        logs_path: Path for event log files.
        state_persistence_interval: Seconds between state snapshots.
    """

    trader_id: str
    binance: BinanceTestnetConfig
    symbols: list[str] = field(default_factory=list)
    strategy_id: int | None = None
    sizing: SizingModuleConfig = field(default_factory=SizingModuleConfig)
    risk: RiskModuleConfig = field(default_factory=RiskModuleConfig)
    db_path: Path | None = None
    logs_path: Path = field(default_factory=lambda: Path("logs/paper"))
    state_persistence_interval: int = 60

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        if not self.trader_id:
            errors.append("trader_id is required")

        if not self.symbols:
            errors.append("At least one symbol is required")

        if self.strategy_id is None:
            errors.append("strategy_id is required")

        _VALID_SIZING_METHODS = {"fixed_fractional", "kelly", "atr"}
        if self.sizing.method not in _VALID_SIZING_METHODS:
            errors.append(f"sizing method must be one of {sorted(_VALID_SIZING_METHODS)}, got '{self.sizing.method}'")

        if self.sizing.risk_per_trade <= 0 or self.sizing.risk_per_trade > Decimal("0.5"):
            errors.append("risk_per_trade must be between 0 and 0.5")

        if self.sizing.max_leverage < 1 or self.sizing.max_leverage > 125:
            errors.append("max_leverage must be between 1 and 125")

        if self.risk.max_drawdown_pct <= 0 or self.risk.max_drawdown_pct > 1:
            errors.append("max_drawdown_pct must be between 0 and 1")

        return errors

    @classmethod
    def create(
        cls,
        trader_id: str,
        symbols: Sequence[str],
        strategy_id: int,
        testnet: bool = True,
        db_path: Path | None = None,
    ) -> PaperTradingConfig:
        """Create paper trading config with defaults.

        Args:
            trader_id: Unique trader identifier.
            symbols: Symbols to trade.
            strategy_id: Strategy ID from database.
            testnet: Use Binance testnet.
            db_path: Database path.

        Returns:
            PaperTradingConfig instance.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        binance = BinanceTestnetConfig.from_env(testnet=testnet)

        config = cls(
            trader_id=trader_id,
            binance=binance,
            symbols=list(symbols),
            strategy_id=strategy_id,
            db_path=db_path,
        )

        errors = config.validate()
        if errors:
            raise ConfigurationError(f"Invalid configuration: {'; '.join(errors)}")

        return config


def create_trading_node_config(config: PaperTradingConfig) -> dict[str, Any]:
    """Create NautilusTrader TradingNodeConfig dictionary.

    This generates the configuration dictionary that can be passed to
    TradingNode for paper trading with Binance testnet.

    Args:
        config: PaperTradingConfig instance.

    Returns:
        Dictionary of TradingNode configuration parameters.

    Note:
        The actual TradingNodeConfig instantiation requires nautilus_trader
        imports which may not be available in all environments. This function
        returns the raw config dict that can be used with TradingNodeConfig.
    """
    return {
        "trader_id": config.trader_id,
        "data_clients": {
            "BINANCE": {
                "api_key": config.binance.api_key,
                "api_secret": config.binance.api_secret,
                "account_type": config.binance.account_type,
                "testnet": config.binance.testnet,
            },
        },
        "exec_clients": {
            "BINANCE": {
                "api_key": config.binance.api_key,
                "api_secret": config.binance.api_secret,
                "account_type": config.binance.account_type,
                "testnet": config.binance.testnet,
            },
        },
        "symbols": config.symbols,
        "strategy_id": config.strategy_id,
        "sizing": {
            "method": config.sizing.method,
            "max_leverage": str(config.sizing.max_leverage),
            "max_position_pct": str(config.sizing.max_position_pct),
            "risk_per_trade": str(config.sizing.risk_per_trade),
        },
        "risk": {
            "max_drawdown_pct": str(config.risk.max_drawdown_pct),
            "max_daily_loss_pct": str(config.risk.max_daily_loss_pct),
            "max_consecutive_losses": config.risk.max_consecutive_losses,
            "max_position_count": config.risk.max_position_count,
        },
        "persistence": {
            "db_path": str(config.db_path) if config.db_path else None,
            "logs_path": str(config.logs_path),
            "state_interval": config.state_persistence_interval,
        },
    }
