"""Configuration for paper trading TradingNode.

Provides configuration dataclasses and factory functions for setting up
NautilusTrader TradingNode with Binance testnet (or, with explicit opt-in, live).

Credentials are read from the environment only and never accepted through
config files or the API:

* testnet: ``BINANCE_TESTNET_API_KEY`` / ``BINANCE_TESTNET_API_SECRET``
* live:    ``BINANCE_API_KEY`` / ``BINANCE_API_SECRET``
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Environment variable names
ENV_BINANCE_API_KEY = "BINANCE_API_KEY"
ENV_BINANCE_API_SECRET = "BINANCE_API_SECRET"
ENV_BINANCE_TESTNET_API_KEY = "BINANCE_TESTNET_API_KEY"
ENV_BINANCE_TESTNET_API_SECRET = "BINANCE_TESTNET_API_SECRET"

#: Default directory for paper event logs (``{trader_id}.jsonl``).
DEFAULT_PAPER_LOGS_PATH = Path("logs/paper")

#: Leverage validation runs use when the run does not specify one
#: (``ValidationRunner._create_venue_config``).
DEFAULT_VALIDATION_LEVERAGE = 10

#: Binance futures margin type matching NT's backtest MARGIN account (one
#: shared wallet = cross margin).
DEFAULT_MARGIN_TYPE = "CROSSED"

#: Sizing methods the compiled strategies actually implement. Kelly/ATR sizers
#: exist in ``vibe_quant.risk.sizing`` but are not wired into generated
#: strategies, so accepting them would silently trade fixed-fractional.
SUPPORTED_SIZING_METHODS = frozenset({"fixed_fractional"})


class ConfigurationError(Exception):
    """Error in paper trading configuration."""

    pass


def credential_env_names(testnet: bool) -> tuple[str, str]:
    """Return ``(key_var, secret_var)`` for the requested environment."""
    if testnet:
        return ENV_BINANCE_TESTNET_API_KEY, ENV_BINANCE_TESTNET_API_SECRET
    return ENV_BINANCE_API_KEY, ENV_BINANCE_API_SECRET


def default_paper_trader_id(run_id: int) -> str:
    """Default trader_id for an API-launched session (``paper_{id}`` aborted NT)."""
    return f"PAPER-{run_id:03d}"


#: NautilusTrader TraderId is ``NAME-TAG``. Constructing ``TraderId`` with an
#: invalid value panics in Rust and ABORTS the process (no Python exception),
#: so validate with a pure-Python pattern. Also restricted to the characters
#: the event-log file name allows (alphanumerics, ``_``, ``-``).
_TRADER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)+")


def is_valid_trader_id(trader_id: str) -> bool:
    """True for ``NAME-TAG`` ids NautilusTrader accepts (e.g. ``PAPER-001``)."""
    return bool(_TRADER_ID_RE.fullmatch(trader_id))


@dataclass(frozen=True)
class BinanceTestnetConfig:
    """Configuration for Binance connection.

    Attributes:
        api_key: Binance API key (from env).
        api_secret: Binance API secret (from env).
        testnet: If True, use Binance testnet. Defaults to True.
        account_type: Account type (USDT_FUTURES for perpetuals).
    """

    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    testnet: bool = True
    account_type: str = "USDT_FUTURES"

    @classmethod
    def from_env(cls, testnet: bool = True) -> BinanceTestnetConfig:
        """Create config from environment variables.

        Testnet reads ``BINANCE_TESTNET_API_*``; live reads ``BINANCE_API_*``.

        Raises:
            ConfigurationError: If required env vars are missing.
        """
        key_var, secret_var = credential_env_names(testnet)
        api_key = os.getenv(key_var)
        api_secret = os.getenv(secret_var)

        if not api_key:
            raise ConfigurationError(f"Missing {key_var} environment variable")
        if not api_secret:
            raise ConfigurationError(f"Missing {secret_var} environment variable")

        return cls(api_key=api_key, api_secret=api_secret, testnet=testnet)


@dataclass
class SizingModuleConfig:
    """Position sizing overrides.

    ``risk_per_trade`` / ``max_position_pct`` map onto the compiled strategy's
    config fields. ``None`` (default) keeps the value the strategy was validated
    with; any explicit value is applied and shows up in the logged config diff.

    Attributes:
        method: Sizing method; only ``fixed_fractional`` is implemented.
        max_leverage: Upper bound for the venue leverage.
        max_position_pct: Max position notional as fraction of equity (override).
        risk_per_trade: Risk per trade as fraction of equity (override).
        kelly_fraction: Unused until Kelly sizing is wired into strategies.
        atr_multiplier: Unused until ATR sizing is wired into strategies.
    """

    method: str = "fixed_fractional"
    max_leverage: Decimal = field(default_factory=lambda: Decimal("20"))
    max_position_pct: Decimal | None = None
    risk_per_trade: Decimal | None = None
    kelly_fraction: Decimal = field(default_factory=lambda: Decimal("0.5"))
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))


@dataclass
class RiskModuleConfig:
    """Risk limits enforced by the node's TradingGuard.

    Attributes:
        max_drawdown_pct: Equity drawdown from high water mark that halts (0.15 = 15%).
        max_daily_loss_pct: Loss vs. UTC-day starting equity that halts until next UTC day.
        max_consecutive_losses: Consecutive losing positions that halt.
        max_position_count: Maximum concurrent open positions (gate-enforced).
        max_portfolio_drawdown_pct: Not enforced (single-account node: same as max_drawdown_pct).
        max_total_exposure_pct: Not enforced.
        max_single_instrument_pct: Not enforced.
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
        trader_id: Unique identifier (NT format ``NAME-TAG``, e.g. ``PAPER-007``).
        binance: Binance connection configuration.
        symbols: Symbols to trade (filled from the validation run when empty).
        strategy_id: Strategy ID from database to deploy.
        validation_run_id: Completed validation run whose exact parameters,
            symbols and leverage paper must reproduce.
        leverage: Venue leverage override (None = validation run's leverage).
        margin_type: Binance futures margin type (``CROSSED`` / ``ISOLATED``).
        sizing: Position sizing overrides.
        risk: Risk limits.
        db_path: Path to SQLite state database.
        logs_path: Path for event log files.
        state_persistence_interval: Seconds between state snapshots.
        control_poll_interval: Seconds between command/kill-switch polls.
    """

    trader_id: str
    binance: BinanceTestnetConfig
    symbols: list[str] = field(default_factory=list)
    strategy_id: int | None = None
    validation_run_id: int | None = None
    leverage: int | None = None
    margin_type: str = DEFAULT_MARGIN_TYPE
    sizing: SizingModuleConfig = field(default_factory=SizingModuleConfig)
    risk: RiskModuleConfig = field(default_factory=RiskModuleConfig)
    db_path: Path | None = None
    logs_path: Path = field(default_factory=lambda: DEFAULT_PAPER_LOGS_PATH)
    state_persistence_interval: int = 60
    control_poll_interval: float = 1.0

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        if not self.trader_id:
            errors.append("trader_id is required")
        elif not is_valid_trader_id(self.trader_id):
            errors.append(
                f"trader_id {self.trader_id!r} is not a valid NautilusTrader TraderId "
                "(needs NAME-TAG with a hyphen, e.g. 'PAPER-001')"
            )

        if not self.symbols and self.validation_run_id is None:
            errors.append("At least one symbol is required (or a validation_run_id)")

        if self.strategy_id is None:
            errors.append("strategy_id is required")

        if self.sizing.method not in SUPPORTED_SIZING_METHODS:
            errors.append(
                f"sizing method '{self.sizing.method}' is not implemented by compiled "
                f"strategies; supported: {sorted(SUPPORTED_SIZING_METHODS)}"
            )

        rpt = self.sizing.risk_per_trade
        if rpt is not None and (rpt <= 0 or rpt > Decimal("0.5")):
            errors.append("risk_per_trade must be between 0 and 0.5")

        mpp = self.sizing.max_position_pct
        if mpp is not None and (mpp <= 0 or mpp > Decimal("125")):
            errors.append("max_position_pct must be > 0 (fraction of equity)")

        if self.sizing.max_leverage < 1 or self.sizing.max_leverage > 125:
            errors.append("max_leverage must be between 1 and 125")

        if self.leverage is not None:
            if self.leverage < 1 or self.leverage > 125:
                errors.append("leverage must be between 1 and 125")
            elif Decimal(self.leverage) > self.sizing.max_leverage:
                errors.append(
                    f"leverage {self.leverage} exceeds max_leverage {self.sizing.max_leverage}"
                )

        if self.margin_type not in ("CROSSED", "ISOLATED"):
            errors.append("margin_type must be CROSSED or ISOLATED")

        if self.risk.max_drawdown_pct <= 0 or self.risk.max_drawdown_pct > 1:
            errors.append("max_drawdown_pct must be between 0 and 1")
        if self.risk.max_daily_loss_pct <= 0 or self.risk.max_daily_loss_pct > 1:
            errors.append("max_daily_loss_pct must be between 0 and 1")
        if self.risk.max_consecutive_losses < 1:
            errors.append("max_consecutive_losses must be >= 1")
        if self.risk.max_position_count < 1:
            errors.append("max_position_count must be >= 1")

        if not self.binance.api_key or not self.binance.api_secret:
            key_var, secret_var = credential_env_names(self.binance.testnet)
            errors.append(f"Binance credentials missing: set {key_var} and {secret_var}")

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
        """Create paper trading config with defaults (credentials from env).

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
    """Summarise the TradingNode configuration as a JSON-safe dict (no secrets).

    Used for logging/inspection only; the node builds the real NautilusTrader
    ``TradingNodeConfig`` itself.
    """
    return {
        "trader_id": config.trader_id,
        "data_clients": {
            "BINANCE": {
                "account_type": config.binance.account_type,
                "testnet": config.binance.testnet,
            },
        },
        "exec_clients": {
            "BINANCE": {
                "account_type": config.binance.account_type,
                "testnet": config.binance.testnet,
                "leverage": config.leverage,
                "margin_type": config.margin_type,
            },
        },
        "symbols": config.symbols,
        "strategy_id": config.strategy_id,
        "validation_run_id": config.validation_run_id,
        "sizing": {
            "method": config.sizing.method,
            "max_leverage": str(config.sizing.max_leverage),
            "max_position_pct": (
                str(config.sizing.max_position_pct)
                if config.sizing.max_position_pct is not None
                else None
            ),
            "risk_per_trade": (
                str(config.sizing.risk_per_trade)
                if config.sizing.risk_per_trade is not None
                else None
            ),
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
