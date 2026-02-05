"""CLI for paper trading subprocess.

Entry point for background paper trading jobs spawned by dashboard.
Handles configuration loading, heartbeat registration, and graceful shutdown.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.jobs.manager import run_with_heartbeat
from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
    RiskModuleConfig,
    SizingModuleConfig,
)
from vibe_quant.paper.node import run_paper_trading

if TYPE_CHECKING:
    from decimal import Decimal


def _decimal_from_str(value: str | float | int) -> Decimal:
    """Convert value to Decimal."""
    from decimal import Decimal

    return Decimal(str(value))


def load_config_from_json(config_path: Path) -> PaperTradingConfig:
    """Load paper trading config from JSON file.

    Args:
        config_path: Path to JSON config file.

    Returns:
        PaperTradingConfig instance.

    Raises:
        ConfigurationError: If config file is invalid.
    """
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        data = json.load(f)

    # Parse binance config
    binance_data = data.get("binance", {})
    binance = BinanceTestnetConfig(
        api_key=binance_data.get("api_key", ""),
        api_secret=binance_data.get("api_secret", ""),
        testnet=binance_data.get("testnet", True),
        account_type=binance_data.get("account_type", "USDT_FUTURES"),
    )

    # Parse sizing config
    sizing_data = data.get("sizing", {})
    sizing = SizingModuleConfig(
        method=sizing_data.get("method", "fixed_fractional"),
        max_leverage=_decimal_from_str(sizing_data.get("max_leverage", 20)),
        max_position_pct=_decimal_from_str(sizing_data.get("max_position_pct", 0.5)),
        risk_per_trade=_decimal_from_str(sizing_data.get("risk_per_trade", 0.02)),
        kelly_fraction=_decimal_from_str(sizing_data.get("kelly_fraction", 0.5)),
        atr_multiplier=_decimal_from_str(sizing_data.get("atr_multiplier", 2.0)),
    )

    # Parse risk config
    risk_data = data.get("risk", {})
    risk = RiskModuleConfig(
        max_drawdown_pct=_decimal_from_str(risk_data.get("max_drawdown_pct", 0.15)),
        max_daily_loss_pct=_decimal_from_str(risk_data.get("max_daily_loss_pct", 0.02)),
        max_consecutive_losses=int(risk_data.get("max_consecutive_losses", 10)),
        max_position_count=int(risk_data.get("max_position_count", 5)),
    )

    # Build config
    db_path_str = data.get("db_path")
    logs_path_str = data.get("logs_path", "logs/paper")

    return PaperTradingConfig(
        trader_id=data["trader_id"],
        binance=binance,
        symbols=data.get("symbols", []),
        strategy_id=data.get("strategy_id"),
        sizing=sizing,
        risk=risk,
        db_path=Path(db_path_str) if db_path_str else None,
        logs_path=Path(logs_path_str),
        state_persistence_interval=int(data.get("state_persistence_interval", 60)),
    )


def save_config_to_json(config: PaperTradingConfig, config_path: Path) -> None:
    """Save paper trading config to JSON file.

    Args:
        config: PaperTradingConfig to save.
        config_path: Path to save JSON file.
    """
    data = {
        "trader_id": config.trader_id,
        "binance": {
            "api_key": config.binance.api_key,
            "api_secret": config.binance.api_secret,
            "testnet": config.binance.testnet,
            "account_type": config.binance.account_type,
        },
        "symbols": config.symbols,
        "strategy_id": config.strategy_id,
        "sizing": {
            "method": config.sizing.method,
            "max_leverage": str(config.sizing.max_leverage),
            "max_position_pct": str(config.sizing.max_position_pct),
            "risk_per_trade": str(config.sizing.risk_per_trade),
            "kelly_fraction": str(config.sizing.kelly_fraction),
            "atr_multiplier": str(config.sizing.atr_multiplier),
        },
        "risk": {
            "max_drawdown_pct": str(config.risk.max_drawdown_pct),
            "max_daily_loss_pct": str(config.risk.max_daily_loss_pct),
            "max_consecutive_losses": config.risk.max_consecutive_losses,
            "max_position_count": config.risk.max_position_count,
        },
        "db_path": str(config.db_path) if config.db_path else None,
        "logs_path": str(config.logs_path),
        "state_persistence_interval": config.state_persistence_interval,
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(data, f, indent=2)


async def run_with_config(config_path: Path, run_id: int | None = None) -> int:
    """Run paper trading with config file.

    Args:
        config_path: Path to JSON config file.
        run_id: Optional run ID for heartbeat registration.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        config = load_config_from_json(config_path)
    except ConfigurationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}", file=sys.stderr)
        return 1

    # Validate config
    errors = config.validate()
    if errors:
        print(f"Invalid configuration: {'; '.join(errors)}", file=sys.stderr)
        return 1

    # Start heartbeat thread if run_id provided
    if run_id is not None:
        run_with_heartbeat(run_id, config.db_path)

    try:
        await run_paper_trading(config)
        return 0
    except Exception as e:
        print(f"Paper trading error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Paper trading CLI",
        prog="python -m vibe_quant.paper.cli",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Start command
    start_parser = subparsers.add_parser("start", help="Start paper trading")
    start_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON config file",
    )
    start_parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run ID for heartbeat registration",
    )

    args = parser.parse_args()

    if args.command == "start":
        return asyncio.run(run_with_config(args.config, args.run_id))

    return 1


if __name__ == "__main__":
    sys.exit(main())
