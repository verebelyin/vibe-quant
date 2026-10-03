"""CLI for paper trading subprocess.

Entry point for background paper trading jobs spawned by dashboard.
Handles configuration loading, heartbeat registration, and graceful shutdown.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.jobs.manager import run_with_heartbeat
from vibe_quant.paper.config import (
    DEFAULT_MARGIN_TYPE,
    DEFAULT_PAPER_LOGS_PATH,
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
    RiskModuleConfig,
    SizingModuleConfig,
    credential_env_names,
)
from vibe_quant.paper.node import run_paper_trading

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe_quant.jobs.manager import BacktestJobManager


def _decimal_from_str(value: str | float | int) -> Decimal:
    """Convert value to Decimal."""
    return Decimal(str(value))


def _optional_decimal(data: dict[str, object], key: str) -> Decimal | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return _decimal_from_str(value)
    raise ConfigurationError(f"{key} must be a number, got {type(value).__name__}")


def load_config_from_json(config_path: Path) -> PaperTradingConfig:
    """Load paper trading config from JSON file.

    Credentials are never read from the file: testnet uses
    ``BINANCE_TESTNET_API_KEY/SECRET``, live uses ``BINANCE_API_KEY/SECRET``.
    ``binance.testnet`` defaults to True when absent.

    Raises:
        ConfigurationError: If config file is invalid or contains credentials.
    """
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        data = json.load(f)

    binance_data = data.get("binance", {})
    if "api_key" in binance_data or "api_secret" in binance_data:
        raise ConfigurationError(
            "Binance credentials must not be stored in config files; "
            "set them in the environment instead"
        )
    testnet = binance_data.get("testnet", True)
    if not isinstance(testnet, bool):
        raise ConfigurationError("binance.testnet must be true or false")
    key_var, secret_var = credential_env_names(testnet)
    binance = BinanceTestnetConfig(
        api_key=os.getenv(key_var, ""),
        api_secret=os.getenv(secret_var, ""),
        testnet=testnet,
        account_type=binance_data.get("account_type", "USDT_FUTURES"),
    )

    sizing_data = data.get("sizing", {})
    sizing = SizingModuleConfig(
        method=sizing_data.get("method", "fixed_fractional"),
        max_leverage=_decimal_from_str(sizing_data.get("max_leverage", 20)),
        max_position_pct=_optional_decimal(sizing_data, "max_position_pct"),
        risk_per_trade=_optional_decimal(sizing_data, "risk_per_trade"),
        kelly_fraction=_decimal_from_str(sizing_data.get("kelly_fraction", 0.5)),
        atr_multiplier=_decimal_from_str(sizing_data.get("atr_multiplier", 2.0)),
    )

    risk_data = data.get("risk", {})
    risk = RiskModuleConfig(
        max_drawdown_pct=_decimal_from_str(risk_data.get("max_drawdown_pct", 0.15)),
        max_daily_loss_pct=_decimal_from_str(risk_data.get("max_daily_loss_pct", 0.02)),
        max_consecutive_losses=int(risk_data.get("max_consecutive_losses", 10)),
        max_position_count=int(risk_data.get("max_position_count", 5)),
    )

    db_path_str = data.get("db_path")
    logs_path_str = data.get("logs_path", str(DEFAULT_PAPER_LOGS_PATH))
    leverage = data.get("leverage")
    validation_run_id = data.get("validation_run_id")

    return PaperTradingConfig(
        trader_id=data["trader_id"],
        binance=binance,
        symbols=list(data.get("symbols") or []),
        strategy_id=data.get("strategy_id"),
        validation_run_id=int(validation_run_id) if validation_run_id is not None else None,
        leverage=int(leverage) if leverage is not None else None,
        margin_type=str(data.get("margin_type", DEFAULT_MARGIN_TYPE)),
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
            # Credentials excluded - must come from env vars
            "testnet": config.binance.testnet,
            "account_type": config.binance.account_type,
        },
        "symbols": config.symbols,
        "strategy_id": config.strategy_id,
        "validation_run_id": config.validation_run_id,
        "leverage": config.leverage,
        "margin_type": config.margin_type,
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


async def run_with_config(
    config_path: Path, run_id: int | None = None, *, allow_live: bool = False
) -> int:
    """Run paper trading with config file.

    Args:
        config_path: Path to JSON config file.
        run_id: Optional run ID for heartbeat registration.
        allow_live: Must be True (``--live``) to run a config with testnet=false.

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

    if not config.binance.testnet and not allow_live:
        print(
            "Refusing to trade LIVE: config has binance.testnet=false; "
            "re-run with --live to confirm real-money trading",
            file=sys.stderr,
        )
        return 1

    # Validate config
    errors = config.validate()
    if errors:
        print(f"Invalid configuration: {'; '.join(errors)}", file=sys.stderr)
        return 1

    heartbeat_manager: BacktestJobManager | None = None
    stop_heartbeat: Callable[[], None] | None = None

    # Start heartbeat thread if run_id provided
    if run_id is not None:
        heartbeat_manager, stop_heartbeat = run_with_heartbeat(run_id, config.db_path)

    try:
        await run_paper_trading(config)
        if run_id is not None and heartbeat_manager is not None:
            with contextlib.suppress(Exception):
                heartbeat_manager.mark_completed(run_id)
        return 0
    except Exception as e:
        if run_id is not None and heartbeat_manager is not None:
            with contextlib.suppress(Exception):
                heartbeat_manager.mark_completed(run_id, error=f"{type(e).__name__}: {e}")
        print(f"Paper trading error: {e}", file=sys.stderr)
        return 1
    finally:
        if stop_heartbeat is not None:
            with contextlib.suppress(Exception):
                stop_heartbeat()
        if heartbeat_manager is not None:
            with contextlib.suppress(Exception):
                heartbeat_manager.close()


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
    start_parser.add_argument(
        "--live",
        action="store_true",
        help="Confirm LIVE (real-money) trading for configs with binance.testnet=false",
    )

    args = parser.parse_args()

    if args.command == "start":
        return asyncio.run(run_with_config(args.config, args.run_id, allow_live=args.live))

    return 1


if __name__ == "__main__":
    sys.exit(main())
