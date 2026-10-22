"""CLI for paper trading subprocess.

Entry point for background paper trading jobs spawned by dashboard.
Handles configuration loading, heartbeat registration, and graceful shutdown.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.alerts.telegram import (
    ConfigurationError as TelegramConfigurationError,
)
from vibe_quant.alerts.telegram import (
    TelegramUnavailableError,
    redact_token,
)
from vibe_quant.jobs.manager import (
    BacktestJobManager,
    process_start_time,
    run_with_heartbeat,
)
from vibe_quant.paper import lineage
from vibe_quant.paper.account_lock import (
    AccountLock,
    acquire_account_lock,
    acquire_trader_lock,
)
from vibe_quant.paper.config import (
    DEFAULT_PAPER_LOGS_PATH,
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
    RiskModuleConfig,
    SizingModuleConfig,
    account_id,
    credential_env_names,
)
from vibe_quant.paper.node import run_paper_trading, validated_leverage_override
from vibe_quant.paper.persistence import StatePersistence
from vibe_quant.sizing_settings import SizingConfigError, SizingSettings

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

#: Refusal errors whose message is curated and operator-facing: every ``raise`` site
#: of these three was audited and interpolates no secret, so their message is printed
#: (redacted again, defence in depth). Subclasses count via ``isinstance``; anything
#: else stays class-only on stderr.
_ALLOWLISTED_ERRORS = (
    ConfigurationError,
    TelegramConfigurationError,
    TelegramUnavailableError,
)


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


def _margin_cushion(sizing_data: dict[str, object]) -> Decimal | None:
    """``sizing.max_margin_pct``: absent = None (paper default); a stored null is an error.

    The T3 validator (:meth:`SizingSettings.from_params`) decides, not a second copy.
    """
    if "max_margin_pct" not in sizing_data:
        return None
    try:
        value = SizingSettings.from_params(
            {"max_margin_pct": sizing_data["max_margin_pct"]}
        ).max_margin_pct
    except SizingConfigError as exc:
        raise ConfigurationError(f"sizing.max_margin_pct: {exc}") from exc
    return None if value is None else Decimal(str(value))


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
        max_margin_pct=_margin_cushion(sizing_data),
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
    raw_leverage = data.get("leverage")
    validation_run_id = data.get("validation_run_id")
    # The override is validated by the shared whole-number rule, never int()-cast
    # (which silently turned 2.5 into 2 and True into 1).
    leverage = validated_leverage_override(raw_leverage) if raw_leverage is not None else None
    # An absent or null margin_type means "derive it from the validation run's
    # liquidation mode"; an explicit CROSSED/ISOLATED is an override. Anything
    # else fails the load instead of silently becoming CROSSED.
    raw_margin_type = data.get("margin_type")
    if raw_margin_type is None:
        margin_type: str | None = None
    elif raw_margin_type in ("CROSSED", "ISOLATED"):
        margin_type = str(raw_margin_type)
    else:
        raise ConfigurationError(
            "margin_type must be CROSSED, ISOLATED or null (derive from the "
            f"validation run), got {raw_margin_type!r}"
        )

    start_fresh = data.get("start_fresh", False)
    if not isinstance(start_fresh, bool):
        raise ConfigurationError("start_fresh must be true or false")

    trade_alerts = data.get("trade_alerts", False)
    if not isinstance(trade_alerts, bool):
        raise ConfigurationError("trade_alerts must be true or false")

    return PaperTradingConfig(
        trader_id=data["trader_id"],
        binance=binance,
        symbols=list(data.get("symbols") or []),
        strategy_id=data.get("strategy_id"),
        validation_run_id=int(validation_run_id) if validation_run_id is not None else None,
        leverage=leverage,
        margin_type=margin_type,
        sizing=sizing,
        risk=risk,
        db_path=Path(db_path_str) if db_path_str else None,
        logs_path=Path(logs_path_str),
        state_persistence_interval=int(data.get("state_persistence_interval", 60)),
        start_fresh=start_fresh,
        trade_alerts=trade_alerts,
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
            **(
                {"max_margin_pct": float(config.sizing.max_margin_pct)}
                if config.sizing.max_margin_pct is not None
                else {}
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
        "start_fresh": config.start_fresh,
        "trade_alerts": config.trade_alerts,
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(data, f, indent=2)


def _has_account_checkpoint(config: PaperTradingConfig) -> bool:
    """Does a LIVE start on this config's account have saved risk state to carry?

    Any trader's row on the account, or this trader's own row on another key of the
    same env (a key rotation; ``vibe_quant.paper.lineage.has_prior_state``).
    Fail closed: a lookup that cannot answer counts as "no checkpoint", so a LIVE
    start is refused rather than starting the limits from zero. Class only in the
    log: this runs on the money path.
    """
    try:
        persistence = StatePersistence(
            db_path=config.db_path, trader_id=config.trader_id
        )
        try:
            return lineage.has_prior_state(
                persistence, account_id(config.binance), config.trader_id
            )
        finally:
            persistence.close()
    except Exception as exc:
        logger.warning("paper account checkpoint lookup failed (%s)", type(exc).__name__)
        return False


def _claim_own_process(manager: BacktestJobManager, run_id: int) -> str | None:
    """qgtdz: record this process in the job row before trading; text = refuse."""
    pid = os.getpid()
    try:
        return manager.claim_process(run_id, pid, process_start_time(pid))
    except Exception as exc:
        logger.error("paper process claim failed (%s)", type(exc).__name__)
        return f"Refusing to trade: could not record this process for run {run_id} ({type(exc).__name__})"


def _persist_refusal(run_id: int, db_path: Path | None, message: str) -> None:
    """Record a consent refusal as the run's and job's terminal failure.

    An API-launched child (e.g. a LIVE ``/restore`` of a trader with no saved
    risk state to carry) exits 1 here; without this its run stays ``running``
    with no error and a later job sync only reports "Process terminated
    unexpectedly". No heartbeat is started and no lock is taken: a refused start
    must leave nothing held. Best effort, class only in the log (money path): the
    refusal and exit code are unchanged.
    """
    try:
        manager = BacktestJobManager(db_path)
        try:
            manager.mark_completed(run_id, error=message)
        finally:
            manager.close()
    except Exception as exc:
        logger.error("paper refusal could not be persisted (%s)", type(exc).__name__)


async def run_with_config(
    config_path: Path,
    run_id: int | None = None,
    *,
    allow_live: bool = False,
    allow_fresh_live: bool = False,
) -> int:
    """Run paper trading with config file.

    Args:
        config_path: Path to JSON config file.
        run_id: Optional run ID for heartbeat registration.
        allow_live: Must be True (``--live``) to run a config with testnet=false.
        allow_fresh_live: Must be True (``--confirm-fresh-live``) to run a LIVE
            config with ``start_fresh`` (skips another trader id's risk state;
            this trader id's own saved limits and halt are kept), or a LIVE start
            with no saved risk state to carry (limits start from current equity).

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

    # v4l2n + 30vki: a LIVE start_fresh skips the risk state ANOTHER trader id left
    # on the account, so --live alone is not enough consent (own state always carries).
    if not config.binance.testnet and config.start_fresh and not allow_fresh_live:
        print(
            "Refusing LIVE start_fresh: it skips the risk state another trader id left on this "
            "account (peak, daily loss, losing streak, ERROR halt), so those limits may count from "
            "current equity; this trader id's own saved limits and halt are kept; re-run with "
            "--confirm-fresh-live to confirm",
            file=sys.stderr,
        )
        return 1

    # 6ub67 + neuvu.2: a LIVE start with no saved risk state to carry (a first
    # session, or a new trader id on a new key: no checkpoint on the account and no
    # earlier live session of this trader on another key) would start the limits
    # from zero, so consent is required. A key rotation by the same trader id carries
    # its risk and halt and needs none. Before the lock and the heartbeat: a refused
    # start must leave nothing held.
    if (
        not config.binance.testnet
        and not allow_fresh_live
        and not _has_account_checkpoint(config)
    ):
        message = (
            "Refusing LIVE start: no saved risk state to carry (no checkpoint on this "
            "account and no earlier live session of this trader on another key): "
            "drawdown, daily-loss and losing-streak limits would start from zero; "
            "re-run with --confirm-fresh-live to confirm"
        )
        print(message, file=sys.stderr)
        if run_id is not None:
            _persist_refusal(run_id, config.db_path, message)
        return 1

    # Validate config
    errors = config.validate()
    if errors:
        print(f"Invalid configuration: {'; '.join(errors)}", file=sys.stderr)
        return 1

    heartbeat_manager: BacktestJobManager | None = None
    stop_heartbeat: Callable[[], None] | None = None
    account_lock: AccountLock | None = None
    trader_lock: AccountLock | None = None

    # Start heartbeat thread if run_id provided
    if run_id is not None:
        heartbeat_manager, stop_heartbeat = run_with_heartbeat(run_id, config.db_path)

    try:
        if run_id is not None and heartbeat_manager is not None:
            refusal = _claim_own_process(heartbeat_manager, run_id)
            if refusal is not None:
                print(refusal, file=sys.stderr)
                return 1
        # rw1g8: one paper session per account. Held for the whole process life;
        # the kernel drops it on death (SIGKILL included), so no stale lock.
        account_lock = acquire_account_lock(account_id(config.binance), config.trader_id)
        # 2q7cv: one session per trader id too, on any key and either venue; the
        # account lock comes first so a same-account start keeps its AccountBusyError.
        trader_lock = acquire_trader_lock(config.trader_id, account_id(config.binance))
        await run_paper_trading(config)
        if run_id is not None and heartbeat_manager is not None:
            try:
                heartbeat_manager.mark_completed(run_id)
            except Exception as exc:
                logger.error("heartbeat mark_completed failed (%s)", type(exc).__name__)
        return 0
    except Exception as e:
        if run_id is not None and heartbeat_manager is not None:
            try:
                heartbeat_manager.mark_completed(run_id, error=f"{type(e).__name__}: {e}")
            except Exception as exc:
                # R29: class only, never ``logger.exception`` - this runs inside an
                # active except, so the chained error's text must not reach the log.
                logger.error("heartbeat mark_completed failed (%s)", type(exc).__name__)
        # u52nu: an allowlisted refusal prints its curated message (redacted again as
        # defence in depth); every other exception keeps the class-only line.
        detail = f": {redact_token(str(e))}" if isinstance(e, _ALLOWLISTED_ERRORS) else ""
        if run_id is not None:
            print(
                f"Paper trading error: {type(e).__name__}{detail} (run {run_id})",
                file=sys.stderr,
            )
        else:
            print(f"Paper trading error: {type(e).__name__}{detail}", file=sys.stderr)
        return 1
    finally:
        if trader_lock is not None:
            trader_lock.release()
        if account_lock is not None:
            account_lock.release()
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
    start_parser.add_argument(
        "--confirm-fresh-live",
        action="store_true",
        help=(
            "Confirm a LIVE start_fresh (skips the risk state another trader id left on the "
            "account) or a LIVE start with no saved risk state to carry (drawdown, daily-loss "
            "and losing-streak limits start from zero)"
        ),
    )

    args = parser.parse_args()

    if args.command == "start":
        return asyncio.run(
            run_with_config(
                args.config,
                args.run_id,
                allow_live=args.live,
                allow_fresh_live=args.confirm_fresh_live,
            )
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
