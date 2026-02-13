"""Result extraction helpers for validation runner.

Extracts metrics and trades from NautilusTrader backtest output
into vibe-quant's ValidationResult format.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vibe_quant.validation.fill_model import SlippageEstimator
from vibe_quant.validation.results import TradeRecord, ValidationResult

if TYPE_CHECKING:
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.backtest.results import BacktestResult

    from vibe_quant.validation.venue import VenueConfig

logger = logging.getLogger(__name__)


def _ns_to_isoformat(ns_timestamp: Any) -> str:
    """Convert a nanosecond Unix timestamp to ISO 8601 string.

    NautilusTrader Position.ts_opened / ts_closed are uint64
    nanosecond timestamps.  ``str()`` gives a bare integer string
    which breaks ``datetime.fromisoformat()``.
    """
    ns = int(ns_timestamp)
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat()


def extract_results(
    run_id: int,
    strategy_name: str,
    bt_result: BacktestResult,
    engine: BacktestEngine,
    venue_config: VenueConfig,
) -> ValidationResult:
    """Extract ValidationResult from NautilusTrader backtest output."""
    result = ValidationResult(
        run_id=run_id,
        strategy_name=strategy_name,
        starting_balance=venue_config.starting_balance_usdt,
    )

    if bt_result is None:
        return result

    result.execution_time_seconds = bt_result.elapsed_time
    result.total_trades = bt_result.total_positions

    extract_stats(result, bt_result)
    extract_trades(result, engine, venue_config)

    return result


def extract_stats(
    result: ValidationResult,
    bt_result: BacktestResult,
) -> None:
    """Extract aggregate statistics from BacktestResult into ValidationResult.

    NT's PortfolioAnalyzer populates stats_pnls with PnL and any
    registered statistics keyed by their ``name`` attribute, and
    stats_returns with the same registered statistics.

    Known key names from NT 1.222 (Rust statistics):
        stats_pnls:  "PnL (total)", "PnL% (total)", "Sharpe Ratio (252 days)",
                     "Sortino Ratio (252 days)", "Max Drawdown", "Win Rate",
                     "Profit Factor", "Expectancy", "Avg Winner", "Avg Loser"
        stats_returns: same statistic names

    Args:
        result: ValidationResult to populate (mutated in place).
        bt_result: NautilusTrader BacktestResult.
    """
    stats_returns = bt_result.stats_returns or {}
    stats_pnls = bt_result.stats_pnls or {}

    _known_pnl_keys = {"pnl (total)", "pnl% (total)", "sharpe", "sortino",
                       "max drawdown", "win rate", "profit factor",
                       "expectancy", "avg winner", "avg loser", "long ratio"}

    # Track which fields were populated from stats_pnls so we only
    # fall back to stats_returns for fields that weren't set.
    _populated: set[str] = set()

    for _currency, pnl_stats in stats_pnls.items():
        for key, value in pnl_stats.items():
            if value is None:
                continue
            key_lower = key.lower()
            fval = float(value)
            if key_lower == "pnl% (total)":
                # NT reports as percentage (e.g. -13.06 for -13.06%);
                # we store as fraction (e.g. -0.1306)
                result.total_return = fval / 100.0
                _populated.add("total_return")
            elif "sharpe" in key_lower:
                result.sharpe_ratio = fval
                _populated.add("sharpe_ratio")
            elif "sortino" in key_lower:
                result.sortino_ratio = fval
                _populated.add("sortino_ratio")
            elif key_lower == "max drawdown":
                result.max_drawdown = abs(fval)
                _populated.add("max_drawdown")
            elif key_lower == "win rate":
                result.win_rate = fval
                _populated.add("win_rate")
            elif key_lower == "profit factor":
                result.profit_factor = fval
                _populated.add("profit_factor")
            elif key_lower == "avg winner":
                result.avg_win = fval
            elif key_lower == "avg loser":
                result.avg_loss = fval
            elif not any(k in key_lower for k in _known_pnl_keys):
                logger.debug("Unmatched PnL stats key: %s = %s", key, value)

    _known_returns_keys = {"sharpe", "sortino", "max drawdown", "win rate",
                           "profit factor", "expectancy", "avg winner",
                           "avg loser", "long ratio"}
    for key, value in stats_returns.items():
        if value is None:
            continue
        key_lower = key.lower()
        fval = float(value)
        if "sharpe" in key_lower and "sharpe_ratio" not in _populated:
            result.sharpe_ratio = fval
        elif "sortino" in key_lower and "sortino_ratio" not in _populated:
            result.sortino_ratio = fval
        elif "max drawdown" in key_lower and "max_drawdown" not in _populated:
            result.max_drawdown = abs(fval)
        elif key_lower == "win rate" and "win_rate" not in _populated:
            result.win_rate = fval
        elif key_lower == "profit factor" and "profit_factor" not in _populated:
            result.profit_factor = fval
        elif not any(k in key_lower for k in _known_returns_keys):
            logger.debug("Unmatched returns stats key: %s = %s", key, value)


def extract_trades(
    result: ValidationResult,
    engine: BacktestEngine,
    venue_config: VenueConfig,
) -> None:
    """Extract individual trade records from the engine's closed positions.

    Uses the Position objects from the engine cache directly, since the
    positions report DataFrame column names can vary across NT versions.

    Args:
        result: ValidationResult to populate trades on (mutated in place).
        engine: BacktestEngine after run.
        venue_config: Venue config for default leverage.
    """
    try:
        positions = engine.kernel.cache.positions()
    except Exception:
        logger.warning("Could not read positions from engine cache", exc_info=True)
        return

    if not positions:
        return

    default_leverage = int(venue_config.default_leverage)
    winning = 0
    losing = 0
    total_fees = 0.0
    total_slippage = 0.0

    fill_cfg = venue_config.fill_config
    impact_k = getattr(fill_cfg, "impact_coefficient", 0.1) if fill_cfg else 0.1
    slippage_estimator = SlippageEstimator(impact_coefficient=impact_k)

    avg_bar_volume, bar_volatility = estimate_market_stats(engine)

    for pos in positions:
        if not pos.is_closed:
            continue

        realized_pnl = float(pos.realized_pnl)
        entry_price = float(pos.avg_px_open)
        exit_price = float(pos.avg_px_close)
        # pos.quantity is 0 for closed positions; use peak_qty for trade size
        quantity = float(pos.peak_qty)

        pos_fees = sum(float(c) for c in pos.commissions())
        total_fees += abs(pos_fees)

        if realized_pnl > 0:
            winning += 1
        elif realized_pnl < 0:
            losing += 1

        slippage_cost = slippage_estimator.estimate_cost(
            entry_price=entry_price,
            order_size=quantity,
            avg_volume=avg_bar_volume,
            volatility=bar_volatility,
            spread=0.0001,
        )
        total_slippage += slippage_cost

        if entry_price > 0 and quantity > 0:
            notional = entry_price * quantity
            roi_pct = (realized_pnl / notional) * 100.0
        else:
            roi_pct = 0.0

        entry_time = _ns_to_isoformat(pos.ts_opened)
        exit_time = _ns_to_isoformat(pos.ts_closed) if pos.ts_closed else None

        direction = "LONG" if str(pos.entry).upper() == "BUY" else "SHORT"
        instrument_id = str(pos.instrument_id)

        trade = TradeRecord(
            symbol=instrument_id,
            direction=direction,
            leverage=default_leverage,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            entry_fee=abs(pos_fees) / 2.0,
            exit_fee=abs(pos_fees) / 2.0,
            slippage_cost=slippage_cost,
            gross_pnl=realized_pnl + abs(pos_fees),
            net_pnl=realized_pnl,
            roi_percent=roi_pct,
            exit_reason="signal",
        )
        result.trades.append(trade)

    result.total_trades = len(result.trades)
    result.trades.sort(key=lambda t: t.entry_time)
    result.winning_trades = winning
    result.losing_trades = losing
    result.total_fees = total_fees
    result.total_slippage = total_slippage
    if result.total_trades > 0:
        result.win_rate = winning / result.total_trades

    compute_extended_metrics(result)


def estimate_market_stats(engine: BacktestEngine) -> tuple[float, float]:
    """Estimate average bar volume and daily volatility from engine cache.

    Reads bars from the engine cache to compute realistic slippage
    parameters instead of using hardcoded values.

    Args:
        engine: BacktestEngine after run.

    Returns:
        Tuple of (avg_bar_volume, daily_volatility). Falls back to
        conservative defaults (1000.0, 0.02) if data is unavailable.
    """
    default_volume = 1000.0
    default_volatility = 0.02

    try:
        bars = engine.kernel.cache.bars()
        if not bars:
            return default_volume, default_volatility

        volumes: list[float] = []
        closes: list[float] = []
        for bar in bars:
            vol = float(bar.volume)
            if vol > 0:
                volumes.append(vol)
            close = float(bar.close)
            if close > 0:
                closes.append(close)

        avg_volume = sum(volumes) / len(volumes) if volumes else default_volume

        volatility = default_volatility
        if len(closes) >= 2:
            log_returns: list[float] = []
            for i in range(1, len(closes)):
                if closes[i - 1] > 0:
                    log_returns.append(math.log(closes[i] / closes[i - 1]))
            if len(log_returns) >= 2:
                mean_r = sum(log_returns) / len(log_returns)
                var = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
                volatility = math.sqrt(var) if var > 0 else default_volatility

        return avg_volume, volatility
    except Exception:
        logger.debug("Could not estimate market stats, using defaults", exc_info=True)
        return default_volume, default_volatility


def compute_extended_metrics(result: ValidationResult) -> None:
    """Compute SPEC-required extended metrics from trades.

    Populates: largest_win/loss, avg_win/loss, max_consecutive_wins/losses,
    avg_trade_duration_hours, cagr, volatility_annual, calmar_ratio.

    Args:
        result: ValidationResult to populate (mutated in place).
    """
    if not result.trades:
        return

    wins: list[float] = []
    losses: list[float] = []
    durations_hours: list[float] = []

    max_con_wins = 0
    max_con_losses = 0
    cur_wins = 0
    cur_losses = 0

    for trade in result.trades:
        pnl = trade.net_pnl
        if pnl > 0:
            wins.append(pnl)
            cur_wins += 1
            max_con_wins = max(max_con_wins, cur_wins)
            cur_losses = 0
        elif pnl < 0:
            losses.append(pnl)
            cur_losses += 1
            max_con_losses = max(max_con_losses, cur_losses)
            cur_wins = 0
        else:
            cur_wins = 0
            cur_losses = 0

        if trade.entry_time and trade.exit_time:
            try:
                entry_dt = datetime.fromisoformat(trade.entry_time.replace("Z", "+00:00"))
                exit_dt = datetime.fromisoformat(trade.exit_time.replace("Z", "+00:00"))
                duration_h = (exit_dt - entry_dt).total_seconds() / 3600.0
                if duration_h >= 0:
                    durations_hours.append(duration_h)
            except (ValueError, TypeError):
                pass

    result.max_consecutive_wins = max_con_wins
    result.max_consecutive_losses = max_con_losses

    if wins:
        result.largest_win = max(wins)
        result.avg_win = sum(wins) / len(wins)
    if losses:
        result.largest_loss = min(losses)
        result.avg_loss = sum(losses) / len(losses)

    if durations_hours:
        result.avg_trade_duration_hours = sum(durations_hours) / len(durations_hours)

    if result.total_return != 0.0 and result.trades:
        try:
            first_entry = datetime.fromisoformat(
                result.trades[0].entry_time.replace("Z", "+00:00")
            )
            last_exit_str = result.trades[-1].exit_time or result.trades[-1].entry_time
            last_exit = datetime.fromisoformat(last_exit_str.replace("Z", "+00:00"))
            days = max((last_exit - first_entry).total_seconds() / 86400.0, 1.0)
            # total_return is stored as a decimal fraction from NT stats
            # (e.g. 0.12 = 12%). Use directly — no heuristic conversion.
            total_return_frac = result.total_return
            if total_return_frac > -1.0:
                result.cagr = ((1.0 + total_return_frac) ** (365.0 / days)) - 1.0
        except (ValueError, TypeError):
            pass

    if len(result.trades) >= 2:
        trade_returns = [t.roi_percent / 100.0 for t in result.trades if t.roi_percent != 0.0]
        if len(trade_returns) >= 2:
            mean_r = sum(trade_returns) / len(trade_returns)
            var = sum((r - mean_r) ** 2 for r in trade_returns) / (len(trade_returns) - 1)
            if durations_hours:
                avg_dur_days = max(sum(durations_hours) / len(durations_hours) / 24.0, 0.01)
                trades_per_year = 365.0 / avg_dur_days
            else:
                trades_per_year = 252.0
            result.volatility_annual = math.sqrt(var * trades_per_year) if var > 0 else 0.0

    if result.max_drawdown > 0 and result.cagr != 0:
        result.calmar_ratio = result.cagr / result.max_drawdown
