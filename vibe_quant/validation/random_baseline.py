"""Random short entry baseline for 1m strategy validation.

Null hypothesis test: enter SHORT randomly on 1m bars with the same
SL/TP as champion strategies, measure Sharpe/DD/PF.  If random entry
with a given SL/TP produces Sharpe ≥ 2, the "alpha" is just SL/TP
geometry + systematic BTC downward drift — not indicator skill.

Usage (CLI):
    python -m vibe_quant.validation.random_baseline \
        --sl-pct 0.59 --tp-pct 10.55 \
        --start 2026-01-10 --end 2026-03-10 \
        --target-trades 50 --monte-carlo 1000

Usage (Python):
    from vibe_quant.validation.random_baseline import (
        load_ohlc, run_random_short_baseline, BaselineConfig
    )
    cfg = BaselineConfig(sl_pct=0.59, tp_pct=10.55, target_trades=50)
    bars = load_ohlc("BTCUSDT", "1m", "2026-01-10", "2026-03-10")
    result = run_random_short_baseline(bars, cfg, n_simulations=1000)
    print(result.summary())
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Taker fee per side (Binance perp default)
DEFAULT_TAKER_FEE = 0.0005  # 0.05%
DEFAULT_ARCHIVE_PATH = Path("data/archive/raw_data.db")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OHLCBar:
    """Single OHLC bar with timestamp."""

    ts: int  # open_time in ms
    open: float
    high: float
    low: float
    close: float


def load_ohlc(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    archive_path: Path = DEFAULT_ARCHIVE_PATH,
) -> list[OHLCBar]:
    """Load OHLC bars from SQLite archive.

    Args:
        symbol: e.g. "BTCUSDT"
        interval: e.g. "1m"
        start_date: ISO date string, inclusive
        end_date: ISO date string, exclusive
        archive_path: Path to raw_data.db

    Returns:
        List of OHLCBar sorted by timestamp.
    """
    from datetime import datetime

    start_ms = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=UTC)
        .timestamp()
        * 1000
    )
    end_ms = int(
        datetime.strptime(end_date, "%Y-%m-%d")
        .replace(tzinfo=UTC)
        .timestamp()
        * 1000
    )

    conn = sqlite3.connect(str(archive_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            "SELECT open_time, open, high, low, close FROM raw_klines "
            "WHERE symbol = ? AND interval = ? "
            "AND open_time >= ? AND open_time < ? "
            "ORDER BY open_time",
            (symbol, interval, start_ms, end_ms),
        ).fetchall()
    finally:
        conn.close()

    bars = [OHLCBar(ts=r[0], open=r[1], high=r[2], low=r[3], close=r[4]) for r in rows]
    logger.info("Loaded %d %s %s bars (%s to %s)", len(bars), symbol, interval, start_date, end_date)
    return bars


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Configuration for random baseline simulation."""

    sl_pct: float  # Stop loss %, e.g. 0.59
    tp_pct: float  # Take profit %, e.g. 10.55
    target_trades: int = 50  # Target number of trades per simulation
    taker_fee: float = DEFAULT_TAKER_FEE
    initial_capital: float = 100_000.0
    position_size_pct: float = 1.0  # % of capital per trade (notional, not margin)


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TradeResult:
    """Single trade outcome."""

    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    pnl_pct: float  # After fees
    hit_tp: bool
    hit_sl: bool


def _simulate_single_run(
    bars: list[OHLCBar],
    entry_indices: np.ndarray,
    sl_pct: float,
    tp_pct: float,
    taker_fee: float,
) -> list[TradeResult]:
    """Simulate SHORT trades at given entry bar indices.

    For each entry:
    - Enter SHORT at the entry bar's close price
    - Walk forward bar-by-bar checking HIGH (SL) and LOW (TP)
    - SL hit if bar HIGH >= entry * (1 + sl_pct/100)
    - TP hit if bar LOW <= entry * (1 - tp_pct/100)
    - If both hit in same bar: assume SL hit first (conservative)
    - If neither hit by end of data: exit at last bar close
    """
    trades: list[TradeResult] = []
    n_bars = len(bars)
    current_exit = 0  # Ensure non-overlapping trades

    for entry_idx in entry_indices:
        if entry_idx < current_exit or entry_idx >= n_bars - 1:
            continue

        entry_price = bars[entry_idx].close
        sl_price = entry_price * (1.0 + sl_pct / 100.0)
        tp_price = entry_price * (1.0 - tp_pct / 100.0)

        hit_tp = False
        hit_sl = False
        exit_idx = n_bars - 1
        exit_price = bars[-1].close

        for j in range(entry_idx + 1, n_bars):
            bar = bars[j]
            # Check SL first (conservative for shorts: price goes UP)
            if bar.high >= sl_price:
                exit_price = sl_price
                exit_idx = j
                hit_sl = True
                break
            # Check TP (price goes DOWN)
            if bar.low <= tp_price:
                exit_price = tp_price
                exit_idx = j
                hit_tp = True
                break

        # PnL: short position → profit when price drops
        raw_pnl_pct = (entry_price - exit_price) / entry_price * 100.0
        # Fees: entry + exit (taker both sides)
        fee_pct = taker_fee * 100.0 * 2.0
        net_pnl_pct = raw_pnl_pct - fee_pct

        trades.append(
            TradeResult(
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=net_pnl_pct,
                hit_tp=hit_tp,
                hit_sl=hit_sl,
            )
        )
        current_exit = exit_idx + 1

    return trades


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimulationMetrics:
    """Metrics from a single simulation run."""

    sharpe: float
    sortino: float
    max_drawdown: float  # Positive fraction, e.g. 0.12 = 12%
    total_return: float  # Decimal fraction, e.g. 0.15 = +15%
    profit_factor: float
    win_rate: float
    total_trades: int
    total_fees_pct: float


def _compute_metrics(trades: list[TradeResult], taker_fee: float) -> SimulationMetrics:
    """Compute performance metrics from a list of trades."""
    if not trades:
        return SimulationMetrics(
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            total_return=0.0,
            profit_factor=0.0,
            win_rate=0.0,
            total_trades=0,
            total_fees_pct=0.0,
        )

    pnls = np.array([t.pnl_pct for t in trades])
    n = len(pnls)

    # Basic stats
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = len(wins) / n if n > 0 else 0.0

    # Total return (compounded)
    equity_curve = np.cumprod(1.0 + pnls / 100.0)
    total_return = equity_curve[-1] - 1.0

    # Max drawdown from equity curve
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe (annualized, assuming ~365*24*60 1m bars per year but
    # we use trade-level returns, so annualize by sqrt(trades_per_year))
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls, ddof=1)) if n > 1 else 1.0
    # Annualization: assume ~252 trading days, scale by trades per day
    sharpe = mean_pnl / std_pnl * np.sqrt(n) if std_pnl > 1e-10 else 0.0

    # Sortino
    downside = pnls[pnls < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1.0
    sortino = mean_pnl / downside_std * np.sqrt(n) if downside_std > 1e-10 else 0.0

    # Profit factor
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    # Total fees
    total_fees_pct = taker_fee * 100.0 * 2.0 * n

    return SimulationMetrics(
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=max_dd,
        total_return=total_return,
        profit_factor=profit_factor,
        win_rate=win_rate,
        total_trades=n,
        total_fees_pct=total_fees_pct,
    )


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
    """Aggregated results from Monte Carlo random baseline."""

    config: BaselineConfig
    n_bars: int
    n_simulations: int
    metrics: list[SimulationMetrics]
    # Aggregated stats
    sharpe_mean: float
    sharpe_median: float
    sharpe_std: float
    sharpe_p5: float  # 5th percentile
    sharpe_p95: float  # 95th percentile
    return_mean: float
    return_median: float
    dd_mean: float
    dd_median: float
    pf_mean: float
    pf_median: float
    wr_mean: float
    trades_mean: float
    # How many simulations achieved Sharpe >= threshold
    pct_sharpe_above_1: float
    pct_sharpe_above_2: float
    pct_sharpe_above_3: float

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Random Short Baseline ({self.n_simulations} simulations) ===",
            f"Config: SL={self.config.sl_pct}% TP={self.config.tp_pct}% "
            f"target_trades={self.config.target_trades}",
            f"Data: {self.n_bars} bars",
            "",
            f"Sharpe:  mean={self.sharpe_mean:.2f}  median={self.sharpe_median:.2f}  "
            f"std={self.sharpe_std:.2f}  [5%={self.sharpe_p5:.2f}, 95%={self.sharpe_p95:.2f}]",
            f"Return:  mean={self.return_mean * 100:.1f}%  median={self.return_median * 100:.1f}%",
            f"MaxDD:   mean={self.dd_mean * 100:.1f}%  median={self.dd_median * 100:.1f}%",
            f"PF:      mean={self.pf_mean:.2f}  median={self.pf_median:.2f}",
            f"WR:      mean={self.wr_mean * 100:.1f}%",
            f"Trades:  mean={self.trades_mean:.0f}",
            "",
            "Sharpe distribution:",
            f"  >= 1.0: {self.pct_sharpe_above_1 * 100:.1f}% of simulations",
            f"  >= 2.0: {self.pct_sharpe_above_2 * 100:.1f}% of simulations",
            f"  >= 3.0: {self.pct_sharpe_above_3 * 100:.1f}% of simulations",
            "",
        ]

        # Verdict
        if self.pct_sharpe_above_2 > 0.10:
            lines.append(
                "VERDICT: >10% of random entries achieve Sharpe >= 2.0. "
                "The SL/TP geometry alone explains much of the 'alpha'. "
                "Indicator skill is questionable."
            )
        elif self.pct_sharpe_above_1 > 0.25:
            lines.append(
                "VERDICT: >25% of random entries achieve Sharpe >= 1.0. "
                "Some alpha from indicators, but directional bias contributes significantly."
            )
        else:
            lines.append(
                "VERDICT: Random entries rarely achieve high Sharpe. "
                "Indicator-based entry timing provides genuine alpha beyond the short bias."
            )
        return "\n".join(lines)


def run_random_short_baseline(
    bars: list[OHLCBar],
    config: BaselineConfig,
    n_simulations: int = 1000,
    seed: int | None = 42,
) -> BaselineResult:
    """Run Monte Carlo random short baseline.

    For each simulation:
    1. Randomly select ~target_trades bar indices as entry points
    2. Simulate SHORT trades with given SL/TP
    3. Compute Sharpe/DD/PF/return

    Aggregate across all simulations.

    Args:
        bars: OHLC data loaded via load_ohlc()
        config: SL/TP and trade parameters
        n_simulations: Number of Monte Carlo runs
        seed: Random seed for reproducibility (None = random)

    Returns:
        BaselineResult with aggregated statistics
    """
    rng = np.random.default_rng(seed)
    n_bars = len(bars)

    if n_bars < 100:
        msg = f"Need at least 100 bars, got {n_bars}"
        raise ValueError(msg)

    # Entry probability to achieve target_trades
    # We select slightly more candidates since non-overlapping trades
    # reduce the actual count
    entry_prob = min(0.5, config.target_trades * 1.5 / n_bars)

    all_metrics: list[SimulationMetrics] = []

    for i in range(n_simulations):
        # Random entry indices
        mask = rng.random(n_bars) < entry_prob
        entry_indices = np.where(mask)[0]

        trades = _simulate_single_run(
            bars, entry_indices, config.sl_pct, config.tp_pct, config.taker_fee
        )
        metrics = _compute_metrics(trades, config.taker_fee)
        all_metrics.append(metrics)

        if (i + 1) % 200 == 0:
            logger.info("Completed %d / %d simulations", i + 1, n_simulations)

    # Aggregate
    sharpes = np.array([m.sharpe for m in all_metrics])
    returns = np.array([m.total_return for m in all_metrics])
    dds = np.array([m.max_drawdown for m in all_metrics])
    pfs = np.array([m.profit_factor for m in all_metrics])
    wrs = np.array([m.win_rate for m in all_metrics])
    trade_counts = np.array([m.total_trades for m in all_metrics])

    return BaselineResult(
        config=config,
        n_bars=n_bars,
        n_simulations=n_simulations,
        metrics=all_metrics,
        sharpe_mean=float(np.mean(sharpes)),
        sharpe_median=float(np.median(sharpes)),
        sharpe_std=float(np.std(sharpes)),
        sharpe_p5=float(np.percentile(sharpes, 5)),
        sharpe_p95=float(np.percentile(sharpes, 95)),
        return_mean=float(np.mean(returns)),
        return_median=float(np.median(returns)),
        dd_mean=float(np.mean(dds)),
        dd_median=float(np.median(dds)),
        pf_mean=float(np.mean(pfs)),
        pf_median=float(np.median(pfs)),
        wr_mean=float(np.mean(wrs)),
        trades_mean=float(np.mean(trade_counts)),
        pct_sharpe_above_1=float(np.mean(sharpes >= 1.0)),
        pct_sharpe_above_2=float(np.mean(sharpes >= 2.0)),
        pct_sharpe_above_3=float(np.mean(sharpes >= 3.0)),
    )


# ---------------------------------------------------------------------------
# Multi-config runner: test multiple champion SL/TP combos
# ---------------------------------------------------------------------------


# Champion strategy SL/TP configs from the 1m journal
CHAMPION_CONFIGS = {
    "sid=212 (B28 STOCH+ATR, tail-win)": BaselineConfig(sl_pct=0.59, tp_pct=10.55, target_trades=50),
    "sid=220 (B30 STOCH+ATR, scalper)": BaselineConfig(sl_pct=7.18, tp_pct=0.85, target_trades=113),
    "sid=186 (B18 STOCH+ATR, ultra-scalper)": BaselineConfig(sl_pct=7.9, tp_pct=0.5, target_trades=59),
    "sid=185 (B18 RSI+STOCH, tail-win)": BaselineConfig(sl_pct=0.63, tp_pct=12.32, target_trades=69),
    "sid=204 (B25 CCI+ROC, scalper)": BaselineConfig(sl_pct=5.44, tp_pct=0.91, target_trades=100),
    "sid=199 (B23 CCI+ROC, ultra-scalper)": BaselineConfig(sl_pct=9.95, tp_pct=0.91, target_trades=88),
}


def run_all_champions(
    bars: list[OHLCBar],
    n_simulations: int = 1000,
    seed: int | None = 42,
) -> dict[str, BaselineResult]:
    """Run random baseline for all champion SL/TP configs.

    Returns dict mapping champion name to BaselineResult.
    """
    results: dict[str, BaselineResult] = {}
    for name, config in CHAMPION_CONFIGS.items():
        logger.info("Running baseline for %s ...", name)
        results[name] = run_random_short_baseline(bars, config, n_simulations, seed)
    return results


def print_comparison_table(
    results: dict[str, BaselineResult],
) -> str:
    """Print comparison table of random baselines vs champion metrics."""
    lines = [
        "=" * 100,
        "RANDOM SHORT BASELINE vs CHAMPION STRATEGIES",
        "=" * 100,
        "",
        f"{'Strategy':<45} {'Rand Sharpe':>12} {'Rand Ret':>10} "
        f"{'Rand DD':>9} {'Rand PF':>9} {'Rand WR':>9} {'%≥S2':>7}",
        "-" * 100,
    ]
    for name, r in results.items():
        lines.append(
            f"{name:<45} {r.sharpe_median:>12.2f} "
            f"{r.return_median * 100:>9.1f}% "
            f"{r.dd_median * 100:>8.1f}% "
            f"{r.pf_median:>9.2f} "
            f"{r.wr_mean * 100:>8.1f}% "
            f"{r.pct_sharpe_above_2 * 100:>6.1f}%"
        )
    lines.append("-" * 100)
    lines.append("")
    lines.append("If %>=S2 > 10%, the SL/TP geometry alone explains the alpha.")
    lines.append("If %>=S2 < 5%, indicator entry timing provides genuine value.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for random baseline testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Random short entry baseline for 1m strategy validation"
    )
    parser.add_argument("--sl-pct", type=float, help="Stop loss %% (e.g. 0.59)")
    parser.add_argument("--tp-pct", type=float, help="Take profit %% (e.g. 10.55)")
    parser.add_argument("--target-trades", type=int, default=50, help="Target trades per simulation")
    parser.add_argument("--start", type=str, default="2026-01-10", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-03-10", help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol")
    parser.add_argument("--interval", type=str, default="1m", help="Bar interval")
    parser.add_argument("--monte-carlo", type=int, default=1000, help="Number of simulations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--archive", type=str, default=str(DEFAULT_ARCHIVE_PATH), help="Archive DB path")
    parser.add_argument(
        "--all-champions", action="store_true",
        help="Run baseline for all champion SL/TP configs",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bars = load_ohlc(args.symbol, args.interval, args.start, args.end, Path(args.archive))
    print(f"Loaded {len(bars)} bars\n")

    if args.all_champions:
        results = run_all_champions(bars, args.monte_carlo, args.seed)
        for _name, r in results.items():
            print(f"\n{r.summary()}")
        print(print_comparison_table(results))
    else:
        if args.sl_pct is None or args.tp_pct is None:
            parser.error("--sl-pct and --tp-pct required unless --all-champions")
        config = BaselineConfig(
            sl_pct=args.sl_pct,
            tp_pct=args.tp_pct,
            target_trades=args.target_trades,
        )
        result = run_random_short_baseline(bars, config, args.monte_carlo, args.seed)
        print(result.summary())


if __name__ == "__main__":
    main()
