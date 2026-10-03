/**
 * Unit helpers for backtest metrics.
 *
 * Backend conventions (vibe_quant/validation/extraction.py, screening/nt_runner.py):
 * - total_return, cagr, max_drawdown, win_rate, volatility_annual are FRACTIONS
 *   (0.0517 = 5.17%). max_drawdown and the /drawdown series are POSITIVE fractions.
 * - total_return is NET of fees, slippage and funding.
 * - total_fees, total_slippage, total_funding, avg/largest win/loss, net_pnl are USDT.
 *   total_funding is signed: positive = paid, negative = received.
 * - trades.roi_percent is already a percent.
 */

/** Fraction → percent (0.0517 → 5.17). */
export function fractionToPercent(value: number | null | undefined): number | null {
  return value == null ? null : value * 100;
}

/** Net PnL in USDT from the fractional total_return. */
export function netPnlUsd(
  totalReturn: number | null | undefined,
  startingBalance: number | null | undefined,
): number | null {
  if (totalReturn == null || startingBalance == null) return null;
  return totalReturn * startingBalance;
}

/** Gross PnL = net + every cost paid (funding signed: received funding lowers it). */
export function grossPnlUsd(
  netPnl: number | null,
  fees: number | null | undefined,
  slippage: number | null | undefined,
  funding: number | null | undefined,
): number | null {
  if (netPnl == null) return null;
  return netPnl + (fees ?? 0) + (slippage ?? 0) + (funding ?? 0);
}

/** "$51.72" / "-$64.60" / "N/A". */
export function formatSignedUsd(value: number | null | undefined, decimals = 2): string {
  if (value == null || Number.isNaN(value)) return "N/A";
  const abs = `$${Math.abs(value).toFixed(decimals)}`;
  return value < 0 ? `-${abs}` : abs;
}

/** Drawdown fractions (positive) → negative percent points for an under-water chart. */
export function drawdownToNegativePercent(drawdown: number): number {
  return -Math.abs(drawdown) * 100;
}
