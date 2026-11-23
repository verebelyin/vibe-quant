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
import { CRYPTO_DAYS_PER_YEAR } from "@/lib/equity";
import { parseServerTimestamp } from "@/lib/time";

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

/**
 * Fees as a fraction of gross PnL: how much of the edge the fees ate.
 *
 * ``null`` when fees are unavailable (null / non-finite) or gross <= 0 / not
 * finite — a strategy that made no money has no meaningful "share lost to
 * fees", and missing fees are not zero fees. Mirrors the backend
 * ``metrics.fee_drag`` (vibe-quant-2gyk1.4 D3-4a); display as a percent
 * (0.195 = 19.5%).
 */
export function feeDrag(
  fees: number | null | undefined,
  gross: number | null | undefined,
): number | null {
  if (fees == null || !Number.isFinite(fees)) return null;
  if (gross == null || !Number.isFinite(gross) || gross <= 0) return null;
  return fees / gross;
}

/**
 * Fees paid in a run, annualized as a fraction of equity: ``fees / balance *
 * 365 / days``. Same assumption as the backend ``fees_pct_equity_per_year``:
 * equity equals the starting balance (no compounding).
 *
 * ``null`` when fees are unavailable, days <= 0, balance <= 0, or any is
 * non-finite.
 */
export function feesPctEquityPerYear(
  fees: number | null | undefined,
  startingBalance: number | null | undefined,
  days: number | null | undefined,
): number | null {
  if (fees == null || startingBalance == null || days == null) return null;
  if (!Number.isFinite(fees) || !Number.isFinite(days) || !Number.isFinite(startingBalance)) {
    return null;
  }
  if (days <= 0 || startingBalance <= 0) return null;
  return ((fees / startingBalance) * 365) / days;
}

/** Days between two server date/datetime strings (null when either is unparseable). */
export function daysBetweenServerDates(
  start: string | null | undefined,
  end: string | null | undefined,
): number | null {
  const from = parseServerTimestamp(start);
  const to = parseServerTimestamp(end);
  if (from === undefined || to === undefined) return null;
  return (to - from) / 86_400_000;
}

/** notes is machine JSON kept as a string; a bad payload must not break a panel. */
export function parseResultNotes(notes: string | null | undefined): Record<string, unknown> | null {
  if (!notes) return null;
  try {
    const parsed: unknown = JSON.parse(notes);
    if (parsed == null || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    return parsed as Record<string, unknown>;
  } catch {
    return null;
  }
}

/**
 * Days the run actually covered. Validation clamps the requested window to the
 * 1m data coverage and records it in ``notes.data_window``
 * (``effective_start`` / ``effective_end``, vibe_quant/validation/runner.py);
 * annualizing over the requested dates would understate fees per year. When
 * ``data_window`` exists but cannot be read the answer is ``null`` (fail
 * closed), not the requested-date fallback.
 */
export function effectiveWindowDays(
  notes: Record<string, unknown> | null,
  runStart: string | null | undefined,
  runEnd: string | null | undefined,
): number | null {
  const window = notes?.data_window;
  if (window != null) {
    if (typeof window !== "object" || Array.isArray(window)) return null;
    const { effective_start: start, effective_end: end } = window as Record<string, unknown>;
    if (typeof start !== "string" || typeof end !== "string") return null;
    return daysBetweenServerDates(start, end);
  }
  return daysBetweenServerDates(runStart, runEnd);
}

export interface CostRatioInput {
  totalReturn: number | null | undefined;
  startingBalance: number | null | undefined;
  /** Raw fees: null / undefined means "unavailable", never zero. */
  fees: number | null | undefined;
  slippage: number | null | undefined;
  funding: number | null | undefined;
  /** The result row's notes JSON string. */
  notes: string | null | undefined;
  runStart: string | null | undefined;
  runEnd: string | null | undefined;
}

export interface CostRatios {
  netPnl: number | null;
  grossPnl: number | null;
  days: number | null;
  feeDrag: number | null;
  feesPctEquityYr: number | null;
}

/** Net/gross PnL, effective window and the two fee ratios for one result row. */
export function costRatios(input: CostRatioInput): CostRatios {
  const netPnl = netPnlUsd(input.totalReturn, input.startingBalance);
  const grossPnl = grossPnlUsd(netPnl, input.fees, input.slippage, input.funding);
  const days = effectiveWindowDays(parseResultNotes(input.notes), input.runStart, input.runEnd);
  return {
    netPnl,
    grossPnl,
    days,
    feeDrag: feeDrag(input.fees, grossPnl),
    feesPctEquityYr: feesPctEquityPerYear(input.fees, input.startingBalance, days),
  };
}

/** Fraction → "19.5%"; "n/a" for null / NaN / Infinity. */
export function formatFractionPct(value: number | null | undefined, decimals = 1): string {
  if (value == null || !Number.isFinite(value)) return "n/a";
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Fixed-decimals number; "n/a" for null / NaN / Infinity. */
export function formatFiniteNumber(value: number | null | undefined, decimals = 2): string {
  if (value == null || !Number.isFinite(value)) return "n/a";
  return value.toFixed(decimals);
}

/**
 * Days/year of every Sharpe/Sortino the engine STORES (NT SharpeRatio, DSR,
 * purged k-fold, gate reasons). Mirror of `vibe_quant/metrics.py`.
 */
export const STORED_SHARPE_DAYS = 252;

/**
 * Stored (√252) → shown (√365): crypto perps trade every calendar day.
 * 1.2035001862952488, pinned to the pytest fixture.
 */
export const SHARPE_DISPLAY_FACTOR = Math.sqrt(CRYPTO_DAYS_PER_YEAR / STORED_SHARPE_DAYS);

/** The unit tag every user-facing line carries once. */
export const SHARPE_UNIT = "ann √365";

export const SHARPE_TOOLTIP =
  "Annualised with 365 days (crypto trades every day): stored engine value × 1.2035 (√(365/252))";

/**
 * A RAW stored Sharpe/Sortino (√252) in display units (√365).
 * `null` for null / NaN / ±Infinity — never NaN on screen.
 *
 * UI code converts the raw field through this helper; it never reads an API
 * display twin (that would convert twice: 0.96 → 1.16).
 */
export function sharpeDisplay(raw: number | null | undefined): number | null {
  if (raw == null || !Number.isFinite(raw)) return null;
  return raw * SHARPE_DISPLAY_FACTOR;
}

/** Displayed Sharpe/Sortino, fixed decimals; "n/a" for null / NaN / ±Infinity. */
export function fmtSharpe(raw: number | null | undefined, digits = 2): string {
  const display = sharpeDisplay(raw);
  return display == null ? "n/a" : display.toFixed(digits);
}

const LEGACY_SHARPE_REASON_KINDS = /^(DSR|Holdout|Cross-window|K-Fold|validation-collapse)\b/i;

/**
 * Tag a stored rejection reason that still prints a √252 Sharpe, so an old run
 * is not read as display units. Bootstrap reasons print a t-stat and buy-hold
 * texts are already √365 — both stay untouched, as does anything already
 * tagged with a basis.
 */
export function annotateLegacySharpeReason(reason: string): string {
  const kind = reason.replace(/^Multi-seed pooled N=\d+:\s*/i, "");
  if (!LEGACY_SHARPE_REASON_KINDS.test(kind)) return reason;
  if (!/sharpe|SR=/i.test(reason)) return reason;
  if (/√365|365-day|252-day/.test(reason)) return reason;
  return `${reason} [252-day basis]`;
}
