/**
 * Equity-curve helpers shared by results charts (vibe-quant-pazbr).
 *
 * The /equity-curve endpoint emits one point per trade close, NOT one per
 * day. Charts labelled "daily returns" / "N-day rolling Sharpe" must first
 * resample to calendar days — treating per-trade points as daily both
 * mislabels the window (30 trades != 30 days) and mis-annualizes Sharpe.
 *
 * Annualization uses sqrt(365): crypto perps trade every calendar day, so
 * the equity curve has up to 365 daily observations per year (the equity
 *-market convention of 252 trading days does not apply).
 */

import type { EquityCurvePoint } from "@/api/generated/models/equityCurvePoint";

export const CRYPTO_DAYS_PER_YEAR = 365;

export interface DailyReturnPoint {
  timestamp: string;
  returnPct: number;
}

export interface SharpePoint {
  timestamp: string;
  sharpe: number;
}

function utcDateKey(ts: string): string {
  const d = new Date(ts);
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/**
 * Resample per-trade equity points to one point per UTC calendar day.
 *
 * Takes the LAST equity value of each day and forward-fills days with no
 * trades, so consecutive points are exactly one day apart and returns
 * between them are true daily returns.
 */
export function resampleDailyEquity(data: EquityCurvePoint[]): EquityCurvePoint[] {
  if (data.length === 0) return [];

  const sorted = [...data].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
  );

  // Last equity per calendar day
  const lastPerDay = new Map<string, number>();
  for (const point of sorted) {
    lastPerDay.set(utcDateKey(point.timestamp), point.equity);
  }

  const firstDay = utcDateKey(sorted[0]!.timestamp);
  const lastDay = utcDateKey(sorted[sorted.length - 1]!.timestamp);

  const result: EquityCurvePoint[] = [];
  const cursor = new Date(`${firstDay}T00:00:00Z`);
  const end = new Date(`${lastDay}T00:00:00Z`);
  let equity = sorted[0]!.equity;

  while (cursor.getTime() <= end.getTime()) {
    const key = cursor.toISOString().slice(0, 10);
    const dayEquity = lastPerDay.get(key);
    if (dayEquity !== undefined) equity = dayEquity;
    result.push({ timestamp: key, equity });
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }

  return result;
}

/** Daily percentage returns from a per-trade equity curve. */
export function computeDailyReturns(data: EquityCurvePoint[]): DailyReturnPoint[] {
  const daily = resampleDailyEquity(data);
  if (daily.length < 2) return [];

  const result: DailyReturnPoint[] = [];
  for (let i = 1; i < daily.length; i++) {
    const prev = daily[i - 1]!.equity;
    const curr = daily[i]!.equity;
    if (prev !== 0) {
      result.push({
        timestamp: daily[i]!.timestamp,
        returnPct: Number.parseFloat((((curr - prev) / prev) * 100).toFixed(4)),
      });
    }
  }
  return result;
}

/**
 * Rolling Sharpe over TRUE daily returns (sample stddev, sqrt-365
 * annualization). `windowDays` is a count of calendar days.
 */
export function computeRollingSharpe(data: EquityCurvePoint[], windowDays: number): SharpePoint[] {
  const daily = resampleDailyEquity(data);
  if (daily.length < windowDays + 1) return [];

  const returns: number[] = [];
  for (let i = 1; i < daily.length; i++) {
    const prev = daily[i - 1]!.equity;
    const curr = daily[i]!.equity;
    returns.push(prev !== 0 ? (curr - prev) / prev : 0);
  }

  const result: SharpePoint[] = [];
  for (let i = windowDays - 1; i < returns.length; i++) {
    const slice = returns.slice(i - windowDays + 1, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance =
      slice.length > 1 ? slice.reduce((a, b) => a + (b - mean) ** 2, 0) / (slice.length - 1) : 0;
    const std = Math.sqrt(variance);
    const sharpe = std !== 0 ? (mean / std) * Math.sqrt(CRYPTO_DAYS_PER_YEAR) : 0;
    result.push({
      timestamp: daily[i + 1]!.timestamp,
      sharpe: Number.parseFloat(sharpe.toFixed(3)),
    });
  }
  return result;
}
