/**
 * Browse-response -> lightweight-charts series data (vibe-quant-v5isx).
 *
 * `/api/data/browse/{symbol}` resamples the archived 1m klines in Python and
 * returns `open_time` in milliseconds with numeric OHLCV. lightweight-charts
 * `setData` throws on times that are not strictly ascending (and on non-finite
 * values), so the mapping normalises here instead of inside the chart effect:
 * rows are sorted by time, deduplicated, and rows with non-finite prices are
 * dropped.
 */
import type { HistogramData, UTCTimestamp } from "lightweight-charts";

export interface CandlePoint {
  time: UTCTimestamp;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const VOLUME_UP_COLOR = "#26a69a80";
const VOLUME_DOWN_COLOR = "#ef535080";

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Candles for the chart: ascending unique times, last row wins per time. */
export function candlesFromBrowseRows(rows: readonly unknown[]): CandlePoint[] {
  const byTime = new Map<number, CandlePoint>();

  for (const row of rows) {
    if (typeof row !== "object" || row === null) continue;
    const raw = row as Record<string, unknown>;
    const openTime = finiteNumber(raw.open_time);
    const open = finiteNumber(raw.open);
    const high = finiteNumber(raw.high);
    const low = finiteNumber(raw.low);
    const close = finiteNumber(raw.close);
    if (openTime === null || open === null || high === null || low === null || close === null) {
      continue;
    }
    const time = Math.floor(openTime / 1000);
    byTime.set(time, {
      time: time as UTCTimestamp,
      open,
      high,
      low,
      close,
      volume: finiteNumber(raw.volume) ?? 0,
    });
  }

  return Array.from(byTime.values()).sort((a, b) => a.time - b.time);
}

/** Volume bars coloured by the candle body direction (flat counts as up). */
export function volumeFromCandles(candles: readonly CandlePoint[]): HistogramData<UTCTimestamp>[] {
  return candles.map((c) => ({
    time: c.time,
    value: c.volume,
    color: c.close >= c.open ? VOLUME_UP_COLOR : VOLUME_DOWN_COLOR,
  }));
}
