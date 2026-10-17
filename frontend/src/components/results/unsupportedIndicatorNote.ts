/**
 * Note text for an indicator the chart overlay could not compute
 * (vibe-quant-5q4r5).
 *
 * `GET /api/data/indicators/{symbol}` reports types it cannot derive from
 * OHLCV bars alone explicitly (`unsupported`) instead of dropping them:
 * context-dependent types (BTC_TREND, FUNDING, ...) and NautilusTrader-only
 * types (NATR). Those entries carry a display label but no data, so the chart
 * has to show a visible note rather than a silent gap.
 *
 * Pure so it can be unit-tested without a DOM (vitest runs in a node env).
 */
export function unsupportedIndicatorNote(series: {
  unsupported?: boolean;
  display_label: string;
}): string | null {
  if (!series.unsupported) return null;
  return `${series.display_label}: not available in chart`;
}
