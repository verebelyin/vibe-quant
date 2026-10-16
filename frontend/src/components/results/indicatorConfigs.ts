/**
 * Indicator overlay configs for the results chart (vibe-quant-6x6h0).
 *
 * A stored strategy's `dsl_config.indicators` is either the UI array format
 * (`[{type: "RSI", period: 14}]`) or the canonical StrategyDSL dict keyed by
 * indicator name (`{adx_exit_0: {type: "ADX", period: 12}}`) — the format 239 of
 * the 241 strategies in the DB carry, which used to render no overlays at all.
 *
 * The overlay endpoint reads `type` and forwards every other non-null key as an
 * indicator param (vibe_quant/data/indicator_compute.py:compute_indicators), so
 * only the dict VALUES are sent: its key is an indicator name, not a parameter.
 */

export type IndicatorConfig = Record<string, unknown>;

/**
 * Entries that are objects (not arrays, not null) carrying a string `type`.
 * Applies to both formats: a dict VALUE and a UI array ENTRY must pass it.
 */
function usableConfigs(values: unknown[]): IndicatorConfig[] {
  const configs: IndicatorConfig[] = [];
  for (const value of values) {
    if (typeof value !== "object" || value === null || Array.isArray(value)) continue;
    const config = value as IndicatorConfig;
    if (typeof config.type !== "string") continue;
    configs.push(config);
  }
  return configs;
}

/** Overlay configs for a strategy dsl_config; [] when it has no usable indicators. */
export function indicatorConfigsFromDsl(dsl: unknown): IndicatorConfig[] {
  if (typeof dsl !== "object" || dsl === null) return [];
  const indicators = (dsl as Record<string, unknown>).indicators;

  if (Array.isArray(indicators)) {
    const configs = usableConfigs(indicators);
    // Keep the caller's array (and thus its identity) when nothing was skipped.
    return configs.length === indicators.length ? (indicators as IndicatorConfig[]) : configs;
  }

  if (typeof indicators !== "object" || indicators === null) return [];
  return usableConfigs(Object.values(indicators));
}
