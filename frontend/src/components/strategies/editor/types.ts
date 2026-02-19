/** Local form shape for the strategy DSL config editor */

export interface DslIndicator {
  type: string;
  params: Record<string, number>;
  timeframe_override?: string;
}

export type IndicatorCategory = "Trend" | "Momentum" | "Volatility" | "Volume";

export interface IndicatorCatalogEntry {
  type: string;
  name: string;
  emoji: string;
  description: string;
  category: IndicatorCategory;
}

export const INDICATOR_CATALOG: IndicatorCatalogEntry[] = [
  {
    type: "SMA",
    name: "Simple Moving Average",
    emoji: "\u{1F4C8}",
    description: "Average price over N periods",
    category: "Trend",
  },
  {
    type: "EMA",
    name: "Exponential Moving Average",
    emoji: "\u{1F4C9}",
    description: "Weighted average favoring recent prices",
    category: "Trend",
  },
  {
    type: "RSI",
    name: "Relative Strength Index",
    emoji: "\u{1F4CA}",
    description: "Momentum oscillator (0-100)",
    category: "Momentum",
  },
  {
    type: "MACD",
    name: "MACD",
    emoji: "\u{1F500}",
    description: "Trend-following momentum indicator",
    category: "Momentum",
  },
  {
    type: "BB",
    name: "Bollinger Bands",
    emoji: "\u{1F4CF}",
    description: "Volatility bands around SMA",
    category: "Volatility",
  },
  {
    type: "ATR",
    name: "Average True Range",
    emoji: "\u{1F4D0}",
    description: "Measures market volatility",
    category: "Volatility",
  },
  {
    type: "VWAP",
    name: "Volume Weighted Avg Price",
    emoji: "\u{1F4E6}",
    description: "Average price weighted by volume",
    category: "Volume",
  },
  {
    type: "STOCH",
    name: "Stochastic Oscillator",
    emoji: "\u{26A1}",
    description: "Compares closing price to range",
    category: "Momentum",
  },
];

export interface DslCondition {
  left: string;
  operator: string;
  right: string;
  logic?: "and" | "or";
}

export interface DslStopLoss {
  type: string;
  value: number;
}

export interface DslTakeProfit {
  type: string;
  value: number;
}

export interface DslPositionSizing {
  type: string;
  value?: number;
}

export interface DslRisk {
  stop_loss: DslStopLoss;
  take_profit: DslTakeProfit;
  position_sizing: DslPositionSizing;
  trailing_stop_pct?: number;
}

export interface DslTime {
  trading_hours?: { start: string; end: string };
  trading_days?: string[];
  sessions?: string[];
  funding_avoidance?: boolean;
}

export interface DslConfig {
  general: {
    strategy_type: string;
    symbols: string[];
    timeframe: string;
    additional_timeframes?: string[];
  };
  indicators: DslIndicator[];
  conditions: {
    entry: DslCondition[];
    exit: DslCondition[];
    long_entry?: DslCondition[];
    long_exit?: DslCondition[];
    short_entry?: DslCondition[];
    short_exit?: DslCondition[];
  };
  risk: DslRisk;
  time: DslTime;
}

export const STRATEGY_TYPES = [
  "momentum",
  "mean_reversion",
  "breakout",
  "trend_following",
  "arbitrage",
  "volatility",
] as const;

export const TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"] as const;

export const INDICATOR_TYPES = ["SMA", "EMA", "RSI", "MACD", "BB", "ATR", "VWAP", "STOCH"] as const;

export const OPERATORS = [
  { value: ">", label: ">" },
  { value: "<", label: "<" },
  { value: ">=", label: ">=" },
  { value: "<=", label: "<=" },
  { value: "==", label: "==" },
  { value: "crosses_above", label: "Crosses Above" },
  { value: "crosses_below", label: "Crosses Below" },
] as const;

export const DAYS_OF_WEEK = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
] as const;

export const SESSIONS = ["Asian", "European", "US"] as const;

/** Default indicator params per type */
export function getDefaultParams(type: string): Record<string, number> {
  switch (type) {
    case "SMA":
    case "EMA":
      return { period: 20 };
    case "RSI":
      return { period: 14, overbought: 70, oversold: 30 };
    case "MACD":
      return { fast: 12, slow: 26, signal: 9 };
    case "BB":
      return { period: 20, std_dev: 2 };
    case "ATR":
      return { period: 14 };
    case "VWAP":
      return {};
    case "STOCH":
      return { k_period: 14, d_period: 3 };
    default:
      return {};
  }
}

/** Build operand options from configured indicators */
export function buildOperandOptions(indicators: DslIndicator[]): string[] {
  const options: string[] = ["price", "volume"];
  for (const ind of indicators) {
    const paramVals = Object.values(ind.params);
    const paramStr = paramVals.length > 0 ? `(${paramVals.join(",")})` : "";
    options.push(`${ind.type}${paramStr}`);
  }
  return options;
}

/** Build empty DslConfig */
export function emptyDslConfig(): DslConfig {
  return {
    general: { strategy_type: "momentum", symbols: [], timeframe: "1h" },
    indicators: [],
    conditions: { entry: [], exit: [] },
    risk: {
      stop_loss: { type: "fixed_pct", value: 2 },
      take_profit: { type: "fixed_pct", value: 4 },
      position_sizing: { type: "percent_equity", value: 10 },
    },
    time: {},
  };
}

/** Parse unknown dsl_config from API into typed DslConfig */
export function parseDslConfig(raw: Record<string, unknown>): DslConfig {
  const base = emptyDslConfig();

  const general = raw.general as Record<string, unknown> | undefined;
  if (general) {
    if (typeof general.strategy_type === "string")
      base.general.strategy_type = general.strategy_type;
    if (Array.isArray(general.symbols)) base.general.symbols = general.symbols as string[];
    if (typeof general.timeframe === "string") base.general.timeframe = general.timeframe;
  }

  if (Array.isArray(raw.indicators)) {
    base.indicators = (raw.indicators as Array<Record<string, unknown>>).map((ind) => ({
      type: typeof ind.type === "string" ? ind.type : "SMA",
      params: (ind.params as Record<string, number>) ?? {},
    }));
  }

  const conds = raw.conditions as Record<string, unknown> | undefined;
  if (conds) {
    if (Array.isArray(conds.entry)) base.conditions.entry = conds.entry as DslCondition[];
    if (Array.isArray(conds.exit)) base.conditions.exit = conds.exit as DslCondition[];
    if (Array.isArray(conds.long_entry)) base.conditions.long_entry = conds.long_entry as DslCondition[];
    if (Array.isArray(conds.long_exit)) base.conditions.long_exit = conds.long_exit as DslCondition[];
    if (Array.isArray(conds.short_entry)) base.conditions.short_entry = conds.short_entry as DslCondition[];
    if (Array.isArray(conds.short_exit)) base.conditions.short_exit = conds.short_exit as DslCondition[];
  }

  const risk = raw.risk as Record<string, unknown> | undefined;
  if (risk) {
    if (risk.stop_loss) base.risk.stop_loss = risk.stop_loss as DslStopLoss;
    if (risk.take_profit) base.risk.take_profit = risk.take_profit as DslTakeProfit;
    if (risk.position_sizing) base.risk.position_sizing = risk.position_sizing as DslPositionSizing;
    if (typeof risk.trailing_stop_pct === "number") base.risk.trailing_stop_pct = risk.trailing_stop_pct;
  }

  const time = raw.time as Record<string, unknown> | undefined;
  if (time) {
    if (time.trading_hours)
      base.time.trading_hours = time.trading_hours as { start: string; end: string };
    if (Array.isArray(time.trading_days)) base.time.trading_days = time.trading_days as string[];
    if (Array.isArray(time.sessions)) base.time.sessions = time.sessions as string[];
    if (typeof time.funding_avoidance === "boolean")
      base.time.funding_avoidance = time.funding_avoidance;
  }

  const gen = raw.general as Record<string, unknown> | undefined;
  if (gen && Array.isArray(gen.additional_timeframes)) {
    base.general.additional_timeframes = gen.additional_timeframes as string[];
  }

  return base;
}
