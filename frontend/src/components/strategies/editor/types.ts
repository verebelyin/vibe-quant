/** Local form shape for the strategy DSL config editor */

export interface DslIndicator {
  type: string;
  params: Record<string, number>;
  timeframe_override?: string | undefined;
}

export interface DslCondition {
  left: string;
  operator: string;
  right: string;
  logic?: "and" | "or" | undefined;
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
  value?: number | undefined;
}

export interface DslRisk {
  stop_loss: DslStopLoss;
  take_profit: DslTakeProfit;
  position_sizing: DslPositionSizing;
  trailing_stop_pct?: number | undefined;
}

export interface DslTime {
  trading_hours?: { start: string; end: string } | undefined;
  trading_days?: string[] | undefined;
  sessions?: string[] | undefined;
  funding_avoidance?: boolean | undefined;
}

export interface DslConfig {
  general: {
    strategy_type: string;
    symbols: string[];
    timeframe: string;
    additional_timeframes?: string[] | undefined;
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

/** Build operand options from configured indicators */
export function buildOperandOptions(indicators: DslIndicator[]): string[] {
  const options: string[] = ["price", "close", "open", "high", "low", "volume"];
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
