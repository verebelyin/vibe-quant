/**
 * Canonical StrategyDSL <-> editor DslConfig bridge (vibe-quant-o1vvm).
 *
 * Templates and discovery-exported champions are stored as canonical StrategyDSL
 * (`indicators` is an object, `entry_conditions`/`stop_loss`/`time_filters` at the top level).
 * The visual editor works on the nested DslConfig model, which cannot hold everything canonical
 * does (sweeps, ATR stop indicator refs, price source, multi-session filters, ...).
 *
 * Lossless strategy: the canonical dict stays the source of truth. `toCanonicalDsl` starts from it
 * and overwrites only the fields the user actually changed relative to the parse of that dict.
 */
import {
  DAYS_OF_WEEK,
  type DslCondition,
  type DslConfig,
  type DslDirectionalRisk,
  type DslIndicator,
} from "./types";

type Dict = Record<string, unknown>;

export const DIRECTIONAL_RISK_KEYS = [
  "stop_loss_long",
  "stop_loss_short",
  "take_profit_long",
  "take_profit_short",
] as const;

function directionalRisk(v: unknown): DslDirectionalRisk | undefined {
  const cfg = asDict(v);
  if (typeof cfg.type !== "string") return undefined;
  const out: DslDirectionalRisk = { type: cfg.type };
  if (typeof cfg.indicator === "string") out.indicator = cfg.indicator;
  for (const key of ["percent", "atr_multiplier", "risk_reward_ratio"] as const)
    if (typeof cfg[key] === "number") out[key] = cfg[key];
  return out;
}

/**
 * Read the canonical TradingView-port fields the editor shows read-only (position mode, refresh,
 * directional stop/target overrides). They stay in `base`, so saving an unrelated edit keeps them.
 */
export function readPortFields(raw: Dict, out: DslConfig): void {
  if (raw.position_mode === "reverse") out.position_mode = "reverse";
  if (typeof raw.refresh_on_signal === "boolean") out.refresh_on_signal = raw.refresh_on_signal;
  for (const key of DIRECTIONAL_RISK_KEYS) {
    const leg = directionalRisk(raw[key]);
    if (leg) out[key] = leg;
  }
}

const OPERATOR_RE = /^\s*(\S+)\s+(crosses_above|crosses_below|>=|<=|==|>|<)\s+(\S+)\s*$/;

function asDict(v: unknown): Dict {
  return v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Dict) : {};
}

/** Mirror of translator._is_dsl_config_format: canonical = no `general` and no indicator array. */
export function isCanonicalDsl(raw: Dict): boolean {
  return Object.keys(raw).length > 0 && !("general" in raw) && !Array.isArray(raw.indicators);
}

/** Number of indicators in a stored dsl_config, whichever format it is in. */
export function countStoredIndicators(raw: Dict): number {
  const ind = raw.indicators;
  if (Array.isArray(ind)) return ind.length;
  return Object.keys(asDict(ind)).length;
}

function conditionFromString(s: string): DslCondition {
  const m = OPERATOR_RE.exec(s);
  // Forms the editor cannot express (e.g. "rsi between 30 70") stay as raw text: operator "".
  if (!m) return { left: s, operator: "", right: "" };
  return { left: m[1] ?? "", operator: m[2] ?? ">", right: m[3] ?? "" };
}

function conditionList(v: unknown): DslCondition[] {
  return Array.isArray(v)
    ? v.filter((c): c is string => typeof c === "string").map(conditionFromString)
    : [];
}

/**
 * Inline message for `risk.max_bars_in_trade`; null when the time exit is off or N is a whole
 * number in [1, 100000] (the backend's StrategyDSL bounds).
 */
export function maxBarsInTradeProblem(v: number | undefined): string | null {
  if (v === undefined) return null;
  if (Number.isInteger(v) && v >= 1 && v <= 100000) return null;
  return "Max bars in trade must be a whole number from 1 to 100000";
}

function riskValue(cfg: Dict, fallback: number): number {
  const key =
    cfg.type === "fixed_pct"
      ? "percent"
      : cfg.type === "risk_reward"
        ? "risk_reward_ratio"
        : "atr_multiplier";
  const v = cfg[key];
  return typeof v === "number" ? v : fallback;
}

/** Canonical StrategyDSL dict -> editor model (display + edit). */
export function canonicalToUi(raw: Dict, base: DslConfig): DslConfig {
  const out = base;
  if (typeof raw.timeframe === "string") out.general.timeframe = raw.timeframe;
  if (Array.isArray(raw.additional_timeframes) && raw.additional_timeframes.length > 0)
    out.general.additional_timeframes = raw.additional_timeframes as string[];

  out.indicators = Object.entries(asDict(raw.indicators)).map(([name, v]): DslIndicator => {
    const cfg = asDict(v);
    const params: Record<string, number> = {};
    for (const [k, val] of Object.entries(cfg)) if (typeof val === "number") params[k] = val;
    const ind: DslIndicator = {
      name,
      type: typeof cfg.type === "string" ? cfg.type : "SMA",
      params,
    };
    if (typeof cfg.timeframe === "string") ind.timeframe_override = cfg.timeframe;
    return ind;
  });

  const entry = asDict(raw.entry_conditions);
  const exit = asDict(raw.exit_conditions);
  out.conditions = {
    entry: [],
    exit: [],
    long_entry: conditionList(entry.long),
    short_entry: conditionList(entry.short),
    long_exit: conditionList(exit.long),
    short_exit: conditionList(exit.short),
  };

  const sl = asDict(raw.stop_loss);
  const tp = asDict(raw.take_profit);
  if (typeof sl.type === "string") out.risk.stop_loss = { type: sl.type, value: riskValue(sl, 2) };
  if (typeof tp.type === "string")
    out.risk.take_profit = { type: tp.type, value: riskValue(tp, 4) };
  if (typeof raw.max_bars_in_trade === "number") out.risk.max_bars_in_trade = raw.max_bars_in_trade;
  readPortFields(raw, out);

  const tf = asDict(raw.time_filters);
  const sessions = Array.isArray(tf.allowed_sessions) ? tf.allowed_sessions.map(asDict) : [];
  const first = sessions[0];
  if (
    sessions.length === 1 &&
    first &&
    typeof first.start === "string" &&
    typeof first.end === "string"
  )
    out.time.trading_hours = { start: first.start, end: first.end };
  const blocked = Array.isArray(tf.blocked_days) ? (tf.blocked_days as string[]) : [];
  if (blocked.length > 0) out.time.trading_days = DAYS_OF_WEEK.filter((d) => !blocked.includes(d));
  if (asDict(tf.avoid_around_funding).enabled === true) out.time.funding_avoidance = true;
  return out;
}

function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function resolveOperand(token: string, names: Set<string>, labels: Map<string, string>): string {
  if (names.has(token)) return token;
  if (token.toLowerCase() === "price") return "close";
  return labels.get(token) ?? token;
}

function conditionToString(
  c: DslCondition,
  names: Set<string>,
  labels: Map<string, string>,
): string {
  if (c.operator === "") return c.left; // raw canonical text the editor could not parse
  const l = resolveOperand(c.left, names, labels);
  const r = resolveOperand(c.right, names, labels);
  return `${l} ${c.operator} ${r}`;
}

/** Editor-added indicators have no canonical name yet: derive `<type>_<period>` (unique). */
function nameIndicators(inds: DslIndicator[]): string[] {
  const taken = new Set(inds.map((i) => i.name).filter((n): n is string => !!n));
  return inds.map((ind) => {
    if (ind.name) return ind.name;
    const period = ind.params.period ?? ind.params.period_k;
    const root = `${ind.type.toLowerCase()}${period === undefined ? "" : `_${period}`}`;
    let name = root;
    for (let n = 2; taken.has(name); n++) name = `${root}_${n}`;
    taken.add(name);
    return name;
  });
}

function buildIndicators(inds: DslIndicator[], names: string[], prior: Dict): Dict {
  const out: Dict = {};
  inds.forEach((ind, i) => {
    const old = asDict(prior[names[i] ?? ""]);
    // Keep non-numeric extras (source, ...) of an unchanged type; the editor owns the numeric
    // params and the timeframe.
    const keep: Dict = {};
    if (old.type === ind.type)
      for (const [k, v] of Object.entries(old))
        if (typeof v !== "number" && k !== "timeframe") keep[k] = v;
    out[names[i] ?? ""] = {
      ...keep,
      type: ind.type,
      ...ind.params,
      ...(ind.timeframe_override ? { timeframe: ind.timeframe_override } : {}),
    };
  });
  return out;
}

/** Names an older editor wrote (and may still sit in YAML text) -> backend types. */
const LEGACY_RISK_TYPES: Record<string, string> = {
  atr_based: "atr_fixed",
  rr_ratio: "risk_reward",
};

function buildRisk(
  ui: { type: string; value: number },
  old: Dict,
  indicators: Dict,
  label: string,
): Dict {
  const type = LEGACY_RISK_TYPES[ui.type] ?? ui.type;
  const out: Dict = { type };
  if (type === "fixed_pct") out.percent = ui.value;
  else if (type === "risk_reward") out.risk_reward_ratio = ui.value;
  else out.atr_multiplier = ui.value;
  if (type.startsWith("atr")) {
    const atr = Object.keys(indicators).find((n) => asDict(indicators[n]).type === "ATR");
    const ref = old.indicator ?? atr;
    if (ref === undefined)
      throw new Error(
        `${label} type '${type}' needs an ATR indicator; add one in the Indicators tab`,
      );
    out.indicator = ref;
  }
  return out;
}

/**
 * Editor model -> canonical StrategyDSL: `base` (the stored canonical dict) with only the
 * fields changed since `baseline` (= canonicalToUi(base)) overwritten.
 *
 * @throws Error for edits canonical cannot express (OR conditions, trailing_stop_pct).
 */
export function toCanonicalDsl(config: DslConfig, base: Dict, baseline: DslConfig): Dict {
  const out = structuredClone(base) as Dict;

  if (config.general.timeframe !== baseline.general.timeframe)
    out.timeframe = config.general.timeframe;
  if (!same(config.general.additional_timeframes, baseline.general.additional_timeframes))
    out.additional_timeframes = config.general.additional_timeframes ?? [];

  const names = nameIndicators(config.indicators);
  if (!same(config.indicators, baseline.indicators))
    out.indicators = buildIndicators(config.indicators, names, asDict(base.indicators));
  const indicators = asDict(out.indicators);

  // Operand resolution: canonical names pass through, editor labels ("EMA(50)") map to names.
  const known = new Set(Object.keys(indicators));
  const labels = new Map<string, string>();
  config.indicators.forEach((ind, i) => {
    const label = `${ind.type}(${Object.values(ind.params).join(",")})`;
    const name = names[i];
    if (name && !labels.has(label)) labels.set(label, name);
  });

  const cond = config.conditions;
  const lists: Array<[keyof DslConfig["conditions"], string, "long" | "short"]> = [
    ["long_entry", "entry_conditions", "long"],
    ["short_entry", "entry_conditions", "short"],
    ["long_exit", "exit_conditions", "long"],
    ["short_exit", "exit_conditions", "short"],
  ];
  const touched = (key: keyof DslConfig["conditions"]) =>
    !same(cond[key], baseline.conditions[key]);
  const toStrings = (list: DslCondition[] | undefined, key: string) => {
    (list ?? []).forEach((c, i) => {
      if (i > 0 && c.logic === "or")
        throw new Error(`${key}[${i}] uses OR logic; the backtest engine only supports AND`);
    });
    return (list ?? []).map((c) => conditionToString(c, known, labels));
  };
  for (const [key, section, side] of lists) {
    if (!touched(key)) continue;
    const dest = asDict(out[section]);
    dest[side] = toStrings(cond[key], key);
    out[section] = dest;
  }
  // Generic entry/exit (added in the non-split condition view) apply like translator.py: entry
  // -> long, exit -> every direction that has entries.
  if (touched("entry") && cond.entry.length > 0) {
    const entry = asDict(out.entry_conditions);
    entry.long = toStrings(cond.entry, "entry");
    out.entry_conditions = entry;
  }
  if (touched("exit") && cond.exit.length > 0) {
    const exit = asDict(out.exit_conditions);
    const strings = toStrings(cond.exit, "exit");
    const entry = asDict(out.entry_conditions);
    for (const side of ["long", "short"] as const)
      if (Array.isArray(entry[side]) && entry[side].length > 0) exit[side] = strings;
    out.exit_conditions = exit;
  }

  if (config.risk.trailing_stop_pct)
    throw new Error("risk.trailing_stop_pct is not supported by the backtest engine");
  if (!same(config.risk.position_sizing, baseline.risk.position_sizing))
    throw new Error("position sizing is not part of the strategy DSL (choose it when launching)");
  if (!same(config.time.sessions, baseline.time.sessions))
    throw new Error("Session filters (Asian/European/US) are not supported; set trading hours");
  if (!same(config.risk.stop_loss, baseline.risk.stop_loss))
    out.stop_loss = buildRisk(
      config.risk.stop_loss,
      asDict(base.stop_loss),
      indicators,
      "stop_loss",
    );
  if (!same(config.risk.take_profit, baseline.risk.take_profit))
    out.take_profit = buildRisk(
      config.risk.take_profit,
      asDict(base.take_profit),
      indicators,
      "take_profit",
    );

  const bars = config.risk.max_bars_in_trade;
  const barsProblem = maxBarsInTradeProblem(bars);
  if (barsProblem) throw new Error(barsProblem);
  if (!same(bars, baseline.risk.max_bars_in_trade)) {
    if (bars === undefined) delete out.max_bars_in_trade;
    else out.max_bars_in_trade = bars;
  }

  const tf = asDict(out.time_filters);
  const t = config.time;
  const b = baseline.time;
  if (!same(t.trading_hours, b.trading_hours)) {
    const prev = asDict(base.time_filters).allowed_sessions;
    const tz = Array.isArray(prev) ? asDict(prev[0]).timezone : undefined;
    tf.allowed_sessions = t.trading_hours
      ? [{ start: t.trading_hours.start, end: t.trading_hours.end, timezone: tz ?? "UTC" }]
      : [];
  }
  if (!same(t.trading_days, b.trading_days))
    tf.blocked_days = t.trading_days
      ? DAYS_OF_WEEK.filter((d) => !t.trading_days?.includes(d))
      : [];
  if (!same(t.funding_avoidance, b.funding_avoidance))
    tf.avoid_around_funding = {
      ...asDict(tf.avoid_around_funding),
      enabled: t.funding_avoidance === true,
    };
  if (Object.keys(tf).length > 0) out.time_filters = tf;
  return out;
}

/**
 * Warning when setting one trading window would replace several stored sessions (the editor can
 * show only a single window), else null.
 */
export function sessionReplaceWarning(
  stored: Dict,
  current: DslConfig,
  baseline: DslConfig,
): string | null {
  const sessions = asDict(stored.time_filters).allowed_sessions;
  if (!Array.isArray(sessions) || sessions.length < 2) return null;
  if (same(current.time.trading_hours, baseline.time.trading_hours)) return null;
  return `Setting a trading window: the ${sessions.length} stored sessions will be replaced by this single window.`;
}

/**
 * Message when saving would silently shrink the stored indicator set because the editor could
 * not read it (parse loss), else null. An explicit template load or indicator edit lifts it.
 */
export function saveGuardMessage(
  stored: Dict,
  initial: DslConfig,
  current: DslConfig,
  templateLoaded: boolean,
): string | null {
  if (templateLoaded) return null;
  const storedCount = countStoredIndicators(stored);
  if (storedCount <= initial.indicators.length) return null;
  if (!same(current.indicators, initial.indicators)) return null;
  return `Save blocked: the editor read ${initial.indicators.length} of ${storedCount} stored indicators; saving would delete the rest.`;
}
