/**
 * The ONE run-sizing rule on the frontend (bead vibe-quant-2gyk1.2, S6): the
 * mirror of `vibe_quant/sizing_settings.py`.
 *
 * `max_position_pct` is an exposure CAP, not a target: a run sizes
 * `min(equity * risk_per_trade / stop_distance, equity * max_position_pct / price)`
 * and skips a position below the instrument's min qty / min notional, so the
 * actual exposure is the smaller of the two. The cap is PER SYMBOL (N symbols
 * hold up to N × cap, D-S2) and leverage is a separate setting that must be at
 * least the cap.
 *
 * The launch forms enter the cap in multiples of balance and the risk in PERCENT;
 * `riskPctToFraction`/`riskFractionToPct` convert between the two with the
 * `toPrecision(12)` rounding that turns the measured `0.07 * 100`
 * (7.000000000000001) back into 7, so a default launch (50% cap, 2% risk) stores
 * no keys. Both fields are validated in every mode, and a recorded value that is
 * not a finite number in range is an error rather than a silent default.
 */

import { marginPctText } from "@/components/backtest/marginLimit";

/** The one default exposure cap: at most 50% of equity per position. */
export const DEFAULT_MAX_POSITION_PCT = 0.5;

/** The one default risk budget: 2% of equity risked per trade. */
export const DEFAULT_RISK_PER_TRADE = 0.02;

/** The default risk as a percent, for the launch forms. */
export const DEFAULT_RISK_PCT = 2;

/** The exposure cap upper bound, in multiples of balance. */
export const MAX_MAX_POSITION_PCT = 125;

/** The risk upper bound, in percent. */
export const MAX_RISK_PCT = 50;

/** The risk upper bound as a fraction (mirrors `MAX_RISK_PER_TRADE`). */
const MAX_RISK_PER_TRADE = MAX_RISK_PCT / 100;

/** The leverage bounds a cap is checked against (mirrors `liquidation.ts`). */
const MIN_LEVERAGE = 1;
const MAX_LEVERAGE = 125;

/** The two sizing modes (mirrors `SIZING_MODES`): the stop distance the risk divides by. */
export type SizingMode = "risk" | "atr";

/** The ATR period default, the window the stop distance averages (mirrors `DEFAULT_ATR_PERIOD`). */
export const DEFAULT_ATR_PERIOD = 14;

/** The ATR multiplier default (mirrors `DEFAULT_ATR_MULTIPLIER`). */
export const DEFAULT_ATR_MULTIPLIER = 2;

/** The shortest ATR window the backend accepts (mirrors `MIN_ATR_PERIOD`). */
export const MIN_ATR_PERIOD = 2;

/** The longest ATR window the backend accepts (mirrors `MAX_ATR_PERIOD`). */
export const MAX_ATR_PERIOD = 10_000;

/** The multiplier upper bound (mirrors `MAX_ATR_MULTIPLIER`). */
export const MAX_ATR_MULTIPLIER = 20;

/** The ATR stop distance a run sizes on: `ATR(period) * multiplier`. */
export interface AtrSizing {
  period: number;
  multiplier: number;
}

/** The validated settings of a run, plus how they were read. */
interface ResolvedSizing {
  maxPositionPct: number;
  riskPerTrade: number;
  legacy: boolean;
  maxMarginPct?: number;
  /** Set only in ATR mode; a risk run sizes on the strategy's own stop distance. */
  atr?: AtrSizing;
}

/** A run's stored sizing: the validated settings, or why they are unreadable. */
export type RunSizing = ResolvedSizing | { invalid: string };

/** The margin limit shown on the chip: a stored fraction in (0, 1], else nothing. */
function storedMarginPct(params: Record<string, unknown>): number | undefined {
  const value = params.max_margin_pct;
  return isAmount(value, 1) ? value : undefined;
}

/** The launch fields sizing contributes: only the keys moved off the default. */
export interface SizingPayload {
  max_position_pct?: number;
  risk_per_trade?: number;
  sizing_mode?: "atr";
  sizing_atr_period?: number;
  sizing_atr_multiplier?: number;
}

/**
 * The ATR inputs a launch body sends: present only in ATR mode with both fields
 * filled, so a risk run (or an ATR form with an empty field) sends nothing new.
 */
export function atrSizingFor(
  mode: SizingMode,
  period: number | "",
  multiplier: number | "",
): AtrSizing | null {
  return mode === "atr" && period !== "" && multiplier !== "" ? { period, multiplier } : null;
}

/** Percent -> fraction (the forms enter percent, the payload stores fractions). */
export function riskPctToFraction(pct: number): number {
  return Number((pct / 100).toPrecision(12));
}

/** Fraction -> percent (for the chip and the form's initial value). */
export function riskFractionToPct(fraction: number): number {
  return Number((fraction * 100).toPrecision(12));
}

/**
 * The payload fields to store: only the keys a run moved off the default, so an
 * explicit default stores byte-identically to a default launch
 * (`SizingSettings.to_params`).
 *
 * `atr` present means ATR mode, and ATR mode stores ALL THREE of its keys, defaults
 * included, so the record names exactly what ran (the cap and risk stay off the
 * payload unless moved). Risk mode passes no `atr` and stores nothing new.
 */
export function sizingPayload(
  maxPositionPct: number,
  riskPerTrade: number,
  atr?: AtrSizing | null,
): SizingPayload {
  const payload: SizingPayload = {};
  if (maxPositionPct !== DEFAULT_MAX_POSITION_PCT) payload.max_position_pct = maxPositionPct;
  if (riskPerTrade !== DEFAULT_RISK_PER_TRADE) payload.risk_per_trade = riskPerTrade;
  if (atr != null) {
    payload.sizing_mode = "atr";
    payload.sizing_atr_period = atr.period;
    payload.sizing_atr_multiplier = atr.multiplier;
  }
  return payload;
}

/** The entered cap text; an empty field reads as "nothing". */
function entered(value: number | ""): string {
  return value === "" ? "nothing" : `${value}`;
}

/** The entered risk text (a percent, or "nothing" for an empty field). */
function enteredPct(value: number | ""): string {
  return value === "" ? "nothing" : `${value}%`;
}

/**
 * The inline problems of the launch forms, in order: cap bounds, risk bounds,
 * then cap-vs-leverage (`[]` when the form is valid). The cap is entered in
 * multiples of balance and the risk in percent; both fields are REQUIRED, so an
 * empty field is a problem ("you entered nothing").
 *
 * The cap-vs-leverage check runs only for a valid cap and a leverage inside
 * [1, 125]: each field has its own validation, so the cross-field problem never
 * double-reports or masks the field's own message. Equality is allowed — a cap
 * exactly at the leverage is fully marginable.
 */
export function sizingProblems(
  maxPos: number | "",
  riskPct: number | "",
  leverage: number,
): string[] {
  const problems: string[] = [];

  const capValid = typeof maxPos === "number" && maxPos > 0 && maxPos <= MAX_MAX_POSITION_PCT;
  if (!capValid) {
    problems.push(
      `Max exposure must be more than 0 and at most ${MAX_MAX_POSITION_PCT} × balance (you entered ${entered(maxPos)})`,
    );
  }

  const riskValid = typeof riskPct === "number" && riskPct > 0 && riskPct <= MAX_RISK_PCT;
  if (!riskValid) {
    problems.push(
      `Risk per trade must be more than 0% and at most ${MAX_RISK_PCT}% of equity per trade (you entered ${enteredPct(riskPct)})`,
    );
  }

  const leverageValid = leverage >= MIN_LEVERAGE && leverage <= MAX_LEVERAGE;
  if (typeof maxPos === "number" && capValid && leverageValid && maxPos > leverage) {
    problems.push(`Max exposure (${maxPos} × balance) must not exceed leverage (${leverage})`);
  }

  return problems;
}

/**
 * The inline problems of the ATR inputs, in order: period bounds, then multiplier
 * bounds (`[]` when the form is valid). Called only in ATR mode, so a risk run never
 * reports a problem for fields it does not show; both fields are REQUIRED, so an
 * empty field is a problem ("you entered nothing"). The bounds mirror
 * `sizing_settings.py`: a whole period in [2, 10000] and a multiplier in (0, 20],
 * equality included at each edge.
 */
export function atrSizingProblems(period: number | "", multiplier: number | ""): string[] {
  const problems: string[] = [];

  const periodValid =
    typeof period === "number" &&
    Number.isInteger(period) &&
    period >= MIN_ATR_PERIOD &&
    period <= MAX_ATR_PERIOD;
  if (!periodValid) {
    problems.push(
      `ATR period must be a whole number from ${MIN_ATR_PERIOD} to ${MAX_ATR_PERIOD} (you entered ${entered(period)})`,
    );
  }

  const multiplierValid =
    typeof multiplier === "number" && isAmount(multiplier, MAX_ATR_MULTIPLIER);
  if (!multiplierValid) {
    problems.push(
      `ATR multiplier must be more than 0 and at most ${MAX_ATR_MULTIPLIER} (you entered ${entered(multiplier)})`,
    );
  }

  return problems;
}

/** The raw value rendered as the Python rule does (JSON.stringify, U5). */
function render(value: unknown): string {
  return JSON.stringify(value) ?? "undefined";
}

/** Whether `value` is a finite JS number in `(0, upper]` (a bool is not). */
function isAmount(value: unknown, upper: number): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 && value <= upper;
}

function capError(value: unknown): string {
  return `max_position_pct must be a number in (0, ${MAX_MAX_POSITION_PCT}], got ${render(value)}`;
}

function riskError(value: unknown): string {
  return `risk_per_trade must be more than 0 and at most ${MAX_RISK_PER_TRADE} (${MAX_RISK_PCT}% of equity per trade), got ${render(value)}`;
}

/** Whether `value` is a whole number in [2, 10000] (a bool is not, `14.5` is not). */
function isAtrPeriod(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= MIN_ATR_PERIOD &&
    value <= MAX_ATR_PERIOD
  );
}

function modeError(value: unknown): string {
  return `sizing_mode must be 'risk' or 'atr', got ${render(value)}`;
}

function atrOnlyError(name: string, value: unknown): string {
  return `${name} is only valid with sizing_mode 'atr', got ${name}=${render(value)} with sizing_mode 'risk'`;
}

function atrPeriodError(value: unknown): string {
  return `sizing_atr_period must be a whole number in [${MIN_ATR_PERIOD}, ${MAX_ATR_PERIOD}], got ${render(value)}`;
}

function atrMultiplierError(value: unknown): string {
  return `sizing_atr_multiplier must be a number in (0, ${MAX_ATR_MULTIPLIER}], got ${render(value)}`;
}

/**
 * The one rule for reading sizing settings recorded with a run
 * (`recorded_sizing` + `from_params`).
 *
 * A legacy row whose parameters are absent or not a mapping (null, a list, a
 * string, a number) reads as the defaults. In a mapping, a missing key means its
 * default and an explicit value must be a finite number in range — a stored
 * `null`/string/bool/out-of-range value is an error rather than a silent
 * default. `max_position_pct` is reported first when both are bad.
 */
export function runSizing(params: Record<string, unknown> | null | undefined): RunSizing {
  if (params == null || typeof params !== "object" || Array.isArray(params)) {
    return {
      maxPositionPct: DEFAULT_MAX_POSITION_PCT,
      riskPerTrade: DEFAULT_RISK_PER_TRADE,
      legacy: true,
    };
  }

  const legacy =
    !Object.hasOwn(params, "max_position_pct") &&
    !Object.hasOwn(params, "risk_per_trade") &&
    !Object.hasOwn(params, "sizing_mode");

  const cap = Object.hasOwn(params, "max_position_pct")
    ? params.max_position_pct
    : DEFAULT_MAX_POSITION_PCT;
  if (!isAmount(cap, MAX_MAX_POSITION_PCT)) return { invalid: capError(cap) };

  const risk = Object.hasOwn(params, "risk_per_trade")
    ? params.risk_per_trade
    : DEFAULT_RISK_PER_TRADE;
  if (!isAmount(risk, MAX_RISK_PER_TRADE)) return { invalid: riskError(risk) };

  // ATR sizing (vibe-quant-hrhok AC3): a missing mode is "risk" and stores nothing
  // new; ATR mode reads the period/multiplier (defaulting each absent one) through
  // the same bounds as `SizingSettings`. An explicit null is an error, as there.
  const mode = Object.hasOwn(params, "sizing_mode") ? params.sizing_mode : "risk";
  if (mode !== "risk" && mode !== "atr") return { invalid: modeError(mode) };
  const rawPeriod = Object.hasOwn(params, "sizing_atr_period")
    ? params.sizing_atr_period
    : undefined;
  const rawMultiplier = Object.hasOwn(params, "sizing_atr_multiplier")
    ? params.sizing_atr_multiplier
    : undefined;
  let atr: AtrSizing | undefined;
  if (mode === "risk") {
    // A stored ATR field with "risk" is corrupt, never silently ignored.
    if (rawPeriod !== undefined) return { invalid: atrOnlyError("sizing_atr_period", rawPeriod) };
    if (rawMultiplier !== undefined) {
      return { invalid: atrOnlyError("sizing_atr_multiplier", rawMultiplier) };
    }
  } else {
    const period = rawPeriod === undefined ? DEFAULT_ATR_PERIOD : rawPeriod;
    if (!isAtrPeriod(period)) return { invalid: atrPeriodError(period) };
    const multiplier = rawMultiplier === undefined ? DEFAULT_ATR_MULTIPLIER : rawMultiplier;
    if (!isAmount(multiplier, MAX_ATR_MULTIPLIER)) {
      return { invalid: atrMultiplierError(multiplier) };
    }
    atr = { period, multiplier };
  }

  // T6: the optional run-level margin limit rides along; an absent or unreadable
  // value adds nothing (it never makes a run "invalid" or non-legacy).
  const maxMarginPct = storedMarginPct(params);
  const resolved: ResolvedSizing = { maxPositionPct: cap, riskPerTrade: risk, legacy };
  if (maxMarginPct !== undefined) resolved.maxMarginPct = maxMarginPct;
  if (atr !== undefined) resolved.atr = atr;
  return resolved;
}

/**
 * The sizing a discovery run recorded in its machine `notes` JSON (Z1, bead
 * vibe-quant-zrh23): the frontend mirror of `_discovery_recorded_sizing` and the
 * child resolver in `api/routers/discovery.py`.
 *
 * `null` when the notes are absent, unreadable, not a mapping, or carry no
 * `sizing` key — the run was launched at the defaults, so the caller keeps the
 * run parameters' (legacy) sizing. The recorded settings when the key holds an
 * object. A `sizing` key that is present but NOT an object, or that holds a value
 * out of range, is an error rather than a silent default (S23b/S23d).
 */
export function notesSizing(notes: string | null | undefined): RunSizing | null {
  if (notes == null) return null;

  let payload: unknown;
  try {
    payload = JSON.parse(notes);
  } catch {
    return null;
  }
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return null;
  if (!Object.hasOwn(payload, "sizing")) return null;

  const raw = (payload as Record<string, unknown>).sizing;
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return { invalid: `notes.sizing must be an object, got ${render(raw)}` };
  }
  return runSizing(raw as Record<string, unknown>);
}

/** The exposure part of the chip: per-symbol cap, plus the run total for many. */
function chipExposure(maxPositionPct: number, nSymbols: number): string {
  if (nSymbols > 1) {
    return `Max exposure ${maxPositionPct}× per symbol, up to ${nSymbols} × ${maxPositionPct}× total`;
  }
  return `Max exposure ${maxPositionPct}×`;
}

/**
 * U6-U8: the results chip label. One symbol reads "Max exposure 0.5× · Risk 2%";
 * several read the per-symbol cap and the run total; a legacy run (no stored
 * sizing keys) is marked "(default)"; unreadable sizing reads
 * "Sizing: invalid (<the Python text>)".
 *
 * ATR sizing (vibe-quant-hrhok AC3) names the mode by prefixing
 * "ATR(<period>)x<multiplier> · " to the whole label, so the exposure, risk and
 * margin parts stay exactly as a risk run's; a risk run's label is unchanged.
 */
export function sizingChipLabel(sizing: RunSizing, nSymbols: number): string {
  if ("invalid" in sizing) return `Sizing: invalid (${sizing.invalid})`;
  const exposure = chipExposure(sizing.maxPositionPct, nSymbols);
  const risk = `Risk ${riskFractionToPct(sizing.riskPerTrade)}%`;
  const suffix = sizing.legacy ? " (default)" : "";
  const margin =
    sizing.maxMarginPct === undefined ? "" : ` · Max margin ${marginPctText(sizing.maxMarginPct)}%`;
  const mode =
    sizing.atr === undefined ? "" : `ATR(${sizing.atr.period})x${sizing.atr.multiplier} · `;
  return `${mode}${exposure} · ${risk}${suffix}${margin}`;
}
