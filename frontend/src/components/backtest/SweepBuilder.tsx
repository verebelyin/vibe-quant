import { useMemo, useState } from "react";
import { isCanonicalDsl } from "@/components/strategies/editor/canonical";
import { parseDslConfig } from "@/components/strategies/editor/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type SweepValueType = "fixed" | "range" | "list";

export interface SweepParam {
  /** Canonical sweep key the backend resolves onto a compiled config field, e.g. "ema_fast_period". */
  key: string;
  label: string;
  type: SweepValueType;
  fixedValue: number;
  rangeMin: number;
  rangeMax: number;
  rangeStep: number;
  listValues: string;
}

export interface SweepConfig {
  params: SweepParam[];
}

/** A strategy parameter the backend can sweep, keyed by its compiled-config field name. */
export interface SweepableParam {
  /** `<indicator_name>_<param>`, e.g. "macd_fast_period" (StrategyDSL.sweep key). */
  key: string;
  label: string;
  currentValue: number;
  /** Compiled config field is int-typed: fractional sweep values fail every backtest's decode. */
  integer: boolean;
}

/** Indicator catalog entries by type name (GET /api/indicators/catalog), the params specs declare. */
export type SweepCatalog = Record<
  string,
  { default_params: Record<string, number>; param_schema: Record<string, string> }
>;

/** Hard limits so a typo (step 0.0001) cannot freeze the tab or flood the screening pool. */
export const MAX_VALUES_PER_PARAM = 1000;
export const MAX_SWEEP_COMBINATIONS = 10_000;

/** Int-typed DSL-native fields (IndicatorConfig); std_dev / atr_multiplier are floats. */
const NATIVE_INT_FIELDS = new Set([
  "period",
  "fast_period",
  "slow_period",
  "signal_period",
  "d_period",
]);
const NATIVE_FIELDS = new Set([...NATIVE_INT_FIELDS, "std_dev", "atr_multiplier"]);
/** Editor type spellings the backend translator renames (dsl/translator.py _TYPE_ALIASES). */
const TYPE_ALIASES: Record<string, string> = { BB: "BBANDS" };

/** Synonym groups of dsl/translator.py: canonical field -> accepted spellings. */
const PARAM_SYNONYMS: Array<[string, string[]]> = [
  ["fast_period", ["fast_period", "fast"]],
  ["slow_period", ["slow_period", "slow"]],
  ["signal_period", ["signal_period", "signal"]],
];
/** STOCH only: k/d periods (`period`, `k_period`, `period_k` -> period; `d_period`, `period_d`). */
const STOCH_SYNONYMS: Array<[string, string[]]> = [
  ["period", ["period", "k_period", "period_k"]],
  ["d_period", ["d_period", "period_d"]],
];
/** Spellings a registry spec may declare natively: the legacy side of `PARAM_SYNONYMS`. Only
 * these can be a plugin's own param, so a type the catalog cannot describe may still own them. */
const SPEC_NATIVE_ALIASES = new Set<string>();
for (const [field, spellings] of PARAM_SYNONYMS) {
  for (const key of spellings) if (key !== field) SPEC_NATIVE_ALIASES.add(key);
}

interface LegacyFields {
  /** StrategyDSL field -> source param key (the spelling whose value the translator uses) */
  fields: Map<string, string>;
  /** Why translation raises, in the translator's own order (first = the launch failure text) */
  problems: string[];
  /** Every spelling the translator consumes (never an "extra") */
  handled: Set<string>;
}

/**
 * Param names the indicator's catalog spec declares natively (dsl/translator.py
 * `_native_param_names`), or undefined when the catalog cannot describe this type -- still
 * loading, failed, or no entry for it. Without them nothing can tell a plugin's own `fast`
 * from a legacy alias, so the checks that depend on it stay silent (the backend decides).
 */
function nativeParamNames(typeKey: string, catalog?: SweepCatalog): Set<string> | undefined {
  const entry = catalog?.[TYPE_ALIASES[typeKey] ?? typeKey];
  return entry === undefined ? undefined : new Set(Object.keys(entry.param_schema ?? {}));
}

/**
 * UI-format params -> StrategyDSL fields exactly as dsl/translator.py `_canonical_params`:
 * equal spellings (by `_conflict_key`) fold, different values are a conflict (the launch fails),
 * nothing is last-wins. A spelling the catalog declares natively is the indicator's own param,
 * not an alias of the canonical field, so it is left to pass through (STOCH's registry-native
 * period_k/period_d are the deliberate exception). Non-STOCH `period` is handled on its own,
 * then std_dev / atr_multiplier.
 */
function legacyNativeFields(
  type: string,
  params: Record<string, number>,
  native?: Set<string>,
): LegacyFields {
  const fields = new Map<string, string>();
  const problems: string[] = [];
  const handled = new Set(["period", "std_dev", "atr_multiplier"]);
  if ("period" in params) fields.set("period", "period");
  const groups = type === "STOCH" ? [...PARAM_SYNONYMS, ...STOCH_SYNONYMS] : PARAM_SYNONYMS;
  const stochFields = new Set(STOCH_SYNONYMS.map(([field]) => field));
  const isNative = (k: string) => native?.has(k) === true;
  for (const [field, spellings] of groups) {
    // STOCH's k/d spellings fold even though the registry declares them (translator: no native
    // precedence there); every other group is subject to it.
    const nativePrecedence = !stochFields.has(field);
    const isNativeAlias = (k: string) => nativePrecedence && k !== field && isNative(k);
    if (nativePrecedence && field in params) {
      // dsl/translator.py raises as soon as the canonical field and a spec-native spelling are
      // both set: the plugin's own param and the legacy alias ask for two different things.
      const shadowed = spellings.find((k) => k !== field && isNative(k) && k in params);
      if (shadowed !== undefined) {
        problems.push(
          `${type} sets both its native param '${shadowed}'=${pyRepr(params[shadowed])} and ` +
            `the legacy alias '${field}'=${pyRepr(params[field])}; '${shadowed}' is a native ` +
            `${type} param, not an alias of '${field}': remove one`,
        );
        continue;
      }
    }
    const found = spellings.filter((k) => k in params && !isNativeAlias(k));
    for (const k of spellings) if (!isNativeAlias(k)) handled.add(k);
    if (new Set(found.map((k) => conflictKey(params[k]))).size > 1) {
      const shown = found.map((k) => `'${k}'=${pyRepr(params[k])}`).join(", ");
      problems.push(`Conflicting ${type} params for '${field}': ${shown}`);
      fields.delete(field);
      continue;
    }
    const first = found[0];
    if (first !== undefined) fields.set(field, first);
  }
  for (const k of ["std_dev", "atr_multiplier"]) if (k in params) fields.set(k, k);
  return { fields, problems, handled };
}

// Plain decimal numeral, as dsl/translator.py `_DECIMAL`: `Number()` also takes `0x10`, Python `float` `1_000`.
const DECIMAL = /^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$/;

/**
 * dsl/translator.py `_as_float`: the value as a finite number, else null.
 *
 * A number must be finite. A string must be a plain decimal numeral (`DECIMAL`) that `Number()`
 * reads the way Python `float()` does -- `Number("0x10")` is 16 but `float("0x10")` raises, and
 * `float("1_000")` is 1000 while `Number("1_000")` is NaN, so both are rejected. A bool is
 * excluded (`Number(true)` is 1, but a flag is not a period) and so is anything `JSON.parse`
 * cannot make a number (null, objects, arrays).
 */
export function asFloat(value: unknown): number | null {
  if (typeof value === "boolean" || value === null || value === undefined) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value !== "string" || !DECIMAL.test(value.trim())) return null;
  const number = Number(value.trim());
  return Number.isFinite(number) ? number : null;
}

/**
 * Python `repr` of a finite float: the shortest round-trip digits JS `toString` and Python `repr`
 * share, laid out the way Python does. Python always signs the exponent and pads it to two digits,
 * and switches to exponent notation below `1e-4` (exp < -4) and at `1e16` (exp >= 16), where JS
 * switches at `1e-6` and `1e21` instead.
 */
function pyFloatRepr(value: number): string {
  const [mantissa = "", exponentText] = Math.abs(value).toString().split("e");
  let digits: string;
  let exponent: number;
  if (exponentText !== undefined) {
    // JS exponential form ("d" or "d.ddd"): the leading digit sits at 10^exponentText.
    digits = mantissa.replace(".", "");
    exponent = Number(exponentText);
  } else {
    const [integerPart = "", fractionPart = ""] = mantissa.split(".");
    if (integerPart !== "0") {
      digits = integerPart + fractionPart;
      exponent = integerPart.length - 1;
    } else {
      const zeros = fractionPart.length - fractionPart.replace(/^0+/, "").length;
      digits = fractionPart.slice(zeros);
      exponent = -(zeros + 1);
    }
  }
  const lead = digits[0] ?? "0";
  const rest = digits.slice(1);
  let body: string;
  if (exponent < -4 || exponent >= 16) {
    const sign = exponent < 0 ? "-" : "+";
    body = `${lead}${rest ? `.${rest}` : ""}e${sign}${String(Math.abs(exponent)).padStart(2, "0")}`;
  } else if (exponent >= digits.length - 1) {
    body = `${digits}${"0".repeat(exponent - (digits.length - 1))}.0`;
  } else if (exponent >= 0) {
    body = `${digits.slice(0, exponent + 1)}.${digits.slice(exponent + 1)}`;
  } else {
    body = `0.${"0".repeat(-exponent - 1)}${digits}`;
  }
  return value < 0 ? `-${body}` : body;
}

/**
 * Python `repr` of a JSON-parsed value, for translator messages built with `{v!r}`.
 *
 * A number mirrors how Python's `json` reads the editor's `JSON.stringify` output: an integer form
 * (no `.`, no exponent) becomes a Python `int`, so it prints as exact decimal digits (`BigInt`);
 * anything else is a float and prints via {@link pyFloatRepr}. JS cannot tell `14` from `14.0`, so
 * a whole number typed as `14.0` prints as `14`, not `14.0` -- not fixable in JS.
 *
 * A string escapes `\` `\n` `\r` `\t` and every other code point `< 0x20` or `= 0x7f` as `\xNN`
 * (lowercase hex); a printable non-ASCII char stays literal and the quote flips to `"` when the
 * string holds a `'` and no `"` -- as Python `repr` does.
 *
 * Exported so tests can pin it against Python `repr` directly, as `asFloat` is exported.
 */
export function pyRepr(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") {
    if (Number.isNaN(value)) return "nan";
    if (!Number.isFinite(value)) return value > 0 ? "inf" : "-inf";
    const json = JSON.stringify(value);
    if (json === undefined || /[.eE]/.test(json)) return pyFloatRepr(value);
    return BigInt(json).toString();
  }
  if (typeof value === "string") {
    const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
    let body = "";
    for (const char of value) {
      const code = char.codePointAt(0) ?? 0;
      if (char === "\\") body += "\\\\";
      else if (char === "\n") body += "\\n";
      else if (char === "\r") body += "\\r";
      else if (char === "\t") body += "\\t";
      else if (char === quote) body += `\\${quote}`;
      else if (code < 0x20 || code === 0x7f) body += `\\x${code.toString(16).padStart(2, "0")}`;
      else body += char;
    }
    return `${quote}${body}${quote}`;
  }
  return JSON.stringify(value) ?? "None";
}

/**
 * dsl/translator.py `_conflict_key`: the comparison key for two spellings of one param.
 * Numbers and numeric strings compare numerically; everything else by its `repr`, so a bool
 * never equals the number it would coerce to (`Number(true)` is 1, `_as_float(True)` is None).
 */
function conflictKey(value: unknown): number | string {
  return asFloat(value) ?? pyRepr(value);
}

/**
 * Mirrors dsl/translator.py `_int_param` for one legacy int field: a number or a numeric string
 * is parsed, a bool or non-numeric string is rejected, and a fractional value is rejected (never
 * silently truncated). Returns the translator's message, or null when the value is a whole number.
 */
function intParamProblem(type: string, field: string, value: unknown): string | null {
  const shown = typeof value === "string" ? `'${value}'` : `${value}`;
  const problem = (reason: string) => `${type} param '${field}'=${shown} ${reason}`;
  let num: number;
  if (typeof value === "number") num = value;
  else if (typeof value === "string" && DECIMAL.test(value.trim())) num = Number(value);
  else return problem("must be a number");
  if (!Number.isFinite(num)) return problem("must be a number");
  if (!Number.isInteger(num)) return problem("must be a whole number");
  return null;
}

/**
 * Why launching this UI-format strategy would fail in the translator (null = fine): one
 * indicator sets both a spec-native param and its legacy alias spelling, two spellings of one
 * param hold different values, or an int param holds a value the translator rejects instead of
 * silently truncating. Applies to every launch.
 *
 * `catalog` is what tells a plugin's native `fast` from a legacy alias; while it is unknown the
 * int check for those spellings is skipped (fail open: the backend validates and answers 422).
 */
export function strategyTranslationProblem(
  raw: Record<string, unknown>,
  catalog?: SweepCatalog,
): string | null {
  if (isCanonicalDsl(raw)) return null;
  for (const ind of parseDslConfig(raw).indicators) {
    const type = ind.type.toUpperCase();
    const native = nativeParamNames(type, catalog);
    const { fields, problems } = legacyNativeFields(type, ind.params, native);
    if (problems[0]) return problems[0];
    for (const [field, key] of fields) {
      if (!NATIVE_INT_FIELDS.has(field)) continue;
      if (native === undefined && SPEC_NATIVE_ALIASES.has(key)) continue;
      const problem = intParamProblem(type, field, ind.params[key]);
      if (problem) return problem;
    }
  }
  return null;
}

/**
 * Sweepable params of a stored strategy dsl_config, keyed exactly as the compiled strategy config.
 *
 * Canonical StrategyDSL: indicator key + numeric param (`ema_fast` + `period` -> `ema_fast_period`).
 * UI-format: the backend translator renames indicators (`<type>_<int period>`, `_2` on clashes) and
 * maps synonyms (`fast`/`fast_period`, STOCH `k_period`/`period_k` -> `period`), so the key mirrors
 * that (a backend test pins the match). Catalog-declared extras the translator does not map (plugin
 * `alpha`, ICHIMOKU `tenkan`, ...) are still compiled config fields (spec defaults), so they stay
 * sweepable; `catalog` also supplies their int/float type.
 */
export function sweepableParamsFor(
  raw: Record<string, unknown>,
  catalog?: SweepCatalog,
): SweepableParam[] {
  const canonical = isCanonicalDsl(raw);
  const out: SweepableParam[] = [];
  const taken = new Set<string>();
  for (const ind of parseDslConfig(raw).indicators) {
    const typeKey = ind.type.toUpperCase();
    const entry = catalog?.[TYPE_ALIASES[typeKey] ?? typeKey];
    const declared = new Set([
      ...Object.keys(entry?.param_schema ?? {}),
      ...Object.keys(entry?.default_params ?? {}),
    ]);
    const isInt = (field: string, key: string) =>
      NATIVE_INT_FIELDS.has(field) || entry?.param_schema[key] === "int";
    const numeric = (key: string) => asFloat(ind.params[key]);

    let name = ind.name ?? "";
    // [field, source param key] in display order
    let fields: Array<[string, string]>;
    if (canonical) {
      // an undeclared numeric key would be rejected by the compiled config: drop it once the catalog is known
      fields = Object.keys(ind.params)
        .filter((k) => !entry || NATIVE_FIELDS.has(k) || declared.has(k))
        .map((k) => [k, k]);
    } else {
      const period = asFloat(ind.params.period);
      const base =
        period !== null
          ? `${ind.type.toLowerCase()}_${Math.trunc(period)}`
          : ind.type.toLowerCase();
      name = base;
      for (let n = 2; taken.has(name); n++) name = `${base}_${n}`;
      const legacy = legacyNativeFields(typeKey, ind.params, nativeParamNames(typeKey, catalog));
      fields = [...legacy.fields]; // a conflicting spelling group has no field: the launch fails
      for (const key of Object.keys(ind.params)) {
        if (!legacy.handled.has(key) && declared.has(key)) fields.push([key, key]);
      }
    }
    taken.add(name);
    for (const [field, key] of fields) {
      const value = numeric(key);
      if (value === null) continue;
      const integer = isInt(field, key);
      out.push({
        key: `${name}_${field}`,
        label: `${name}.${field}`,
        currentValue: value,
        integer,
      });
    }
  }
  return out;
}

function listNumbers(p: SweepParam): number[] {
  return p.listValues
    .split(",")
    .map((v) => v.trim())
    .filter((v) => v !== "")
    .map(Number)
    .filter((n) => Number.isFinite(n));
}

/** Expand a sweep param to its concrete values ([] = invalid or over MAX_VALUES_PER_PARAM) */
export function expandParam(p: SweepParam): number[] {
  switch (p.type) {
    case "fixed":
      return Number.isFinite(p.fixedValue) ? [p.fixedValue] : [];
    case "range": {
      if (!(p.rangeStep > 0) || !(p.rangeMax >= p.rangeMin)) return [];
      // floor (+eps for float drift) so the last value never overshoots rangeMax
      const count = Math.floor((p.rangeMax - p.rangeMin) / p.rangeStep + 1e-9) + 1;
      if (count > MAX_VALUES_PER_PARAM) return [];
      const vals: number[] = [];
      for (let i = 0; i < count; i++) {
        vals.push(Number((p.rangeMin + i * p.rangeStep).toPrecision(12)));
      }
      return vals;
    }
    case "list": {
      const vals = listNumbers(p);
      return vals.length > MAX_VALUES_PER_PARAM ? [] : vals;
    }
    default:
      return [];
  }
}

/** Number of values a single param generates */
function paramValueCount(p: SweepParam): number {
  return expandParam(p).length;
}

/**
 * Why this sweep cannot launch (null = fine). Launching it would fail at screening start.
 * `current` is sweepableParamsFor under the live catalog: int-ness is read from it at check time
 * (never cached in the sweep state), and a param it no longer lists cannot be swept.
 */
export function sweepProblem(sweep: SweepConfig, current: SweepableParam[]): string | null {
  const byKey = new Map(current.map((c) => [c.key, c]));
  for (const p of sweep.params) {
    if (!byKey.has(p.key)) return `${p.label} is not a sweepable parameter of this strategy.`;
    if (paramValueCount(p) === 0)
      return `${p.label} has no valid values (check min/max/step, list, or more than ${MAX_VALUES_PER_PARAM} values).`;
  }
  for (const p of sweep.params) {
    const bad = byKey.get(p.key)?.integer
      ? expandParam(p).find((v) => !Number.isInteger(v))
      : undefined;
    if (bad !== undefined)
      return `${p.label} is an integer parameter; ${bad} is not a whole number.`;
  }
  const total = sweep.params.reduce((acc, p) => acc * paramValueCount(p), 1);
  if (total > MAX_SWEEP_COMBINATIONS)
    return `${total.toLocaleString()} combinations exceed the ${MAX_SWEEP_COMBINATIONS.toLocaleString()} limit.`;
  return null;
}

/** Convert sweep config to StrategyDSL.sweep (`{field: [values]}`) for the API */
export function sweepToPayload(sweep: SweepConfig): Record<string, number[]> {
  const result: Record<string, number[]> = {};
  for (const p of sweep.params) result[p.key] = expandParam(p);
  return result;
}

interface SweepBuilderProps {
  /** Strategy params to sweep (from sweepableParamsFor) */
  params: SweepableParam[];
  /** Callback when sweep config changes */
  onChange: (config: SweepConfig) => void;
  /** Current sweep config */
  value: SweepConfig;
}

const PRESETS = {
  quick: { label: "Quick Scan", description: "3 values per range param" },
  fine: { label: "Fine Grid", description: "10+ values per range param" },
  custom: { label: "Custom", description: "Manual configuration" },
} as const;

type PresetKey = keyof typeof PRESETS;

function applyPreset(params: SweepableParam[], preset: "quick" | "fine"): SweepParam[] {
  return params.map((p) => {
    const { key, label } = p;
    const base = p.currentValue;

    if (preset === "quick") {
      // 3 values: 0.5x, 1x, 1.5x of current
      const step = Math.max(1, Math.round(base * 0.5));
      return {
        key,
        label,
        type: "range" as const,
        fixedValue: base,
        rangeMin: Math.max(1, base - step),
        rangeMax: base + step,
        rangeStep: step,
        listValues: "",
      };
    }
    // fine: 10 steps from 0.25x to 2x
    const min = Math.max(1, Math.round(base * 0.25));
    const max = Math.round(base * 2);
    const step = Math.max(1, Math.round((max - min) / 10));
    return {
      key,
      label,
      type: "range" as const,
      fixedValue: base,
      rangeMin: min,
      rangeMax: max,
      rangeStep: step,
      listValues: "",
    };
  });
}

export function SweepBuilder({ params: indicatorParams, onChange, value }: SweepBuilderProps) {
  const [preset, setPreset] = useState<PresetKey>("custom");

  const totalCombos = useMemo(() => {
    if (value.params.length === 0) return 0;
    return value.params.reduce((acc, p) => acc * paramValueCount(p), 1);
  }, [value.params]);
  const problem = sweepProblem(value, indicatorParams);

  const handleInitialize = () => {
    const params = indicatorParams.map((p) => ({
      key: p.key,
      label: p.label,
      type: "fixed" as const,
      fixedValue: p.currentValue,
      rangeMin: 1,
      rangeMax: p.currentValue * 2,
      rangeStep: 1,
      listValues: "",
    }));
    onChange({ params });
  };

  const handleApplyPreset = (p: PresetKey) => {
    setPreset(p);
    if (p === "custom") return;
    const params = applyPreset(indicatorParams, p);
    onChange({ params });
  };

  const updateParam = (idx: number, updates: Partial<SweepParam>) => {
    const next = [...value.params];
    const existing = next[idx]!;
    next[idx] = { ...existing, ...updates } as SweepParam;
    setPreset("custom");
    onChange({ params: next });
  };

  if (indicatorParams.length === 0) {
    return (
      <div className="rounded-lg border border-dashed py-6 text-center text-sm text-muted-foreground">
        No sweepable parameters. Add indicators to the strategy first.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Parameter Sweep</h3>
          {totalCombos > 0 && (
            <p className="text-xs text-muted-foreground">
              {totalCombos.toLocaleString()} total combination{totalCombos !== 1 ? "s" : ""}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {(Object.keys(PRESETS) as PresetKey[]).map((key) => (
            <Button
              key={key}
              variant={preset === key ? "default" : "outline"}
              size="sm"
              onClick={() => handleApplyPreset(key)}
            >
              {PRESETS[key].label}
            </Button>
          ))}
        </div>
      </div>

      {/* Initialize if empty */}
      {value.params.length === 0 ? (
        <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed py-6">
          <p className="text-sm text-muted-foreground">
            {indicatorParams.length} parameter{indicatorParams.length !== 1 ? "s" : ""} available
            for sweep
          </p>
          <Button variant="outline" size="sm" onClick={handleInitialize}>
            Initialize Sweep Parameters
          </Button>
        </div>
      ) : (
        <div className="space-y-3">
          {value.params.map((param, idx) => (
            <Card key={param.key}>
              <CardContent className="space-y-3 p-4">
                <div className="flex items-center justify-between">
                  <Label className="font-mono text-sm">{param.label}</Label>
                  <Select
                    value={param.type}
                    onValueChange={(v) => updateParam(idx, { type: v as SweepValueType })}
                  >
                    <SelectTrigger className="w-[120px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fixed">Fixed</SelectItem>
                      <SelectItem value="range">Range</SelectItem>
                      <SelectItem value="list">List</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {param.type === "fixed" && (
                  <div className="space-y-1">
                    <Label className="text-xs">Value</Label>
                    <Input
                      type="number"
                      value={param.fixedValue}
                      onChange={(e) => updateParam(idx, { fixedValue: Number(e.target.value) })}
                      className="h-8 w-32 text-xs"
                    />
                  </div>
                )}

                {param.type === "range" && (
                  <div className="grid grid-cols-3 gap-3">
                    <div className="space-y-1">
                      <Label className="text-xs">Min</Label>
                      <Input
                        type="number"
                        value={param.rangeMin}
                        onChange={(e) => updateParam(idx, { rangeMin: Number(e.target.value) })}
                        className="h-8 text-xs"
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">Max</Label>
                      <Input
                        type="number"
                        value={param.rangeMax}
                        onChange={(e) => updateParam(idx, { rangeMax: Number(e.target.value) })}
                        className="h-8 text-xs"
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">Step</Label>
                      <Input
                        type="number"
                        value={param.rangeStep}
                        onChange={(e) => updateParam(idx, { rangeStep: Number(e.target.value) })}
                        className="h-8 text-xs"
                      />
                    </div>
                    <p className="col-span-3 text-xs text-muted-foreground">
                      {paramValueCount(param)} value{paramValueCount(param) !== 1 ? "s" : ""}
                    </p>
                  </div>
                )}

                {param.type === "list" && (
                  <div className="space-y-1">
                    <Label className="text-xs">Values (comma-separated)</Label>
                    <Input
                      type="text"
                      value={param.listValues}
                      onChange={(e) => updateParam(idx, { listValues: e.target.value })}
                      placeholder="e.g. 5, 10, 20, 50"
                      className="h-8 text-xs"
                    />
                    <p className="text-xs text-muted-foreground">
                      {paramValueCount(param)} value{paramValueCount(param) !== 1 ? "s" : ""}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {problem && (
        <div className="rounded-md border border-destructive bg-destructive/10 px-3 py-2">
          <p className="text-sm text-destructive">{problem}</p>
        </div>
      )}

      {/* Combo warning */}
      {totalCombos > 1000 && !problem && (
        <div className="rounded-md border border-yellow-600 bg-yellow-600/10 px-3 py-2">
          <p className="text-sm text-yellow-600">
            {totalCombos.toLocaleString()} combinations may take a long time to run. Consider
            reducing ranges or using Quick Scan preset.
          </p>
        </div>
      )}
    </div>
  );
}
