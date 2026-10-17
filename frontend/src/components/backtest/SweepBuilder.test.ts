/**
 * Sweep payload sent by the backtest launch form must be StrategyDSL.sweep: `{<indicator>_<param>:
 * [values]}` (vibe-quant-tsxf8). The payloads are snapshotted to __fixtures__/*.json; a python
 * test (tests/unit/test_ui_sweep_payload.py) feeds them to the real screening runner.
 */
import { describe, expect, it } from "vitest";
import templates from "../strategies/editor/__fixtures__/templates.json";
import catalogDefaults from "./__fixtures__/catalog_ui_strategy.json";
import folded from "./__fixtures__/folded_ui_strategy.json";
import catalog from "./__fixtures__/indicator_catalog.json";
import legacy from "./__fixtures__/legacy_ui_strategy.json";
import {
  asFloat,
  expandParam,
  MAX_SWEEP_COMBINATIONS,
  MAX_VALUES_PER_PARAM,
  pyRepr,
  type SweepableParam,
  type SweepCatalog,
  type SweepConfig,
  type SweepParam,
  strategyTranslationProblem,
  sweepableParamsFor,
  sweepProblem,
  sweepToPayload,
} from "./SweepBuilder";

type Dict = Record<string, unknown>;
/** JSON with inline number arrays: the shape biome's formatter keeps, so fixtures lint clean. */
const fixtureJson = (v: unknown): string =>
  `${JSON.stringify(v, null, 2).replace(/\[([^\]]*)\]/g, (_m, inner: string) => `[${inner.replace(/\s*\n\s*/g, " ").trim()}]`)}\n`;
const CATALOG = catalog as SweepCatalog;
/** Synthetic plugin spec (never shipped): declares its own `fast` (float) and `signal` (int). */
const NATIVE_CATALOG: SweepCatalog = {
  TEST_NATIVE_ALIAS: {
    default_params: { fast: 0.5, signal: 3 },
    param_schema: { fast: "float", signal: "int" },
  },
};
const TEMPLATES = templates as Array<{ name: string; dsl_config: Dict }>;

/** What a user gets after "Initialize" + switching every param to a 2-value list `v, v+1`. */
function twoValueSweep(params: SweepableParam[]): SweepConfig {
  return {
    params: params.map((p) => ({
      key: p.key,
      label: p.label,
      type: "list",
      fixedValue: p.currentValue,
      rangeMin: 1,
      rangeMax: 2,
      rangeStep: 1,
      listValues: `${p.currentValue}, ${p.currentValue + 1}`,
    })),
  };
}

const range = (over: Partial<SweepParam>): SweepParam => ({
  key: "ema_period",
  label: "ema.period",
  type: "range",
  fixedValue: 0,
  rangeMin: 1,
  rangeMax: 3,
  rangeStep: 1,
  listValues: "",
  ...over,
});

describe("sweepableParamsFor canonical strategies (templates)", () => {
  it("keys are <indicator_name>_<param> for every numeric indicator param", () => {
    for (const t of TEMPLATES) {
      const expected: string[] = [];
      for (const [name, cfg] of Object.entries(t.dsl_config.indicators as Record<string, Dict>)) {
        for (const [param, v] of Object.entries(cfg)) {
          if (typeof v === "number") expected.push(`${name}_${param}`);
        }
      }
      expect(
        sweepableParamsFor(t.dsl_config).map((p) => p.key),
        t.name,
      ).toEqual(expected);
    }
  });

  it("dual_ema_crossover 2-value sweep is a dict of value lists", () => {
    const t = TEMPLATES.find((x) => x.name === "dual_ema_crossover");
    const sweep = twoValueSweep(sweepableParamsFor(t!.dsl_config));
    expect(sweepToPayload(sweep)).toEqual({
      ema_fast_period: [12, 13],
      ema_slow_period: [26, 27],
      atr_period: [14, 15],
    });
  });

  it("macd keeps its *_period field names", () => {
    const t = TEMPLATES.find((x) => x.name === "macd_crossover");
    const keys = sweepableParamsFor(t!.dsl_config).map((p) => p.key);
    expect(keys).toContain("macd_fast_period");
    expect(keys).toContain("macd_signal_period");
  });

  it("template sweep payloads match the snapshot the backend test consumes", async () => {
    const payloads = Object.fromEntries(
      TEMPLATES.map((t) => [
        t.name,
        sweepToPayload(twoValueSweep(sweepableParamsFor(t.dsl_config))),
      ]),
    );
    await expect(fixtureJson(payloads)).toMatchFileSnapshot(
      "./__fixtures__/template_sweep_payloads.json",
    );
  });
});

describe("sweepableParamsFor UI-format strategies (backend translator renames indicators)", () => {
  it("keys follow translator naming: <type>_<period>, _2 on clashes, fast -> fast_period", () => {
    expect(sweepableParamsFor(legacy as Dict).map((p) => p.key)).toEqual([
      "ema_12_period",
      "ema_12_2_period",
      "macd_fast_period",
      "macd_slow_period",
      "macd_signal_period",
      "bbands_20_period",
      "bbands_20_std_dev",
      "stoch_period",
      "stoch_d_period",
      "atr_14_period",
    ]);
  });

  it("legacy sweep payload matches the snapshot the backend test consumes", async () => {
    const payload = sweepToPayload(twoValueSweep(sweepableParamsFor(legacy as Dict)));
    expect(payload.macd_fast_period).toEqual([12, 13]);
    await expect(fixtureJson(payload)).toMatchFileSnapshot(
      "./__fixtures__/legacy_sweep_payload.json",
    );
  });
});

describe("sweepableParamsFor with catalog defaults (what the editor really stores)", () => {
  const keysOf = (raw: Dict) =>
    sweepableParamsFor(raw, CATALOG).map((p) => `${p.key}${p.integer ? ":int" : ":float"}`);

  it("keeps registry spellings, aliases period_k/period_d and catalog extras", () => {
    expect(keysOf(catalogDefaults as Dict)).toEqual([
      "ema_12_period:int",
      "atr_14_period:int",
      "macd_fast_period:int",
      "macd_slow_period:int",
      "macd_signal_period:int",
      "stoch_period:int",
      "stoch_d_period:int",
      "ichimoku_tenkan:int",
      "ichimoku_kijun:int",
      "ichimoku_senkou:int",
      "adaptive_rsi_14_period:int",
      "adaptive_rsi_14_alpha:float",
      "squeeze_mom_length:int",
      "bbands_20_period:int",
      "bbands_20_std_dev:float",
    ]);
  });

  it("a single MACD or STOCH indicator is still sweepable", () => {
    const only = (type: string, params: Record<string, number>) =>
      sweepableParamsFor({ general: {}, indicators: [{ type, params }] }, CATALOG).map(
        (p) => p.key,
      );
    expect(only("MACD", { fast_period: 12, slow_period: 26, signal_period: 9 })).toEqual([
      "macd_fast_period",
      "macd_slow_period",
      "macd_signal_period",
    ]);
    expect(only("STOCH", { period_k: 14, period_d: 3 })).toEqual([
      "stoch_period",
      "stoch_d_period",
    ]);
    // without the catalog (still loading) the natives and aliases still resolve
    expect(
      sweepableParamsFor({
        general: {},
        indicators: [{ type: "STOCH", params: { period_k: 14 } }],
      }).map((p) => p.key),
    ).toEqual(["stoch_period"]);
  });

  it("BB keeps its editor name (bb_20); synonyms are STOCH-only for k/d", () => {
    const raw = {
      general: {},
      indicators: [
        { type: "BB", params: { period: 20, std_dev: 2 } },
        { type: "RSI", params: { period: 14, k_period: 3 } },
      ],
    };
    expect(sweepableParamsFor(raw, CATALOG).map((p) => p.key)).toEqual([
      "bb_20_period",
      "bb_20_std_dev",
      "rsi_14_period",
    ]);
  });

  it("equal spellings fold into one field (translator: nothing is last-wins)", () => {
    expect(strategyTranslationProblem(folded as Dict)).toBeNull();
    expect(sweepableParamsFor(folded as Dict, CATALOG).map((p) => p.key)).toEqual([
      "ema_20_period",
      "atr_14_period",
      "macd_fast_period",
      "macd_slow_period",
      "macd_signal_period",
      "stoch_14_period",
      "stoch_14_d_period",
    ]);
  });

  it("different values for two spellings: no sweep key, and the launch is flagged", () => {
    const conflict = (type: string, params: Record<string, number>) => ({
      general: {},
      indicators: [{ type, params }],
    });
    const stoch = conflict("STOCH", { period: 10, k_period: 14, d_period: 3 });
    expect(strategyTranslationProblem(stoch)).toBe(
      "Conflicting STOCH params for 'period': 'period'=10, 'k_period'=14",
    );
    expect(sweepableParamsFor(stoch, CATALOG).map((p) => p.key)).toEqual(["stoch_10_d_period"]);
    const macd = conflict("MACD", { fast: 12, fast_period: 9, slow: 26 });
    expect(strategyTranslationProblem(macd)).toMatch(/Conflicting MACD params for 'fast_period'/);
    expect(sweepableParamsFor(macd, CATALOG).map((p) => p.key)).toEqual(["macd_slow_period"]);
    // canonical strategies never go through the translator
    const templ = TEMPLATES[0]?.dsl_config as Dict;
    expect(strategyTranslationProblem(templ)).toBeNull();
  });

  it("folded payload matches the snapshot the backend test decodes", async () => {
    const payload = sweepToPayload(twoValueSweep(sweepableParamsFor(folded as Dict, CATALOG)));
    await expect(fixtureJson(payload)).toMatchFileSnapshot(
      "./__fixtures__/folded_sweep_payload.json",
    );
  });

  it("a fractional value for a legacy int param is flagged, not truncated", () => {
    const raw = { general: {}, indicators: [{ type: "EMA", params: { period: 12.5 } }] };
    expect(strategyTranslationProblem(raw)).toBe("EMA param 'period'=12.5 must be a whole number");
    // the real value is surfaced (and the launch blocked above), never silently int()ed
    const [ema] = sweepableParamsFor(raw, CATALOG);
    expect(ema).toMatchObject({ key: "ema_12_period", currentValue: 12.5, integer: true });
  });

  it("flags a fractional legacy int param on EMA, MACD, STOCH and BB alike", () => {
    const raw = (type: string, params: Record<string, number>) => ({
      general: {},
      indicators: [{ type, params }],
    });
    // Without the catalog a `fast` is ambiguous (a plugin may declare it natively), so only the
    // loaded catalog can tell MACD's legacy alias from a native param: pin the block with it.
    expect(
      strategyTranslationProblem(raw("MACD", { fast: 12.5, slow: 26, signal: 9 }), CATALOG),
    ).toBe("MACD param 'fast_period'=12.5 must be a whole number");
    expect(strategyTranslationProblem(raw("STOCH", { k_period: 14.5, d_period: 3 }))).toBe(
      "STOCH param 'period'=14.5 must be a whole number",
    );
    expect(strategyTranslationProblem(raw("BB", { period: 20.5, std_dev: 2 }))).toBe(
      "BB param 'period'=20.5 must be a whole number",
    );
  });

  it("accepts a numeric string for a legacy int param, like the backend translator", () => {
    const raw = { general: {}, indicators: [{ type: "EMA", params: { period: "14" } }] };
    expect(strategyTranslationProblem(raw)).toBeNull();
  });

  it("accepts only plain decimal numerals for a string legacy int param, like the backend", () => {
    const raw = (period: string) => ({
      general: {},
      indicators: [{ type: "EMA", params: { period } }],
    });
    for (const ok of [" 14 ", "1e1", "+7", "5."]) {
      expect(strategyTranslationProblem(raw(ok))).toBeNull();
    }
    // Number("0x10") is 16 and Python float("1_000") is 1000, so both sides reject them
    for (const bad of ["0x10", "0b11", "1_000", "1 000", "--1", "1e", "+"]) {
      expect(strategyTranslationProblem(raw(bad))).toBe(
        `EMA param 'period'='${bad}' must be a number`,
      );
    }
  });

  it("rejects a boolean or a non-numeric string for a legacy int param", () => {
    const raw = (params: Record<string, unknown>) => ({
      general: {},
      indicators: [{ type: "EMA", params }],
    });
    expect(strategyTranslationProblem(raw({ period: true }))).toBe(
      "EMA param 'period'=true must be a number",
    );
    expect(strategyTranslationProblem(raw({ period: "abc" }))).toBe(
      "EMA param 'period'='abc' must be a number",
    );
  });

  it("catalog payload matches the snapshot the backend test decodes", async () => {
    const payload = sweepToPayload(
      twoValueSweep(sweepableParamsFor(catalogDefaults as Dict, CATALOG)),
    );
    await expect(fixtureJson(payload)).toMatchFileSnapshot(
      "./__fixtures__/catalog_sweep_payload.json",
    );
  });
});

describe("expandParam / sweepToPayload", () => {
  it("range expands to a plain list and never overshoots max", () => {
    expect(
      sweepToPayload({ params: [range({ rangeMin: 1, rangeMax: 10, rangeStep: 2 })] }),
    ).toEqual({
      ema_period: [1, 3, 5, 7, 9],
    });
  });

  it("fixed becomes a one-value list", () => {
    expect(sweepToPayload({ params: [range({ type: "fixed", fixedValue: 14 })] })).toEqual({
      ema_period: [14],
    });
  });

  it("tiny steps are not collapsed by rounding", () => {
    const vals = expandParam(range({ rangeMin: 1, rangeMax: 1.0000000003, rangeStep: 1e-10 }));
    expect(new Set(vals).size).toBe(4);
  });

  it("float steps carry no drift", () => {
    expect(expandParam(range({ rangeMin: 0.1, rangeMax: 0.5, rangeStep: 0.1 }))).toEqual([
      0.1, 0.2, 0.3, 0.4, 0.5,
    ]);
  });

  it("step <= 0, max < min, empty list and over-large expansions are empty (invalid)", () => {
    expect(expandParam(range({ rangeStep: 0 }))).toEqual([]);
    expect(expandParam(range({ rangeStep: -1 }))).toEqual([]);
    expect(expandParam(range({ rangeMin: 5, rangeMax: 1 }))).toEqual([]);
    expect(expandParam(range({ type: "list", listValues: " , x" }))).toEqual([]);
    expect(
      expandParam(range({ rangeMin: 1, rangeMax: MAX_VALUES_PER_PARAM + 1, rangeStep: 1 })),
    ).toEqual([]);
  });
});

/** What sweepableParamsFor yields under the live catalog for the `ema_period` test param. */
const sweepable = (integer: boolean, key = "ema_period"): SweepableParam[] => [
  { key, label: "ema.period", currentValue: 12, integer },
];

describe("sweepProblem integer parameters", () => {
  const ints = sweepable(true);

  it("rejects fractional values for an int-typed field (every backtest would fail decoding)", () => {
    const check = (over: Partial<SweepParam>) => sweepProblem({ params: [range(over)] }, ints);
    expect(check({ rangeMin: 12, rangeMax: 13, rangeStep: 0.5 })).toMatch(
      /ema\.period is an integer parameter; 12\.5 is not a whole number/,
    );
    expect(check({ type: "fixed", fixedValue: 12.5 })).toMatch(/12\.5/);
    expect(check({ type: "list", listValues: "10, 11.5" })).toMatch(/11\.5/);
  });

  it("accepts whole numbers on int fields and fractions on float fields", () => {
    expect(
      sweepProblem({ params: [range({ rangeMin: 10, rangeMax: 14, rangeStep: 2 })] }, ints),
    ).toBeNull();
    expect(
      sweepProblem(
        { params: [range({ rangeMin: 1, rangeMax: 2, rangeStep: 0.5 })] },
        sweepable(false),
      ),
    ).toBeNull();
  });

  it("a sweep built before the catalog arrived is checked against the catalog's int types", () => {
    // canonical ICHIMOKU: tenkan is only known to be int-typed once the catalog is loaded
    const raw = {
      indicators: { ichi: { type: "ICHIMOKU", tenkan: 9, kijun: 26, senkou: 52 } },
      entry_conditions: { long: [] },
    };
    const early = sweepableParamsFor(raw);
    expect(early.find((p) => p.key === "ichi_tenkan")?.integer).toBe(false);
    const sweep = twoValueSweep(early); // "Initialize" before the catalog loads
    const tenkan = sweep.params.find((p) => p.key === "ichi_tenkan") as SweepParam;
    tenkan.listValues = "9, 9.5"; // user edits after the catalog arrives
    expect(sweepProblem(sweep, early)).toBeNull(); // stale metadata cannot see it

    const current = sweepableParamsFor(raw, CATALOG);
    expect(current.find((p) => p.key === "ichi_tenkan")?.integer).toBe(true);
    expect(sweepProblem(sweep, current)).toMatch(
      /ichi\.tenkan is an integer parameter; 9\.5 is not a whole number/,
    );
  });

  it("a swept param the current strategy metadata no longer lists blocks the launch", () => {
    expect(sweepProblem({ params: [range({})] }, [])).toMatch(
      /ema\.period is not a sweepable parameter/,
    );
  });
});

describe("sweepProblem", () => {
  it("is null for a valid sweep", () => {
    expect(sweepProblem({ params: [range({})] }, sweepable(true))).toBeNull();
  });

  it("names the param with no valid values", () => {
    expect(sweepProblem({ params: [range({ rangeStep: 0 })] }, sweepable(true))).toMatch(
      /ema\.period has no valid/,
    );
  });

  it("rejects more combinations than the cap", () => {
    const wide = (key: string) =>
      range({ key, label: key, rangeMin: 1, rangeMax: 101, rangeStep: 1 });
    expect(MAX_SWEEP_COMBINATIONS).toBe(10_000);
    expect(
      sweepProblem({ params: [wide("a"), wide("b")] }, [
        ...sweepable(true, "a"),
        ...sweepable(true, "b"),
      ]),
    ).toMatch(/10[\s\u00a0\u202f,.]?201 combinations exceed/); // 101 * 101
  });
});

describe("asFloat mirrors the translator's _as_float", () => {
  it("parses a finite number or a plain decimal numeral string, and nothing else", () => {
    const TABLE: Array<[unknown, number | null]> = [
      [14, 14],
      [14.0, 14],
      [14.5, 14.5],
      [-3, -3],
      ["14", 14],
      [" 14 ", 14],
      ["1e1", 10],
      ["+7", 7],
      ["5.", 5],
      ["-2.5", -2.5],
      // Number() and Python float() disagree on these, and Python rejects them too
      ["0x10", null],
      ["0b11", null],
      ["1_000", null],
      ["1 000", null],
      ["--1", null],
      ["1e", null],
      ["+", null],
      ["abc", null],
      ["", null],
      [" ", null],
      // non-finite, on both sides: a literal or the "inf"/"nan" strings a YAML tab can produce
      [Infinity, null],
      [-Infinity, null],
      [NaN, null],
      ["inf", null],
      ["-inf", null],
      ["nan", null],
      ["1e400", null],
      // a bool is not a period (Number(true) is 1), and neither is null/an object
      [true, null],
      [false, null],
      [null, null],
      [undefined, null],
      [{}, null],
      [[], null],
    ];
    for (const [value, expected] of TABLE) {
      expect(asFloat(value), JSON.stringify(value)).toBe(expected);
    }
  });
});

describe("pyRepr mirrors Python 3.13 repr for floats and control-character strings", () => {
  it("prints each value the way Python repr() does", () => {
    const TABLE: Array<[unknown, string]> = [
      [1e-7, "1e-07"],
      [0.0001, "0.0001"],
      [0.00001, "1e-05"],
      [1.5, "1.5"],
      // A whole number stringifies to digits only -> the editor sends an int -> Python int repr;
      // 1e21 stringifies with an exponent -> Python parses a float -> Python float repr.
      [2.5e16, "25000000000000000"],
      [1e21, "1e+21"],
      [1.5e300, "1.5e+300"],
      [14, "14"],
      [-0.000012, "-1.2e-05"],
      ["a\bb", "'a\\x08b'"],
      ["tab\t", "'tab\\t'"],
      ["\u007f", "'\\x7f'"],
      ["é", "'é'"],
      ["it's", '"it\'s"'],
    ];
    for (const [value, expected] of TABLE) {
      expect(pyRepr(value), JSON.stringify(value)).toBe(expected);
    }
  });

  it("a float conflict prints the exact backend translator message", () => {
    const macd = {
      general: {},
      indicators: [{ type: "MACD", params: { fast_period: 1e-7, fast: true } }],
    };
    expect(strategyTranslationProblem(macd)).toBe(
      "Conflicting MACD params for 'fast_period': 'fast_period'=1e-07, 'fast'=True",
    );
  });
});

describe("sweepableParamsFor UI-format numeric strings mirror the translator naming", () => {
  // BB is used so the sweepable std_dev key survives even when the period is unusable
  const bb = (period: unknown) =>
    sweepableParamsFor({
      general: {},
      indicators: [{ type: "BB", params: { period, std_dev: 2 } }],
    });

  it("names the indicator <type>_<int period> for every usable period spelling", () => {
    for (const [value, name] of [
      [14, "bb_14"],
      [14.0, "bb_14"],
      ["14", "bb_14"],
      [" 14 ", "bb_14"],
      ["1e1", "bb_10"],
    ] as Array<[unknown, string]>) {
      expect(
        bb(value).map((p) => p.key),
        JSON.stringify(value),
      ).toEqual([`${name}_period`, `${name}_std_dev`]);
    }
  });

  it("an unusable period leaves the plain <type> name and is not swept", () => {
    for (const value of ["0x10", "1_000", "abc", "", true, null, Infinity, NaN, "inf"]) {
      expect(
        bb(value).map((p) => p.key),
        JSON.stringify(value),
      ).toEqual(["bb_std_dev"]);
    }
  });

  it("a numeric string period is swept with its parsed value", () => {
    expect(bb("14")[0]).toMatchObject({ key: "bb_14_period", currentValue: 14, integer: true });
    expect(bb("1e1")[0]).toMatchObject({ key: "bb_10_period", currentValue: 10, integer: true });
  });
});

describe("legacy spelling conflicts compare like the translator's _conflict_key", () => {
  const macd = (params: Record<string, unknown>) => ({
    general: {},
    indicators: [{ type: "MACD", params }],
  });
  /** The conflict message for a param set, or null when the translator would not raise one. */
  const conflictOf = (params: Record<string, unknown>): string | null => {
    const problem = strategyTranslationProblem(macd(params));
    return problem?.startsWith("Conflicting") ? problem : null;
  };

  it("a bool never equals the number it would coerce to (Number(true) === 1)", () => {
    expect(strategyTranslationProblem(macd({ fast_period: 1, fast: true, slow: 26 }))).toBe(
      "Conflicting MACD params for 'fast_period': 'fast_period'=1, 'fast'=True",
    );
    expect(strategyTranslationProblem(macd({ fast_period: 0, fast: false, slow: 26 }))).toBe(
      "Conflicting MACD params for 'fast_period': 'fast_period'=0, 'fast'=False",
    );
  });

  it("numbers and numeric strings compare numerically (no conflict)", () => {
    for (const equal of [
      { fast_period: 1, fast: "1", slow: 26 },
      { fast_period: 1, fast: 1.0, slow: 26 },
      { fast_period: 12, fast: " 12 ", slow: 26 },
    ]) {
      expect(conflictOf(equal), JSON.stringify(equal)).toBeNull();
    }
  });

  it("two null spellings agree by repr (the launch then fails the int check)", () => {
    expect(conflictOf({ fast_period: null, fast: null, slow: 26 })).toBeNull();
  });

  it("non-numeric strings conflict and print Python reprs", () => {
    expect(strategyTranslationProblem(macd({ fast_period: "a", fast: "b", slow: 26 }))).toBe(
      "Conflicting MACD params for 'fast_period': 'fast_period'='a', 'fast'='b'",
    );
  });
});

describe("plugin-native fast/slow/signal are the plugin's own params, not legacy aliases", () => {
  // NATIVE_CATALOG's param_schema lets the UI tell a plugin param from a MACD alias.
  const NATIVE = NATIVE_CATALOG;
  const raw = (params: Record<string, unknown>) => ({
    general: {},
    indicators: [{ type: "TEST_NATIVE_ALIAS", params }],
  });

  it("sweeps the plugin's own <name>_fast key, never <name>_fast_period", () => {
    expect(
      sweepableParamsFor(raw({ fast: 0.5 }), NATIVE).map(
        (p) => `${p.key}${p.integer ? ":int" : ":float"}`,
      ),
    ).toEqual(["test_native_alias_fast:float"]);
  });

  it("does not block a native float fast (the translator passes it through typed)", () => {
    expect(strategyTranslationProblem(raw({ fast: 0.5 }), NATIVE)).toBeNull();
    expect(strategyTranslationProblem(raw({ signal: 3 }), NATIVE)).toBeNull();
  });

  it("still folds the legacy MACD alias (no registry spec declares a native fast there)", () => {
    const macdRaw = {
      general: {},
      indicators: [{ type: "MACD", params: { fast: 12, slow: 26, signal: 9 } }],
    };
    expect(sweepableParamsFor(macdRaw, CATALOG).map((p) => p.key)).toEqual([
      "macd_fast_period",
      "macd_slow_period",
      "macd_signal_period",
    ]);
    expect(strategyTranslationProblem(macdRaw, CATALOG)).toBeNull();
  });

  it("mirrors the backend's native+legacy conflict for the same indicator", () => {
    // the exact translator.py text, so both layers say the same thing about the same config
    expect(strategyTranslationProblem(raw({ fast: 18, fast_period: 12 }), NATIVE)).toBe(
      "TEST_NATIVE_ALIAS sets both its native param 'fast'=18 and the legacy alias " +
        "'fast_period'=12; 'fast' is a native TEST_NATIVE_ALIAS param, not an alias of " +
        "'fast_period': remove one",
    );
    expect(strategyTranslationProblem(raw({ signal: 3, signal_period: 9 }), NATIVE)).toBe(
      "TEST_NATIVE_ALIAS sets both its native param 'signal'=3 and the legacy alias " +
        "'signal_period'=9; 'signal' is a native TEST_NATIVE_ALIAS param, not an alias of " +
        "'signal_period': remove one",
    );
  });
});

describe("strategyTranslationProblem fails open while the catalog is unknown", () => {
  const plugin = (params: Record<string, number>) => ({
    general: {},
    indicators: [{ type: "TEST_NATIVE_ALIAS", params }],
  });
  const macd = (params: Record<string, number>) => ({
    general: {},
    indicators: [{ type: "MACD", params }],
  });

  it("does not false-block a plugin's native float fast (catalog loading, or the fetch failed)", () => {
    expect(strategyTranslationProblem(plugin({ fast: 0.5 }))).toBeNull();
    expect(strategyTranslationProblem(plugin({ signal: 3 }))).toBeNull();
    // loaded: the spec declares `fast` natively, so it is the plugin's own param, not an alias
    expect(strategyTranslationProblem(plugin({ fast: 0.5 }), NATIVE_CATALOG)).toBeNull();
  });

  it("blocks MACD's legacy alias once the catalog says the spec declares no native fast", () => {
    // without the catalog the two are indistinguishable, so the backend's 422 decides
    expect(strategyTranslationProblem(macd({ fast: 12.5, slow: 26 }))).toBeNull();
    expect(strategyTranslationProblem(macd({ fast: 12.5, slow: 26 }), CATALOG)).toBe(
      "MACD param 'fast_period'=12.5 must be a whole number",
    );
  });

  it("only the int check fails open: other spellings and conflicts are still reported", () => {
    const ema = { general: {}, indicators: [{ type: "EMA", params: { period: 12.5 } }] };
    expect(strategyTranslationProblem(ema)).toBe("EMA param 'period'=12.5 must be a whole number");
    expect(strategyTranslationProblem(plugin({ fast_period: 1, fast: 2 }))).toBe(
      "Conflicting TEST_NATIVE_ALIAS params for 'fast_period': 'fast_period'=1, 'fast'=2",
    );
  });
});
