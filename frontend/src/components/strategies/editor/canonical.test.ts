/**
 * Canonical StrategyDSL <-> editor model (vibe-quant-o1vvm / 4es7v).
 *
 * Fixture = the real GET /api/strategies/templates payloads (a python test keeps it in sync).
 */
import { describe, expect, it } from "vitest";
import templates from "./__fixtures__/templates.json";
import {
  countStoredIndicators,
  maxBarsInTradeProblem,
  saveGuardMessage,
  sessionReplaceWarning,
  toCanonicalDsl,
} from "./canonical";
import {
  type DslConfig,
  emptyDslConfig,
  parseDslConfig,
  STOP_LOSS_TYPES,
  TAKE_PROFIT_TYPES,
} from "./types";
import { parseYamlToDsl, serializeConfig } from "./yamlConfig";

type Dict = Record<string, unknown>;
interface Template {
  name: string;
  strategy_type: string;
  dsl_config: Dict;
}
const TEMPLATES = templates as Template[];
const byName = (n: string): Dict => {
  const t = TEMPLATES.find((x) => x.name === n);
  if (!t) throw new Error(`no template ${n}`);
  return t.dsl_config;
};
const asDict = (v: unknown): Dict => v as Dict;

describe("parseDslConfig on canonical StrategyDSL", () => {
  it("has all 12 templates", () => {
    expect(TEMPLATES).toHaveLength(12);
  });

  it.each(TEMPLATES.map((t) => [t.name, t.dsl_config] as const))(
    "%s: indicators, conditions, risk and timeframe are read",
    (_name, raw) => {
      const cfg = parseDslConfig(raw);
      const canonicalNames = Object.keys(asDict(raw.indicators));
      expect(canonicalNames.length).toBeGreaterThan(0);
      expect(cfg.indicators.map((i) => i.name)).toEqual(canonicalNames);
      expect(cfg.indicators.map((i) => i.type)).toEqual(
        canonicalNames.map((n) => asDict(asDict(raw.indicators)[n]).type),
      );
      expect(cfg.general.timeframe).toBe(raw.timeframe);
      const entry = asDict(raw.entry_conditions);
      expect(cfg.conditions.long_entry).toHaveLength((entry.long as string[]).length);
      expect(cfg.conditions.short_entry).toHaveLength((entry.short as string[]).length);
      expect(cfg.risk.stop_loss.type).toBe(asDict(raw.stop_loss).type);
      expect(cfg.risk.take_profit.type).toBe(asDict(raw.take_profit).type);
    },
  );

  it("reads concrete values: indicator params, condition operands, stop value, filters", () => {
    const cfg = parseDslConfig(byName("macd_crossover"));
    expect(cfg.indicators.find((i) => i.name === "macd")?.params).toEqual({
      fast_period: 12,
      slow_period: 26,
      signal_period: 9,
    });
    expect(cfg.conditions.long_entry?.[0]).toEqual({
      left: "ema_fast",
      operator: "crosses_above",
      right: "ema_slow",
    });
    expect(cfg.risk.stop_loss).toEqual({ type: "atr_fixed", value: 1.5 });
    expect(cfg.time.funding_avoidance).toBe(true);
    expect(cfg.time.trading_days).toHaveLength(5);
    const mtf = parseDslConfig(byName("mtf_trend_scalp"));
    expect(mtf.indicators.find((i) => i.name === "ema_trend_4h")?.timeframe_override).toBe("4h");
    expect(mtf.general.additional_timeframes).toEqual(["1h", "4h"]);
  });

  it("keeps UI-format configs on the old path", () => {
    const cfg = parseDslConfig({
      general: { strategy_type: "breakout", symbols: ["BTCUSDT"], timeframe: "4h" },
      indicators: [{ type: "RSI", params: { period: 14 } }],
      conditions: { entry: [{ left: "RSI(14)", operator: "<", right: "30" }], exit: [] },
    });
    expect(cfg.indicators).toEqual([{ type: "RSI", params: { period: 14 } }]);
    expect(cfg.general.symbols).toEqual(["BTCUSDT"]);
  });
});

describe("toCanonicalDsl keeps what the editor cannot show", () => {
  it.each(TEMPLATES.map((t) => [t.name, t.dsl_config] as const))(
    "%s: unedited config round-trips to the stored dict",
    (_name, raw) => {
      const cfg = parseDslConfig(raw);
      expect(toCanonicalDsl(cfg, raw, cfg)).toEqual(raw);
    },
  );

  it.each(TEMPLATES.map((t) => [t.name, t.dsl_config] as const))(
    "%s: rebuilding every editable field from the editor model is lossless",
    (_name, raw) => {
      const cfg = parseDslConfig(raw);
      // baseline = blank editor => every field counts as edited and is rewritten
      expect(toCanonicalDsl(cfg, raw, emptyDslConfig())).toEqual(raw);
    },
  );

  it("an edit changes only that field; sweep, ATR stop indicator and sources survive", () => {
    const raw = byName("keltner_trend");
    const cfg = parseDslConfig(raw);
    const rsi = cfg.indicators.find((i) => i.name === "rsi");
    if (!rsi) throw new Error("rsi missing");
    rsi.params = { period: 21 };
    cfg.risk.stop_loss = { type: "atr_trailing", value: 3 };
    const out = toCanonicalDsl(cfg, raw, parseDslConfig(raw));
    const expected = structuredClone(raw) as Dict;
    asDict(asDict(expected.indicators).rsi).period = 21;
    asDict(expected.stop_loss).atr_multiplier = 3;
    expect(out).toEqual(expected);
    expect(out.sweep).toEqual(raw.sweep);
    expect(asDict(out.stop_loss).indicator).toBe("atr");
    expect(asDict(asDict(out.indicators).kc).source).toBe("close");
  });

  it("new indicator + condition on its editor label gets a canonical name and reference", () => {
    const raw = byName("dual_ema_crossover");
    const base = parseDslConfig(raw);
    const cfg = parseDslConfig(raw);
    cfg.indicators.push({ type: "RSI", params: { period: 14 } });
    cfg.conditions.long_entry = [
      ...(cfg.conditions.long_entry ?? []),
      { left: "RSI(14)", operator: "<", right: "30" },
      { left: "price", operator: ">", right: "ema_fast" },
    ];
    const out = toCanonicalDsl(cfg, raw, base);
    expect(asDict(out.indicators).rsi_14).toEqual({ type: "RSI", period: 14 });
    const long = asDict(out.entry_conditions).long as string[];
    expect(long.slice(-2)).toEqual(["rsi_14 < 30", "close > ema_fast"]);
    expect(out.stop_loss).toEqual(raw.stop_loss);
  });

  it("time edits rewrite time_filters without touching other keys", () => {
    const raw = byName("macd_crossover");
    const base = parseDslConfig(raw);
    const cfg = parseDslConfig(raw);
    cfg.time.funding_avoidance = false;
    const out = toCanonicalDsl(cfg, raw, base);
    const tf = asDict(out.time_filters);
    expect(asDict(tf.avoid_around_funding)).toEqual({
      enabled: false,
      minutes_before: 5,
      minutes_after: 5,
    });
    expect(tf.allowed_sessions).toEqual(asDict(raw.time_filters).allowed_sessions); // 2 sessions kept
  });

  it("refuses edits canonical cannot express instead of dropping them", () => {
    const raw = byName("dual_ema_crossover");
    const base = parseDslConfig(raw);
    const orCfg = parseDslConfig(raw);
    orCfg.conditions.long_entry = [
      { left: "ema_fast", operator: ">", right: "ema_slow" },
      { left: "ema_fast", operator: ">", right: "close", logic: "or" },
    ];
    expect(() => toCanonicalDsl(orCfg, raw, base)).toThrow(/OR logic/);
    const trailing = parseDslConfig(raw);
    trailing.risk.trailing_stop_pct = 1;
    expect(() => toCanonicalDsl(trailing, raw, base)).toThrow(/trailing_stop_pct/);
  });
});

describe("max_bars_in_trade time exit", () => {
  const MSG = "Max bars in trade must be a whole number from 1 to 100000";
  const withBars = (v: unknown): Dict => {
    const raw = structuredClone(byName("macd_crossover")) as Dict;
    raw.max_bars_in_trade = v;
    return raw;
  };

  it("max_bars_in_trade round-trips and clears", () => {
    const raw = withBars(12);
    const cfg = parseDslConfig(raw);
    expect(cfg.risk.max_bars_in_trade).toBe(12);
    expect(toCanonicalDsl(cfg, raw, parseDslConfig(raw))).toEqual(raw);
    expect(asDict(toCanonicalDsl(cfg, raw, parseDslConfig(raw))).max_bars_in_trade).toBe(12);

    const base = parseDslConfig(raw);
    const cleared = parseDslConfig(raw);
    cleared.risk.max_bars_in_trade = undefined;
    const out = toCanonicalDsl(cleared, raw, base);
    expect("max_bars_in_trade" in out).toBe(false);
    expect(out).toEqual(byName("macd_crossover"));
  });

  it("keeps an explicit null off and untouched on an unchanged save", () => {
    const raw = withBars(null);
    const cfg = parseDslConfig(raw);
    expect(cfg.risk.max_bars_in_trade).toBeUndefined();
    expect(toCanonicalDsl(cfg, raw, parseDslConfig(raw))).toEqual(raw);
  });

  it("editor-format configs carry the value through the YAML tab", () => {
    const raw: Dict = {
      general: { strategy_type: "momentum", symbols: ["BTCUSDT"], timeframe: "4h" },
      indicators: [{ type: "RSI", params: { period: 2 } }],
      conditions: { entry: [], exit: [] },
      risk: {
        stop_loss: { type: "fixed_pct", value: 50 },
        take_profit: { type: "fixed_pct", value: 100 },
        position_sizing: { type: "percent_equity", value: 10 },
        max_bars_in_trade: 12,
      },
    };
    const cfg = parseDslConfig(raw);
    expect(cfg.risk.max_bars_in_trade).toBe(12);
    expect(typeIntoYaml(cfg).risk.max_bars_in_trade).toBe(12);
  });

  it("refuses to save an invalid N through the canonical path", () => {
    const raw = withBars(12);
    const base = parseDslConfig(raw);
    for (const bad of [0, 12.5, 100001, Number.NaN, Number.POSITIVE_INFINITY]) {
      const cfg = parseDslConfig(raw);
      cfg.risk.max_bars_in_trade = bad;
      expect(() => toCanonicalDsl(cfg, raw, base)).toThrow(MSG);
    }
  });

  const ROWS: Array<[number | undefined, string | null]> = [
    [undefined, null],
    [1, null],
    [12, null],
    [100000, null],
    [0, MSG],
    [-3, MSG],
    [12.5, MSG],
    [100001, MSG],
    [Number.NaN, MSG],
    [Number.POSITIVE_INFINITY, MSG],
    [Number.NEGATIVE_INFINITY, MSG],
  ];

  it.each(ROWS)("maxBarsInTradeProblem(%s) -> %s", (value, expected) => {
    expect(maxBarsInTradeProblem(value)).toBe(expected);
  });
});

describe("canonical roundtrip keeps position_mode and signal_level", () => {
  const EDIT_INDICATOR: Record<string, (cfg: DslConfig) => void> = {
    macd_crossover: (cfg) => {
      const macd = cfg.indicators.find((i) => i.name === "macd");
      if (!macd) throw new Error("macd missing");
      macd.params = { ...macd.params, fast_period: 11 };
    },
    keltner_trend: (cfg) => {
      const rsi = cfg.indicators.find((i) => i.name === "rsi");
      if (!rsi) throw new Error("rsi missing");
      rsi.params = { period: 21 };
    },
  };
  const EXPECTED_PARAM: Record<string, [string, string, number]> = {
    macd_crossover: ["macd", "fast_period", 11],
    keltner_trend: ["rsi", "period", 21],
  };

  it("canonical roundtrip keeps position_mode and signal_level", () => {
    for (const name of ["macd_crossover", "keltner_trend"]) {
      const raw = structuredClone(byName(name)) as Dict;
      raw.position_mode = "reverse";
      raw.refresh_on_signal = true;
      asDict(raw.indicators).lv = { type: "SIGNAL_LEVELS", stop: "close", target: "close" };
      raw.stop_loss_long = { type: "signal_level", indicator: "lv.stop" };
      raw.take_profit_short = { type: "signal_level", indicator: "lv.target" };

      const base = parseDslConfig(raw);
      const cfg = parseDslConfig(raw);
      EDIT_INDICATOR[name]?.(cfg);

      const out = toCanonicalDsl(cfg, raw, base);
      const [indName, param, value] = EXPECTED_PARAM[name] ?? ["", "", 0];
      const expected = structuredClone(raw) as Dict;
      asDict(asDict(expected.indicators)[indName])[param] = value;
      expect(out).toEqual(expected);
      expect(out.position_mode).toBe("reverse");
      expect(out.refresh_on_signal).toBe(true);
      expect(out.stop_loss_long).toEqual({ type: "signal_level", indicator: "lv.stop" });
      expect(out.take_profit_short).toEqual({ type: "signal_level", indicator: "lv.target" });
    }
  });
});

describe("saveGuardMessage", () => {
  it("blocks an untouched save when the editor read fewer indicators than stored", () => {
    const stored = { indicators: { a: { type: "EMA" }, b: { type: "RSI" } } };
    const lossy = emptyDslConfig(); // what the old parser returned for canonical input
    expect(countStoredIndicators(stored)).toBe(2);
    expect(saveGuardMessage(stored, lossy, lossy, false)).toMatch(/read 0 of 2 stored indicators/);
  });

  it("allows saving after an explicit indicator edit or template load, and for faithful parses", () => {
    const stored = { indicators: { a: { type: "EMA" }, b: { type: "RSI" } } };
    const lossy = emptyDslConfig();
    const edited = { ...lossy, indicators: [{ type: "EMA", params: { period: 9 } }] };
    expect(saveGuardMessage(stored, lossy, edited, false)).toBeNull();
    expect(saveGuardMessage(stored, lossy, lossy, true)).toBeNull();
    const raw = byName("ema_ribbon");
    const full = parseDslConfig(raw);
    expect(saveGuardMessage(raw, full, full, false)).toBeNull();
  });
});

/** What the YAML tab does on a keystroke: dump the model, edit the text, parse it back. */
function typeIntoYaml(cfg: DslConfig): DslConfig {
  const { config, error } = parseYamlToDsl(`${serializeConfig(cfg)} `);
  expect(error).toBeNull();
  return config;
}

describe("YAML tab round trip (a keystroke must not change the stored strategy)", () => {
  it.each(TEMPLATES.map((t) => [t.name, t.dsl_config] as const))(
    "%s: typing in the YAML tab keeps indicator names, timeframes and the stored dict",
    (_name, raw) => {
      const cfg = parseDslConfig(raw);
      const back = typeIntoYaml(cfg);
      expect(back.indicators).toEqual(cfg.indicators);
      expect(toCanonicalDsl(back, raw, cfg)).toEqual(raw);
    },
  );

  it("keeps an indicator's timeframe override and name through YAML", () => {
    const raw = byName("mtf_trend_scalp");
    const back = typeIntoYaml(parseDslConfig(raw));
    const trend = back.indicators.find((i) => i.name === "ema_trend_4h");
    expect(trend?.timeframe_override).toBe("4h");
    expect(
      asDict(asDict(toCanonicalDsl(back, raw, parseDslConfig(raw)).indicators).ema_trend_4h)
        .timeframe,
    ).toBe("4h");
  });

  it("a UI-format YAML indicator without a name still gets a derived canonical name", () => {
    const { config } = parseYamlToDsl("indicators:\n  - type: RSI\n    params: {period: 14}\n");
    expect(config.indicators).toEqual([{ type: "RSI", params: { period: 14 } }]);
  });
});

describe("risk type choices always emit backend-valid types", () => {
  it("offers exactly the types the backend validates", () => {
    expect(STOP_LOSS_TYPES.map((t) => t.value)).toEqual(["fixed_pct", "atr_fixed", "atr_trailing"]);
    expect(TAKE_PROFIT_TYPES.map((t) => t.value)).toEqual([
      "fixed_pct",
      "atr_fixed",
      "risk_reward",
    ]);
  });

  it("maps the legacy editor names typed in YAML to canonical types", () => {
    const raw = byName("keltner_trend");
    const cfg = parseDslConfig(raw);
    cfg.risk.stop_loss = { type: "atr_based", value: 2 };
    cfg.risk.take_profit = { type: "rr_ratio", value: 3 };
    const out = toCanonicalDsl(cfg, raw, parseDslConfig(raw));
    expect(out.stop_loss).toEqual({ type: "atr_fixed", atr_multiplier: 2, indicator: "atr" });
    expect(out.take_profit).toEqual({ type: "risk_reward", risk_reward_ratio: 3 });
  });

  it("refuses an ATR stop when the strategy has no ATR indicator to reference", () => {
    const raw = structuredClone(byName("dual_ema_crossover")) as Dict;
    delete asDict(raw.indicators).atr;
    raw.stop_loss = { type: "fixed_pct", percent: 2 };
    raw.take_profit = { type: "fixed_pct", percent: 4 };
    const cfg = parseDslConfig(raw);
    cfg.risk.stop_loss = { type: "atr_fixed", value: 2 };
    expect(() => toCanonicalDsl(cfg, raw, parseDslConfig(raw))).toThrow(/ATR indicator/);
  });

  it("keeps the stored ATR reference rather than the first ATR indicator", () => {
    const raw = structuredClone(byName("keltner_trend")) as Dict;
    const inds = asDict(raw.indicators);
    inds.atr_aaa = { type: "ATR", period: 7 }; // sorts/creates before the referenced one
    raw.indicators = { atr_aaa: inds.atr_aaa, ...inds };
    asDict(raw.stop_loss).indicator = "atr";
    const cfg = parseDslConfig(raw);
    cfg.risk.stop_loss = { type: "atr_trailing", value: 2.5 };
    const out = toCanonicalDsl(cfg, raw, parseDslConfig(raw));
    expect(asDict(out.stop_loss).indicator).toBe("atr");
  });

  it("canonical edits and emitted dicts are pinned in editor_outputs.json (validated by python)", async () => {
    const outputs: Record<string, { ok: Dict } | { error: string }> = {};
    for (const t of TEMPLATES) {
      const raw = t.dsl_config;
      const base = parseDslConfig(raw);
      const scenarios: Array<[string, (c: DslConfig) => DslConfig]> = [
        ["yaml_keystroke", typeIntoYaml],
        ...STOP_LOSS_TYPES.map(
          ({ value }) =>
            [
              `sl_${value}`,
              (c: DslConfig) => ({
                ...c,
                risk: { ...c.risk, stop_loss: { ...c.risk.stop_loss, type: value } },
              }),
            ] as [string, (c: DslConfig) => DslConfig],
        ),
        ...TAKE_PROFIT_TYPES.map(
          ({ value }) =>
            [
              `tp_${value}`,
              (c: DslConfig) => ({
                ...c,
                risk: { ...c.risk, take_profit: { ...c.risk.take_profit, type: value } },
              }),
            ] as [string, (c: DslConfig) => DslConfig],
        ),
      ];
      for (const [scenario, edit] of scenarios) {
        const key = `${t.name}::${scenario}`;
        try {
          outputs[key] = { ok: toCanonicalDsl(edit(structuredClone(base)), raw, base) };
        } catch (e) {
          outputs[key] = { error: e instanceof Error ? e.message : String(e) };
        }
      }
    }
    await expect(`${JSON.stringify(outputs, null, 2)}\n`).toMatchFileSnapshot(
      "./__fixtures__/editor_outputs.json",
    );
  });
});

describe("time edits on canonical strategies", () => {
  it("keeps the session timezone when only the hours change", () => {
    const raw = structuredClone(byName("dual_ema_crossover")) as Dict;
    raw.time_filters = {
      allowed_sessions: [{ start: "09:30", end: "16:00", timezone: "America/New_York" }],
    };
    const base = parseDslConfig(raw);
    const cfg = parseDslConfig(raw);
    cfg.time.trading_hours = { start: "10:00", end: "15:00" };
    const out = toCanonicalDsl(cfg, raw, base);
    expect(asDict(out.time_filters).allowed_sessions).toEqual([
      { start: "10:00", end: "15:00", timezone: "America/New_York" },
    ]);
  });

  it("defaults the timezone to UTC when there was no session before", () => {
    const raw = byName("dual_ema_crossover");
    const base = parseDslConfig(raw);
    const cfg = parseDslConfig(raw);
    cfg.time.trading_hours = { start: "08:00", end: "12:00" };
    const out = toCanonicalDsl(cfg, raw, base);
    expect(asDict(out.time_filters).allowed_sessions).toEqual([
      { start: "08:00", end: "12:00", timezone: "UTC" },
    ]);
  });

  it("warns that a single trading window replaces the other stored sessions", () => {
    const raw = byName("macd_crossover"); // 2 sessions
    const base = parseDslConfig(raw);
    const cfg = parseDslConfig(raw);
    expect(sessionReplaceWarning(raw, cfg, base)).toBeNull();
    cfg.time.trading_hours = { start: "10:00", end: "12:00" };
    expect(sessionReplaceWarning(raw, cfg, base)).toMatch(/2 stored sessions will be replaced/);
    const single = byName("vwap_intraday"); // 1 session: the window edit is a faithful rewrite
    const sb = parseDslConfig(single);
    const sc = parseDslConfig(single);
    sc.time.trading_hours = { start: "10:00", end: "12:00" };
    expect(sessionReplaceWarning(single, sc, sb)).toBeNull();
  });

  it("refuses edits canonical cannot express: session names, position sizing", () => {
    const raw = byName("dual_ema_crossover");
    const base = parseDslConfig(raw);
    const sessions = parseDslConfig(raw);
    sessions.time.sessions = ["Asian"];
    expect(() => toCanonicalDsl(sessions, raw, base)).toThrow(/session filters/i);
    const sizing = parseDslConfig(raw);
    sizing.risk.position_sizing = { type: "fixed", value: 3 };
    expect(() => toCanonicalDsl(sizing, raw, base)).toThrow(/position sizing/i);
  });
});

describe("generic entry/exit conditions (non-split condition view)", () => {
  it("entry -> long; exit -> every side that has entries; names and labels resolve", () => {
    const raw = byName("macd_crossover"); // long and short entries
    const base = parseDslConfig(raw);
    const cfg = parseDslConfig(raw);
    cfg.conditions.exit = [{ left: "price", operator: "<", right: "ema_slow" }];
    const out = toCanonicalDsl(cfg, raw, base);
    const exit = asDict(out.exit_conditions);
    expect(exit.long).toEqual(["close < ema_slow"]);
    expect(exit.short).toEqual(["close < ema_slow"]);
    expect(asDict(out.entry_conditions)).toEqual(raw.entry_conditions);

    const cfg2 = parseDslConfig(raw);
    cfg2.conditions.entry = [{ left: "ema_fast", operator: ">", right: "ema_slow" }];
    const out2 = toCanonicalDsl(cfg2, raw, base);
    expect(asDict(out2.entry_conditions).long).toEqual(["ema_fast > ema_slow"]);
    expect(asDict(out2.entry_conditions).short).toEqual(asDict(raw.entry_conditions).short);
  });

  it("leaves the exit of a side without entries as stored", () => {
    const raw = structuredClone(byName("dual_ema_crossover")) as Dict;
    raw.entry_conditions = { long: ["ema_fast > ema_slow"], short: [] };
    const base = parseDslConfig(raw);
    const cfg = parseDslConfig(raw);
    cfg.conditions.exit = [{ left: "ema_fast", operator: "<", right: "ema_slow" }];
    const exit = asDict(toCanonicalDsl(cfg, raw, base).exit_conditions);
    expect(exit.long).toEqual(["ema_fast < ema_slow"]);
    expect(exit.short).toEqual(asDict(raw.exit_conditions).short);
  });
});

describe("parseYamlToDsl edge input", () => {
  it("reports empty or blank YAML as 'must be a mapping', not a parser exception", () => {
    for (const text of ["", "   \n"]) {
      const { error } = parseYamlToDsl(text);
      expect(error?.message).toBe("YAML must be a mapping (object), not a scalar or array");
    }
  });
});
