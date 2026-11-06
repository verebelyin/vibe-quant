/**
 * Tests for the frontend sizing rule (bead vibe-quant-2gyk1.2, S6): the mirror
 * of `vibe_quant/sizing_settings.py` (contract U1-U7).
 *
 * The cap is entered in multiples of balance and the risk in PERCENT; the
 * conversion rounds to 12 significant digits so a default launch (50% cap, 2%
 * risk) stores no keys and the chip shows 7 for a stored 0.07. A recorded value
 * that is not a finite number in range is invalid, with the Python text.
 */
import { describe, expect, it } from "vitest";
import {
  atrSizingProblems,
  DEFAULT_ATR_MULTIPLIER,
  DEFAULT_ATR_PERIOD,
  DEFAULT_MAX_POSITION_PCT,
  DEFAULT_RISK_PCT,
  DEFAULT_RISK_PER_TRADE,
  MAX_ATR_MULTIPLIER,
  MAX_ATR_PERIOD,
  MAX_MAX_POSITION_PCT,
  MAX_RISK_PCT,
  MIN_ATR_PERIOD,
  notesSizing,
  type RunSizing,
  riskFractionToPct,
  riskPctToFraction,
  runSizing,
  sizingChipLabel,
  sizingPayload,
  sizingProblems,
} from "@/lib/sizing";

const LEGACY: RunSizing = { maxPositionPct: 0.5, riskPerTrade: 0.02, legacy: true };
const INVALID_CAP = "max_position_pct must be a number in (0, 125], got -1";

describe("percent conversion", () => {
  it("converts percent both ways", () => {
    expect(DEFAULT_MAX_POSITION_PCT).toBe(0.5);
    expect(DEFAULT_RISK_PER_TRADE).toBe(0.02);
    expect(DEFAULT_RISK_PCT).toBe(2);
    expect(MAX_MAX_POSITION_PCT).toBe(125);
    expect(MAX_RISK_PCT).toBe(50);

    expect(riskPctToFraction(2)).toBe(0.02);
    expect(riskPctToFraction(1)).toBe(0.01);
    expect(riskPctToFraction(33.3)).toBe(0.333);
    expect(riskPctToFraction(0.1)).toBe(0.001);
    expect(riskPctToFraction(50)).toBe(0.5);

    expect(riskFractionToPct(0.07)).toBe(7);
    expect(riskFractionToPct(0.029)).toBe(2.9);
    // The measured trap the rounding exists for (raw 0.07 * 100 is not 7).
    expect(0.07 * 100).not.toBe(7);

    // The default percent pair stores nothing.
    expect(sizingPayload(riskPctToFraction(50), riskPctToFraction(2))).toStrictEqual({});
  });
});

describe("sizingPayload", () => {
  it("sends only the non-default keys", () => {
    expect(sizingPayload(DEFAULT_MAX_POSITION_PCT, DEFAULT_RISK_PER_TRADE)).toStrictEqual({});
    expect(sizingPayload(1, DEFAULT_RISK_PER_TRADE)).toStrictEqual({ max_position_pct: 1 });
    expect(sizingPayload(DEFAULT_MAX_POSITION_PCT, 0.01)).toStrictEqual({ risk_per_trade: 0.01 });
    expect(sizingPayload(1, 0.01)).toStrictEqual({ max_position_pct: 1, risk_per_trade: 0.01 });
  });
});

describe("sizingProblems", () => {
  it("risk percent bounds", () => {
    const problem = (entered: string) =>
      `Risk per trade must be more than 0% and at most 50% of equity per trade (you entered ${entered})`;

    expect(sizingProblems(1, 0, 10)).toStrictEqual([problem("0%")]);
    expect(sizingProblems(1, -1, 10)).toStrictEqual([problem("-1%")]);
    expect(sizingProblems(1, 50.0001, 10)).toStrictEqual([problem("50.0001%")]);
    expect(sizingProblems(1, "", 10)).toStrictEqual([problem("nothing")]);
    // 50% is allowed (the upper edge).
    expect(sizingProblems(1, 50, 10)).toStrictEqual([]);
  });

  it("exposure bounds and leverage", () => {
    const problem = (entered: string) =>
      `Max exposure must be more than 0 and at most 125 × balance (you entered ${entered})`;

    expect(sizingProblems(0, 2, 10)).toStrictEqual([problem("0")]);
    expect(sizingProblems(125.5, 2, 10)).toStrictEqual([problem("125.5")]);
    expect(sizingProblems("", 2, 10)).toStrictEqual([problem("nothing")]);
    // 125 is allowed (the upper edge); 125.0001 is not.
    expect(sizingProblems(125, 2, 125)).toStrictEqual([]);
    expect(sizingProblems(125.0001, 2, 125)).toStrictEqual([problem("125.0001")]);

    expect(sizingProblems(20, 2, 10)).toStrictEqual([
      "Max exposure (20 × balance) must not exceed leverage (10)",
    ]);
    // The cap may equal the leverage.
    expect(sizingProblems(10, 2, 10)).toStrictEqual([]);

    // Leverage outside [1, 125] has its own problem; no exposure-vs-leverage one.
    expect(sizingProblems(20, 2, 0)).toStrictEqual([]);
    expect(sizingProblems(20, 2, 200)).toStrictEqual([]);
    expect(sizingProblems(20, 2, 3)).toStrictEqual([
      "Max exposure (20 × balance) must not exceed leverage (3)",
    ]);

    // An invalid cap reports only its own problem (no redundant leverage one).
    expect(sizingProblems(125.5, 2, 10)).toStrictEqual([problem("125.5")]);
  });
});

describe("runSizing", () => {
  it("runSizing", () => {
    expect(runSizing({})).toStrictEqual({
      maxPositionPct: 0.5,
      riskPerTrade: 0.02,
      legacy: true,
    });
    expect(runSizing(null)).toStrictEqual({
      maxPositionPct: 0.5,
      riskPerTrade: 0.02,
      legacy: true,
    });
    expect(runSizing(undefined)).toStrictEqual({
      maxPositionPct: 0.5,
      riskPerTrade: 0.02,
      legacy: true,
    });
    // A non-mapping (legacy row whose parameters are a list/string/number).
    expect(runSizing([] as unknown as Record<string, unknown>)).toStrictEqual({
      maxPositionPct: 0.5,
      riskPerTrade: 0.02,
      legacy: true,
    });

    expect(runSizing({ max_position_pct: 1 })).toStrictEqual({
      maxPositionPct: 1,
      riskPerTrade: 0.02,
      legacy: false,
    });
    expect(runSizing({ risk_per_trade: 0.01 })).toStrictEqual({
      maxPositionPct: 0.5,
      riskPerTrade: 0.01,
      legacy: false,
    });

    expect(runSizing({ max_position_pct: -1 })).toStrictEqual({ invalid: INVALID_CAP });
    expect(runSizing({ max_position_pct: 0 })).toStrictEqual({
      invalid: "max_position_pct must be a number in (0, 125], got 0",
    });
    expect(runSizing({ max_position_pct: 125.0001 })).toStrictEqual({
      invalid: "max_position_pct must be a number in (0, 125], got 125.0001",
    });
    expect(runSizing({ max_position_pct: 125 })).toStrictEqual({
      maxPositionPct: 125,
      riskPerTrade: 0.02,
      legacy: false,
    });
    // A string is rejected (rendered with JSON.stringify).
    expect(runSizing({ max_position_pct: "0.5" })).toStrictEqual({
      invalid: 'max_position_pct must be a number in (0, 125], got "0.5"',
    });
    expect(runSizing({ max_position_pct: null })).toStrictEqual({
      invalid: "max_position_pct must be a number in (0, 125], got null",
    });
    expect(runSizing({ max_position_pct: true })).toStrictEqual({
      invalid: "max_position_pct must be a number in (0, 125], got true",
    });

    const badRisk = "risk_per_trade must be more than 0 and at most 0.5 (50% of equity per trade)";
    expect(runSizing({ risk_per_trade: -1 })).toStrictEqual({ invalid: `${badRisk}, got -1` });
    expect(runSizing({ risk_per_trade: null })).toStrictEqual({ invalid: `${badRisk}, got null` });
    expect(runSizing({ risk_per_trade: 0.5000001 })).toStrictEqual({
      invalid: `${badRisk}, got 0.5000001`,
    });
    expect(runSizing({ risk_per_trade: 0.5 })).toStrictEqual({
      maxPositionPct: 0.5,
      riskPerTrade: 0.5,
      legacy: false,
    });

    // The cap is reported first when both are bad.
    expect(runSizing({ max_position_pct: -1, risk_per_trade: -1 })).toStrictEqual({
      invalid: INVALID_CAP,
    });
  });
});

describe("sizingChipLabel", () => {
  it("chip label", () => {
    expect(sizingChipLabel(LEGACY, 1)).toBe("Max exposure 0.5× · Risk 2% (default)");
    expect(sizingChipLabel({ maxPositionPct: 1, riskPerTrade: 0.01, legacy: false }, 1)).toBe(
      "Max exposure 1× · Risk 1%",
    );
    // One non-default key: no "(default)".
    expect(sizingChipLabel({ maxPositionPct: 1, riskPerTrade: 0.02, legacy: false }, 1)).toBe(
      "Max exposure 1× · Risk 2%",
    );
    expect(sizingChipLabel({ invalid: INVALID_CAP }, 1)).toBe(
      "Sizing: invalid (max_position_pct must be a number in (0, 125], got -1)",
    );
  });

  it("chip label spans symbols", () => {
    expect(sizingChipLabel(LEGACY, 3)).toBe(
      "Max exposure 0.5× per symbol, up to 3 × 0.5× total · Risk 2% (default)",
    );
    expect(sizingChipLabel({ maxPositionPct: 1, riskPerTrade: 0.01, legacy: false }, 3)).toBe(
      "Max exposure 1× per symbol, up to 3 × 1× total · Risk 1%",
    );
  });
});

describe("notesSizing", () => {
  it("reads the sizing key of a discovery run's notes", () => {
    expect(notesSizing(JSON.stringify({ sizing: { max_position_pct: 1 } }))).toStrictEqual({
      maxPositionPct: 1,
      riskPerTrade: 0.02,
      legacy: false,
    });
    // A present but empty object is the defaults, not an error.
    expect(notesSizing(JSON.stringify({ sizing: {} }))).toStrictEqual(LEGACY);
  });

  it("absent, unreadable or non-mapping notes read as no sizing", () => {
    expect(notesSizing(JSON.stringify({ seed: 1 }))).toBeNull();
    expect(notesSizing(null)).toBeNull();
    expect(notesSizing(undefined)).toBeNull();
    expect(notesSizing("not json")).toBeNull();
    expect(notesSizing(JSON.stringify([1, 2]))).toBeNull();
    expect(notesSizing(JSON.stringify("str"))).toBeNull();
  });

  it("a present but unreadable sizing is invalid", () => {
    expect(notesSizing(JSON.stringify({ sizing: null }))).toStrictEqual({
      invalid: "notes.sizing must be an object, got null",
    });
    expect(notesSizing(JSON.stringify({ sizing: "x" }))).toStrictEqual({
      invalid: 'notes.sizing must be an object, got "x"',
    });
    expect(notesSizing(JSON.stringify({ sizing: { risk_per_trade: -1 } }))).toStrictEqual({
      invalid:
        "risk_per_trade must be more than 0 and at most 0.5 (50% of equity per trade), got -1",
    });
  });
});

/**
 * ATR sizing (bead vibe-quant-hrhok AC2 + AC3): the frontend mirror of
 * `sizing_settings.py`'s `sizing_mode` / `sizing_atr_period` / `sizing_atr_multiplier`.
 * Risk mode stores nothing new, so every risk-mode body and label stays byte-identical.
 */
describe("ATR sizing constants", () => {
  it("mirrors sizing_settings.py", () => {
    expect(DEFAULT_ATR_PERIOD).toBe(14);
    expect(DEFAULT_ATR_MULTIPLIER).toBe(2);
    expect(MIN_ATR_PERIOD).toBe(2);
    expect(MAX_ATR_PERIOD).toBe(10000);
    expect(MAX_ATR_MULTIPLIER).toBe(20);
  });
});

describe("atrSizingProblems", () => {
  const periodProblem = (entered: string) =>
    `ATR period must be a whole number from 2 to 10000 (you entered ${entered})`;
  const multiplierProblem = (entered: string) =>
    `ATR multiplier must be more than 0 and at most 20 (you entered ${entered})`;

  it("accepts a valid period and multiplier", () => {
    expect(atrSizingProblems(14, 2)).toStrictEqual([]);
  });

  it("accepts the inclusive period bounds and rejects just outside them", () => {
    expect(atrSizingProblems(2, 2)).toStrictEqual([]);
    expect(atrSizingProblems(10000, 2)).toStrictEqual([]);
    expect(atrSizingProblems(1, 2)).toStrictEqual([periodProblem("1")]);
    expect(atrSizingProblems(10001, 2)).toStrictEqual([periodProblem("10001")]);
  });

  it("rejects an empty, zero, negative or fractional period", () => {
    expect(atrSizingProblems("", 2)).toStrictEqual([periodProblem("nothing")]);
    expect(atrSizingProblems(0, 2)).toStrictEqual([periodProblem("0")]);
    expect(atrSizingProblems(-1, 2)).toStrictEqual([periodProblem("-1")]);
    expect(atrSizingProblems(14.5, 2)).toStrictEqual([periodProblem("14.5")]);
  });

  it("accepts the inclusive multiplier edge and rejects just outside it", () => {
    expect(atrSizingProblems(14, 20)).toStrictEqual([]);
    expect(atrSizingProblems(14, 20.0001)).toStrictEqual([multiplierProblem("20.0001")]);
  });

  it("rejects an empty, zero, negative or NaN multiplier", () => {
    expect(atrSizingProblems(14, "")).toStrictEqual([multiplierProblem("nothing")]);
    expect(atrSizingProblems(14, 0)).toStrictEqual([multiplierProblem("0")]);
    expect(atrSizingProblems(14, -1)).toStrictEqual([multiplierProblem("-1")]);
    expect(atrSizingProblems(14, Number.NaN)).toStrictEqual([multiplierProblem("NaN")]);
  });

  it("lists the period problem first when both are bad", () => {
    expect(atrSizingProblems(1.5, 0)).toStrictEqual([periodProblem("1.5"), multiplierProblem("0")]);
  });
});

describe("sizingPayload ATR mode", () => {
  it("keeps a risk-mode payload byte-identical (no third argument)", () => {
    expect(sizingPayload(0.5, 0.02, null)).toStrictEqual({});
    expect(sizingPayload(0.5, 0.02, undefined)).toStrictEqual({});
    expect(sizingPayload(1, 0.01)).toStrictEqual({ max_position_pct: 1, risk_per_trade: 0.01 });
  });

  it("stores all three ATR keys, defaults included, and no untouched cap/risk", () => {
    expect(sizingPayload(0.5, 0.02, { period: 14, multiplier: 2 })).toStrictEqual({
      sizing_mode: "atr",
      sizing_atr_period: 14,
      sizing_atr_multiplier: 2,
    });
  });

  it("stores the moved cap and risk next to the ATR keys", () => {
    expect(sizingPayload(1, 0.01, { period: 20, multiplier: 1.5 })).toStrictEqual({
      max_position_pct: 1,
      risk_per_trade: 0.01,
      sizing_mode: "atr",
      sizing_atr_period: 20,
      sizing_atr_multiplier: 1.5,
    });
  });
});

describe("runSizing ATR mode", () => {
  it("reads a missing sizing_mode as risk with no ATR fields", () => {
    expect(runSizing({ max_position_pct: 1 })).toStrictEqual({
      maxPositionPct: 1,
      riskPerTrade: 0.02,
      legacy: false,
    });
  });

  it("reads an explicit risk mode as non-legacy with no ATR fields", () => {
    expect(runSizing({ sizing_mode: "risk" })).toStrictEqual({
      maxPositionPct: 0.5,
      riskPerTrade: 0.02,
      legacy: false,
    });
  });

  it("reads the ATR defaults and the stored period/multiplier", () => {
    expect(runSizing({ sizing_mode: "atr" })).toStrictEqual({
      maxPositionPct: 0.5,
      riskPerTrade: 0.02,
      legacy: false,
      atr: { period: 14, multiplier: 2 },
    });
    expect(
      runSizing({ sizing_mode: "atr", sizing_atr_period: 20, sizing_atr_multiplier: 1.5 }),
    ).toStrictEqual({
      maxPositionPct: 0.5,
      riskPerTrade: 0.02,
      legacy: false,
      atr: { period: 20, multiplier: 1.5 },
    });
  });

  it("keeps the margin limit next to the ATR fields", () => {
    expect(
      runSizing({
        sizing_mode: "atr",
        sizing_atr_period: 14,
        sizing_atr_multiplier: 2,
        max_margin_pct: 0.4,
      }),
    ).toStrictEqual({
      maxPositionPct: 0.5,
      riskPerTrade: 0.02,
      legacy: false,
      atr: { period: 14, multiplier: 2 },
      maxMarginPct: 0.4,
    });
  });

  it("accepts the inclusive ATR bounds", () => {
    expect(runSizing({ sizing_mode: "atr", sizing_atr_period: 2 })).toMatchObject({
      atr: { period: 2, multiplier: 2 },
    });
    expect(runSizing({ sizing_mode: "atr", sizing_atr_period: 10000 })).toMatchObject({
      atr: { period: 10000, multiplier: 2 },
    });
    expect(runSizing({ sizing_mode: "atr", sizing_atr_multiplier: 20 })).toMatchObject({
      atr: { period: 14, multiplier: 20 },
    });
  });

  it("refuses an unknown mode", () => {
    expect(runSizing({ sizing_mode: "scalp" })).toStrictEqual({
      invalid: "sizing_mode must be 'risk' or 'atr', got \"scalp\"",
    });
  });

  it("refuses ATR fields in risk mode", () => {
    expect(runSizing({ sizing_mode: "risk", sizing_atr_period: 14 })).toStrictEqual({
      invalid:
        "sizing_atr_period is only valid with sizing_mode 'atr', got sizing_atr_period=14 with sizing_mode 'risk'",
    });
    expect(runSizing({ sizing_mode: "risk", sizing_atr_multiplier: 2 })).toStrictEqual({
      invalid:
        "sizing_atr_multiplier is only valid with sizing_mode 'atr', got sizing_atr_multiplier=2 with sizing_mode 'risk'",
    });
  });

  it("refuses an out-of-range or non-whole stored period", () => {
    const problem = (value: string) =>
      `sizing_atr_period must be a whole number in [2, 10000], got ${value}`;
    expect(runSizing({ sizing_mode: "atr", sizing_atr_period: 1 })).toStrictEqual({
      invalid: problem("1"),
    });
    expect(runSizing({ sizing_mode: "atr", sizing_atr_period: 10001 })).toStrictEqual({
      invalid: problem("10001"),
    });
    expect(runSizing({ sizing_mode: "atr", sizing_atr_period: 14.5 })).toStrictEqual({
      invalid: problem("14.5"),
    });
    expect(runSizing({ sizing_mode: "atr", sizing_atr_period: null })).toStrictEqual({
      invalid: problem("null"),
    });
    expect(runSizing({ sizing_mode: "atr", sizing_atr_period: "14" })).toStrictEqual({
      invalid: problem('"14"'),
    });
  });

  it("refuses an out-of-range or non-numeric stored multiplier", () => {
    const problem = (value: string) =>
      `sizing_atr_multiplier must be a number in (0, 20], got ${value}`;
    expect(runSizing({ sizing_mode: "atr", sizing_atr_multiplier: 0 })).toStrictEqual({
      invalid: problem("0"),
    });
    expect(runSizing({ sizing_mode: "atr", sizing_atr_multiplier: -1 })).toStrictEqual({
      invalid: problem("-1"),
    });
    expect(runSizing({ sizing_mode: "atr", sizing_atr_multiplier: 20.0001 })).toStrictEqual({
      invalid: problem("20.0001"),
    });
    expect(runSizing({ sizing_mode: "atr", sizing_atr_multiplier: null })).toStrictEqual({
      invalid: problem("null"),
    });
  });

  it("reports the cap error first when it is also bad", () => {
    expect(runSizing({ max_position_pct: -1, sizing_mode: "scalp" })).toStrictEqual({
      invalid: INVALID_CAP,
    });
  });
});

describe("sizingChipLabel ATR mode", () => {
  it("prefixes the mode to today's one-symbol label", () => {
    expect(
      sizingChipLabel(
        {
          maxPositionPct: 0.5,
          riskPerTrade: 0.005,
          legacy: false,
          atr: { period: 14, multiplier: 2 },
        },
        1,
      ),
    ).toBe("ATR(14)x2 · Max exposure 0.5× · Risk 0.5%");
  });

  it("prints a fractional multiplier and a moved cap", () => {
    expect(
      sizingChipLabel(
        {
          maxPositionPct: 1,
          riskPerTrade: 0.02,
          legacy: false,
          atr: { period: 20, multiplier: 1.5 },
        },
        1,
      ),
    ).toBe("ATR(20)x1.5 · Max exposure 1× · Risk 2%");
  });

  it("keeps the multi-symbol total text", () => {
    expect(
      sizingChipLabel(
        {
          maxPositionPct: 1,
          riskPerTrade: 0.02,
          legacy: false,
          atr: { period: 14, multiplier: 2 },
        },
        2,
      ),
    ).toBe("ATR(14)x2 · Max exposure 1× per symbol, up to 2 × 1× total · Risk 2%");
  });

  it("keeps the margin part last", () => {
    expect(
      sizingChipLabel(
        {
          maxPositionPct: 0.5,
          riskPerTrade: 0.005,
          legacy: false,
          atr: { period: 14, multiplier: 2 },
          maxMarginPct: 0.5,
        },
        1,
      ),
    ).toBe("ATR(14)x2 · Max exposure 0.5× · Risk 0.5% · Max margin 50%");
  });

  it("leaves a risk-mode label byte-identical", () => {
    expect(sizingChipLabel({ maxPositionPct: 0.5, riskPerTrade: 0.02, legacy: true }, 1)).toBe(
      "Max exposure 0.5× · Risk 2% (default)",
    );
  });
});

describe("notesSizing ATR mode", () => {
  it("reads the ATR fields of a CLI discovery run's notes.sizing", () => {
    expect(
      notesSizing(
        JSON.stringify({
          sizing: { sizing_mode: "atr", sizing_atr_period: 20, sizing_atr_multiplier: 1.5 },
        }),
      ),
    ).toStrictEqual({
      maxPositionPct: 0.5,
      riskPerTrade: 0.02,
      legacy: false,
      atr: { period: 20, multiplier: 1.5 },
    });
  });
});
