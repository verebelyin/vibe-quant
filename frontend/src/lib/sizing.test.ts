/**
 * Tests for the frontend sizing rule (bead vibe-quant-2gyk1.2, S6): the mirror
 * of `vibe_quant/sizing_settings.py` (contract U1-U7).
 *
 * The cap is entered in multiples of equity and the risk in PERCENT; the
 * conversion rounds to 12 significant digits so a default launch (50% cap, 2%
 * risk) stores no keys and the chip shows 7 for a stored 0.07. A recorded value
 * that is not a finite number in range is invalid, with the Python text.
 */
import { describe, expect, it } from "vitest";
import {
  DEFAULT_MAX_POSITION_PCT,
  DEFAULT_RISK_PCT,
  DEFAULT_RISK_PER_TRADE,
  MAX_MAX_POSITION_PCT,
  MAX_RISK_PCT,
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
      `Max exposure must be more than 0 and at most 125 × equity (you entered ${entered})`;

    expect(sizingProblems(0, 2, 10)).toStrictEqual([problem("0")]);
    expect(sizingProblems(125.5, 2, 10)).toStrictEqual([problem("125.5")]);
    expect(sizingProblems("", 2, 10)).toStrictEqual([problem("nothing")]);
    // 125 is allowed (the upper edge); 125.0001 is not.
    expect(sizingProblems(125, 2, 125)).toStrictEqual([]);
    expect(sizingProblems(125.0001, 2, 125)).toStrictEqual([problem("125.0001")]);

    expect(sizingProblems(20, 2, 10)).toStrictEqual([
      "Max exposure (20 × equity) must not exceed leverage (10)",
    ]);
    // The cap may equal the leverage.
    expect(sizingProblems(10, 2, 10)).toStrictEqual([]);

    // Leverage outside [1, 125] has its own problem; no exposure-vs-leverage one.
    expect(sizingProblems(20, 2, 0)).toStrictEqual([]);
    expect(sizingProblems(20, 2, 200)).toStrictEqual([]);
    expect(sizingProblems(20, 2, 3)).toStrictEqual([
      "Max exposure (20 × equity) must not exceed leverage (3)",
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
