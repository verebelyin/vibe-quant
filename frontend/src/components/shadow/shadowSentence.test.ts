import { describe, expect, it } from "vitest";
import { type ShadowSentenceRow, shadowSentence } from "./shadowSentence";

const base: ShadowSentenceRow = {
  state: "tracking",
  state_reason: null,
  state_changed_at: null,
  verdict_day: null,
  first_day: "2026-11-24",
  slot: 2,
  is_test: false,
  max_dd_limit: 0.2,
  proposal_stale: false,
  engine_epoch_last: 1,
  stats: {
    days: 143,
    total_return: 0.042,
    sharpe_ann: 0.9,
    se_ann: 1.4,
    ub95: 3.2,
    max_dd: 0.061,
    blocks: 20,
    e_value: 1.8,
  },
};

describe("shadowSentence", () => {
  it("names the epoch the evidence was measured on for a stale pass", () => {
    const row: ShadowSentenceRow = { ...base, proposal_stale: true, engine_epoch_last: 2 };
    expect(shadowSentence(row)).toContain(
      "Evidence measured on engine epoch 2; a paper decision must re-check it under the current engine.",
    );
  });

  it("never rounds evidence up to the bar or prints a negative zero return", () => {
    const row: ShadowSentenceRow = {
      ...base,
      stats: { ...base.stats!, e_value: 99.96, total_return: -0.0004 },
    };
    const text = shadowSentence(row);
    expect(text).toContain("Forward return 0.0 %,");
    expect(text).toContain("Evidence 99.9 of 100 needed");
  });

  it("quotes the design example (143 days, slot 2 of 5)", () => {
    expect(shadowSentence(base)).toBe(
      "Tracking 143 days (20 full weeks). Forward return +4.2 %, Sharpe 0.9 (365-day basis), worst drawdown 6.1 % on daily closes (limit 20 %). Evidence 1.8 of 100 needed (slot 2 of 5).",
    );
  });

  it("says no full week yet below 7 days", () => {
    const row: ShadowSentenceRow = {
      ...base,
      slot: 1,
      max_dd_limit: 0.125,
      stats: {
        days: 5,
        total_return: -0.0131,
        sharpe_ann: -2.37,
        se_ann: 9.1,
        ub95: 12.6,
        max_dd: 0.0204,
        blocks: 0,
        e_value: 1,
      },
    };
    expect(shadowSentence(row)).toBe(
      "Tracking 5 days (no full week yet). Forward return -1.3 %, Sharpe -2.4 (365-day basis), worst drawdown 2.0 % on daily closes (limit 12.5 %). Evidence 1.0 of 100 needed (slot 1 of 5).",
    );
  });

  it("says no forward day yet before the first scored day", () => {
    expect(shadowSentence({ ...base, stats: null })).toBe(
      "Tracking starts on 2026-11-24; no forward day scored yet.",
    );
  });

  it("shows no Sharpe when every return is identical", () => {
    const row: ShadowSentenceRow = {
      ...base,
      slot: null,
      stats: {
        days: 200,
        total_return: 0,
        sharpe_ann: null,
        se_ann: null,
        ub95: null,
        max_dd: 0,
        blocks: 28,
        e_value: 1,
      },
    };
    expect(shadowSentence(row)).toBe(
      "Tracking 200 days (28 full weeks). Forward return +0.0 %, Sharpe n/a (no variation), worst drawdown 0.0 % on daily closes (limit 20 %). Evidence 1.0 of 100 needed (no evidence slot left; a new family needs your OK).",
    );
  });

  it("names the futility rule and its reason when dropped on the Sharpe bound", () => {
    const row: ShadowSentenceRow = {
      ...base,
      state: "futile",
      state_reason: "forward Sharpe -0.41, 95% upper bound 0.4870 < 0.5 after 120 days",
      verdict_day: "2027-03-23",
      stats: {
        days: 120,
        total_return: -0.0712,
        sharpe_ann: -0.41,
        se_ann: 0.54,
        ub95: 0.487,
        max_dd: 0.118,
        blocks: 17,
        e_value: 0.42,
      },
    };
    expect(shadowSentence(row)).toBe(
      "Dropped on 2027-03-23 by the futility rule: forward Sharpe -0.41, 95% upper bound 0.4870 < 0.5 after 120 days. Tracked 120 days (17 full weeks). Forward return -7.1 %, Sharpe -0.4 (365-day basis), worst drawdown 11.8 % on daily closes (limit 20 %). Evidence 0.4 of 100 needed (slot 2 of 5).",
    );
  });

  it("names the drawdown limit when dropped on drawdown", () => {
    const row: ShadowSentenceRow = {
      ...base,
      state: "futile",
      state_reason: "max drawdown 15.30% > limit 15.00%",
      verdict_day: "2027-01-09",
      max_dd_limit: 0.15,
      stats: {
        days: 46,
        total_return: -0.1402,
        sharpe_ann: -3.05,
        se_ann: 2.9,
        ub95: 1.72,
        max_dd: 0.153,
        blocks: 6,
        e_value: 1,
      },
    };
    expect(shadowSentence(row)).toBe(
      "Dropped on 2027-01-09 by the futility rule: max drawdown 15.30% > limit 15.00%. Tracked 46 days (6 full weeks). Forward return -14.0 %, Sharpe -3.0 (365-day basis), worst drawdown 15.3 % on daily closes (limit 15 %). Evidence 1.0 of 100 needed (slot 2 of 5).",
    );
  });

  it("never reads a pass as a recommendation while proposals are off", () => {
    const row: ShadowSentenceRow = {
      ...base,
      state: "passed",
      state_reason: "evidence 131.4 >= 100",
      verdict_day: "2028-06-04",
      is_test: true,
      slot: null,
      stats: {
        days: 560,
        total_return: 0.6123,
        sharpe_ann: 2.81,
        se_ann: 0.8,
        ub95: 4.1,
        max_dd: 0.0941,
        blocks: 80,
        e_value: 131.4,
      },
    };
    expect(shadowSentence(row)).toBe(
      "[TEST] Passed on 2028-06-04: evidence bar met — not proposed (proposals off until the pre-registered result + your OK). Tracked 560 days (80 full weeks). Forward return +61.2 %, Sharpe 2.8 (365-day basis), worst drawdown 9.4 % on daily closes (limit 20 %). Evidence 131.4 of 100 needed (test row, no slot).",
    );
  });
});
