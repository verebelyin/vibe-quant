import { describe, expect, it } from "vitest";
import type { ShadowStrategy } from "@/api/generated/models/shadowStrategy";
import {
  buildEnrolRequest,
  defaultEnrolForm,
  type EnrolForm,
  preRegistrationShortBy,
  shadowEligible,
  trackedRowForRun,
} from "./enrolRequest";

const PREREG = "Stop if the forward Sharpe is below 0.5 after 120 days of trading.";

const form: EnrolForm = {
  name: "ema_trend (run 41)",
  preRegistration: PREREG,
  maxDdPercent: "20",
  warmupDays: "180",
};

describe("shadowEligible", () => {
  it("only a completed validation run", () => {
    expect(shadowEligible({ run_mode: "validation", status: "completed" })).toBe(true);
    expect(shadowEligible({ run_mode: "screening", status: "completed" })).toBe(false);
    expect(shadowEligible({ run_mode: "discovery", status: "completed" })).toBe(false);
    expect(shadowEligible({ run_mode: "validation", status: "running" })).toBe(false);
    expect(shadowEligible({ run_mode: "validation", status: "failed" })).toBe(false);
  });
});

describe("defaultEnrolForm", () => {
  it("names the row after the strategy and run, 20 % drawdown, 180 warm-up days", () => {
    expect(defaultEnrolForm("rsi_meanrev", 1149)).toEqual({
      name: "rsi_meanrev (run 1149)",
      preRegistration: "",
      maxDdPercent: "20",
      warmupDays: "180",
    });
  });
});

describe("preRegistrationShortBy", () => {
  it("39 characters are one short, 40 are enough, outer spaces do not count", () => {
    expect(preRegistrationShortBy("x".repeat(39))).toBe(1);
    expect(preRegistrationShortBy("x".repeat(40))).toBe(0);
    expect(preRegistrationShortBy(`   ${"x".repeat(39)}   `)).toBe(1);
    expect(preRegistrationShortBy("")).toBe(40);
  });
});

describe("buildEnrolRequest", () => {
  it("single-symbol run: the run's DSL, symbol and parameters, 20 % -> 0.2", () => {
    const body = buildEnrolRequest(
      {
        id: 41,
        symbols: ["BTCUSDT"],
        parameters: { leverage: 1, risk_per_trade: 0.5, liquidation_mode: "isolated" },
      },
      { strategy_id: 7, dsl_json: '{"name":"ema_trend","timeframe":"4h"}' },
      form,
    );
    expect(body).toEqual({
      name: "ema_trend (run 41)",
      source_kind: "strategy",
      dsl_json: '{"name":"ema_trend","timeframe":"4h"}',
      symbols: ["BTCUSDT"],
      parameters: { leverage: 1, risk_per_trade: 0.5, liquidation_mode: "isolated" },
      pre_registration: PREREG,
      max_dd_limit: 0.2,
      warmup_days: 180,
      source_ref: { run_id: 41 },
      strategy_id: 7,
    });
  });

  it("two-symbol run without parameters, 12.5 % and 365 days", () => {
    const body = buildEnrolRequest(
      { id: 1149, symbols: ["ETHUSDT", "SOLUSDT"], parameters: null },
      { strategy_id: 312, dsl_json: '{"name":"rsi_meanrev"}' },
      {
        name: "rsi two coins",
        preRegistration: "x".repeat(39),
        maxDdPercent: "12.5",
        warmupDays: "365",
      },
    );
    expect(body).toEqual({
      name: "rsi two coins",
      source_kind: "strategy",
      dsl_json: '{"name":"rsi_meanrev"}',
      symbols: ["ETHUSDT", "SOLUSDT"],
      parameters: {},
      pre_registration: "x".repeat(39), // sent as typed: the API's 422 is the refusal
      max_dd_limit: 0.125,
      warmup_days: 365,
      source_ref: { run_id: 1149 },
      strategy_id: 312,
    });
  });
});

describe("trackedRowForRun", () => {
  const row = (
    id: number,
    runId: unknown,
    state: ShadowStrategy["state"] = "tracking",
    isTest = false,
  ) => ({
    id,
    name: `row ${id}`,
    state,
    is_test: isTest,
    source_ref: runId === undefined ? {} : { run_id: runId },
  });

  it("finds the live real row enrolled from this run", () => {
    const rows = [row(1, 40), row(2, undefined), row(3, 41, "futile")];
    expect(trackedRowForRun(rows, 41)?.id).toBe(3);
    expect(trackedRowForRun(rows, 42)).toBeUndefined();
  });

  it("a retired or test row does not count (the API frees retired inputs)", () => {
    expect(trackedRowForRun([row(1, 41, "retired"), row(2, 41, "tracking", true)], 41)).toBe(
      undefined,
    );
  });
});
