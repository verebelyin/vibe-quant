/**
 * Tests for daily equity resampling (vibe-quant-pazbr).
 *
 * The /equity-curve endpoint emits one point per TRADE close. "Daily
 * returns" and "30-day rolling Sharpe" charts must resample to calendar
 * days first — otherwise a 30-trade window masquerades as 30 days and
 * Sharpe is annualized on the wrong frequency.
 */
import { describe, expect, it } from "vitest";
import { computeDailyReturns, computeRollingSharpe, resampleDailyEquity } from "@/lib/equity";

describe("resampleDailyEquity", () => {
  it("takes the last equity of each day", () => {
    const data = [
      { timestamp: "2024-01-01T03:00:00Z", equity: 100 },
      { timestamp: "2024-01-01T15:00:00Z", equity: 105 },
      { timestamp: "2024-01-01T22:00:00Z", equity: 103 },
      { timestamp: "2024-01-02T10:00:00Z", equity: 110 },
    ];
    const daily = resampleDailyEquity(data);
    expect(daily).toEqual([
      { timestamp: "2024-01-01", equity: 103 },
      { timestamp: "2024-01-02", equity: 110 },
    ]);
  });

  it("forward-fills days without trades", () => {
    const data = [
      { timestamp: "2024-01-01T12:00:00Z", equity: 100 },
      { timestamp: "2024-01-04T12:00:00Z", equity: 120 },
    ];
    const daily = resampleDailyEquity(data);
    expect(daily.map((p) => p.timestamp)).toEqual([
      "2024-01-01",
      "2024-01-02",
      "2024-01-03",
      "2024-01-04",
    ]);
    expect(daily.map((p) => p.equity)).toEqual([100, 100, 100, 120]);
  });

  it("sorts unordered input", () => {
    const data = [
      { timestamp: "2024-01-02T12:00:00Z", equity: 110 },
      { timestamp: "2024-01-01T12:00:00Z", equity: 100 },
    ];
    const daily = resampleDailyEquity(data);
    expect(daily.map((p) => p.equity)).toEqual([100, 110]);
  });

  it("returns empty for empty input", () => {
    expect(resampleDailyEquity([])).toEqual([]);
  });
});

describe("computeDailyReturns", () => {
  it("computes returns between calendar days, not trades", () => {
    // 3 trades on day 1, 1 trade on day 2 -> exactly ONE daily return
    const data = [
      { timestamp: "2024-01-01T03:00:00Z", equity: 100 },
      { timestamp: "2024-01-01T15:00:00Z", equity: 90 },
      { timestamp: "2024-01-01T22:00:00Z", equity: 105 },
      { timestamp: "2024-01-02T10:00:00Z", equity: 110.25 },
    ];
    const returns = computeDailyReturns(data);
    expect(returns).toHaveLength(1);
    expect(returns[0]!.returnPct).toBeCloseTo(5.0, 4); // 105 -> 110.25
  });

  it("gap days produce zero returns (forward-filled)", () => {
    const data = [
      { timestamp: "2024-01-01T12:00:00Z", equity: 100 },
      { timestamp: "2024-01-03T12:00:00Z", equity: 110 },
    ];
    const returns = computeDailyReturns(data);
    expect(returns.map((r) => r.returnPct)).toEqual([0, 10]);
  });
});

describe("computeRollingSharpe daily semantics", () => {
  it("windows over days: many trades on one day collapse to one observation", () => {
    // 20 trades all on 2 days -> only 1 daily return -> not enough for a
    // 5-day window. The old per-trade version would happily emit points.
    const data = Array.from({ length: 20 }, (_, i) => ({
      timestamp: `2024-01-0${(i % 2) + 1}T${String(i % 24).padStart(2, "0")}:00:00Z`,
      equity: 100 + i,
    }));
    expect(computeRollingSharpe(data, 5)).toEqual([]);
  });

  it("emits one point per day once the window fills", () => {
    const data = Array.from({ length: 10 }, (_, i) => ({
      timestamp: `2024-01-${String(i + 1).padStart(2, "0")}T12:00:00Z`,
      equity: 100 * 1.01 ** i + (i % 2), // drift + wiggle so std > 0
    }));
    const result = computeRollingSharpe(data, 5);
    // 10 days -> 9 returns -> 5 rolling points
    expect(result).toHaveLength(5);
  });
});
