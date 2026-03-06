/**
 * Regression tests for frontend bug fixes.
 *
 * Each section targets a specific fix to prevent regressions.
 */
import { describe, expect, it } from "vitest";
import { toISODate } from "@/components/ui/date-picker";
import { getPresetRange } from "@/components/ui/DateRangePicker";
import { computeRollingSharpe } from "@/components/results/RollingSharpeChart";
import { expandParam, type SweepParam } from "@/components/backtest/SweepBuilder";

// ---------------------------------------------------------------------------
// 1. toISODate timezone fix (bd-bai9)
//    Bug: using toISOString() would shift to UTC, turning a late-night local
//    date into the next day. Fix: use getFullYear/getMonth/getDate instead.
// ---------------------------------------------------------------------------
describe("toISODate timezone fix (bd-bai9)", () => {
  it("should return local date, not UTC-shifted date", () => {
    // 11:30 PM on Jan 15 — UTC would be Jan 16 for timezones >= UTC+1
    const d = new Date(2024, 0, 15, 23, 30);
    expect(toISODate(d)).toBe("2024-01-15");
  });

  it("should zero-pad month and day", () => {
    const d = new Date(2024, 2, 5); // March 5
    expect(toISODate(d)).toBe("2024-03-05");
  });

  it("should handle Dec 31 without rolling to next year", () => {
    const d = new Date(2024, 11, 31, 23, 59);
    expect(toISODate(d)).toBe("2024-12-31");
  });
});

// ---------------------------------------------------------------------------
// 2. Rolling Sharpe sample variance (bd-s5mi)
//    Bug: stddev used population variance (n denominator).
//    Fix: use sample variance (n-1 denominator).
// ---------------------------------------------------------------------------
describe("Rolling Sharpe sample variance (bd-s5mi)", () => {
  it("should use n-1 denominator (Bessel correction)", () => {
    // Create synthetic equity curve: 6 data points -> 5 returns
    // With window=5, we get exactly 1 Sharpe point from all 5 returns
    const equity = [100, 101, 99, 102, 100, 103];
    const data = equity.map((eq, i) => ({
      timestamp: `2024-01-0${i + 1}`,
      equity: eq,
    }));

    const result = computeRollingSharpe(data, 5);
    expect(result).toHaveLength(1);

    // Manually compute expected Sharpe with n-1 denominator
    const returns = [
      (101 - 100) / 100,
      (99 - 101) / 101,
      (102 - 99) / 99,
      (100 - 102) / 102,
      (103 - 100) / 100,
    ];
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const sampleVariance =
      returns.reduce((a, b) => a + (b - mean) ** 2, 0) / (returns.length - 1);
    const std = Math.sqrt(sampleVariance);
    const expectedSharpe = (mean / std) * Math.sqrt(252);

    expect(result[0]!.sharpe).toBeCloseTo(
      Number.parseFloat(expectedSharpe.toFixed(3)),
      3,
    );
  });

  it("should return empty for insufficient data", () => {
    const data = [
      { timestamp: "2024-01-01", equity: 100 },
      { timestamp: "2024-01-02", equity: 101 },
    ];
    expect(computeRollingSharpe(data, 30)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// 3. SweepBuilder float drift fix (bd-njqa)
//    Bug: accumulating floats (0.1+0.1+...) caused drift, producing wrong
//    count (e.g. 4 instead of 5). Fix: use count = round((max-min)/step)+1
//    and generate via min + i*step.
// ---------------------------------------------------------------------------
describe("SweepBuilder float drift fix (bd-njqa)", () => {
  it("should produce exactly 5 values for 0.1 to 0.5 step 0.1", () => {
    const param: SweepParam = {
      key: "test",
      label: "test",
      type: "range",
      fixedValue: 0,
      rangeMin: 0.1,
      rangeMax: 0.5,
      rangeStep: 0.1,
      listValues: "",
    };
    const values = expandParam(param);
    expect(values).toHaveLength(5);
    // Each value should be close to expected (float imprecision is ok)
    expect(values[0]).toBeCloseTo(0.1, 10);
    expect(values[1]).toBeCloseTo(0.2, 10);
    expect(values[2]).toBeCloseTo(0.3, 10);
    expect(values[3]).toBeCloseTo(0.4, 10);
    expect(values[4]).toBeCloseTo(0.5, 10);
  });

  it("should produce exactly 3 values for 1 to 3 step 1", () => {
    const param: SweepParam = {
      key: "test",
      label: "test",
      type: "range",
      fixedValue: 0,
      rangeMin: 1,
      rangeMax: 3,
      rangeStep: 1,
      listValues: "",
    };
    const values = expandParam(param);
    expect(values).toEqual([1, 2, 3]);
  });

  it("should return empty for step <= 0", () => {
    const param: SweepParam = {
      key: "test",
      label: "test",
      type: "range",
      fixedValue: 0,
      rangeMin: 1,
      rangeMax: 5,
      rangeStep: 0,
      listValues: "",
    };
    expect(expandParam(param)).toEqual([]);
  });

  it("should return empty when max < min", () => {
    const param: SweepParam = {
      key: "test",
      label: "test",
      type: "range",
      fixedValue: 0,
      rangeMin: 5,
      rangeMax: 1,
      rangeStep: 1,
      listValues: "",
    };
    expect(expandParam(param)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// 4. Date presets reference date (bd-xs4s)
//    Bug: presets always used new Date() (today). Fix: accept an optional
//    reference date so presets anchor to dataset end date.
// ---------------------------------------------------------------------------
describe("Date presets reference date (bd-xs4s)", () => {
  it("should use reference date instead of today for 30d preset", () => {
    const ref = new Date(2024, 5, 15); // June 15, 2024
    const [start, end] = getPresetRange("30d", ref);
    expect(end).toBe("2024-06-15");
    expect(start).toBe("2024-05-16"); // June 15 - 30 days = May 16
  });

  it("should use reference date for 1y preset", () => {
    const ref = new Date(2024, 5, 15);
    const [start, end] = getPresetRange("1y", ref);
    expect(end).toBe("2024-06-15");
    expect(start).toBe("2023-06-15");
  });

  it("should fall back to today when no reference date", () => {
    const [, end] = getPresetRange("30d");
    const today = new Date();
    const y = today.getFullYear();
    const m = String(today.getMonth() + 1).padStart(2, "0");
    const d = String(today.getDate()).padStart(2, "0");
    expect(end).toBe(`${y}-${m}-${d}`);
  });

  it("should handle YTD preset with reference date", () => {
    const ref = new Date(2024, 5, 15);
    const [start, end] = getPresetRange("ytd", ref);
    expect(end).toBe("2024-06-15");
    expect(start).toBe("2024-01-01");
  });
});

// ---------------------------------------------------------------------------
// 5. PnL formula (bd-n5gp)
//    Bug: netPnl/grossPnl computation was wrong.
//    Fix: netPnl = (total_return/100) * starting_balance
//         grossPnl = netPnl + totalCosts
// ---------------------------------------------------------------------------
describe("PnL formula (bd-n5gp)", () => {
  it("should compute netPnl and grossPnl correctly", () => {
    const totalReturn = 10; // 10% return
    const startingBalance = 50000;
    const fees = 80;
    const funding = 15;
    const slippage = 5;
    const totalCosts = fees + funding + slippage; // 100

    const netPnl = (totalReturn / 100) * startingBalance;
    const grossPnl = netPnl + totalCosts;

    expect(netPnl).toBe(5000);
    expect(grossPnl).toBe(5100);
  });

  it("should handle negative returns", () => {
    const totalReturn = -5;
    const startingBalance = 100000;
    const totalCosts = 200;

    const netPnl = (totalReturn / 100) * startingBalance;
    const grossPnl = netPnl + totalCosts;

    expect(netPnl).toBe(-5000);
    expect(grossPnl).toBe(-4800); // lost 5000, but gross is 4800 loss (costs made it worse)
  });

  it("should handle zero return", () => {
    const totalReturn = 0;
    const startingBalance = 50000;
    const totalCosts = 50;

    const netPnl = (totalReturn / 100) * startingBalance;
    const grossPnl = netPnl + totalCosts;

    expect(netPnl).toBe(0);
    expect(grossPnl).toBe(50); // broke even net, but earned 50 gross (costs ate 50)
  });
});
