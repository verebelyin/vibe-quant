/**
 * Audit e70tl.22 follow-ups: other results components with unit/sign mistakes.
 */
import { describe, expect, it } from "vitest";
import type { BacktestResultResponse } from "@/api/generated/models";
import type { TradeResponse } from "@/api/generated/models/tradeResponse";
import { buildHistogram, histogramRange } from "@/components/charts/TradeDistributionChart";
import { getBestWorst } from "@/components/results/ComparisonView";
import { computeDirectionStats } from "@/components/results/LongShortSplit";
import { overallVerdict, overfittingChecks } from "@/components/results/OverfittingBadges";
import { computeYearlyReturns } from "@/components/results/YearlyReturnsChart";

describe("ComparisonView best/worst for lower-is-better metrics", () => {
  it("smallest max drawdown is best", () => {
    const runs = [
      { run_id: 1, max_drawdown: 0.2 },
      { run_id: 2, max_drawdown: 0.05 },
      { run_id: 3, max_drawdown: 0.1 },
    ] as unknown as BacktestResultResponse[];
    expect(getBestWorst(runs, "max_drawdown", false)).toEqual({ bestId: 2, worstId: 1 });
  });
});

describe("LongShortSplit counts backend 'LONG'/'SHORT' trades", () => {
  it("matches case-insensitively", () => {
    const trades = [
      { direction: "LONG", net_pnl: 5 },
      { direction: "LONG", net_pnl: -1 },
      { direction: "SHORT", net_pnl: 2 },
    ] as unknown as TradeResponse[];
    expect(computeDirectionStats(trades, "long")).toEqual({
      count: 2,
      winRate: 50,
      totalPnl: 4,
      avgPnl: 2,
    });
    expect(computeDirectionStats(trades, "short").count).toBe(1);
  });
});

describe("YearlyReturnsChart chains years on the previous close", () => {
  it("keeps the first trade of each year", () => {
    const curve = [
      { timestamp: "2024-03-01T00:00:00Z", equity: 1000 },
      { timestamp: "2024-12-30T00:00:00Z", equity: 1100 },
      // first point of 2025 is AFTER its first closed trade (+50)
      { timestamp: "2025-01-05T00:00:00Z", equity: 1150 },
      { timestamp: "2025-06-01T00:00:00Z", equity: 1210 },
    ];
    const years = computeYearlyReturns(curve);
    expect(years.map((y) => [y.year, y.returnPct])).toEqual([
      ["2024", 10],
      ["2025", 10],
    ]);
  });
});

describe("TradeDistributionChart bins follow the data", () => {
  it("spreads small ROI% values over the bins", () => {
    const rois = [-0.8, -0.2, 0.1, 0.3, 0.9];
    const { min, max, step } = histogramRange(rois);
    expect([min, max]).toEqual([-0.8, 0.9]);
    const bins = buildHistogram(
      rois.map((roi_percent) => ({ roi_percent })),
      min,
      max,
      step,
    );
    expect(bins).toHaveLength(20);
    expect(bins.reduce((n, b) => n + b.count, 0)).toBe(5);
    expect(bins.filter((b) => b.count > 0).length).toBe(5);
  });
});

describe("OverfittingBadges thresholds match the backend", () => {
  const base = {} as BacktestResultResponse;
  const status = (data: Partial<BacktestResultResponse>, tf?: string) =>
    Object.fromEntries(
      overfittingChecks({ ...base, ...data } as BacktestResultResponse, tf).map((c) => [
        c.label,
        c.status,
      ]),
    );

  it("no tests ran → Not Run (win rate / calmar are not overfitting tests)", () => {
    const checks = overfittingChecks(
      { win_rate: 0.6, calmar_ratio: 2 } as unknown as BacktestResultResponse,
      "4h",
    );
    expect(overallVerdict(checks)).toBe("not_run");
  });

  it("DSR is a z-score: 1.0 fails, 1.7 passes", () => {
    expect(status({ deflated_sharpe: 1.0 })["Deflated Sharpe Ratio"]).toBe("fail");
    expect(status({ deflated_sharpe: 1.7 })["Deflated Sharpe Ratio"]).toBe("pass");
  });

  it("purged k-fold needs mean OOS sharpe > 0.5", () => {
    expect(status({ purged_kfold_mean_sharpe: 0.3 })["Purged K-Fold"]).toBe("fail");
    expect(status({ purged_kfold_mean_sharpe: 0.6 })["Purged K-Fold"]).toBe("pass");
  });

  it("bootstrap floor: 4h/1d 0.0, 1m 0.5, run override wins", () => {
    expect(status({ bootstrap_sharpe_lower: 0.2 }, "4h")["Bootstrap CI (Sharpe lower)"]).toBe(
      "pass",
    );
    expect(status({ bootstrap_sharpe_lower: 0.2 }, "1m")["Bootstrap CI (Sharpe lower)"]).toBe(
      "fail",
    );
    expect(status({ bootstrap_sharpe_lower: 0.7 }, "1h")["Bootstrap CI (Sharpe lower)"]).toBe(
      "fail",
    );
    expect(
      status({ bootstrap_sharpe_lower: 0.2, bootstrap_min_sharpe: 0.3 }, "4h")[
        "Bootstrap CI (Sharpe lower)"
      ],
    ).toBe("fail");
  });

  it("walk-forward uses the discovery verdict when present", () => {
    expect(status({ wfa_passed: false, walk_forward_efficiency: 3.0 })["Walk-Forward"]).toBe(
      "fail",
    );
    expect(status({ walk_forward_efficiency: 0.5 })["Walk-Forward"]).toBe("fail");
  });
});
