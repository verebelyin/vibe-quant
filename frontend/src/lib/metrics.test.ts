/**
 * Audit e70tl.22: money numbers shown in the results UI.
 * Backend sends fractions (total_return 0.0517 = 5.17%) and USDT costs.
 */
import { describe, expect, it } from "vitest";
import type { BacktestResultResponse } from "@/api/generated/models";
import { metricsToCSV, metricsToReport } from "@/components/results/ExportPanel";
import {
  drawdownToNegativePercent,
  formatSignedUsd,
  fractionToPercent,
  grossPnlUsd,
  netPnlUsd,
} from "@/lib/metrics";

// Run 870 (validation, strategy 240) as stored in the state DB.
const RUN_870 = {
  id: 526,
  run_id: 870,
  total_return: 0.05171515398049868,
  starting_balance: 1000,
  win_rate: 0.5409836065573771,
  max_drawdown: 0.0312,
  cagr: 0.0734,
  volatility_annual: 0.081,
  sharpe_ratio: 0.5827484921332822,
  total_trades: 61,
  total_fees: 7.5,
  total_slippage: 1.25,
  total_funding: -0.4,
  notes: '{"consistency": {"flags": []}}',
  user_notes: "my note",
} as unknown as BacktestResultResponse;

describe("net PnL from fractional total_return", () => {
  it("run 870 Cost Breakdown net PnL is $51.72", () => {
    expect(formatSignedUsd(netPnlUsd(RUN_870.total_return, RUN_870.starting_balance))).toBe(
      "$51.72",
    );
  });

  it("negative return shows negative PnL", () => {
    expect(formatSignedUsd(netPnlUsd(-0.06460843666999995, 1000))).toBe("-$64.61");
  });

  it("gross adds back costs; received funding lowers gross", () => {
    const net = netPnlUsd(0.05, 1000);
    expect(grossPnlUsd(net, 7.5, 1.25, -0.4)).toBeCloseTo(58.35, 10);
  });

  it("missing inputs → N/A", () => {
    expect(netPnlUsd(null, 1000)).toBeNull();
    expect(formatSignedUsd(null)).toBe("N/A");
  });
});

describe("fractions displayed as percent", () => {
  it("converts", () => {
    expect(fractionToPercent(0.05171515398049868)).toBeCloseTo(5.1715, 4);
    expect(fractionToPercent(null)).toBeNull();
  });

  it("drawdown renders below zero with the right magnitude", () => {
    expect(drawdownToNegativePercent(0.0312)).toBeCloseTo(-3.12, 10);
  });
});

describe("ExportPanel CSV/text", () => {
  it("CSV percent columns hold percents", () => {
    const [header = "", values = ""] = metricsToCSV(RUN_870).split("\n");
    const cells = values.split(",");
    const row = Object.fromEntries(header.split(",").map((k, i) => [k, cells[i]] as const));
    expect(Number(row["Total Return (%)"])).toBeCloseTo(5.1715, 4);
    expect(Number(row["Win Rate (%)"])).toBeCloseTo(54.098, 3);
    expect(Number(row["Max Drawdown (%)"])).toBeCloseTo(3.12, 10);
    expect(Number(row["CAGR (%)"])).toBeCloseTo(7.34, 10);
    expect(Number(row["Annual Volatility (%)"])).toBeCloseTo(8.1, 10);
  });

  it("text report prints 5.17% and win rate in %", () => {
    const report = metricsToReport(RUN_870);
    expect(report).toContain("Total Return:     5.17%");
    expect(report).toContain("Win Rate:         54.1%");
    expect(report).toContain("Max Drawdown:     3.12%");
    expect(report).toContain("Net PnL:          $51.72");
  });

  it("report shows the user's notes, never the machine JSON", () => {
    const report = metricsToReport(RUN_870);
    expect(report).toContain("my note");
    expect(report).not.toContain("consistency");
  });
});
