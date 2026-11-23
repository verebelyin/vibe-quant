/**
 * T5 (vibe-quant-clmmc.15): every results-side render of a stored √252
 * Sharpe/Sortino goes through lib/metrics.ts and shows √365; exports carry both
 * columns; n/a for a missing value. Design contract rows R32-R38.
 */
import { describe, expect, it } from "vitest";
import type { BacktestResultResponse } from "@/api/generated/models";
import type { SweepResultResponse } from "@/api/generated/models/sweepResultResponse";
import { screenBadgeClasses, screenBadgeText } from "@/components/research/ScreenBadge";
import { metricsToCSV } from "@/components/results/ExportPanel";
import { sharpeCard } from "@/components/results/MetricsPanel";
import { overfittingChecks } from "@/components/results/OverfittingBadges";
import { filterSweeps, sweepChartData, sweepCsvContent } from "@/components/results/SweepAnalysis";
import { SHARPE_TOOLTIP, SHARPE_UNIT } from "@/lib/metrics";

const STRATEGY_239_SCREENING_SHARPE = 0.8004025908124951;

function result(over: Partial<BacktestResultResponse>): BacktestResultResponse {
  return { ...over } as BacktestResultResponse;
}

function sweep(over: Partial<SweepResultResponse> & { id: number }): SweepResultResponse {
  return {
    run_id: 1,
    parameters: { period: 20 },
    sharpe_ratio: null,
    sortino_ratio: null,
    max_drawdown: null,
    total_return: null,
    profit_factor: null,
    win_rate: null,
    total_trades: null,
    total_fees: null,
    total_funding: null,
    execution_time_seconds: null,
    is_pareto_optimal: false,
    passed_deflated_sharpe: null,
    passed_walk_forward: null,
    passed_purged_kfold: null,
    liquidation_count: null,
    liquidation_forfeited: null,
    exit_timing: null,
    sharpe_display: null,
    sortino_display: null,
    ...over,
  } as SweepResultResponse;
}

/** Split one CSV line, honouring quoted cells (a column name contains a comma). */
function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line.charAt(i);
    if (quoted) {
      if (ch === '"' && line.charAt(i + 1) === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      cells.push(cell);
      cell = "";
    } else {
      cell += ch;
    }
  }
  cells.push(cell);
  return cells;
}

function csvRow(csv: string): Record<string, string> {
  const [header = "", values = ""] = csv.split("\n");
  const cells = splitCsvLine(values);
  return Object.fromEntries(splitCsvLine(header).map((k, i) => [k, cells[i] ?? ""] as const));
}

describe("R32 MetricsPanel Sharpe card", () => {
  it("strategy 239 screening Sharpe shows 0.96 with the basis tooltip", () => {
    expect(sharpeCard(result({ sharpe_ratio: STRATEGY_239_SCREENING_SHARPE }))).toEqual({
      value: "0.96",
      title: SHARPE_TOOLTIP,
      subtitle: SHARPE_UNIT,
    });
  });

  it("another stored value converts independently (never double-converted)", () => {
    expect(sharpeCard(result({ sharpe_ratio: 2 })).value).toBe("2.41");
  });

  it("a null Sharpe shows n/a, not N/A or a NaN", () => {
    expect(sharpeCard(result({ sharpe_ratio: null })).value).toBe("n/a");
    expect(sharpeCard(result({})).value).toBe("n/a");
  });

  it("the Sortino card uses the same basis", () => {
    expect(sharpeCard(result({ sortino_ratio: 0.8004025908124951 }), "sortino_ratio")).toEqual({
      value: "0.96",
      title: SHARPE_TOOLTIP,
      subtitle: SHARPE_UNIT,
    });
  });
});

describe("R34 OverfittingBadges: verdict raw, text display", () => {
  const check = (data: Partial<BacktestResultResponse>) =>
    overfittingChecks(result(data), undefined).find((c) => c.label === "Purged K-Fold");

  it("raw 0.6 passes and reads 0.72 with the >0.60 display threshold", () => {
    const c = check({ purged_kfold_mean_sharpe: 0.6 });
    expect(c?.status).toBe("pass");
    expect(c?.detail).toBe("mean OOS sharpe: 0.72 (thr: >0.60)");
  });

  it("raw 0.45 still fails and reads 0.54", () => {
    const c = check({ purged_kfold_mean_sharpe: 0.45 });
    expect(c?.status).toBe("fail");
    expect(c?.detail).toBe("mean OOS sharpe: 0.54 (thr: >0.60)");
  });

  it("the bootstrap bound is a t-stat and is not converted", () => {
    const bootstrap = overfittingChecks(result({ bootstrap_sharpe_lower: 0.2 }), undefined).find(
      (c) => c.label === "Bootstrap CI (t-stat lower)",
    );
    expect(bootstrap?.detail).toContain("≥0.20");
    expect(bootstrap?.detail).not.toContain("0.24");
  });
});

describe("R35 Sweep Min Sharpe filter is typed in √365", () => {
  const rows = [
    sweep({ id: 1, sharpe_ratio: 0.49, max_drawdown: 0.25 }),
    sweep({ id: 2, sharpe_ratio: 0.5, max_drawdown: 0.1 }),
  ];

  it("raw 0.49 (shown 0.59) is hidden by '0.60'; raw 0.5 (0.6018) is kept", () => {
    expect(filterSweeps(rows, "0.60", "").map((s) => s.id)).toEqual([2]);
  });

  it("an empty filter keeps both; the drawdown filter still works", () => {
    expect(filterSweeps(rows, "", "").map((s) => s.id)).toEqual([1, 2]);
    expect(filterSweeps(rows, "", "20").map((s) => s.id)).toEqual([2]);
  });
});

describe("R35/R36 sweep chart + CSV carry display values", () => {
  it("chart y values are √365 and sorted rows keep pareto flags", () => {
    const points = sweepChartData([
      sweep({
        id: 1,
        sharpe_ratio: STRATEGY_239_SCREENING_SHARPE,
        max_drawdown: 0.2,
        total_return: 0.1,
      }),
      sweep({
        id: 2,
        sharpe_ratio: 2,
        max_drawdown: 0.3,
        total_return: -0.1,
        is_pareto_optimal: true,
      }),
      sweep({ id: 3, sharpe_ratio: null, max_drawdown: 0.1, total_return: 0.05 }),
    ]);
    expect(points).toEqual([
      {
        sharpe_ratio: 0.9632846671540376,
        max_drawdown: 0.2,
        total_return: 0.1,
        is_pareto_optimal: false,
      },
      {
        sharpe_ratio: 2.4070003725904976,
        max_drawdown: 0.3,
        total_return: -0.1,
        is_pareto_optimal: true,
      },
    ]);
  });

  it("sweep CSV has Sharpe/Sortino in both bases", () => {
    const row = csvRow(
      sweepCsvContent([
        sweep({
          id: 1,
          sharpe_ratio: STRATEGY_239_SCREENING_SHARPE,
          sortino_ratio: 2,
        }),
      ]),
    );
    expect(row["Sharpe (ann √365)"]).toBe("0.9632846671540376");
    expect(row["Sharpe (stored √252)"]).toBe("0.8004025908124951");
    expect(row["Sortino (ann √365)"]).toBe("2.4070003725904976");
    expect(row["Sortino (stored √252)"]).toBe("2");
  });

  it("result CSV has Sharpe/Sortino/Purged K-Fold in both bases", () => {
    const row = csvRow(
      metricsToCSV(
        result({
          run_id: 870,
          sharpe_ratio: STRATEGY_239_SCREENING_SHARPE,
          sortino_ratio: null,
          purged_kfold_mean_sharpe: 0.6,
        }),
      ),
    );
    expect(row["Sharpe Ratio (ann √365)"]).toBe("0.9632846671540376");
    expect(row["Sharpe Ratio (stored, ann √252)"]).toBe("0.8004025908124951");
    expect(row["Sortino Ratio (ann √365)"]).toBe("");
    expect(row["Sortino Ratio (stored, ann √252)"]).toBe("");
    expect(row["Purged K-Fold Mean Sharpe (ann √365)"]).toBe("0.7221001117771493");
    expect(row["Purged K-Fold Mean Sharpe (stored, ann √252)"]).toBe("0.6");
  });
});

describe("R37 ScreenBadge: winner rule raw, text display", () => {
  const done = (screen_sharpe: number) => ({
    screen_status: "done",
    screen_trades: 60,
    screen_sharpe,
  });

  it("raw 1.0 is a winner and reads 1.20", () => {
    expect(screenBadgeClasses(done(1))).toContain("green");
    expect(screenBadgeText(done(1))).toBe("Sharpe: 1.20");
  });

  it("raw 0.99 is not a winner and reads 1.19", () => {
    expect(screenBadgeClasses(done(0.99))).not.toContain("green");
    expect(screenBadgeText(done(0.99))).toBe("Sharpe: 1.19");
  });

  it("missing/failed statuses keep their existing texts", () => {
    expect(screenBadgeText({ screen_status: "done", screen_sharpe: null })).toBe("Sharpe: n/a");
    expect(screenBadgeText({ screen_status: "failed" })).toBe("Sharpe: ✕");
  });
});
