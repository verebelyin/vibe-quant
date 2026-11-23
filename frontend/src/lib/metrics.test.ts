/**
 * Audit e70tl.22: money numbers shown in the results UI.
 * Backend sends fractions (total_return 0.0517 = 5.17%) and USDT costs.
 */
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import type { BacktestResultResponse } from "@/api/generated/models";
import { metricsToCSV, metricsToReport } from "@/components/results/ExportPanel";
import {
  annotateLegacySharpeReason,
  costRatios,
  daysBetweenServerDates,
  drawdownToNegativePercent,
  feeDrag,
  feesPctEquityPerYear,
  fmtSharpe,
  formatFiniteNumber,
  formatFractionPct,
  formatSignedUsd,
  fractionToPercent,
  grossPnlUsd,
  netPnlUsd,
  parseResultNotes,
  SHARPE_DISPLAY_FACTOR,
  SHARPE_TOOLTIP,
  SHARPE_UNIT,
  sharpeDisplay,
} from "@/lib/metrics";
// The SAME fixture pytest uses (vibe_quant/metrics.py fee_drag /
// fees_pct_equity_per_year, D3-4a) — the frontend must agree case for case.
import feeDragVectors from "../../../tests/fixtures/fee_drag_vectors.json";

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

/** Split one CSV line, honouring quoted cells (a column name contains a comma). */
function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line.charAt(i);
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
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

describe("ExportPanel CSV/text", () => {
  it("CSV percent columns hold percents", () => {
    const [header = "", values = ""] = metricsToCSV(RUN_870).split("\n");
    const cells = splitCsvLine(values);
    const row = Object.fromEntries(splitCsvLine(header).map((k, i) => [k, cells[i]] as const));
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

// Shared vector with pytest (tests/unit/test_fee_drag.py reads the same file):
// every gross / fee_drag / fees_pct_equity_yr case must match the backend
// (vibe_quant/metrics.py, D3-4a). Backend null = JSON null.
describe("fee drag and annualized fees (shared vector tests/fixtures/fee_drag_vectors.json)", () => {
  it("the vector has the D3-4a cases", () => {
    expect(feeDragVectors.length).toBeGreaterThanOrEqual(7);
  });

  it.each(feeDragVectors)(
    "net=$net fees=$fees slippage=$slippage funding=$funding balance=$starting_balance days=$days",
    (v) => {
      const gross = grossPnlUsd(v.net, v.fees, v.slippage, v.funding);
      expect(gross).toBeCloseTo(v.expected_gross, 10);

      const drag = feeDrag(v.fees, gross);
      if (v.expected_fee_drag == null) {
        expect(drag).toBeNull();
      } else {
        expect(drag).toBeCloseTo(v.expected_fee_drag, 12);
      }

      const perYear = feesPctEquityPerYear(v.fees, v.starting_balance, v.days);
      if (v.expected_fees_pct_equity_yr == null) {
        expect(perYear).toBeNull();
      } else {
        expect(perYear).toBeCloseTo(v.expected_fees_pct_equity_yr, 12);
      }
    },
  );

  it("non-finite gross and inputs yield null, never NaN", () => {
    expect(feeDrag(10, Number.POSITIVE_INFINITY)).toBeNull();
    expect(feeDrag(10, Number.NaN)).toBeNull();
    expect(feeDrag(10, 0)).toBeNull();
    expect(feeDrag(10, -5)).toBeNull();
    expect(feesPctEquityPerYear(10, Number.NaN, 30)).toBeNull();
    expect(feesPctEquityPerYear(10, 1000, Number.POSITIVE_INFINITY)).toBeNull();
    expect(feesPctEquityPerYear(10, 1000, null)).toBeNull();
  });
});

describe("days between two run dates (annualizing fees needs the window)", () => {
  it("plain dates count whole days", () => {
    expect(daysBetweenServerDates("2026-06-01", "2026-08-01")).toBeCloseTo(61, 10);
    expect(daysBetweenServerDates("2026-01-01", "2026-01-01")).toBe(0);
  });

  it("zone-less server datetimes are read as UTC", () => {
    expect(daysBetweenServerDates("2026-01-01 00:00:00", "2026-01-31 12:00:00")).toBeCloseTo(
      30.5,
      10,
    );
  });

  it("missing or unparseable dates → null", () => {
    expect(daysBetweenServerDates(null, "2026-01-31")).toBeNull();
    expect(daysBetweenServerDates("2026-01-01", undefined)).toBeNull();
    expect(daysBetweenServerDates("not a date", "2026-01-31")).toBeNull();
  });
});

describe("zone-less server datetimes ignore the machine timezone (mutant: parse as local)", () => {
  beforeAll(() => {
    // New York observes DST on 2026-03-08: a local-time parse sees 47h, UTC sees 48h.
    vi.stubEnv("TZ", "America/New_York");
  });
  afterAll(() => {
    vi.unstubAllEnvs();
  });

  it("an interval across a DST change is still whole UTC days", () => {
    expect(daysBetweenServerDates("2026-03-07 00:00:00", "2026-03-09 00:00:00")).toBe(2);
  });

  it("a zone-less start and an explicit-zone end are compared in UTC", () => {
    expect(daysBetweenServerDates("2026-03-07 00:00:00", "2026-03-09T00:00:00Z")).toBe(2);
  });
});

describe("notes JSON", () => {
  it("parses an object and rejects bad payloads", () => {
    expect(parseResultNotes('{"a": 1}')).toEqual({ a: 1 });
    expect(parseResultNotes("not json")).toBeNull();
    expect(parseResultNotes("[1]")).toBeNull();
    expect(parseResultNotes(null)).toBeNull();
  });
});

describe("costRatios (fee drag + fees/equity/yr from a result row)", () => {
  const base = {
    totalReturn: 0.1,
    startingBalance: 1000,
    fees: 10,
    slippage: 0,
    funding: 0,
    runStart: "2026-01-01",
    runEnd: "2026-02-01",
  };

  it("uses the requested run dates when there is no clamp", () => {
    const r = costRatios({ ...base, notes: null });
    expect(r.days).toBe(31);
    expect(r.feesPctEquityYr).toBeCloseTo(((10 / 1000) * 365) / 31, 12);
    expect(r.feeDrag).toBeCloseTo(10 / 110, 12);
  });

  it("prefers the effective (clamped) data_window over the requested dates", () => {
    // 31 requested days but validation clamped to 10: 36.50%, not 11.77%.
    const notes = JSON.stringify({
      data_window: {
        requested_start: "2026-01-01",
        requested_end: "2026-02-01",
        effective_start: "2026-01-22",
        effective_end: "2026-02-01",
      },
    });
    const r = costRatios({ ...base, notes });
    expect(r.days).toBe(10);
    expect(r.feesPctEquityYr).toBeCloseTo(0.365, 12);
  });

  it("a data_window that cannot be read gives n/a, never the requested-date ratio", () => {
    const notes = JSON.stringify({ data_window: { effective_start: "garbage" } });
    expect(costRatios({ ...base, notes }).feesPctEquityYr).toBeNull();
  });

  it("unavailable fees stay unavailable instead of becoming a 0% ratio", () => {
    for (const fees of [null, undefined, Number.NaN]) {
      const r = costRatios({ ...base, fees, notes: null });
      expect(r.feeDrag).toBeNull();
      expect(r.feesPctEquityYr).toBeNull();
    }
  });

  it("missing starting balance → every ratio null", () => {
    const r = costRatios({ ...base, startingBalance: null, notes: null });
    expect(r.netPnl).toBeNull();
    expect(r.feeDrag).toBeNull();
    expect(r.feesPctEquityYr).toBeNull();
  });
});

describe("Sharpe/Sortino displayed in √365 units (stored values are √252)", () => {
  it("display factor is sqrt(365/252), pinned to the same double as pytest", () => {
    expect(SHARPE_DISPLAY_FACTOR).toBe(Math.sqrt(365 / 252));
    expect(SHARPE_DISPLAY_FACTOR).toBeCloseTo(1.2035001862952488, 15);
  });

  it("strategy 239 screening Sharpe 0.8004025908124951 → 0.9632846671540376 → '0.96'", () => {
    expect(sharpeDisplay(0.8004025908124951)).toBe(0.9632846671540376);
    expect(fmtSharpe(0.8004025908124951)).toBe("0.96");
  });

  it("two unrelated stored values scale by the same factor", () => {
    expect(sharpeDisplay(0.5827484921332822)).toBe(0.7013379188456804);
    expect(sharpeDisplay(2)).toBe(2.4070003725904976);
  });

  it("zero, negative and large Sharpes keep their sign and magnitude", () => {
    expect(sharpeDisplay(0)).toBe(0);
    expect(fmtSharpe(0)).toBe("0.00");
    expect(sharpeDisplay(-0.12)).toBe(-0.14442002235542986);
    expect(fmtSharpe(-0.12)).toBe("-0.14");
    expect(sharpeDisplay(2)).toBe(2.4070003725904976);
    expect(fmtSharpe(2)).toBe("2.41");
  });

  it("null, undefined, NaN and ±Infinity render n/a, never NaN", () => {
    for (const raw of [
      null,
      undefined,
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
    ]) {
      expect(sharpeDisplay(raw)).toBeNull();
      expect(fmtSharpe(raw)).toBe("n/a");
    }
  });

  it("honours the digits argument", () => {
    expect(fmtSharpe(0.8004025908124951, 3)).toBe("0.963");
    expect(fmtSharpe(undefined, 3)).toBe("n/a");
  });

  it("carries one unit tag and a tooltip naming both bases", () => {
    expect(SHARPE_UNIT).toBe("ann √365");
    expect(SHARPE_TOOLTIP).toBe(
      "Annualised with 365 days (crypto trades every day): stored engine value × 1.2035 (√(365/252))",
    );
  });

  it("annotates only legacy stored-√252 reasons, never bootstrap t-stats or already-365 texts", () => {
    // q95 DSR reason: stored √252 value, no unit tag yet.
    expect(
      annotateLegacySharpeReason("DSR not significant at q95 (ann SR=0.544, T=1689, N=400)"),
    ).toBe("DSR not significant at q95 (ann SR=0.544, T=1689, N=400) [252-day basis]");
    // Pooled DSR reason, another stored 252 Sharpe.
    expect(
      annotateLegacySharpeReason("DSR not significant (pooled daily SR=0.500 ann, T=181, N=55)"),
    ).toBe("DSR not significant (pooled daily SR=0.500 ann, T=181, N=55) [252-day basis]");
    // Old bootstrap reasons print a t-stat: leave them alone.
    expect(annotateLegacySharpeReason("Bootstrap CI lower bound 0.12 (t-stat)")).toBe(
      "Bootstrap CI lower bound 0.12 (t-stat)",
    );
    expect(annotateLegacySharpeReason("Bootstrap: observed Sharpe=0.88 (t-stat)")).toBe(
      "Bootstrap: observed Sharpe=0.88 (t-stat)",
    );
    // New T3 text, buy-hold (already √365) and non-Sharpe reasons are untouched.
    expect(annotateLegacySharpeReason("Holdout: sharpe 0.54 <= 0.00 (ann √365)")).toBe(
      "Holdout: sharpe 0.54 <= 0.00 (ann √365)",
    );
    expect(annotateLegacySharpeReason("Buy-hold: sharpe 0.54 <= bar 0.60 (ann √365, T=760)")).toBe(
      "Buy-hold: sharpe 0.54 <= bar 0.60 (ann √365, T=760)",
    );
    expect(annotateLegacySharpeReason("K-Fold CV not robust: folds 2/5 (ann √365)")).toBe(
      "K-Fold CV not robust: folds 2/5 (ann √365)",
    );
    expect(annotateLegacySharpeReason("Min trades not met")).toBe("Min trades not met");
  });
});

describe("finite-safe display helpers", () => {
  it("never print NaN or Infinity", () => {
    expect(formatFractionPct(Number.NaN)).toBe("n/a");
    expect(formatFractionPct(Number.POSITIVE_INFINITY)).toBe("n/a");
    expect(formatFractionPct(null)).toBe("n/a");
    expect(formatFractionPct(undefined)).toBe("n/a");
    expect(formatFractionPct(0.195, 1)).toBe("19.5%");
    expect(formatFiniteNumber(Number.NaN)).toBe("n/a");
    expect(formatFiniteNumber(1.2345, 2)).toBe("1.23");
  });
});

it("tags a multi-seed pooled legacy reason (rev-c15t15-1 S1)", () => {
  const r =
    "Multi-seed pooled N=55: DSR q95 not significant: stat=-1.5730 < spread=1.0854 (ann SR=0.544, T=1689, N=400)";
  expect(annotateLegacySharpeReason(r)).toBe(`${r} [252-day basis]`);
});
