import { describe, expect, it } from "vitest";
import {
  type CostEdgeLine,
  costEdgeLines,
  costEdgeTier,
  rejectionEdgeText,
} from "@/components/results/costEdge";

const text = (lines: CostEdgeLine[]): string[] =>
  lines.map((line) => line.map((part) => part.text).join(""));
const bold = (lines: CostEdgeLine[]): string[] =>
  lines.flatMap((line) => line.filter((part) => part.strong).map((part) => part.text));

const NOT_RECORDED = "Edge per side: n/a · Cost per side: n/a (not recorded)";
const NO_FILLS = "Edge per side: n/a · Cost per side: n/a (no fills)";
const FOOTNOTE = "Screening prices already include up to 1 tick of slippage.";

describe("costEdgeLines", () => {
  it("says not recorded for a null, absent or non-object cost_edge", () => {
    for (const ce of [null, undefined, [1], "x", 3]) {
      expect(text(costEdgeLines(ce, "validation"))).toEqual([NOT_RECORDED]);
    }
  });

  it("says no fills for turnover 0 with null bps (R-fe-3), never NaN", () => {
    const ce = {
      turnover: 0,
      gross_edge_bps: null,
      cost_paid_bps: null,
      breakeven_fee_bps: null,
    };
    expect(text(costEdgeLines(ce, "validation"))).toEqual([NO_FILLS]);
    expect(text(costEdgeLines(ce, "screening"))).toEqual([NO_FILLS, FOOTNOTE]);
  });

  it("says not recorded when turnover > 0 but a bps field is null or non-finite", () => {
    const base = {
      turnover: 630,
      gross_edge_bps: 157.1,
      cost_paid_bps: 6.2,
      breakeven_fee_bps: 155.8,
    };
    expect(text(costEdgeLines({ ...base, cost_paid_bps: null }, "validation"))).toEqual([
      NOT_RECORDED,
    ]);
    expect(text(costEdgeLines({ ...base, breakeven_fee_bps: Number.NaN }, "validation"))).toEqual([
      NOT_RECORDED,
    ]);
    expect(text(costEdgeLines({ ...base, turnover: null }, "validation"))).toEqual([NOT_RECORDED]);
  });

  it("states earned, paid and break-even fee for the R1 ledger (validation)", () => {
    const ce = {
      turnover: 630,
      gross_edge_bps: 157.14285714285717,
      cost_paid_bps: 6.2698412698412715,
      breakeven_fee_bps: 155.87301587301587,
    };
    const lines = costEdgeLines(ce, "validation");
    expect(text(lines)).toEqual([
      "Earns 157.1 bps per side before costs, pays 6.3 bps.",
      "Would break even at a 1.559% fee per side (slippage as modelled)",
    ]);
    expect(bold(lines)).toEqual(["157.1 bps", "6.3 bps", "1.559%"]);
  });

  it("states the 239 screening numbers with the slippage footnote", () => {
    const ce = {
      turnover: 65193.86350000001,
      gross_edge_bps: 29.264037684473333,
      cost_paid_bps: 4.670940247313308,
      breakeven_fee_bps: 29.264037684473333,
    };
    expect(text(costEdgeLines(ce, "screening"))).toEqual([
      "Earns 29.3 bps per side before costs, pays 4.7 bps.",
      "Would break even at a 0.293% fee per side",
      FOOTNOTE,
    ]);
  });

  it("says loses for gross edge 0 and below, no fee breaks it even", () => {
    const ce = { turnover: 100, gross_edge_bps: -12.34, cost_paid_bps: 5, breakeven_fee_bps: -9.1 };
    const lines = costEdgeLines(ce, "validation");
    expect(text(lines)).toEqual(["Loses 12.3 bps per side before costs — no fee breaks it even."]);
    expect(bold(lines)).toEqual(["12.3 bps"]);
    expect(text(costEdgeLines({ ...ce, gross_edge_bps: 0 }, "validation"))).toEqual([
      "Loses 0.0 bps per side before costs — no fee breaks it even.",
    ]);
  });

  it("says slippage alone exceeds the edge when gross > 0 but break-even <= 0", () => {
    const ce = { turnover: 100, gross_edge_bps: 3.25, cost_paid_bps: 8, breakeven_fee_bps: -1.5 };
    const want = [
      "Earns 3.3 bps before costs, but modelled slippage alone exceeds it — no fee breaks it even.",
    ];
    expect(text(costEdgeLines(ce, "validation"))).toEqual(want);
    expect(text(costEdgeLines({ ...ce, breakeven_fee_bps: 0 }, "validation"))).toEqual(want);
  });

  it("gives the break-even sentence at the smallest positive edge and break-even", () => {
    const ce = { turnover: 1, gross_edge_bps: 0.04, cost_paid_bps: 0.01, breakeven_fee_bps: 0.04 };
    expect(text(costEdgeLines(ce, null))).toEqual([
      "Earns 0.0 bps per side before costs, pays 0.0 bps.",
      "Would break even at a 0.000% fee per side",
    ]);
  });
});

describe("costEdgeTier", () => {
  it("maps run modes to the cost tier", () => {
    expect(costEdgeTier("validation")).toBe("validation");
    expect(costEdgeTier("regime_cross_oos")).toBe("validation");
    expect(costEdgeTier("screening")).toBe("screening");
    expect(costEdgeTier("discovery")).toBe("screening");
    expect(costEdgeTier("paper")).toBeNull();
    expect(costEdgeTier(undefined)).toBeNull();
  });
});

describe("rejectionEdgeText", () => {
  it("prints edge and cost in bps from a rejection's cost_edge", () => {
    expect(
      rejectionEdgeText({
        turnover: 630,
        gross_edge_bps: 157.14285714285717,
        cost_paid_bps: 6.2698412698412715,
        breakeven_fee_bps: 155.87301587301587,
      }),
    ).toBe(" edge=157.1bps cost=6.3bps");
    expect(
      rejectionEdgeText({
        turnover: 9,
        gross_edge_bps: -2.06,
        cost_paid_bps: 4.44,
        breakeven_fee_bps: -2,
      }),
    ).toBe(" edge=-2.1bps cost=4.4bps");
  });

  it("prints n/a when cost_edge is absent, has no fills or a non-finite field", () => {
    expect(rejectionEdgeText(undefined)).toBe(" edge=n/a cost=n/a");
    expect(rejectionEdgeText(null)).toBe(" edge=n/a cost=n/a");
    expect(
      rejectionEdgeText({
        turnover: 0,
        gross_edge_bps: null,
        cost_paid_bps: null,
        breakeven_fee_bps: null,
      }),
    ).toBe(" edge=n/a cost=n/a");
    expect(
      rejectionEdgeText({ gross_edge_bps: "x", cost_paid_bps: Number.POSITIVE_INFINITY }),
    ).toBe(" edge=n/a cost=n/a");
  });
});
