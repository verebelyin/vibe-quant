/**
 * Cost-edge sentences (vibe-quant-clmmc.11, design arch-fees-1 § 5).
 *
 * The backend computes every bps value (one money rule, one place); this module only
 * words them. A null `cost_edge` was never recorded; turnover 0 means the run had no
 * tier-fee fills. Missing or non-finite inputs read "n/a", never 0 bps or NaN.
 */

/** Fill model the numbers come from: validation books modelled slippage, screening does not. */
export type CostEdgeTier = "screening" | "validation";

/** One rendered line; `strong` parts are the numbers the sentence is about. */
export type CostEdgeLine = { text: string; strong?: boolean }[];

const SCREENING_MODES = new Set(["screening", "discovery", "regime_cross_discovery"]);
/** regime_cross_oos runs the validation engine (discovery/campaign.py, jobs/manager.py). */
const VALIDATION_MODES = new Set(["validation", "regime_cross_oos"]);

export function costEdgeTier(runMode: string | null | undefined): CostEdgeTier | null {
  if (runMode != null && VALIDATION_MODES.has(runMode)) return "validation";
  return runMode != null && SCREENING_MODES.has(runMode) ? "screening" : null;
}

function field(ce: unknown, key: string): number | null {
  if (ce == null || typeof ce !== "object" || Array.isArray(ce)) return null;
  const v = (ce as Record<string, unknown>)[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

const bps = (v: number): string => `${v.toFixed(1)} bps`;

function sentence(ce: unknown, tier: CostEdgeTier | null): CostEdgeLine[] {
  const turnover = field(ce, "turnover");
  const ge = field(ce, "gross_edge_bps");
  const cp = field(ce, "cost_paid_bps");
  const be = field(ce, "breakeven_fee_bps");
  if (turnover === 0) return [[{ text: "Edge per side: n/a · Cost per side: n/a (no fills)" }]];
  if (turnover == null || ge == null || cp == null || be == null) {
    return [[{ text: "Edge per side: n/a · Cost per side: n/a (not recorded)" }]];
  }
  if (ge <= 0) {
    return [
      [
        { text: "Loses " },
        { text: bps(Math.abs(ge)), strong: true },
        { text: " per side before costs — no fee breaks it even." },
      ],
    ];
  }
  if (be <= 0) {
    return [
      [
        {
          text: `Earns ${bps(ge)} before costs, but modelled slippage alone exceeds it — no fee breaks it even.`,
        },
      ],
    ];
  }
  return [
    [
      { text: "Earns " },
      { text: bps(ge), strong: true },
      { text: " per side before costs, pays " },
      { text: bps(cp), strong: true },
      { text: "." },
    ],
    [
      { text: "Would break even at a " },
      { text: `${(be / 100).toFixed(3)}%`, strong: true },
      { text: tier === "validation" ? " fee per side (slippage as modelled)" : " fee per side" },
    ],
  ];
}

/** The CostBreakdown sentence lines for a result's `cost_edge`, plus the screening footnote. */
export function costEdgeLines(ce: unknown, tier: CostEdgeTier | null): CostEdgeLine[] {
  const lines = sentence(ce, tier);
  if (tier === "screening") {
    lines.push([{ text: "Screening prices already include up to 1 tick of slippage." }]);
  }
  return lines;
}

/** Discovery rejection-row suffix, e.g. ` edge=29.3bps cost=4.7bps`; n/a per missing field. */
export function rejectionEdgeText(ce: unknown): string {
  const fmt = (v: number | null): string => (v == null ? "n/a" : `${v.toFixed(1)}bps`);
  return ` edge=${fmt(field(ce, "gross_edge_bps"))} cost=${fmt(field(ce, "cost_paid_bps"))}`;
}
