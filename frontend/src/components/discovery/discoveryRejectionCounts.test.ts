import { describe, expect, it } from "vitest";
import { rejectionCountsLine } from "./DiscoveryResults";

describe("rejectionCountsLine", () => {
  it("counts every rejection by kind, largest first, and names the listed rows", () => {
    const counts = {
      total: 612,
      by_stage: { guardrails: 612 },
      by_kind: { "Bootstrap CI": 61, DSR: 540, "Min trades": 11 },
    };
    expect(rejectionCountsLine(counts, 50)).toBe(
      "612 rejected: DSR 540, Bootstrap CI 61, Min trades 11 (first 50 listed)",
    );
  });

  it("counts capped rows by their verdict (passed every gate), not as rejections", () => {
    const counts = {
      total: 9,
      by_stage: { gate_pool_cap: 2, holdout: 7 },
      by_kind: { Holdout: 7, "Top-K cap": 2 },
    };
    expect(rejectionCountsLine(counts, 9)).toBe(
      "7 rejected: Holdout 7 · 2 passed every gate, left out by the top_k cap",
    );
  });

  it("says when kinds sum above the total (a candidate failing DSR and bootstrap)", () => {
    const counts = {
      total: 3,
      by_stage: { guardrails: 3 },
      by_kind: { "Bootstrap CI": 1, DSR: 2, WFA: 1 },
    };
    expect(rejectionCountsLine(counts, 3)).toBe(
      "3 rejected: DSR 2, Bootstrap CI 1, WFA 1 (a candidate can fail several kinds, so kinds sum to 4)",
    );
  });

  it("equal counts order by kind name; no counts (top_k or legacy run) gives no line", () => {
    const counts = { total: 2, by_stage: { guardrails: 2 }, by_kind: { WFA: 1, DSR: 1 } };
    expect(rejectionCountsLine(counts, 2)).toBe("2 rejected: DSR 1, WFA 1");
    expect(rejectionCountsLine(null, 5)).toBeNull();
    expect(rejectionCountsLine(undefined, 0)).toBeNull();
  });

  it("no line when nothing was rejected (gate-pool all with 0 pool genomes)", () => {
    expect(rejectionCountsLine({ total: 0, by_stage: {}, by_kind: {} }, 0)).toBeNull();
  });
});
