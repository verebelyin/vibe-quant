/**
 * R40/R41 (vibe-quant-clmmc.15): the discovery strategies table renders a stored
 * √252 Sharpe in √365 units through lib/metrics.ts. A pooled genome's Sharpe (the
 * pooled daily MTM × √252) gets no special case: same factor, nothing else moves
 * (design §2.6, rule "UI converts the raw field, never a display twin").
 */
import { describe, expect, it } from "vitest";
import type { DiscoveryResultResponseStrategiesItem } from "@/api/generated/models";
import { strategySharpeCell } from "@/components/discovery/DiscoveryResults";

function strategy(over: Record<string, unknown>): DiscoveryResultResponseStrategiesItem {
  return { ...over } as unknown as DiscoveryResultResponseStrategiesItem;
}

describe("discovery results Sharpe cell (√365 display)", () => {
  it("strategy 239 screening Sharpe 0.8004025908124951 renders 0.96", () => {
    expect(strategySharpeCell(strategy({ sharpe: 0.8004025908124951 }))).toBe("0.96");
  });

  it("a pooled daily-MTM Sharpe (0.5 × √252 stored) renders 0.5 × √365 = 9.55", () => {
    expect(strategySharpeCell(strategy({ sharpe: 7.937253933193772 }))).toBe("9.55");
  });

  it("reads the sharpe_ratio alias when sharpe is absent", () => {
    expect(strategySharpeCell(strategy({ sharpe_ratio: 2 }))).toBe("2.41");
  });

  it("missing, null or non-finite renders n/a", () => {
    expect(strategySharpeCell(strategy({}))).toBe("n/a");
    expect(strategySharpeCell(strategy({ sharpe: null }))).toBe("n/a");
    expect(strategySharpeCell(strategy({ sharpe: Number.NaN }))).toBe("n/a");
    expect(strategySharpeCell(strategy({ sharpe: Number.POSITIVE_INFINITY }))).toBe("n/a");
  });
});
