import { describe, expect, it } from "vitest";
import { totalCosts } from "@/components/results/totalCosts";

describe("totalCosts", () => {
  it("is null when fees, slippage and funding are all null/undefined", () => {
    expect(totalCosts(null, null, null)).toBeNull();
    expect(totalCosts(undefined, undefined, undefined)).toBeNull();
    expect(totalCosts(null, undefined, null)).toBeNull();
  });

  it("is 0 (not null) when one component is an explicit 0 and the rest are null", () => {
    expect(totalCosts(0, null, null)).toBe(0);
    expect(totalCosts(null, 0, undefined)).toBe(0);
    expect(totalCosts(null, null, 0)).toBe(0);
  });

  it("sums the non-null components, treating the missing ones as 0", () => {
    expect(totalCosts(10, 2, 3)).toBe(15);
    expect(totalCosts(10, null, 3)).toBe(13);
    expect(totalCosts(null, 2, null)).toBe(2);
  });

  it("sums a negative funding rebate correctly", () => {
    expect(totalCosts(10, 2, -3)).toBe(9);
    expect(totalCosts(5, null, -5)).toBe(0);
  });
});
