/**
 * Tests for the launch forms' margin-limit field (bead vibe-quant-t8nrv, T6).
 *
 * The forms enter the limit in PERCENT and the payload stores the fraction the
 * backend's `SizingSettings` validates, so the field's whole domain is pinned here,
 * one case per entry the design lists: '' -> off (no key), '50' -> 0.5, '0' / '-5' /
 * '100.5' / 'abc' -> a refused value that is never silently read as off, '100' -> 1.0
 * (the inclusive edge), '25.5' -> 0.255.
 */
import { describe, expect, it } from "vitest";
import { marginPctText, parseMaxMarginPct } from "@/components/backtest/marginLimit";

describe("parseMaxMarginPct", () => {
  it("reads an empty field as off", () => {
    expect(parseMaxMarginPct("")).toStrictEqual({ kind: "off" });
    expect(parseMaxMarginPct("   ")).toStrictEqual({ kind: "off" });
  });

  it("reads a percentage as the fraction the run stores", () => {
    expect(parseMaxMarginPct("50")).toStrictEqual({ kind: "on", fraction: 0.5 });
    expect(parseMaxMarginPct("25.5")).toStrictEqual({ kind: "on", fraction: 0.255 });
    expect(parseMaxMarginPct("0.5")).toStrictEqual({ kind: "on", fraction: 0.005 });
    expect(parseMaxMarginPct(" 50 ")).toStrictEqual({ kind: "on", fraction: 0.5 });
  });

  it("accepts a single decimal comma", () => {
    expect(parseMaxMarginPct("25,5")).toStrictEqual({ kind: "on", fraction: 0.255 });
    expect(parseMaxMarginPct("1,000.5")).toMatchObject({ kind: "invalid" });
    expect(parseMaxMarginPct("1,2,3")).toMatchObject({ kind: "invalid" });
  });

  it("accepts the inclusive 100% edge", () => {
    expect(parseMaxMarginPct("100")).toStrictEqual({ kind: "on", fraction: 1 });
  });

  it("refuses a value outside (0, 100] rather than reading it as off", () => {
    const refused = "Max margin in use must be more than 0% and at most 100% of equity";
    expect(parseMaxMarginPct("0")).toStrictEqual({
      kind: "invalid",
      message: `${refused} (you entered 0)`,
    });
    expect(parseMaxMarginPct("-5")).toStrictEqual({
      kind: "invalid",
      message: `${refused} (you entered -5)`,
    });
    expect(parseMaxMarginPct("100.5")).toStrictEqual({
      kind: "invalid",
      message: `${refused} (you entered 100.5)`,
    });
  });

  it("refuses text that is not a number", () => {
    expect(parseMaxMarginPct("abc")).toStrictEqual({
      kind: "invalid",
      message:
        "Max margin in use must be more than 0% and at most 100% of equity (you entered abc)",
    });
    expect(parseMaxMarginPct("NaN")).toStrictEqual({
      kind: "invalid",
      message:
        "Max margin in use must be more than 0% and at most 100% of equity (you entered NaN)",
    });
  });
});

describe("marginPctText", () => {
  it("renders the stored fraction back as the entered percentage", () => {
    expect(marginPctText(0.5)).toBe("50");
    expect(marginPctText(1)).toBe("100");
    expect(marginPctText(0.255)).toBe("25.5");
    expect(marginPctText(0.005)).toBe("0.5");
  });
});
