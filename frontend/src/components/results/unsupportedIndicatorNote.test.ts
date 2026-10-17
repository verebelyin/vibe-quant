import { describe, expect, it } from "vitest";
import { unsupportedIndicatorNote } from "@/components/results/unsupportedIndicatorNote";

describe("unsupportedIndicatorNote", () => {
  it("is null for a normally computed series", () => {
    expect(unsupportedIndicatorNote({ unsupported: false, display_label: "KAMA(10)" })).toBeNull();
    expect(unsupportedIndicatorNote({ display_label: "KAMA(10)" })).toBeNull();
  });

  it("names the indicator and says it is not available in the chart", () => {
    expect(unsupportedIndicatorNote({ unsupported: true, display_label: "KAMA(10)" })).toBe(
      "KAMA(10): not available in chart",
    );
    expect(unsupportedIndicatorNote({ unsupported: true, display_label: "NATR(14)" })).toBe(
      "NATR(14): not available in chart",
    );
  });
});
