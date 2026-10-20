import { describe, expect, it } from "vitest";
import { positionsEmptyView, SNAPSHOT_WARNING_TITLE } from "@/components/paper/snapshotWarning";

describe("positionsEmptyView", () => {
  it("no errors reported: no warning and the plain empty text", () => {
    expect(positionsEmptyView(undefined)).toEqual({
      warnings: [],
      emptyText: "No open positions.",
    });
    expect(positionsEmptyView(null)).toEqual({
      warnings: [],
      emptyText: "No open positions.",
    });
    expect(positionsEmptyView([])).toEqual({
      warnings: [],
      emptyText: "No open positions.",
    });
  });

  it("a failed snapshot: warnings verbatim and an incomplete empty text", () => {
    const view = positionsEmptyView(["positions: RuntimeError"]);
    expect(view.warnings).toEqual(["positions: RuntimeError"]);
    expect(view.emptyText).not.toBe("No open positions.");
    expect(view.emptyText).toContain("incomplete");
  });

  it("passes every error through in order", () => {
    expect(positionsEmptyView(["positions: RuntimeError", "order O-2: KeyError"]).warnings).toEqual(
      ["positions: RuntimeError", "order O-2: KeyError"],
    );
  });

  it("drops blank entries", () => {
    expect(positionsEmptyView(["", "   ", "positions: RuntimeError", "\t"]).warnings).toEqual([
      "positions: RuntimeError",
    ]);
  });

  it("only-blank entries never claim 'No open positions.'", () => {
    const view = positionsEmptyView(["", "  "]);
    expect(view.warnings).toEqual([]);
    expect(view.emptyText).toContain("incomplete");
  });

  it("names the warning title", () => {
    expect(SNAPSHOT_WARNING_TITLE).toBe("Position snapshot incomplete");
  });
});
