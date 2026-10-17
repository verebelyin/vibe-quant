import { describe, expect, it } from "vitest";
import { buildSymbolOptions, HAS_DATA_LABEL } from "@/components/data/symbolOptions";

describe("buildSymbolOptions", () => {
  it("offers a supported coin with no archived data (the whole point)", () => {
    const options = buildSymbolOptions(["BTCUSDT", "HYPEUSDT"], ["BTCUSDT"]);
    expect(options).toEqual([
      { symbol: "BTCUSDT", hasData: true },
      { symbol: "HYPEUSDT", hasData: false },
    ]);
  });

  it("keeps supported order and then appends archive-only extras", () => {
    const options = buildSymbolOptions(["SOLUSDT", "BTCUSDT"], ["DOGEUSDT", "BTCUSDT"]);
    expect(options.map((o) => o.symbol)).toEqual(["SOLUSDT", "BTCUSDT", "DOGEUSDT"]);
  });

  it("lists a symbol present in both sources exactly once, marked as archived", () => {
    const options = buildSymbolOptions(["ETHUSDT"], ["ETHUSDT", "ETHUSDT"]);
    expect(options).toEqual([{ symbol: "ETHUSDT", hasData: true }]);
  });

  it("marks nothing as archived when the archive is empty", () => {
    const options = buildSymbolOptions(["BTCUSDT", "ETHUSDT"], []);
    expect(options).toEqual([
      { symbol: "BTCUSDT", hasData: false },
      { symbol: "ETHUSDT", hasData: false },
    ]);
  });

  it("still lists archived symbols when the supported list is empty", () => {
    const options = buildSymbolOptions([], ["BTCUSDT"]);
    expect(options).toEqual([{ symbol: "BTCUSDT", hasData: true }]);
  });

  it("returns nothing for two empty sources", () => {
    expect(buildSymbolOptions([], [])).toEqual([]);
  });

  it("labels archived symbols 'has data'", () => {
    expect(HAS_DATA_LABEL).toBe("has data");
  });
});
