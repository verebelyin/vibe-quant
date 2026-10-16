import { describe, expect, it } from "vitest";
import { candlesFromBrowseRows, volumeFromCandles } from "@/components/results/candleSeriesData";

describe("candlesFromBrowseRows", () => {
  it("maps archive rows (open_time in ms) to candles whose time is seconds", () => {
    const candles = candlesFromBrowseRows([
      { open_time: 1704067200000, open: 1, high: 3, low: 0.5, close: 2, volume: 10 },
    ]);

    expect(candles).toEqual([
      { time: 1704067200, open: 1, high: 3, low: 0.5, close: 2, volume: 10 },
    ]);
  });

  it("sorts rows ascending by time (setData throws on unordered data)", () => {
    const candles = candlesFromBrowseRows([
      { open_time: 7200000, open: 3, high: 3, low: 3, close: 3, volume: 1 },
      { open_time: 3600000, open: 2, high: 2, low: 2, close: 2, volume: 1 },
      { open_time: 0, open: 1, high: 1, low: 1, close: 1, volume: 1 },
    ]);

    expect(candles.map((c) => c.time)).toEqual([0, 3600, 7200]);
  });

  it("keeps one candle per time, the last row wins (duplicate archive rows)", () => {
    const candles = candlesFromBrowseRows([
      { open_time: 0, open: 1, high: 1, low: 1, close: 1, volume: 1 },
      { open_time: 0, open: 9, high: 9, low: 9, close: 9, volume: 9 },
    ]);

    expect(candles).toHaveLength(1);
    expect(candles[0]?.close).toBe(9);
  });

  it("drops rows whose prices are not finite numbers (null, string, missing)", () => {
    const candles = candlesFromBrowseRows([
      { open_time: 0, open: 1, high: 1, low: 1, close: 1, volume: 1 },
      { open_time: 3600000, open: null, high: 1, low: 1, close: 1, volume: 1 },
      { open_time: 7200000, open: "2", high: "3", low: "1", close: "2", volume: 1 },
      { open_time: 10800000, open: 1, high: 1, low: 1 },
    ]);

    expect(candles.map((c) => c.time)).toEqual([0]);
  });

  it("treats a missing volume as 0", () => {
    const candles = candlesFromBrowseRows([{ open_time: 0, open: 1, high: 1, low: 1, close: 1 }]);

    expect(candles[0]?.volume).toBe(0);
  });

  it("returns [] when the run has no bars", () => {
    expect(candlesFromBrowseRows([])).toEqual([]);
  });
});

describe("volumeFromCandles", () => {
  const candle = (open: number, close: number) => ({
    time: 0 as never,
    open,
    high: Math.max(open, close),
    low: Math.min(open, close),
    close,
    volume: 5,
  });

  it("colours a rising candle green and a falling candle red", () => {
    expect(volumeFromCandles([candle(1, 2), candle(2, 1)])).toEqual([
      { time: 0, value: 5, color: "#26a69a80" },
      { time: 0, value: 5, color: "#ef535080" },
    ]);
  });

  it("colours a flat candle (close === open) green", () => {
    expect(volumeFromCandles([candle(2, 2)])[0]?.color).toBe("#26a69a80");
  });
});
