/**
 * Tests for the TradeChart indicator-overlay config extractor (vibe-quant-6x6h0).
 *
 * A stored strategy carries either the UI array format (`[{type, period}]`) or
 * the canonical StrategyDSL dict keyed by indicator name (239 of 241 strategies
 * in the DB). Only the VALUES are sent to the overlay endpoint: it passes every
 * non-null key except `type` as an indicator param, so a leaked dict key would
 * arrive as a bogus parameter.
 */
import { describe, expect, it } from "vitest";
import { indicatorConfigsFromDsl } from "@/components/results/indicatorConfigs";

describe("indicatorConfigsFromDsl", () => {
  it("maps a canonical dict to its configs in insertion order", () => {
    expect(
      indicatorConfigsFromDsl({
        indicators: {
          adx_exit_0: { type: "ADX", period: 12 },
          kama_ma_exit_0_fast: { type: "KAMA", period: 25 },
        },
      }),
    ).toStrictEqual([
      { type: "ADX", period: 12 },
      { type: "KAMA", period: 25 },
    ]);
  });

  it("does not send the dict key: the serialized configs carry only value fields", () => {
    const configs = indicatorConfigsFromDsl({
      indicators: { adx_exit_0: { type: "ADX", period: 12 } },
    });
    expect(configs).toHaveLength(1);
    expect(Object.keys(configs[0]!)).toStrictEqual(["type", "period"]);
    expect(JSON.stringify(configs)).not.toContain("adx_exit_0");
    expect(JSON.stringify(configs)).toBe('[{"type":"ADX","period":12}]');
  });

  it("keeps the UI array format as the same array", () => {
    const indicators = [{ type: "RSI", period: 14 }];
    expect(indicatorConfigsFromDsl({ indicators })).toBe(indicators);
  });

  it("returns [] for null, missing, and non-object input", () => {
    expect(indicatorConfigsFromDsl(null)).toStrictEqual([]);
    expect(indicatorConfigsFromDsl(undefined)).toStrictEqual([]);
    expect(indicatorConfigsFromDsl({})).toStrictEqual([]);
    expect(indicatorConfigsFromDsl({ indicators: null })).toStrictEqual([]);
    expect(indicatorConfigsFromDsl("ADX")).toStrictEqual([]);
    expect(indicatorConfigsFromDsl(7)).toStrictEqual([]);
  });

  it("skips array entries that are not objects with a string type", () => {
    expect(
      indicatorConfigsFromDsl({
        indicators: [null, { period: 14 }, { type: 12 }, { type: "RSI", period: 14 }],
      }),
    ).toStrictEqual([{ type: "RSI", period: 14 }]);
  });

  it("skips non-object array entries and keeps usable ones in order", () => {
    expect(
      indicatorConfigsFromDsl({
        indicators: [
          "ADX",
          14,
          true,
          [{ type: "ADX" }],
          { type: "ADX", period: 12 },
          { type: "KAMA", period: 25 },
        ],
      }),
    ).toStrictEqual([
      { type: "ADX", period: 12 },
      { type: "KAMA", period: 25 },
    ]);
  });

  it("returns [] when every array entry is unusable", () => {
    expect(
      indicatorConfigsFromDsl({ indicators: [null, { period: 14 }, { type: 12 }] }),
    ).toStrictEqual([]);
  });

  it("returns [] when `indicators` is neither an array nor an object", () => {
    expect(indicatorConfigsFromDsl({ indicators: "ADX" })).toStrictEqual([]);
    expect(indicatorConfigsFromDsl({ indicators: 14 })).toStrictEqual([]);
    expect(indicatorConfigsFromDsl({ indicators: true })).toStrictEqual([]);
  });

  it("skips dict entries that are not objects with a string type", () => {
    expect(
      indicatorConfigsFromDsl({
        indicators: {
          null_entry: null,
          string_entry: "ADX",
          array_entry: [{ type: "ADX" }],
          number_entry: 14,
          missing_type: { period: 14 },
          non_string_type: { type: 12, period: 12 },
          ok: { type: "ADX", period: 12 },
        },
      }),
    ).toStrictEqual([{ type: "ADX", period: 12 }]);
  });

  it("keeps a genuine config whose only field is the type", () => {
    expect(indicatorConfigsFromDsl({ indicators: { vwap_0: { type: "VWAP" } } })).toStrictEqual([
      { type: "VWAP" },
    ]);
  });
});
