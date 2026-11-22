import { describe, expect, it } from "vitest";
import type { DiscoveryLaunchRequest, DiscoveryPreflightResponse } from "@/api/generated/models";
import { fullHistoryParams, visiblePreflight } from "./usePowerPreflight";

const body = (extra: Partial<DiscoveryLaunchRequest>): DiscoveryLaunchRequest => ({
  symbols: ["BTCUSDT", "HYPEUSDT"],
  timeframes: ["4h"],
  ...extra,
});

describe("fullHistoryParams", () => {
  it("asks by symbol set and MARK clamping, independent of the dates", () => {
    expect(fullHistoryParams(body({}))).toEqual({ symbols: "BTCUSDT,HYPEUSDT", mark: false });
    expect(fullHistoryParams(body({ start_date: "2026-10-21", end_date: "2026-11-10" }))).toEqual({
      symbols: "BTCUSDT,HYPEUSDT",
      mark: false,
    });
    expect(fullHistoryParams(body({ liquidation_mode: "off" }))).toEqual({
      symbols: "BTCUSDT,HYPEUSDT",
      mark: false,
    });
    expect(fullHistoryParams(body({ liquidation_mode: "isolated" }))).toEqual({
      symbols: "BTCUSDT,HYPEUSDT",
      mark: true,
    });
    expect(
      fullHistoryParams(
        body({
          symbols: ["ETHUSDT"],
          bankroll_grid: { modes: ["cross"], leverages: [5], max_margin_pcts: [0.25] },
        }),
      ),
    ).toEqual({ symbols: "ETHUSDT", mark: true });
  });

  it("asks nothing without a body or symbols", () => {
    expect(fullHistoryParams(null)).toBeNull();
    expect(fullHistoryParams(body({ symbols: [] }))).toBeNull();
  });
});

describe("visiblePreflight", () => {
  const data = { line: "Power: N=1000" } as DiscoveryPreflightResponse;
  const old = JSON.stringify(body({ population: 50, generations: 20 }));
  const edited = JSON.stringify(body({ population: 50, generations: 100 }));

  it("shows a settled answer only for the body it was asked about", () => {
    const ok = { key: old, result: { kind: "ok" as const, data, body: JSON.parse(old) } };
    expect(visiblePreflight(ok, old)).toEqual(ok.result);
    // generations 20 -> 100: the N=1000 line must not stand for the 5000 budget
    expect(visiblePreflight(ok, edited)).toEqual({ kind: "idle" });
    expect(visiblePreflight(ok, "")).toEqual({ kind: "idle" });
    expect(visiblePreflight(null, old)).toEqual({ kind: "idle" });
  });

  it("drops an error for an older body too", () => {
    const err = { key: old, result: { kind: "error" as const, message: "range too short" } };
    expect(visiblePreflight(err, old)).toEqual(err.result);
    expect(visiblePreflight(err, edited)).toEqual({ kind: "idle" });
  });
});
