import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import type { RunSummaryItem } from "@/api/generated/models";
import { RunsTable } from "./results";

function run(over: Partial<RunSummaryItem> & { run_id: number }): RunSummaryItem {
  return {
    run_mode: "validation",
    symbols: ["BTCUSDT"],
    timeframe: "4h",
    status: "completed",
    // Computed twin the API always sends; the table sorts and renders the raw value.
    sharpe_display: null,
    ...over,
  };
}

const RUNS: RunSummaryItem[] = [
  run({
    run_id: 1,
    strategy_name: "strat-a",
    sharpe_ratio: 1,
    total_return: 0.3,
    created_at: "2026-01-03T00:00:00Z",
  }),
  run({
    run_id: 2,
    strategy_name: "strat-b",
    sharpe_ratio: 3,
    total_return: -0.1,
    created_at: "2026-01-01T00:00:00Z",
  }),
  run({
    run_id: 3,
    strategy_name: "strat-c",
    sharpe_ratio: 2,
    total_return: 0.05,
    created_at: "2026-01-02T00:00:00Z",
  }),
];

function rowOrder(sorting: { id: string; desc: boolean }[], runs = RUNS): string[] {
  const html = renderToStaticMarkup(
    createElement(RunsTable, {
      runs,
      sorting,
      onSortingChange: () => undefined,
      onRowClick: () => undefined,
    }),
  );
  return [...html.matchAll(/strat-[a-z]/g)].map((m) => m[0]);
}

describe("results RunsTable sorting", () => {
  // A non-UTC zone makes a dropped "Z" on zone-less SQLite text observable here,
  // independent of the machine's TZ (CI runs UTC).
  beforeAll(() => {
    vi.stubEnv("TZ", "Europe/Berlin");
  });
  afterAll(() => {
    vi.unstubAllEnvs();
  });

  it("sorts by sharpe descending and ascending", () => {
    expect(rowOrder([{ id: "sharpe_ratio", desc: true }])).toEqual([
      "strat-b",
      "strat-c",
      "strat-a",
    ]);
    expect(rowOrder([{ id: "sharpe_ratio", desc: false }])).toEqual([
      "strat-a",
      "strat-c",
      "strat-b",
    ]);
  });

  it("sorts by created_at (the page default, newest first)", () => {
    expect(rowOrder([{ id: "created_at", desc: true }])).toEqual(["strat-a", "strat-c", "strat-b"]);
  });

  it("sorts text columns alphabetically", () => {
    const reversed = [...RUNS].reverse();
    expect(rowOrder([{ id: "strategy_name", desc: false }], reversed)).toEqual([
      "strat-a",
      "strat-b",
      "strat-c",
    ]);
  });

  it("renders every column's formatted cell for a row", () => {
    const html = renderToStaticMarkup(
      createElement(RunsTable, {
        runs: [RUNS[1] as RunSummaryItem],
        sorting: [],
        onSortingChange: () => undefined,
        onRowClick: () => undefined,
      }),
    );
    expect(html).toContain("-10.00%");
    // Raw 3 (stored √252) is shown as 3.61 (√365); sorting above stays on the raw value.
    expect(html).toContain("3.61");
    expect(html).toContain("BTCUSDT");
  });

  it("keeps null metrics at one end, not interleaved, when sorting descending", () => {
    const withNull = [
      ...RUNS,
      run({ run_id: 4, strategy_name: "strat-d", sharpe_ratio: null, created_at: null }),
    ];
    const order = rowOrder([{ id: "sharpe_ratio", desc: true }], withNull);
    expect(order.slice(0, 3)).toEqual(["strat-b", "strat-c", "strat-a"]);
    expect(order).toHaveLength(4);
  });

  describe("strategy name sorting is explicit natural/case-insensitive, missing last", () => {
    const named = (names: (string | null)[]) =>
      names.map((strategy_name, i) => run({ run_id: i + 1, strategy_name }));
    const order = (names: (string | null)[], desc: boolean) => {
      const html = renderToStaticMarkup(
        createElement(RunsTable, {
          runs: named(names),
          sorting: [{ id: "strategy_name", desc }],
          onSortingChange: () => undefined,
          onRowClick: () => undefined,
        }),
      );
      return [...html.matchAll(/strat-\d+|alpha|Beta|Unnamed run/g)].map((m) => m[0]);
    };

    it("orders strat-2 before strat-10 in both directions", () => {
      const names = ["strat-10", "strat-2", "strat-1"];
      expect(order(names, false)).toEqual(["strat-1", "strat-2", "strat-10"]);
      expect(order(names, true)).toEqual(["strat-10", "strat-2", "strat-1"]);
    });

    it("applies natural order even when the first ten rows have no digits", () => {
      // v9 auto-inference samples the first 10 rows and would pick plain text here
      const names = [..."abcdefghij"].map((c) => `alpha-${c}`).concat(["strat-10", "strat-2"]);
      const html = renderToStaticMarkup(
        createElement(RunsTable, {
          runs: named(names),
          sorting: [{ id: "strategy_name", desc: false }],
          onSortingChange: () => undefined,
          onRowClick: () => undefined,
        }),
      );
      const got = [...html.matchAll(/strat-\d+/g)].map((m) => m[0]);
      expect(got).toEqual(["strat-2", "strat-10"]);
    });

    it("ignores case and keeps null/empty names last whichever the direction", () => {
      const names = ["Beta", null, "alpha", ""];
      expect(order(names, false)).toEqual(["alpha", "Beta", "Unnamed run", "Unnamed run"]);
      expect(order(names, true)).toEqual(["Beta", "alpha", "Unnamed run", "Unnamed run"]);
    });
  });

  describe("created_at sorts by instant, missing dates last", () => {
    // Each run is named strat-a, strat-b, ... so the rendered row order identifies it.
    const dateOrder = (dates: (string | null)[], desc: boolean) => {
      const runs = dates.map((created_at, i) =>
        run({ run_id: i + 1, strategy_name: `strat-${"abcdefgh"[i]}`, created_at }),
      );
      return rowOrder([{ id: "created_at", desc }], runs);
    };

    it("orders offset timestamps by the instant they name", () => {
      // 2026-01-02 00:30 at +02:00 is 2026-01-01 22:30Z, earlier than 23:00Z
      const dates = ["2026-01-02T00:30:00+02:00", "2026-01-01T23:00:00Z"];
      expect(dateOrder(dates, false)).toEqual(["strat-a", "strat-b"]);
      expect(dateOrder(dates, true)).toEqual(["strat-b", "strat-a"]);
    });

    it("orders fractional seconds by time, not by character", () => {
      const dates = ["2026-01-01T10:00:00.5Z", "2026-01-01T10:00:00Z"];
      expect(dateOrder(dates, false)).toEqual(["strat-b", "strat-a"]);
      expect(dateOrder(dates, true)).toEqual(["strat-a", "strat-b"]);
    });

    it("reads SQLite datetime('now') text as UTC", () => {
      // Guard: a non-UTC zone is what makes a missing "Z" change this order.
      expect(new Date(2026, 0, 1).getTimezoneOffset()).toBe(-60);
      // No zone in the text means UTC: 23:00 is later than 22:30Z
      const dates = ["2026-01-01 23:00:00", "2026-01-01T22:30:00Z"];
      expect(dateOrder(dates, false)).toEqual(["strat-b", "strat-a"]);
      expect(dateOrder(dates, true)).toEqual(["strat-a", "strat-b"]);
    });

    it("keeps null and empty dates last in both directions", () => {
      const dates = [null, "2026-01-01T00:00:00Z", "", "2026-01-02T00:00:00Z"];
      expect(dateOrder(dates, false)).toEqual(["strat-b", "strat-d", "strat-a", "strat-c"]);
      expect(dateOrder(dates, true)).toEqual(["strat-d", "strat-b", "strat-a", "strat-c"]);
    });
  });
});
