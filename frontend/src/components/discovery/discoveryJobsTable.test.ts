import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { DiscoveryJobResponse } from "@/api/generated/models";
import { DiscoveryJobsTable } from "./DiscoveryResultsPage";

function job(over: Partial<DiscoveryJobResponse> & { run_id: number }): DiscoveryJobResponse {
  return {
    status: "completed",
    started_at: null,
    symbols: ["BTCUSDT"],
    timeframe: "4h",
    generations: 10,
    population: 20,
    // Computed twin the API always sends; the table sorts and renders the raw value.
    best_sharpe_display: null,
    ...over,
  };
}

const JOBS: DiscoveryJobResponse[] = [
  job({ run_id: 101, started_at: "2026-01-02T00:00:00Z", progress: { best_fitness: 1.5 } }),
  job({ run_id: 102, started_at: "2026-01-03T00:00:00Z", progress: { best_fitness: 0.5 } }),
  job({ run_id: 103, started_at: "2026-01-01T00:00:00Z", progress: { fitness: 2.5 } }),
];

function render(
  jobs: DiscoveryJobResponse[],
  sorting: { id: string; desc: boolean }[],
  extra: { expanded?: Record<string, boolean>; isKilling?: boolean } = {},
): string {
  return renderToStaticMarkup(
    createElement(DiscoveryJobsTable, {
      jobs,
      sorting,
      onSortingChange: () => undefined,
      expanded: extra.expanded ?? {},
      onExpandedChange: () => undefined,
      onKill: () => undefined,
      isKilling: extra.isKilling ?? false,
    }),
  );
}

const killButton = (html: string) =>
  html.match(/<button[^>]*data-variant="destructive"[^>]*>/)?.[0] ?? "";
const order = (html: string) => [...html.matchAll(/#(10\d)/g)].map((m) => m[1]);

describe("discovery DiscoveryJobsTable", () => {
  it("sorts by best fitness (accessorFn over progress) descending", () => {
    expect(order(render(JOBS, [{ id: "best_fitness", desc: true }]))).toEqual([
      "103",
      "101",
      "102",
    ]);
  });

  it("sorts by started_at ascending and descending", () => {
    expect(order(render(JOBS, [{ id: "started_at", desc: false }]))).toEqual(["103", "101", "102"]);
    expect(order(render(JOBS, [{ id: "started_at", desc: true }]))).toEqual(["102", "101", "103"]);
  });

  it("puts jobs without started_at first under the default desc sort, last under asc", () => {
    const jobs = [
      job({ run_id: 101, started_at: null }),
      job({ run_id: 102, started_at: "2026-01-02T00:00:00Z" }),
      job({ run_id: 103, started_at: "2026-01-01T00:00:00Z" }),
    ];
    expect(order(render(jobs, [{ id: "started_at", desc: true }]))).toEqual(["101", "102", "103"]);
    expect(order(render(jobs, [{ id: "started_at", desc: false }]))).toEqual(["103", "102", "101"]);
  });

  it("sorts started_at by instant, not by string, when UTC offsets differ", () => {
    const jobs = [
      // 07:00Z, lexically later than 10:00Z because of the "12" hour
      job({ run_id: 101, started_at: "2026-01-01T12:00:00+05:00" }),
      job({ run_id: 102, started_at: "2026-01-01T10:00:00+00:00" }),
    ];
    expect(order(render(jobs, [{ id: "started_at", desc: false }]))).toEqual(["101", "102"]);
  });

  it("sorts by run id", () => {
    expect(order(render([...JOBS].reverse(), [{ id: "run_id", desc: false }]))).toEqual([
      "101",
      "102",
      "103",
    ]);
  });

  it("shows a Kill button only for running jobs and disables it while killing", () => {
    const jobs = [job({ run_id: 101, status: "running" }), job({ run_id: 102 })];
    const idle = render(jobs, [{ id: "run_id", desc: false }]);
    expect(idle.match(/Kill/g)).toHaveLength(1);
    expect(killButton(idle)).not.toContain('disabled=""');
    const killing = render(jobs, [{ id: "run_id", desc: false }], { isKilling: true });
    expect(killButton(killing)).toContain('disabled=""');
  });

  it("expands only the row whose id is in the expanded state and shows its error", () => {
    const jobs = [
      job({ run_id: 101, status: "failed", error_message: "boom-101" }),
      job({ run_id: 102, status: "failed", error_message: "boom-102" }),
    ];
    const html = render(jobs, [{ id: "run_id", desc: false }], { expanded: { "102": true } });
    expect(html).toContain("boom-102");
    expect(html).not.toContain("boom-101");
  });
});
