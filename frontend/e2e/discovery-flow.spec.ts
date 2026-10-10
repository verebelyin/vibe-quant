import { type APIRequestContext, expect, test } from "@playwright/test";

/**
 * End-to-end discovery -> champion export -> validation launch -> results flow.
 *
 * Runs against the throwaway ui-check stack (backend :8001 + Vite :5188) with
 * E2E_BASE_URL set (see playwright.config.ts). Real rows are picked from the
 * API copy of the state DB rather than hard-coded ids.
 */
const API = process.env.E2E_API_URL ?? "http://localhost:8001";

/**
 * The GA genome's indicator pool (discovery.genome.build_indicator_pool(include_context=True)
 * -> the names the pipeline accepts). Fixed on purpose: the UI and
 * /api/discovery/indicator-pool both render the same data, so comparing them to
 * each other cannot catch a display-name mismatch (vibe-quant-rubp1 served
 * "Stochastic"/"BollingerBands" while the GA wants STOCH/BBANDS). Update this
 * list when an indicator is added to / removed from the GA pool.
 */
const GA_INDICATOR_POOL = [
  "ADAPTIVE_RSI",
  "ADX",
  "BBANDS",
  "BTC_ROC",
  "BTC_TREND",
  "CCI",
  "DONCHIAN",
  "FUNDING",
  "FUNDING_Z",
  "MACD",
  "MFI",
  "NATR",
  "OWN_TREND",
  "PRICE_POSITION",
  "RAMS",
  "ROC",
  "RSI",
  "SQUEEZE_MOM",
  "SQUEEZE_RATIO",
  "STOCH",
  "TREND_ENSEMBLE",
  "WILLR",
];

/** Discovery: how long a launched run must survive (a bad pool name failed within ~8 s). */
const DISCOVERY_MIN_ALIVE_MS = 20_000;
const DISCOVERY_MAX_WAIT_MS = 120_000;
const VALIDATION_MAX_WAIT_MS = 180_000;

interface DiscoveryJob {
  run_id: number;
  status: string;
  progress?: Record<string, unknown> | null;
  error_message?: string | null;
  strategies_found?: number | null;
  symbols?: string[] | null;
  timeframe?: string | null;
}

interface RunRow {
  id: number;
  run_mode: string;
  status: string;
  error_message?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  symbols?: string[] | null;
  timeframe?: string | null;
}

interface DiscoveryResult {
  strategies?: unknown[] | null;
}

async function getJson<T>(request: APIRequestContext, path: string): Promise<T> {
  const resp = await request.get(`${API}${path}`);
  expect(resp.ok(), `GET ${path} -> HTTP ${resp.status()}`).toBeTruthy();
  return (await resp.json()) as T;
}

async function listDiscoveryJobs(request: APIRequestContext): Promise<DiscoveryJob[]> {
  return getJson<DiscoveryJob[]>(request, "/api/discovery/jobs");
}

/** Newest completed discovery run that actually persisted champions. */
async function pickDiscoveryRunWithChampions(
  request: APIRequestContext,
): Promise<{ runId: number; champions: number }> {
  const jobs = await listDiscoveryJobs(request);
  const completed = jobs
    .filter((j) => (j.status ?? "").toLowerCase() === "completed")
    .sort((a, b) => b.run_id - a.run_id);
  for (const job of completed) {
    const resp = await request.get(`${API}/api/discovery/results/${job.run_id}`);
    if (!resp.ok()) continue;
    const data = (await resp.json()) as DiscoveryResult;
    if (Array.isArray(data.strategies) && data.strategies.length > 0) {
      return { runId: job.run_id, champions: data.strategies.length };
    }
  }
  throw new Error("no completed discovery run with champions found in the DB copy");
}

/** Newest completed validation run whose headline metrics and equity curve are non-empty. */
async function pickCompletedValidationRun(request: APIRequestContext): Promise<number> {
  const { runs } = await getJson<{ runs: RunRow[] }>(request, "/api/results/runs");
  const candidates = runs
    .filter((r) => r.run_mode === "validation" && r.status === "completed")
    .sort((a, b) => b.id - a.id);
  for (const run of candidates) {
    const summaryResp = await request.get(`${API}/api/results/runs/${run.id}`);
    if (!summaryResp.ok()) continue;
    const summary = (await summaryResp.json()) as Record<string, unknown>;
    if (typeof summary.sharpe_ratio !== "number") continue;
    if (typeof summary.total_trades !== "number" || summary.total_trades <= 0) continue;
    const equityResp = await request.get(`${API}/api/results/runs/${run.id}/equity-curve`);
    if (!equityResp.ok()) continue;
    const points = (await equityResp.json()) as unknown[];
    if (Array.isArray(points) && points.length > 0) return run.id;
  }
  throw new Error("no completed validation run with metrics + equity curve found in the DB copy");
}

/** Poll until `read()` returns a non-undefined verdict; throws with `describe()` on timeout. */
async function pollUntil<T>(
  read: () => Promise<T | undefined>,
  opts: { timeoutMs: number; intervalMs: number; what: string },
): Promise<T> {
  const deadline = Date.now() + opts.timeoutMs;
  while (Date.now() < deadline) {
    const verdict = await read();
    if (verdict !== undefined) return verdict;
    await new Promise((resolve) => setTimeout(resolve, opts.intervalMs));
  }
  throw new Error(`timed out after ${opts.timeoutMs} ms waiting for ${opts.what}`);
}

test.describe("discovery flow: launch -> export -> validation -> results", () => {
  test.describe.configure({ mode: "serial" });

  // Set by the export scenario; the validation scenario launches this strategy.
  let exportedStrategyName: string | undefined;

  test("discovery launch: GA pool names are ticked and the run survives past launch", async ({
    page,
    request,
  }) => {
    test.setTimeout(DISCOVERY_MAX_WAIT_MS + 90_000);
    // Guards vibe-quant-rubp1: the pool endpoint served display names the GA
    // rejects, so a run launched with those names failed seconds after the 201.
    const pool = await getJson<Array<{ name: string }>>(request, "/api/discovery/indicator-pool");
    expect(pool.map((p) => p.name).sort()).toEqual([...GA_INDICATOR_POOL].sort());

    await page.goto("/discovery");
    await expect(page.getByRole("heading", { name: "Discovery" })).toBeVisible();

    const indicatorLabels = page.locator('label[for^="ind-"]');
    await expect(indicatorLabels.first()).toBeVisible({ timeout: 20_000 });
    const uiNames = (await indicatorLabels.allInnerTexts()).map((s) => s.trim()).sort();
    expect(uiNames).toEqual([...GA_INDICATOR_POOL].sort());

    // Tick the whole pool so the launch request carries explicit names
    // (an empty selection sends indicator_pool: null and never exercises them).
    await page.getByRole("button", { name: "Select All", exact: true }).click();
    await expect(
      page.getByText(`${GA_INDICATOR_POOL.length}/${GA_INDICATOR_POOL.length}`),
    ).toBeVisible();
    await expect(page.locator("#ind-STOCH")).toHaveAttribute("data-state", "checked");

    // TINY config: 1 symbol BTCUSDT, 4h, pop 4, 1 generation, ~3-month window
    await page.locator("#population").fill("4");
    await page.locator("#generations").fill("1");
    await expect(page.locator("#symbols")).toHaveValue("BTCUSDT");
    await expect(page.locator("#disc-timeframe")).toContainText("4 hours");
    // The date pickers auto-populate only after /api/data/coverage resolves.
    await expect(page.locator("#disc-start-date")).toHaveText(/\d{4}/, { timeout: 30_000 });
    await expect(page.locator("#disc-end-date")).toHaveText(/\d{4}/, { timeout: 30_000 });

    const before = Math.max(0, ...(await listDiscoveryJobs(request)).map((j) => j.run_id));

    await page.getByRole("button", { name: "Launch Discovery" }).click();
    await expect(page.getByText("Discovery launched").first()).toBeVisible({ timeout: 20_000 });

    const jobs = await listDiscoveryJobs(request);
    const created = jobs.find((j) => j.run_id > before);
    expect(created, "a discovery run was created").toBeTruthy();
    const runId = created?.run_id as number;
    expect(created?.symbols).toContain("BTCUSDT");
    expect(created?.timeframe).toBe("4h");

    const meta = await getJson<RunRow>(request, `/api/results/runs/${runId}/meta`);
    const spanDays =
      (Date.parse(`${meta.end_date}`) - Date.parse(`${meta.start_date}`)) / 86_400_000;
    expect(spanDays, `window ${meta.start_date}..${meta.end_date}`).toBeGreaterThan(60);
    expect(spanDays, `window ${meta.start_date}..${meta.end_date}`).toBeLessThan(125);

    // The 201 only means the subprocess was spawned. Watch the run: it must
    // stay alive past the window in which a bad pool name used to kill it, and
    // then be running-with-progress or completed — never failed.
    const launchedAt = Date.now();
    let final: DiscoveryJob | undefined;
    try {
      final = await pollUntil<DiscoveryJob>(
        async () => {
          const job = (await listDiscoveryJobs(request)).find((j) => j.run_id === runId);
          const status = (job?.status ?? "").toLowerCase();
          expect(
            ["pending", "running", "completed"],
            `discovery run ${runId} went '${status}': ${job?.error_message ?? "(no error_message)"}`,
          ).toContain(status);
          if (Date.now() - launchedAt < DISCOVERY_MIN_ALIVE_MS) return undefined;
          if (status === "completed") return job;
          if (status === "running" && job?.progress && Object.keys(job.progress).length > 0) {
            return job;
          }
          return undefined;
        },
        {
          timeoutMs: DISCOVERY_MAX_WAIT_MS,
          intervalMs: 2_000,
          what: "discovery progress/completion",
        },
      );
    } finally {
      // Do not leave the throwaway discovery subprocess running for the suite.
      const del = await request.delete(`${API}/api/discovery/jobs/${runId}`);
      // 404 = "Job not running" (it already finished); anything else is a real problem.
      expect([204, 404], `DELETE discovery job ${runId}`).toContain(del.status());
    }
    expect(["running", "completed"]).toContain((final?.status ?? "").toLowerCase());
  });

  test("champion export: exporting a discovered strategy succeeds", async ({ page, request }) => {
    const { runId, champions } = await pickDiscoveryRunWithChampions(request);
    expect(champions).toBeGreaterThan(0);

    await page.goto("/discovery/results");
    await expect(page.getByRole("heading", { name: "Discovery Results" }).first()).toBeVisible();

    // Expand the run row to load its champions table.
    await page.getByText(`#${runId}`, { exact: true }).first().click();

    const exportButton = page.getByRole("button", { name: "Export", exact: true }).first();
    await expect(exportButton).toBeVisible({ timeout: 15_000 });
    const [exportResp] = await Promise.all([
      page.waitForResponse(
        (r) =>
          r.url().includes(`/api/discovery/results/${runId}/export/`) &&
          r.request().method() === "POST",
      ),
      exportButton.click(),
    ]);
    expect(exportResp.status()).toBe(201);
    const exported = (await exportResp.json()) as { strategy_id: number; name: string };
    expect(exported.strategy_id).toBeGreaterThan(0);
    exportedStrategyName = exported.name;

    await expect(page.getByText("Strategy exported to library").first()).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByRole("button", { name: "Exported", exact: true }).first()).toBeVisible();
  });

  test("validation launch: the exported champion runs to completion on BTCUSDT 1m", async ({
    page,
    request,
  }) => {
    test.skip(!exportedStrategyName, "champion export scenario did not produce a strategy");
    test.setTimeout(VALIDATION_MAX_WAIT_MS + 60_000);
    const { runs } = await getJson<{ runs: RunRow[] }>(request, "/api/results/runs");
    const maxBefore = Math.max(0, ...runs.map((r) => r.id));

    await page.goto("/backtest");

    await page.locator("#strategy-select").click();
    const escapedName = (exportedStrategyName as string).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    await page.getByRole("option", { name: new RegExp(`^${escapedName} \\(v`) }).click();

    await page.locator("#sym-BTCUSDT").click();
    await expect(page.locator("#sym-BTCUSDT")).toHaveAttribute("data-state", "checked");

    // Validation mode (default timeframe is 1m) on a short window inside coverage.
    await page.getByRole("button", { name: "validation", exact: true }).click();
    await page.getByRole("button", { name: "6M", exact: true }).click();

    await page.getByRole("button", { name: "Launch Validation" }).click();

    await expect(page.getByText("Backtest launched successfully").first()).toBeVisible({
      timeout: 20_000,
    });
    await expect(page.getByText("Launch failed")).toHaveCount(0);

    const created = await pollUntil<RunRow>(
      async () =>
        (await getJson<{ runs: RunRow[] }>(request, "/api/results/runs")).runs
          .filter((r) => r.run_mode === "validation" && r.id > maxBefore)
          .sort((a, b) => b.id - a.id)[0],
      { timeoutMs: 15_000, intervalMs: 500, what: "the validation run row" },
    );
    const meta = await getJson<RunRow>(request, `/api/results/runs/${created.id}/meta`);
    expect(meta.run_mode).toBe("validation");
    expect(meta.symbols).toContain("BTCUSDT");

    // Guards vibe-quant-wrlea: the launch is a 201 + "running" regardless; the
    // config crash ("Backtest engine not found") only lands asynchronously as
    // status=failed. Wait for the terminal state and require 'completed'.
    let last: RunRow = meta;
    try {
      last = await pollUntil<RunRow>(
        async () => {
          const row = await getJson<RunRow>(request, `/api/results/runs/${created.id}/meta`);
          expect(
            row.status,
            `validation run ${created.id} -> '${row.status}': ${row.error_message ?? "(no error_message)"}`,
          ).not.toMatch(/failed|killed|cancelled/i);
          return row.status === "completed" ? row : undefined;
        },
        { timeoutMs: VALIDATION_MAX_WAIT_MS, intervalMs: 3_000, what: "validation completion" },
      );
    } finally {
      if (last.status !== "completed") {
        // Timed out (or asserted) while still running: stop the subprocess.
        const del = await request.delete(`${API}/api/backtest/jobs/${created.id}`);
        expect([204, 404], `DELETE backtest job ${created.id}`).toContain(del.status());
      }
    }
    expect(last.status).toBe("completed");
  });

  test("results render: headline metrics and the equity chart are non-empty", async ({
    page,
    request,
  }) => {
    const runId = await pickCompletedValidationRun(request);

    await page.goto(`/results/${runId}`);
    await expect(page.getByRole("heading", { name: `Run #${runId}` }).first()).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByText("Performance Metrics")).toBeVisible();

    for (const label of ["Sharpe Ratio", "Total Trades", "Total Return"]) {
      const value = page
        .getByText(label, { exact: true })
        .first()
        .locator("xpath=..")
        .locator("span.font-mono")
        .first();
      await expect(value).toBeVisible();
      const text = (await value.innerText()).trim();
      expect(text, `${label} renders a value`).not.toBe("");
      expect(text, `${label} is non-empty`).not.toBe("N/A");
    }

    // Equity chart container exists (default Charts tab).
    await expect(page.getByRole("tab", { name: "Equity" })).toBeVisible();
    await expect(page.locator(".recharts-surface").first()).toBeVisible({ timeout: 15_000 });
  });
});
