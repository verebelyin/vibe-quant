import { type APIRequestContext, expect, test } from "@playwright/test";

// 127.0.0.1, not localhost: uvicorn binds IPv4 only (see discovery-flow.spec.ts).
const API = process.env.E2E_API_URL ?? "http://127.0.0.1:8001";

interface RunRow {
  id: number;
  strategy_id: number | null;
  run_mode: string;
  status: string;
  symbols: string[];
  timeframe: string;
  start_date: string | null;
  end_date: string | null;
}

/**
 * Newest completed validation run whose chart gets an indicator overlay: the
 * overlay response recreates the chart AFTER the candles arrived, which is the
 * path the chartGeneration dependency in TradeChart.tsx protects.
 */
async function pickRunWithOverlay(request: APIRequestContext): Promise<RunRow | null> {
  const resp = await request.get(`${API}/api/results/runs`);
  expect(resp.ok()).toBeTruthy();
  const { runs } = (await resp.json()) as { runs: RunRow[] };
  const candidates = runs
    .filter((r) => r.run_mode === "validation" && r.status === "completed" && r.strategy_id != null)
    .sort((a, b) => b.id - a.id)
    .slice(0, 40);
  for (const run of candidates) {
    const symbol = run.symbols[0];
    if (!symbol || !run.start_date || !run.end_date) continue;
    const strategy = await request.get(`${API}/api/strategies/${run.strategy_id}`);
    if (!strategy.ok()) continue;
    const dsl = ((await strategy.json()) as { dsl_config?: { indicators?: unknown } }).dsl_config;
    const indicators = dsl?.indicators;
    const configs = Array.isArray(indicators)
      ? indicators
      : indicators && typeof indicators === "object"
        ? Object.values(indicators)
        : [];
    if (configs.length === 0) continue;
    const params = new URLSearchParams({
      interval: run.timeframe,
      start: run.start_date,
      end: run.end_date,
      indicators: JSON.stringify(configs),
    });
    const overlay = await request.get(`${API}/api/data/indicators/${symbol}?${params}`);
    if (!overlay.ok()) continue;
    const { series } = (await overlay.json()) as { series: { unsupported?: boolean }[] };
    if (series.some((s) => !s.unsupported)) return run;
  }
  return null;
}

test.describe("Results TradeChart", () => {
  test("candles stay drawn after the indicator overlay recreates the chart", async ({
    page,
    request,
  }) => {
    // The picker's untimed overlay request warms the backend process (imports, indicator
    // code path); there is no response cache. The 60 s wait below covers a cold Vite
    // compile plus a cold backend on the first run after stack boot.
    test.setTimeout(150_000);
    const run = await pickRunWithOverlay(request);
    test.skip(
      run === null,
      "no completed validation run on this DB copy gets an indicator overlay",
    );
    if (!run) return;

    // Hold the overlay back so the chart is first built from candles alone and
    // then recreated when the overlay lands (the order that dropped the candles).
    await page.route("**/api/data/indicators/**", async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 2_000));
      await route.continue();
    });
    const overlayLanded = page.waitForResponse("**/api/data/indicators/**", { timeout: 60_000 });
    await page.goto(`/results/${run.id}`);
    await expect(page.getByRole("heading", { name: "Trade Chart" })).toBeVisible({
      timeout: 20_000,
    });
    await overlayLanded;
    // Legend chip = the overlay series were applied to the recreated chart.
    await expect(page.locator("button.rounded-full").first()).toBeVisible({ timeout: 10_000 });

    // Candles are painted on the chart canvases in the up/down colours
    // (#26a69a / #ef5350); an empty recreated chart has none. Count painted pixels.
    const candlePixels = () =>
      page.evaluate(() => {
        const heading = [...document.querySelectorAll("h3")].find(
          (h) => h.textContent?.trim() === "Trade Chart",
        );
        const canvases = heading?.parentElement?.parentElement?.querySelectorAll("canvas") ?? [];
        let count = 0;
        for (const canvas of canvases) {
          const ctx = canvas.getContext("2d");
          if (!ctx || canvas.width === 0 || canvas.height === 0) continue;
          const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
          for (let i = 0; i < data.length; i += 4) {
            const [r, g, b, a] = [data[i]!, data[i + 1]!, data[i + 2]!, data[i + 3]!];
            const up = Math.abs(r - 0x26) < 8 && Math.abs(g - 0xa6) < 8 && Math.abs(b - 0x9a) < 8;
            const down = Math.abs(r - 0xef) < 8 && Math.abs(g - 0x53) < 8 && Math.abs(b - 0x50) < 8;
            if (a > 200 && (up || down)) count++;
          }
        }
        return count;
      });
    // Floor 200: measured 5410 painted pixels on the picked run, 0 with the generation
    // wiring removed; 200 leaves room for a short window yet no empty chart reaches it.
    await expect.poll(candlePixels, { timeout: 10_000 }).toBeGreaterThan(200);
  });
});
