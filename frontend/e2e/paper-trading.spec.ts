import { type APIRequestContext, expect, test } from "@playwright/test";

// 127.0.0.1, not localhost: uvicorn binds IPv4 only (see discovery-flow.spec.ts).
const API = process.env.E2E_API_URL ?? "http://127.0.0.1:8001";

interface StrategyRow {
  id: number;
  name: string;
  version: number;
}

interface RunRow {
  run_mode: string;
  status: string;
  strategy_id: number | null;
}

async function getJson<T>(request: APIRequestContext, path: string): Promise<T> {
  const resp = await request.get(`${API}${path}`);
  expect(resp.ok(), `GET ${path} -> HTTP ${resp.status()}`).toBeTruthy();
  return (await resp.json()) as T;
}

/**
 * Option labels the Strategy select must show, derived from the API copy the same way
 * SessionControl does: "validated" = has a completed validation run. Sorted, so the
 * comparison is order-independent.
 */
async function expectedStrategyLabels(
  request: APIRequestContext,
  validatedOnly: boolean,
): Promise<string[]> {
  const { strategies } = await getJson<{ strategies: StrategyRow[] }>(request, "/api/strategies");
  const { runs } = await getJson<{ runs: RunRow[] }>(request, "/api/results/runs");
  const validated = new Set(
    runs
      .filter((run) => run.run_mode === "validation" && run.status === "completed")
      .map((run) => run.strategy_id),
  );
  return strategies
    .filter((s) => !validatedOnly || validated.has(s.id))
    .map((s) => `${s.name} (v${s.version})`)
    .sort();
}

test.describe("Paper Trading page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/paper-trading");
  });

  test("renders page heading", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Paper Trading", exact: true })).toBeVisible();
  });

  test("renders start paper trading form when idle", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "START PAPER TRADING" })).toBeVisible();
  });

  test("strategy selector is present", async ({ page }) => {
    await expect(page.getByText("Select a strategy...").first()).toBeVisible();
  });

  test("testnet checkbox defaults to checked", async ({ page }) => {
    // Radix checkbox (button[role=checkbox]) labelled "Binance testnet ...".
    await expect(page.getByRole("checkbox", { name: /binance testnet/i })).toBeChecked();
  });

  test("start fresh checkbox defaults to unchecked", async ({ page }) => {
    // 3fqw9: inheriting the account's risk state is the default; the operator opts out.
    // pp-t3: the label and help must name the consequence, not just say "ignore previous limits".
    await expect(
      page.getByRole("checkbox", {
        name: "Start fresh: skip what another trader id left on this account",
        exact: true,
      }),
    ).not.toBeChecked();
    await expect(page.getByText(/this trader id's own limits and halt\s+still carry\./)).toBeVisible();
  });

  test("live start fresh needs its own confirmation", async ({ page }) => {
    // v4l2n + 6ub67: a LIVE session always shows the fresh/risk-state confirmation.
    // It is required client-side with start fresh; without it the wording explains
    // that an account with no saved risk state would start its limits from zero.
    const testnet = page.getByRole("checkbox", { name: /binance testnet/i });
    const startFresh = page.getByRole("checkbox", {
      name: "Start fresh: skip what another trader id left on this account",
      exact: true,
    });
    const confirmLive = page.getByRole("checkbox", {
      name: /I understand this trades REAL funds/i,
    });
    const optionalFresh = page.getByRole("checkbox", { name: /may have no saved risk state/i });
    const freshOnStart = page.getByRole("checkbox", {
      name: /starting fresh skips the LIVE/i,
    });

    // Testnet: no second confirmation is asked for.
    await expect(optionalFresh).toHaveCount(0);
    await expect(freshOnStart).toHaveCount(0);

    await testnet.click();
    await expect(testnet).not.toBeChecked();
    await expect(optionalFresh).toBeVisible();
    await expect(optionalFresh).not.toBeChecked();
    await expect(freshOnStart).toHaveCount(0);

    await confirmLive.click();
    await expect(confirmLive).toBeChecked();

    await startFresh.click();
    await expect(startFresh).toBeChecked();
    await expect(freshOnStart).toBeVisible();
    await expect(freshOnStart).not.toBeChecked();
    await expect(optionalFresh).toHaveCount(0);

    // Turning start fresh off returns to the optional (no saved state) wording.
    await startFresh.click();
    await expect(startFresh).not.toBeChecked();
    await expect(optionalFresh).toBeVisible();
    await expect(freshOnStart).toHaveCount(0);
  });

  test("testnet hides the live account fresh confirmation", async ({ page }) => {
    // 6ub67: the LIVE block exists only for live; testnet reuses the account's
    // saved limits without any extra consent.
    await expect(page.getByRole("checkbox", { name: /may have no saved risk state/i })).toHaveCount(
      0,
    );
    await expect(
      page.getByRole("checkbox", { name: /starting fresh resets my LIVE/i }),
    ).toHaveCount(0);
  });

  test("sizing configuration fields are visible", async ({ page }) => {
    await expect(page.getByLabel("Max Leverage")).toBeVisible();
    await expect(page.getByLabel("Max Position %")).toBeVisible();
    await expect(page.getByLabel("Risk Per Trade %")).toBeVisible();
  });

  test("start button disabled without strategy", async ({ page }) => {
    const btn = page.getByRole("button", { name: /start paper trading/i });
    await expect(btn).toBeDisabled();
  });

  test("validated only checkbox filters strategies", async ({ page, request }) => {
    const validated = await expectedStrategyLabels(request, true);
    const all = await expectedStrategyLabels(request, false);
    // The filter must remove something on this DB copy, or the on/off comparison proves nothing.
    expect(validated.length).toBeGreaterThan(0);
    expect(validated.length).toBeLessThan(all.length);

    const checkbox = page.getByRole("checkbox", { name: /validated only/i });
    const trigger = page.locator("#paper-strategy");
    // Opens the Strategy select and returns the option labels it lists, then closes it.
    const listedOptions = async (expected: number): Promise<string[]> => {
      await trigger.click();
      const options = page.getByRole("option");
      await expect(options).toHaveCount(expected);
      const labels = (await options.allTextContents()).map((label) => label.trim()).sort();
      await page.keyboard.press("Escape");
      await expect(page.getByRole("listbox")).toHaveCount(0);
      return labels;
    };

    // Default: checked, only validated strategies are selectable.
    await expect(checkbox).toBeChecked();
    expect(await listedOptions(validated.length)).toEqual(validated);

    // Unchecked: every strategy is selectable.
    await checkbox.click();
    await expect(checkbox).not.toBeChecked();
    expect(await listedOptions(all.length)).toEqual(all);

    // Re-checked: back to the validated subset.
    await checkbox.click();
    await expect(checkbox).toBeChecked();
    expect(await listedOptions(validated.length)).toEqual(validated);
  });

  test("auto-stop limits render their defaults", async ({ page }) => {
    // Defaults from SessionControl useState: drawdown 20, daily loss 5, consecutive losses 5,
    // positions 3. Read from the rendered inputs, not from the source.
    await expect(page.getByText("Auto-Stop Limits")).toBeVisible();
    await expect(page.getByLabel("Max Drawdown %", { exact: true })).toHaveValue("20");
    await expect(page.getByLabel("Max Daily Loss %", { exact: true })).toHaveValue("5");
    await expect(page.getByLabel("Max Consec. Losses", { exact: true })).toHaveValue("5");
    await expect(page.getByLabel("Max Positions", { exact: true })).toHaveValue("3");
  });
});
