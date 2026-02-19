import { expect, test } from "@playwright/test";

test.describe("Backtest Launch page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/backtest");
  });

  test("renders launch form", async ({ page }) => {
    // BacktestLaunchForm should render with strategy/config fields
    await expect(page.getByText(/backtest|launch|strategy/i).first()).toBeVisible();
  });

  test("renders preflight status component", async ({ page }) => {
    // PreflightStatus is part of BacktestLaunchForm
    await expect(page.getByText(/preflight|status|ready/i).first()).toBeVisible();
  });

  test("renders active jobs panel", async ({ page }) => {
    // ActiveJobsPanel shows running/queued jobs
    await expect(page.getByText(/active|jobs|queue/i).first()).toBeVisible();
  });

  test("strategy selector renders with placeholder", async ({ page }) => {
    await expect(page.getByLabel("Strategy")).toBeVisible();
    await expect(page.getByText("Select a strategy...").first()).toBeVisible();
  });

  test("mode toggle shows Screening and Validation buttons", async ({ page }) => {
    await expect(page.getByRole("button", { name: "screening" })).toBeVisible();
    await expect(page.getByRole("button", { name: "validation" })).toBeVisible();
  });

  test("clicking Validation mode shows validation settings", async ({ page }) => {
    await page.getByRole("button", { name: "validation" }).click();
    await expect(page.getByText("Validation Settings")).toBeVisible();
    await expect(page.getByLabel("Latency Preset")).toBeVisible();
  });

  test("clicking Screening mode hides validation settings", async ({ page }) => {
    // Switch to validation first
    await page.getByRole("button", { name: "validation" }).click();
    await expect(page.getByText("Validation Settings")).toBeVisible();

    // Switch back to screening
    await page.getByRole("button", { name: "screening" }).click();
    await expect(page.getByText("Validation Settings")).toBeHidden();
  });

  test("symbols section renders", async ({ page }) => {
    await expect(page.getByText(/Symbols \(\d+ selected\)/)).toBeVisible();
  });

  test("date preset buttons are visible", async ({ page }) => {
    for (const label of ["1M", "3M", "6M", "1Y", "2Y"]) {
      await expect(page.getByRole("button", { name: label, exact: true })).toBeVisible();
    }
  });

  test("date preset populates start and end dates", async ({ page }) => {
    await page.getByRole("button", { name: "3M", exact: true }).click();

    const startInput = page.getByLabel("Start Date");
    const endInput = page.getByLabel("End Date");
    // After clicking preset, both should have values
    await expect(startInput).not.toHaveValue("");
    await expect(endInput).not.toHaveValue("");
  });

  test("launch button is disabled without strategy selected", async ({ page }) => {
    // The Launch button should be disabled when no strategy is selected
    const launchBtn = page.getByRole("button", { name: /launch screening/i });
    await expect(launchBtn).toBeDisabled();
  });

  test("initial balance and leverage inputs render with defaults", async ({ page }) => {
    const balanceInput = page.getByLabel("Initial Balance (USD)");
    const leverageInput = page.getByLabel("Leverage (1-125)");
    await expect(balanceInput).toBeVisible();
    await expect(leverageInput).toBeVisible();
    await expect(balanceInput).toHaveValue("10000");
    await expect(leverageInput).toHaveValue("10");
  });

  test("overfitting filters section renders", async ({ page }) => {
    await expect(page.getByText("Overfitting Filters")).toBeVisible();
    await expect(page.getByText("Deflated Sharpe Ratio (DSR)")).toBeVisible();
    await expect(page.getByText("Walk-Forward Analysis (WFA)")).toBeVisible();
    await expect(page.getByText("Purged K-Fold CV")).toBeVisible();
  });
});
