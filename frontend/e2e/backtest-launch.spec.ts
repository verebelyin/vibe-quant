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
});
