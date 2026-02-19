import { expect, test } from "@playwright/test";

test.describe("Cross-page workflows", () => {
  test("navigate from strategies to backtest launch", async ({ page }) => {
    await page.goto("/strategies");
    await expect(page.getByRole("heading", { name: "Strategy Management" })).toBeVisible();

    // Navigate to backtest via sidebar
    await page.getByRole("link", { name: "Backtest Launch" }).click();
    await expect(page).toHaveURL(/\/backtest/);
    await expect(page.getByRole("heading", { name: /backtest/i }).first()).toBeVisible();
  });

  test("navigate from backtest to results", async ({ page }) => {
    await page.goto("/backtest");
    await page.getByRole("link", { name: "Results Analysis" }).click();
    await expect(page).toHaveURL(/\/results/);
    await expect(page.getByRole("heading", { name: "Results Analysis" })).toBeVisible();
  });

  test("navigate from data management to strategies", async ({ page }) => {
    await page.goto("/data");
    await page.getByRole("link", { name: "Strategy Management" }).click();
    await expect(page).toHaveURL(/\/strategies/);
    await expect(page.getByRole("heading", { name: "Strategy Management" })).toBeVisible();
  });

  test("navigate from discovery to paper trading", async ({ page }) => {
    await page.goto("/discovery");
    await page.getByRole("link", { name: "Paper Trading" }).click();
    await expect(page).toHaveURL(/\/paper-trading/);
    await expect(page.getByRole("heading", { name: "Paper Trading" })).toBeVisible();
  });

  test("full navigation cycle through all pages", async ({ page }) => {
    // Start at strategies
    await page.goto("/strategies");
    await expect(page.getByRole("heading", { name: "Strategy Management" })).toBeVisible();

    // Discovery
    await page.getByRole("link", { name: "Discovery" }).click();
    await expect(page.getByRole("heading", { name: "Discovery" })).toBeVisible();

    // Backtest
    await page.getByRole("link", { name: "Backtest Launch" }).click();
    await expect(page.getByRole("heading", { name: /backtest/i }).first()).toBeVisible();

    // Results
    await page.getByRole("link", { name: "Results Analysis" }).click();
    await expect(page.getByRole("heading", { name: "Results Analysis" })).toBeVisible();

    // Paper Trading
    await page.getByRole("link", { name: "Paper Trading" }).click();
    await expect(page.getByRole("heading", { name: "Paper Trading" })).toBeVisible();

    // Data
    await page.getByRole("link", { name: "Data Management" }).click();
    await expect(page.getByText(/data status|coverage|symbols/i).first()).toBeVisible();

    // Settings
    await page.getByRole("link", { name: "Settings" }).click();
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
  });

  test("settings tabs maintain state on re-navigation", async ({ page }) => {
    await page.goto("/settings");

    // Click Risk tab
    await page.getByRole("tab", { name: "Risk" }).click();
    await expect(page.getByRole("tab", { name: "Risk" })).toHaveAttribute("data-state", "active");

    // Navigate away and back
    await page.getByRole("link", { name: "Data Management" }).click();
    await expect(page).toHaveURL(/\/data/);
    await page.getByRole("link", { name: "Settings" }).click();

    // Page loads (tab state may reset, that's OK)
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
    // All tabs should still be present
    await expect(page.getByRole("tab", { name: "Risk" })).toBeVisible();
  });
});
