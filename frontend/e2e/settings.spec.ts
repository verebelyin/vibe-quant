import { expect, test } from "@playwright/test";

const tabs = ["Sizing", "Risk", "Latency", "Database", "System"] as const;

test.describe("Settings page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/settings");
  });

  test("renders page heading", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
  });

  test("all 5 tabs are visible", async ({ page }) => {
    for (const tab of tabs) {
      await expect(page.getByRole("tab", { name: tab })).toBeVisible();
    }
  });

  for (const tab of tabs) {
    test(`${tab} tab renders content`, async ({ page }) => {
      await page.getByRole("tab", { name: tab }).click();

      // Each tab should render some content below the tab bar
      // Wait for tab content to appear (lazy-loaded components)
      const tabPanel = page.locator("[role=tabpanel]");
      await expect(tabPanel).toBeVisible();
      // Content should not be empty
      await expect(tabPanel).not.toBeEmpty();
    });
  }
});
