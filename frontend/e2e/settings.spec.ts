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

  test("Sizing tab New Config button shows form", async ({ page }) => {
    // Sizing tab is the default
    await page.getByRole("tab", { name: "Sizing" }).click();
    const tabPanel = page.locator("[role=tabpanel]");
    await expect(tabPanel).toBeVisible();

    // Click New Config -- may be in EmptyState or in the header
    const newConfigBtn = tabPanel.getByRole("button", { name: "New Config" });
    await expect(newConfigBtn.first()).toBeVisible();
    await newConfigBtn.first().click();

    // Form should now appear with Name input and method selector
    await expect(tabPanel.getByText("Name").first()).toBeVisible();
    await expect(tabPanel.getByText("Method").first()).toBeVisible();
    await expect(tabPanel.getByRole("button", { name: "Create" })).toBeVisible();
    await expect(tabPanel.getByRole("button", { name: "Cancel" })).toBeVisible();
  });

  test("Sizing tab form Cancel hides form", async ({ page }) => {
    await page.getByRole("tab", { name: "Sizing" }).click();
    const tabPanel = page.locator("[role=tabpanel]");

    const newConfigBtn = tabPanel.getByRole("button", { name: "New Config" });
    await newConfigBtn.first().click();
    await expect(tabPanel.getByRole("button", { name: "Create" })).toBeVisible();

    await tabPanel.getByRole("button", { name: "Cancel" }).click();
    // Form should be hidden, New Config button should reappear
    await expect(newConfigBtn.first()).toBeVisible();
  });

  test("switching tabs preserves tab panel rendering", async ({ page }) => {
    // Click Risk tab
    await page.getByRole("tab", { name: "Risk" }).click();
    const riskPanel = page.locator("[role=tabpanel]");
    await expect(riskPanel).toBeVisible();
    await expect(riskPanel).not.toBeEmpty();

    // Click Latency tab
    await page.getByRole("tab", { name: "Latency" }).click();
    const latencyPanel = page.locator("[role=tabpanel]");
    await expect(latencyPanel).toBeVisible();
    await expect(latencyPanel).not.toBeEmpty();

    // Click back to Risk
    await page.getByRole("tab", { name: "Risk" }).click();
    await expect(page.locator("[role=tabpanel]")).toBeVisible();
    await expect(page.locator("[role=tabpanel]")).not.toBeEmpty();
  });
});
