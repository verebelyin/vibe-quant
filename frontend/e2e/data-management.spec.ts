import { expect, test } from "@playwright/test";

test.describe("Data Management page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/data");
  });

  test("renders data status dashboard", async ({ page }) => {
    // DataStatusDashboard should be visible at top of page
    await expect(page.getByText(/data status|coverage|symbols/i).first()).toBeVisible();
  });

  test("renders coverage table with headers", async ({ page }) => {
    // CoverageTable renders inside DataStatusDashboard
    const table = page.locator("table").first();
    await expect(table).toBeVisible();

    // Verify table has header cells
    const headers = table.locator("th");
    await expect(headers.first()).toBeVisible();
  });

  test("renders ingest form", async ({ page }) => {
    // IngestForm should have input fields for data ingestion
    await expect(page.getByText(/download|ingest|exchange/i).first()).toBeVisible();
  });
});
