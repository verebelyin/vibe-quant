import { expect, test } from "@playwright/test";

test.describe("Strategy Management page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/strategies");
  });

  test("renders page heading", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Strategy Management" })).toBeVisible();
  });

  test("Create New button opens dialog", async ({ page }) => {
    await page.getByRole("button", { name: /create new/i }).click();

    // Dialog should appear with form fields
    await expect(page.getByRole("dialog")).toBeVisible();
  });

  test("Wizard button opens wizard view", async ({ page }) => {
    await page.getByRole("button", { name: /wizard/i }).click();

    // Wizard replaces the page content -- should see wizard UI
    await expect(page.getByText(/wizard|step|template/i).first()).toBeVisible();
  });
});
