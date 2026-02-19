import { expect, test } from "@playwright/test";

test.describe("Results Analysis page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/results");
  });

  test("renders page heading", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Results Analysis" })).toBeVisible();
  });

  test("renders run selector", async ({ page }) => {
    // RunSelector component should be visible
    await expect(page.getByText(/select.*run|no run selected/i).first()).toBeVisible();
  });

  test("renders Single Run and Compare tabs", async ({ page }) => {
    await expect(page.getByRole("tab", { name: /single run/i })).toBeVisible();
    await expect(page.getByRole("tab", { name: /compare/i })).toBeVisible();
  });

  test("Compare tab shows comparison view", async ({ page }) => {
    await page.getByRole("tab", { name: /compare/i }).click();

    // ComparisonView should render
    await expect(page.getByText(/compare|comparison|select.*runs/i).first()).toBeVisible();
  });
});
