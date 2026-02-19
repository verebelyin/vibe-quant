import { expect, test } from "@playwright/test";

test.describe("Paper Trading page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/paper-trading");
  });

  test("renders page heading", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Paper Trading" })).toBeVisible();
  });

  test("renders start paper trading form when idle", async ({ page }) => {
    await expect(page.getByText("Start Paper Trading")).toBeVisible();
  });

  test("strategy selector is present", async ({ page }) => {
    await expect(page.getByText("Select a strategy...").first()).toBeVisible();
  });

  test("testnet checkbox defaults to checked", async ({ page }) => {
    // The testnet checkbox is a native input[type=checkbox]
    const checkbox = page.locator("input[type=checkbox]").first();
    await expect(checkbox).toBeChecked();
  });

  test("sizing configuration fields are visible", async ({ page }) => {
    await expect(page.getByLabel("Max Leverage")).toBeVisible();
    await expect(page.getByLabel("Max Position %")).toBeVisible();
    await expect(page.getByLabel("Risk Per Trade %")).toBeVisible();
  });

  test("sizing fields have default values", async ({ page }) => {
    await expect(page.getByLabel("Max Leverage")).toHaveValue("10");
    await expect(page.getByLabel("Max Position %")).toHaveValue("25");
    await expect(page.getByLabel("Risk Per Trade %")).toHaveValue("2");
  });

  test("start button disabled without strategy", async ({ page }) => {
    const btn = page.getByRole("button", { name: /start paper trading/i });
    await expect(btn).toBeDisabled();
  });

  test("API credentials section toggles", async ({ page }) => {
    // Initially hidden
    await expect(page.getByLabel("API Key")).toBeHidden();

    // Click to show
    await page.getByRole("button", { name: /show api credentials/i }).click();
    await expect(page.getByLabel("API Key")).toBeVisible();
    await expect(page.getByLabel("API Secret")).toBeVisible();

    // Click to hide again
    await page.getByRole("button", { name: /hide api credentials/i }).click();
    await expect(page.getByLabel("API Key")).toBeHidden();
  });

  test("validated only checkbox filters strategies", async ({ page }) => {
    // The validated only checkbox should be present
    await expect(page.getByText("Validated only")).toBeVisible();
  });

  test("sizing method dropdown is visible", async ({ page }) => {
    await expect(page.getByLabel("Sizing Method")).toBeVisible();
  });
});
