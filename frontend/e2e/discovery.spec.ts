import { expect, test } from "@playwright/test";

test.describe("Discovery page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/discovery");
  });

  test("renders page heading", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Discovery" })).toBeVisible();
  });

  test("renders GA Parameters section", async ({ page }) => {
    await expect(page.getByText("GA Parameters")).toBeVisible();
    await expect(page.getByLabel("Population Size")).toBeVisible();
    await expect(page.getByLabel("Generations")).toBeVisible();
  });

  test("renders all GA parameter inputs", async ({ page }) => {
    await expect(page.getByLabel("Population Size")).toBeVisible();
    await expect(page.getByLabel("Generations")).toBeVisible();
    await expect(page.getByLabel("Crossover Rate")).toBeVisible();
    await expect(page.getByLabel("Mutation Rate")).toBeVisible();
    await expect(page.getByLabel("Elite Count")).toBeVisible();
    await expect(page.getByLabel("Tournament Size")).toBeVisible();
  });

  test("renders Indicator Pool section", async ({ page }) => {
    await expect(page.getByText("Indicator Pool")).toBeVisible();
  });

  test("renders Target Config section", async ({ page }) => {
    await expect(page.getByText("Target Config")).toBeVisible();
  });

  test("GA parameter inputs accept values", async ({ page }) => {
    const popInput = page.getByLabel("Population Size");
    await popInput.fill("100");
    await expect(popInput).toHaveValue("100");
  });

  test("GA parameter inputs have sensible defaults", async ({ page }) => {
    await expect(page.getByLabel("Population Size")).toHaveValue("50");
    await expect(page.getByLabel("Generations")).toHaveValue("100");
    await expect(page.getByLabel("Crossover Rate")).toHaveValue("0.8");
    await expect(page.getByLabel("Mutation Rate")).toHaveValue("0.1");
    await expect(page.getByLabel("Elite Count")).toHaveValue("5");
    await expect(page.getByLabel("Tournament Size")).toHaveValue("3");
  });

  test("Target Config has symbols input with default", async ({ page }) => {
    const symbolsInput = page.getByLabel("Symbols (comma-separated)");
    await expect(symbolsInput).toBeVisible();
    await expect(symbolsInput).toHaveValue("BTCUSDT");
  });

  test("Target Config has date inputs", async ({ page }) => {
    await expect(page.getByLabel("Start Date")).toBeVisible();
    await expect(page.getByLabel("End Date")).toBeVisible();
  });

  test("Launch Discovery button is visible", async ({ page }) => {
    await expect(page.getByRole("button", { name: /launch discovery/i })).toBeVisible();
  });
});
