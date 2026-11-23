import { type APIRequestContext, expect, type Page, test } from "@playwright/test";

// 127.0.0.1, not localhost: uvicorn binds IPv4 only (see discovery-flow.spec.ts).
const API = process.env.E2E_API_URL ?? "http://127.0.0.1:8001";

interface RunRow {
  id: number;
  run_mode: string;
  status: string;
}

interface RunSummary {
  wfa_passed?: boolean | null;
  wfa_consistency?: number | null;
  walk_forward_efficiency?: number | null;
  deflated_sharpe?: number | null;
  purged_kfold_mean_sharpe?: number | null;
  bootstrap_sharpe_lower?: number | null;
  cross_window_passed?: boolean | null;
  cross_regime_results?: unknown[] | null;
}

type Verdict = "PASS" | "FAIL" | "N/A";

// Mirrors OverfittingBadges WFA_MIN_EFFICIENCY (the fallback when wfa_passed is null).
const WFA_MIN_EFFICIENCY = 0.5;

// Mirrors lib/metrics.ts SHARPE_DISPLAY_FACTOR: stored √252 → shown √365.
const SHARPE_DISPLAY_FACTOR = Math.sqrt(365 / 252);
const displaySharpe = (stored: number): string => (stored * SHARPE_DISPLAY_FACTOR).toFixed(2);

const verdictOf = (passed: boolean | null | undefined): Verdict =>
  passed == null ? "N/A" : passed ? "PASS" : "FAIL";

/** Discovery verdict wfa_passed, else efficiency > 0.5 (as OverfittingBadges computes it). */
function walkForwardVerdict(summary: RunSummary): Verdict {
  return verdictOf(
    summary.wfa_passed ??
      (summary.walk_forward_efficiency != null
        ? summary.walk_forward_efficiency > WFA_MIN_EFFICIENCY
        : null),
  );
}

/** N/A without regimes; PASS only when every regime passed. */
function crossRegimeVerdict(summary: RunSummary): Verdict {
  const regimes = summary.cross_regime_results as Array<{ passed?: boolean }> | null | undefined;
  if (!regimes || regimes.length === 0) return "N/A";
  return verdictOf(regimes.every((r) => r?.passed === true));
}

/** Fields OverfittingBadges reads; any one set means at least one check has a verdict. */
function hasOverfittingData(summary: RunSummary): boolean {
  return (
    summary.wfa_passed != null ||
    summary.walk_forward_efficiency != null ||
    summary.deflated_sharpe != null ||
    summary.purged_kfold_mean_sharpe != null ||
    summary.bootstrap_sharpe_lower != null ||
    summary.cross_window_passed != null ||
    (summary.cross_regime_results?.length ?? 0) > 0
  );
}

async function getSummary(request: APIRequestContext, runId: number): Promise<RunSummary | null> {
  const resp = await request.get(`${API}/api/results/runs/${runId}`);
  return resp.ok() ? ((await resp.json()) as RunSummary) : null;
}

/** First run (newest first) whose summary matches `pick`. Batched: ~300 discovery runs on the DB copy. */
async function pickRun(
  request: APIRequestContext,
  runMode: string,
  pick: (summary: RunSummary) => boolean,
): Promise<{ id: number; summary: RunSummary } | null> {
  const resp = await request.get(`${API}/api/results/runs`);
  expect(resp.ok()).toBeTruthy();
  const { runs } = (await resp.json()) as { runs: RunRow[] };
  const candidates = runs
    .filter((run) => run.run_mode === runMode && run.status === "completed")
    .sort((a, b) => b.id - a.id);
  for (let i = 0; i < candidates.length; i += 25) {
    const batch = candidates.slice(i, i + 25);
    const summaries = await Promise.all(batch.map((run) => getSummary(request, run.id)));
    const hit = batch.findIndex((_, j) => {
      const summary = summaries[j];
      return summary != null && pick(summary);
    });
    if (hit >= 0) {
      const summary = summaries[hit];
      if (summary) return { id: batch[hit].id, summary };
    }
  }
  return null;
}

test.describe("Results Analysis page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/results");
  });

  test("renders page heading", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Results Analysis" })).toBeVisible();
  });

  test("renders runs table with mode filters", async ({ page }) => {
    // The page is a single runs list; filters are comboboxes above the table.
    await expect(page.getByRole("combobox").filter({ hasText: "Validation only" })).toBeVisible();
    // The runs list is div-based (no table/columnheader roles); wait for a header
    // to prove the list rendered.
    await expect(page.getByText("Sharpe").first()).toBeVisible({ timeout: 20_000 });
  });
});

test.describe("Results overfitting badges", () => {
  // Row container of one CheckRow: a region named by its label (aria-label).
  const checkRow = (page: Page, label: string) =>
    page.getByRole("region", { name: label, exact: true });

  test("gate data renders as PASS/FAIL/N/A rows with values", async ({ page, request }) => {
    // Only a few discovery runs carry gate data; pick one by data, never by id.
    const picked = await pickRun(request, "discovery", hasOverfittingData);
    test.skip(picked === null, "no discovery run on this DB copy has overfitting data");
    if (!picked) return;
    const { id, summary } = picked;

    await page.goto(`/results/${id}`);
    await expect(page.getByText("Overfitting Filters")).toBeVisible({ timeout: 15_000 });

    // Verdicts (PASS/FAIL/N/A) for Walk-Forward, Cross-Window and Cross-Regime mirror
    // OverfittingBadges. Purged K-Fold, DSR and Bootstrap assert the value only (N/A when
    // absent); their thresholds are covered by src/__tests__/audit-results-display.test.ts.
    const wfa = checkRow(page, "Walk-Forward");
    const wfaVerdict = walkForwardVerdict(summary);
    await expect(wfa).toContainText(wfaVerdict);
    if (summary.wfa_consistency != null) {
      await expect(wfa).toContainText(`consistency ${(summary.wfa_consistency * 100).toFixed(0)}%`);
    }
    const crossWindow = verdictOf(summary.cross_window_passed);
    const crossRegime = crossRegimeVerdict(summary);
    const rows: Array<[string, string]> = [
      [
        "Purged K-Fold",
        summary.purged_kfold_mean_sharpe == null
          ? "N/A"
          : `mean OOS sharpe: ${displaySharpe(summary.purged_kfold_mean_sharpe)}`,
      ],
      [
        "Deflated Sharpe Ratio",
        summary.deflated_sharpe == null ? "N/A" : `z: ${summary.deflated_sharpe.toFixed(2)}`,
      ],
      [
        "Bootstrap CI (t-stat lower)",
        summary.bootstrap_sharpe_lower == null
          ? "N/A"
          : `≥${summary.bootstrap_sharpe_lower.toFixed(2)}`,
      ],
      ["Cross-Window", crossWindow],
      ["Cross-Regime", crossRegime],
    ];
    for (const [label, expected] of rows) {
      await expect(checkRow(page, label)).toContainText(expected);
    }

    // The header counts failed checks; if any asserted verdict is FAIL it must say "<n> Failed".
    if ([wfaVerdict, crossWindow, crossRegime].includes("FAIL")) {
      await expect(page.getByText(/^\d+ Failed$/)).toBeVisible();
    }
  });

  test("a run without gate data shows Not Run and N/A rows", async ({ page, request }) => {
    const picked = await pickRun(request, "validation", (s) => !hasOverfittingData(s));
    test.skip(picked === null, "no completed validation run on this DB copy");
    if (!picked) return;

    await page.goto(`/results/${picked.id}`);
    await expect(page.getByText("Overfitting Filters")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("Not Run", { exact: true })).toBeVisible();
    await expect(checkRow(page, "Walk-Forward")).toContainText("N/A");
    await expect(checkRow(page, "Deflated Sharpe Ratio")).toContainText("N/A");
  });
});
