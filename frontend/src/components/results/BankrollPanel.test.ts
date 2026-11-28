/**
 * Tests for the results-page sizing report panel (Kelly + size stress).
 *
 * Pure: every state decision (launch gate, ETA, stale parent, risk-held flag,
 * recommendation row, empty state, launch error text) lives in exported helpers,
 * so no render is needed (vitest runs in node). The chief checks rendering on screen.
 */
import { describe, expect, it } from "vitest";
import type { BankrollReportResponse } from "@/api/generated/models";
import {
  estimateSeconds,
  formatDuration,
  formatEta,
  formatMove,
  formatMultiple,
  K_TOOLTIP,
  launchErrorText,
  launchGate,
  loadView,
  numberFieldError,
  panelState,
  parseHeadlineMismatch,
  REPLAY_CAPTION,
  RISK_HELD_LABEL,
  RISK_HELD_TOOLTIP,
  recommendationView,
  rowViews,
  staleView,
  sweepView,
} from "@/components/results/BankrollPanel";

/** An error shaped like the one `customInstance` throws. */
function apiError(status: number, body: unknown): Error {
  const err = new Error(`API error: ${status}`) as Error & { status: number; body: unknown };
  err.status = status;
  err.body = body;
  return err;
}

function resp(over: Partial<BankrollReportResponse> = {}): BankrollReportResponse {
  return {
    id: 7,
    run_id: 42,
    job_id: -1_000_000_000_007,
    status: "completed",
    error_message: null,
    created_at: "2026-11-19 10:00:00",
    completed_at: "2026-11-19 10:04:00",
    request: {},
    report: null,
    freshness: "fresh",
    stale_reasons: [],
    ...over,
  };
}

function row(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    k: 1,
    role: "baseline",
    status: "ok",
    error: null,
    requested_risk: 0.2,
    effective_risk: 0.2,
    effective_cap: 0.5,
    leverage: 10,
    mode: "isolated",
    risk_held_at_limit: false,
    eligible: true,
    ineligible_reason: null,
    liquidation_count: 0,
    time_to_first_liquidation_s: null,
    time_to_ruin_s: null,
    trades: 273,
    total_return: 1.7,
    mtm_dd: 0.23,
    ...over,
  };
}

describe("risk held at limit", () => {
  it("risk held at limit tooltip never claims a loss cap", () => {
    expect(RISK_HELD_TOOLTIP).toBe(
      "Risk is held at the 50% sizing-budget limit: the stop distance at the signal price " +
        "may use at most 50% of equity, before minimum quantity, fills and costs. " +
        "It does not cap the realized loss.",
    );
    expect(RISK_HELD_LABEL).toBe("risk held at limit");
    // C9 rg check, in-test: no text claims a bound on the realized loss.
    for (const text of [RISK_HELD_TOOLTIP, RISK_HELD_LABEL, REPLAY_CAPTION]) {
      expect(text).not.toMatch(/caps? (the )?loss|max(imum)? loss/i);
    }
  });

  it("flags only rows the report flagged, with requested and effective risk", () => {
    const report = {
      rows: [
        row({ k: 2, requested_risk: 0.6, effective_risk: 0.5, risk_held_at_limit: true }),
        row({ k: 1.25, requested_risk: 0.5, effective_risk: 0.5, risk_held_at_limit: false }),
      ],
      recommendation: { k: null },
    };
    const views = rowViews(report, null);
    expect(views.map((v) => [v.k, v.riskHeld, v.requestedRisk, v.effectiveRisk])).toEqual([
      [2, true, 0.6, 0.5],
      [1.25, false, 0.5, 0.5],
    ]);
  });
});

describe("recommendation row", () => {
  const report = {
    rows: [
      row({ k: 0.25, effective_cap: 0.125 }),
      row({ k: 0.5, effective_cap: 0.25 }),
      row({ k: 1, eligible: false, ineligible_reason: "dd_over_limit", mtm_dd: 0.41 }),
    ],
    recommendation: {
      k: 0.5,
      ceiling: 3,
      reason: null,
      binding: "dd_limit",
      text: "Kelly overstates: drawdown limit binds",
    },
  };

  it("marks exactly the recommended tested row", () => {
    const views = rowViews(report, 0.5);
    expect(views.filter((v) => v.recommended).map((v) => v.k)).toEqual([0.5]);
    expect(views[2]?.reasonText).toBe("Drawdown over your limit");
    expect(K_TOOLTIP).toContain("(risk held at the 50% limit)");
    expect(K_TOOLTIP).not.toMatch(/caps? (the )?loss|max(imum)? loss/i);
    expect(
      rowViews(report, 0.25)
        .filter((v) => v.recommended)
        .map((v) => v.k),
    ).toEqual([0.25]);
  });

  it("recommends a tested row with its binding text", () => {
    expect(recommendationView(report, "fresh")).toEqual({
      kind: "recommended",
      k: 0.5,
      ceiling: 3,
      text: "Kelly overstates: drawdown limit binds",
    });
  });

  it("never recommends a size that is not a row", () => {
    const ghost = { ...report, recommendation: { ...report.recommendation, k: 0.75 } };
    expect(recommendationView(ghost, "fresh").kind).toBe("none");
  });

  it("no recommendation keeps the reason", () => {
    const none = {
      ...report,
      recommendation: { k: null, ceiling: 0.4567, reason: "no_eligible_row", binding: null },
    };
    expect(recommendationView(none, "fresh")).toEqual({
      kind: "none",
      reason: "no_eligible_row",
      text: "No tested size at or under the ceiling is eligible",
      ceiling: 0.4567,
    });
    const neg = { ...report, recommendation: { k: null, ceiling: 0, reason: "non_positive_mean" } };
    expect(recommendationView(neg, "fresh")).toMatchObject({
      kind: "none",
      text: "Mean daily return is not positive",
    });
  });

  it("a stale report shows no recommendation", () => {
    expect(recommendationView(report, "stale")).toEqual({
      kind: "none",
      reason: "stale",
      text: "Parent changed since this report: re-validate first",
      ceiling: 3,
    });
  });
});

describe("stale parent", () => {
  const message =
    "stale_parent: headline_mismatch: total_return stored=2.36706856637102 " +
    "replay=1.7076461805441767; max_drawdown stored=0.255137435396459 " +
    "replay=0.23426476664979817: re-validate first";

  it("parses both numbers per headline field", () => {
    expect(parseHeadlineMismatch(message)).toEqual([
      { field: "total_return", stored: "2.36706856637102", replay: "1.7076461805441767" },
      { field: "max_drawdown", stored: "0.255137435396459", replay: "0.23426476664979817" },
    ]);
    expect(parseHeadlineMismatch("stale_parent: result_missing: re-validate first")).toEqual([]);
  });

  it("a stale_parent report says re-validate first with both numbers", () => {
    const view = staleView(resp({ status: "stale_parent", error_message: message }));
    expect(view).toEqual({
      title: "Parent results do not reproduce: re-validate first",
      reasons: ["headline_mismatch"],
      headline: parseHeadlineMismatch(message),
    });
  });

  it("an identity mismatch names the differing keys", () => {
    const view = staleView(
      resp({
        status: "stale_parent",
        error_message: "stale_parent: inputs_mismatch: dsl_sha256, end_date: re-validate first",
      }),
    );
    expect(view?.reasons).toEqual(["inputs_mismatch: dsl_sha256, end_date"]);
    expect(view?.headline).toEqual([]);
    expect(view?.title).toBe("Parent results do not reproduce: re-validate first");
  });

  it("a replaced or missing parent result is titled a change, not a reproduction failure", () => {
    for (const reason of ["result_replaced", "result_missing", "run_row_changed; dsl_changed"]) {
      const view = staleView(
        resp({
          status: "stale_parent",
          error_message: `stale_parent: ${reason}: re-validate first`,
        }),
      );
      expect(view).toEqual({
        title: "Parent result changed: re-validate first",
        reasons: [reason],
        headline: [],
      });
    }
  });

  it("a completed report whose parent moved is stale with its reasons", () => {
    const view = staleView(
      resp({ freshness: "stale", stale_reasons: ["dsl_changed", "result_replaced"] }),
    );
    expect(view).toEqual({
      title: "Parent changed since this report: re-validate first",
      reasons: ["dsl_changed", "result_replaced"],
      headline: [],
    });
    expect(staleView(resp())).toBeNull();
  });
});

describe("empty state and panel state", () => {
  it("no report yet is the empty state", () => {
    expect(panelState(null)).toBe("empty");
    expect(panelState(undefined)).toBe("empty");
  });

  it("maps every report status", () => {
    expect(panelState(resp({ status: "running" }))).toBe("running");
    expect(panelState(resp({ status: "failed" }))).toBe("failed");
    expect(panelState(resp({ status: "stale_parent" }))).toBe("stale");
    expect(panelState(resp({ freshness: "stale" }))).toBe("stale");
    expect(panelState(resp())).toBe("completed");
  });
});

describe("launch gate", () => {
  const notesWith = JSON.stringify({ validation_input: { v: 1, dsl_sha256: "ab" } });

  it("only completed validation runs with recorded inputs may launch", () => {
    expect(launchGate({ run_mode: "screening", status: "completed" }, notesWith)).toBe(
      "not_validation",
    );
    expect(launchGate({ run_mode: "validation", status: "running" }, notesWith)).toBe(
      "not_completed",
    );
    expect(launchGate({ run_mode: "validation", status: "completed" }, notesWith)).toBe("ok");
  });

  it("a run without validation_input is unverified", () => {
    const run = { run_mode: "validation", status: "completed" };
    expect(launchGate(run, JSON.stringify({ liquidation: {} }))).toBe("inputs_unverified");
    expect(launchGate(run, null)).toBe("inputs_unverified");
    expect(launchGate(run, "{not json")).toBe("inputs_unverified");
    expect(launchGate(run, JSON.stringify({ validation_input: "x" }))).toBe("inputs_unverified");
  });

  it("Python NaN/Infinity tokens in notes do not hide recorded inputs", () => {
    const run = { run_mode: "validation", status: "completed" };
    const py = '{"validation_input": {"v": 1}, "sharpe": NaN, "pf": Infinity, "x": [-Infinity]}';
    expect(launchGate(run, py)).toBe("ok");
    expect(launchGate(run, '{"sharpe": NaN}')).toBe("inputs_unverified");
  });
});

describe("loading and error states", () => {
  const idle = { isLoading: false, isError: false, error: null };

  it("a loading summary or report is loading, never unverified or empty", () => {
    expect(loadView({ ...idle, isLoading: true }, idle)).toEqual({ kind: "loading" });
    expect(loadView(idle, { ...idle, isLoading: true })).toEqual({ kind: "loading" });
    expect(loadView(idle, idle)).toEqual({ kind: "ready" });
  });

  it("a failed summary or report GET shows its error, not the empty state", () => {
    const boom = apiError(500, { detail: "reconcile failed" });
    expect(loadView(idle, { ...idle, isError: true, error: boom })).toEqual({
      kind: "error",
      text: "Could not load the sizing report: reconcile failed",
    });
    expect(loadView({ ...idle, isError: true, error: new Error("net") }, idle)).toEqual({
      kind: "error",
      text: "Could not load the run summary",
    });
  });
});

describe("launch form fields", () => {
  it("flags a decimal comma and non-numbers before the API sees them", () => {
    expect(numberFieldError("0.3")).toBeNull();
    expect(numberFieldError("2")).toBeNull();
    expect(numberFieldError("0,3")).toBe("Use a dot for decimals (0.3)");
    expect(numberFieldError("")).toBe("Enter a number");
    expect(numberFieldError("abc")).toBe("Enter a number");
  });

  it("a missing multiple shows a dash, never a dangling ×", () => {
    expect(formatMultiple(null)).toBe("—");
    expect(formatMultiple(0.5)).toBe("0.50×");
    expect(formatMultiple(12.345)).toBe("12.35×");
  });
});

describe("time estimate", () => {
  it("mirrors the API's upper estimate: baseline plus cells in worker waves", () => {
    // 8 default multiples incl. 1.0 -> 8 cells; 4 workers -> 2 waves; 80 s * (1 + 2).
    expect(estimateSeconds(80, [0.25, 0.5, 0.75, 1, 2, 3, 5, 10], 4)).toBe(240);
    // 1.0 is always added: [2] -> {1, 2} = 2 cells; 1 worker -> 2 waves.
    expect(estimateSeconds(51, [2], 1)).toBe(153);
    // 9 cells / 4 workers -> 3 waves.
    expect(estimateSeconds(10.4, [0.25, 0.5, 0.75, 2, 3, 4, 5, 10], 4)).toBe(42);
  });

  it("rounds a .5 tie to even like Python round()", () => {
    // 4 cells / 1 worker -> 5x: 2.5 -> 2, 3.5 -> 4, 4.5 -> 4 (Python round, half to even).
    expect(estimateSeconds(0.5, [1, 2, 3, 4], 1)).toBe(2);
    expect(estimateSeconds(0.7, [1, 2, 3, 4], 1)).toBe(4);
    expect(estimateSeconds(0.9, [1, 2, 3, 4], 1)).toBe(4);
  });

  it("margin sweep: the opt-in sweep adds its cells as their own worker waves", () => {
    // Design R27: parent 100 s, 9 size cells, 4 workers -> 3 waves; 12 sweep cells -> 3 more.
    const nine = [0.25, 0.5, 0.75, 2, 3, 4, 5, 10];
    expect(estimateSeconds(100, nine, 4, 12)).toBe(700);
    expect(estimateSeconds(100, nine, 4, 4)).toBe(500);
    expect(estimateSeconds(100, nine, 4, 0)).toBe(400);
    expect(estimateSeconds(100, nine, 4)).toBe(400);
  });

  it("unknown without a recorded validation time", () => {
    expect(estimateSeconds(null, [1], 2)).toBeNull();
    expect(estimateSeconds(Number.NaN, [1], 2)).toBeNull();
    expect(formatEta(null)).toBe("unknown");
    expect(formatEta(240)).toBe("≈ 4 min");
    expect(formatEta(42)).toBe("≈ 42 s");
  });

  it("formats elapsed seconds for time-to-ruin", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(90_000)).toBe("1d 1h");
    expect(formatDuration(3_600 * 5 + 120)).toBe("5h 2m");
  });
});

describe("launch error text", () => {
  it("keeps the API's re-validate message and explains a running report", () => {
    const stale = apiError(422, {
      detail: "stale_parent: strategy changed since validation: re-validate first",
    });
    expect(launchErrorText(stale)).toBe(
      "stale_parent: strategy changed since validation: re-validate first",
    );
    const busy = apiError(409, { detail: "report_running: report 3 is running" });
    expect(launchErrorText(busy)).toBe(
      "Another sizing report is running; one runs at a time. (report_running: report 3 is running)",
    );
    expect(launchErrorText(new Error("boom"))).toBe("Could not start the sizing report");
  });
});

function move(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    status: "ok",
    value: 0.328,
    side: "short",
    legs: [["BTCUSDT", 500]],
    reason: null,
    path_dependent: false,
    ...over,
  };
}

function marginCell(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    ...row({ role: "cell", k: 2 }),
    mode: "isolated",
    leverage: 3,
    max_margin_pct: 0.5,
    reused: false,
    positions: 1,
    remainder: 0,
    move: move(),
    ...over,
  };
}

function sweepReport(sweep: Record<string, unknown> | null | undefined): Record<string, unknown> {
  const report: Record<string, unknown> = { rows: [row()], recommendation: { k: 2 } };
  if (sweep !== undefined) report.margin_sweep = sweep;
  return report;
}

function completedSweep(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    status: "completed",
    reason: null,
    recommendation_reason: null,
    k: 2,
    scenario: { label: "Modeled scenario: test label." },
    cells: [
      marginCell(),
      marginCell({ leverage: 10, max_margin_pct: 1, move: move({ value: 0.0812, side: "long" }) }),
      marginCell({
        mode: "cross",
        leverage: 5,
        max_margin_pct: 0.5,
        positions: 2,
        remainder: 125.5,
        eligible: false,
        ineligible_reason: "cross_path_dependent",
        move: move({ value: 0.6651038178164769, side: "long", path_dependent: true }),
      }),
    ],
    safest: { mode: "isolated", leverage: 3, max_margin_pct: 0.5, move: move() },
    safest_reason: null,
    ...over,
  };
}

describe("margin sweep", () => {
  it("margin sweep: a stale report shows no safest setting", () => {
    // The API masks `safest` on a stale read; the panel must not rely on it.
    const v = sweepView(sweepReport(completedSweep()), "stale");
    expect(v.kind).toBe("completed");
    if (v.kind !== "completed") return;
    expect(v.safest).toEqual({
      kind: "none",
      text: "Parent changed since this report: re-validate first",
    });
    expect(v.rows.some((r) => r.safest)).toBe(false);
  });

  it("margin sweep: move text per status, path-dependent marked", () => {
    expect(formatMove(move())).toBe("32.80% short");
    expect(formatMove(move({ value: 0.0812, side: "long" }))).toBe("8.12% long");
    expect(formatMove(move({ status: "never", value: null, side: null }))).toBe(
      "not liquidatable in this scenario",
    );
    expect(formatMove(move({ status: "at_entry", value: 0, side: "long" }))).toBe(
      "0.00% (at entry)",
    );
    expect(
      formatMove(move({ status: "unavailable", value: null, side: null, reason: "tier_overflow" })),
    ).toBe("unavailable (tier_overflow)");
    expect(
      formatMove(move({ status: "unavailable", value: null, side: null, reason: "no_tier_table" })),
    ).toBe("unavailable (no_tier_table)");
    expect(
      formatMove(move({ value: 0.6651038178164769, side: "long", path_dependent: true })),
    ).toBe("66.51% long (simultaneous entry only)");
    expect(formatMove({})).toBe("—");
  });

  it("margin sweep: a completed sweep names the safest setting and marks only its row", () => {
    const v = sweepView(sweepReport(completedSweep()), "fresh");
    expect(v.kind).toBe("completed");
    if (v.kind !== "completed") return;
    expect(v.title).toBe("Margin settings at k = 2");
    expect(v.label).toBe("Modeled scenario: test label.");
    expect(v.crossNote).toBe(true);
    expect(v.safest).toEqual({
      kind: "safest",
      text: "Safest setting: isolated, leverage 3, margin limit 50% (liquidating move 32.80% short)",
    });
    expect(v.rows.map((r) => [r.mode, r.leverage, r.marginLimit, r.safest])).toEqual([
      ["isolated", 3, 0.5, true],
      ["isolated", 10, 1, false],
      ["cross", 5, 0.5, false],
    ]);
    expect(v.rows.map((r) => r.positions)).toEqual(["1", "1", "2 + 1 smaller"]);
    expect(v.rows.map((r) => r.move)).toEqual([
      "32.80% short",
      "8.12% long",
      "66.51% long (simultaneous entry only)",
    ]);
    expect(v.rows[2]?.reasonText).toBe("Not ranked: multi-coin cross");

    // An unrelated safest cell: the line and the marked row follow it.
    const other = completedSweep({
      k: 0.5,
      safest: {
        mode: "isolated",
        leverage: 10,
        max_margin_pct: 1,
        move: move({ value: 0.0812, side: "long" }),
      },
    });
    const w = sweepView(sweepReport(other), "fresh");
    if (w.kind !== "completed") throw new Error(w.kind);
    expect(w.title).toBe("Margin settings at k = 0.5");
    expect(w.safest.text).toBe(
      "Safest setting: isolated, leverage 10, margin limit 100% (liquidating move 8.12% long)",
    );
    expect(w.rows.map((r) => r.safest)).toEqual([false, true, false]);
  });

  it("margin sweep: isolated-only sweep has no cross note", () => {
    const v = sweepView(sweepReport(completedSweep({ cells: [marginCell()] })), "fresh");
    if (v.kind !== "completed") throw new Error(v.kind);
    expect(v.crossNote).toBe(false);
  });

  it("margin sweep: no safe setting, skipped and failed say why", () => {
    const none = sweepView(
      sweepReport(completedSweep({ safest: null, safest_reason: "no_safe_setting" })),
      "fresh",
    );
    if (none.kind !== "completed") throw new Error(none.kind);
    expect(none.safest).toEqual({ kind: "none", text: "No safe setting" });

    const skipped = completedSweep({
      status: "skipped",
      reason: "no_recommended_size",
      recommendation_reason: "no_eligible_row",
      k: null,
      scenario: null,
      cells: [],
      safest: null,
    });
    expect(sweepView(sweepReport(skipped), "fresh")).toEqual({
      kind: "message",
      title: "Margin settings",
      text: "Sweep skipped: no recommended size",
    });

    const failed = completedSweep({
      status: "failed",
      reason: "KeyError: 'venue'",
      cells: [],
      scenario: null,
      safest: null,
    });
    expect(sweepView(sweepReport(failed), "fresh")).toEqual({
      kind: "message",
      title: "Margin settings at k = 2",
      text: "Sweep failed: KeyError: 'venue'",
    });
  });

  it("margin sweep: a legacy report or a report without the sweep shows no card", () => {
    expect(sweepView(sweepReport(undefined), "fresh")).toEqual({ kind: "none" });
    expect(sweepView(sweepReport(null), "fresh")).toEqual({ kind: "none" });
  });
});
