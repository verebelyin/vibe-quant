/**
 * Tests for the discovery runtime estimate (vibe-quant-fq0pi) and the per-timeframe
 * budget (bead vibe-quant-wnqea).
 *
 * The Discovery form's "Total evaluations" / "Parallel (8 cores)" hint must count
 * the bankroll grid: a grid of N setups runs N times the search, so the number the
 * operator sees before launching is population * generations * seeds * windows * N.
 *
 * P-2 (docs/discovery-journal.md, 2026-10-19) measured the WHOLE-RUN wall seconds per
 * evaluation at 0.673-0.810 on 4h and 0.227-0.362 on 1d, so those two timeframes
 * estimate from that measured figure and judge the budget in wall seconds instead of
 * evaluations. The 4h launch default is the P-2 rung 50x20 with convergence 10.
 *
 * These are pure-function tests (this repo's vitest is node-only, no jsdom).
 */
import { describe, expect, it } from "vitest";
import {
  budgetFor,
  budgetOnTimeframeChange,
  DEFAULT_BUDGET,
  dangerAdvice,
  discoveryEvalEstimate,
  EVAL_CORES,
  evalBudget,
  gridSetupCount,
  RECOMMENDED_BUDGET,
  SEC_PER_EVAL,
  secPerEvalFor,
} from "@/components/discovery/discoveryEstimate";
import type { BankrollGridParse } from "@/components/discovery/discoveryLaunchBody";

describe("discoveryEvalEstimate", () => {
  it("counts a grid-OFF search as one setup (E1)", () => {
    expect(
      discoveryEvalEstimate({
        timeframe: "1m",
        population: 20,
        generations: 15,
        numSeeds: 1,
        crossWindowMultiplier: 1,
        gridCount: 1,
      }),
    ).toStrictEqual({ totalEvals: 300, parallelSec: 570 });
  });

  it("multiplies the estimate by the grid setup count (E2)", () => {
    expect(
      discoveryEvalEstimate({
        timeframe: "1m",
        population: 20,
        generations: 15,
        numSeeds: 1,
        crossWindowMultiplier: 1,
        gridCount: 8,
      }),
    ).toStrictEqual({ totalEvals: 2400, parallelSec: 4500 });
  });

  it("multiplies by every factor of the search (N1)", () => {
    expect(
      discoveryEvalEstimate({
        timeframe: "1m",
        population: 20,
        generations: 15,
        numSeeds: 3,
        crossWindowMultiplier: 2,
        gridCount: 8,
      }),
    ).toStrictEqual({ totalEvals: 14400, parallelSec: 27000 });
  });

  it("divides evals across EVAL_CORES cores at SEC_PER_EVAL seconds each", () => {
    expect(EVAL_CORES).toBe(8);
    expect(SEC_PER_EVAL).toBe(15);
    expect(
      discoveryEvalEstimate({
        timeframe: "1m",
        population: 1,
        generations: 1,
        numSeeds: 1,
        crossWindowMultiplier: 1,
        gridCount: 9,
      }),
    ).toStrictEqual({ totalEvals: 9, parallelSec: 30 });
  });

  it("uses the measured 4h wall seconds per evaluation (U4)", () => {
    expect(
      discoveryEvalEstimate({
        timeframe: "4h",
        population: 50,
        generations: 20,
        numSeeds: 1,
        crossWindowMultiplier: 1,
        gridCount: 1,
      }),
    ).toStrictEqual({ totalEvals: 1000, parallelSec: 800 });
  });

  it("uses the measured 1d wall seconds per evaluation (U4)", () => {
    expect(
      discoveryEvalEstimate({
        timeframe: "1d",
        population: 50,
        generations: 20,
        numSeeds: 1,
        crossWindowMultiplier: 1,
        gridCount: 1,
      }),
    ).toStrictEqual({ totalEvals: 1000, parallelSec: 400 });
  });

  it("rounds a measured-timeframe fraction of an evaluation up", () => {
    expect(
      discoveryEvalEstimate({
        timeframe: "4h",
        population: 1,
        generations: 1,
        numSeeds: 1,
        crossWindowMultiplier: 1,
        gridCount: 1,
      }),
    ).toStrictEqual({ totalEvals: 1, parallelSec: 1 });
    expect(
      discoveryEvalEstimate({
        timeframe: "1d",
        population: 1,
        generations: 1,
        numSeeds: 1,
        crossWindowMultiplier: 1,
        gridCount: 1,
      }),
    ).toStrictEqual({ totalEvals: 1, parallelSec: 1 });
  });

  it("keeps every other timeframe on the core-spread arithmetic (U4)", () => {
    expect(
      discoveryEvalEstimate({
        timeframe: "1m",
        population: 20,
        generations: 15,
        numSeeds: 1,
        crossWindowMultiplier: 1,
        gridCount: 1,
      }),
    ).toStrictEqual({ totalEvals: 300, parallelSec: 570 });
    expect(
      discoveryEvalEstimate({
        timeframe: "1h",
        population: 20,
        generations: 15,
        numSeeds: 1,
        crossWindowMultiplier: 1,
        gridCount: 1,
      }),
    ).toStrictEqual({ totalEvals: 300, parallelSec: 570 });
  });
});

describe("secPerEvalFor", () => {
  it("shows the measured whole-run seconds on 4h and 1d", () => {
    expect(secPerEvalFor("4h")).toBe(0.8);
    expect(secPerEvalFor("1d")).toBe(0.4);
  });

  it("shows the core-spread seconds on every other timeframe", () => {
    expect(secPerEvalFor("1m")).toBe(15);
    expect(secPerEvalFor("1h")).toBe(15);
  });
});

describe("budgetFor", () => {
  it("recommends the P-2 rung 50x20 with convergence 10 on 4h (U1)", () => {
    expect(RECOMMENDED_BUDGET["4h"]).toStrictEqual({
      population: 50,
      generations: 20,
      convergenceWindow: 10,
    });
    expect(budgetFor("4h")).toStrictEqual({
      population: 50,
      generations: 20,
      convergenceWindow: 10,
    });
  });

  it("defaults every other timeframe to 20x15 with convergence 5 (U1)", () => {
    expect(DEFAULT_BUDGET).toStrictEqual({
      population: 20,
      generations: 15,
      convergenceWindow: 5,
    });
    for (const timeframe of ["1s", "5s", "1m", "5m", "15m", "1h", "1d"]) {
      expect(budgetFor(timeframe)).toStrictEqual(DEFAULT_BUDGET);
    }
  });

  it("defaults an unknown timeframe to the default budget", () => {
    expect(budgetFor("7h")).toStrictEqual(DEFAULT_BUDGET);
  });
});

describe("budgetOnTimeframeChange", () => {
  const recommended = { population: 50, generations: 20, convergenceWindow: 10 };

  it("adopts the next timeframe's budget when the current values are untouched (U3)", () => {
    expect(budgetOnTimeframeChange("4h", "1m", recommended)).toStrictEqual({
      population: 20,
      generations: 15,
      convergenceWindow: 5,
    });
  });

  it("adopts the 4h budget when switching from an untouched default (U3)", () => {
    expect(
      budgetOnTimeframeChange("1m", "4h", {
        population: 20,
        generations: 15,
        convergenceWindow: 5,
      }),
    ).toStrictEqual(recommended);
  });

  it("keeps a user-edited population (U3)", () => {
    expect(budgetOnTimeframeChange("4h", "1m", { ...recommended, population: 30 })).toStrictEqual({
      population: 30,
      generations: 20,
      convergenceWindow: 10,
    });
  });

  it("keeps a user edit in generations or in the convergence window (U3)", () => {
    expect(budgetOnTimeframeChange("4h", "1m", { ...recommended, generations: 8 })).toStrictEqual({
      population: 50,
      generations: 8,
      convergenceWindow: 10,
    });
    expect(
      budgetOnTimeframeChange("4h", "1m", { ...recommended, convergenceWindow: 5 }),
    ).toStrictEqual({
      population: 50,
      generations: 20,
      convergenceWindow: 5,
    });
  });
});

describe("gridSetupCount", () => {
  it("is the grid's cell count when the grid is on", () => {
    const grid: BankrollGridParse = {
      kind: "on",
      count: 8,
      payload: { modes: ["cross"], leverages: [5], max_margin_pcts: [0.25] },
    };
    expect(gridSetupCount(grid)).toBe(8);
  });

  it("is 1 when the grid is off or refused (E3)", () => {
    expect(gridSetupCount({ kind: "off" })).toBe(1);
    expect(gridSetupCount({ kind: "invalid", message: "nope" })).toBe(1);
  });

  it("feeds an invalid grid the same estimate as OFF (E3)", () => {
    const invalid: BankrollGridParse = { kind: "invalid", message: "nope" };
    expect(
      discoveryEvalEstimate({
        timeframe: "1m",
        population: 20,
        generations: 15,
        numSeeds: 1,
        crossWindowMultiplier: 1,
        gridCount: gridSetupCount(invalid),
      }),
    ).toStrictEqual({ totalEvals: 300, parallelSec: 570 });
  });
});

describe("evalBudget", () => {
  it("is 'ok' at and below 200 evaluations on an evaluation-judged timeframe", () => {
    expect(evalBudget(0, 0, "1m")).toBe("ok");
    expect(evalBudget(200, 0, "1m")).toBe("ok");
  });

  it("is 'warn' just above 200 and up to 500 evaluations", () => {
    expect(evalBudget(201, 0, "1m")).toBe("warn");
    expect(evalBudget(500, 0, "1m")).toBe("warn");
  });

  it("is 'danger' just above 500 evaluations", () => {
    expect(evalBudget(501, 0, "1m")).toBe("danger");
    expect(evalBudget(2400, 0, "1m")).toBe("danger");
  });

  it("judges 4h by wall seconds: 'ok' at and below 1200 s (U5)", () => {
    expect(evalBudget(1500, 1200, "4h")).toBe("ok");
    expect(evalBudget(0, 0, "4h")).toBe("ok");
  });

  it("judges 4h by wall seconds: 'warn' just above 1200 s up to 3600 s (U5)", () => {
    expect(evalBudget(1501, 1201, "4h")).toBe("warn");
    expect(evalBudget(4500, 3600, "4h")).toBe("warn");
  });

  it("judges 4h by wall seconds: 'danger' just above 3600 s (U5)", () => {
    expect(evalBudget(4501, 3601, "4h")).toBe("danger");
  });

  it("judges 1d by wall seconds too (U5)", () => {
    expect(evalBudget(3000, 1200, "1d")).toBe("ok");
    expect(evalBudget(3001, 1201, "1d")).toBe("warn");
    expect(evalBudget(9001, 3601, "1d")).toBe("danger");
  });

  it("keeps the evaluation thresholds on every other timeframe, whatever the seconds (U5)", () => {
    expect(evalBudget(200, 999999, "5m")).toBe("ok");
    expect(evalBudget(201, 0, "15m")).toBe("warn");
    expect(evalBudget(501, 0, "1h")).toBe("danger");
  });

  it("calls the 4h 50x20 budget 'ok' and only a larger run 'danger' (bead vibe-quant-wnqea)", () => {
    const oneSeed = discoveryEvalEstimate({
      timeframe: "4h",
      population: 50,
      generations: 20,
      numSeeds: 1,
      crossWindowMultiplier: 1,
      gridCount: 1,
    });
    expect(oneSeed).toStrictEqual({ totalEvals: 1000, parallelSec: 800 });
    expect(evalBudget(oneSeed.totalEvals, oneSeed.parallelSec, "4h")).toBe("ok");

    const fiveSeeds = discoveryEvalEstimate({
      timeframe: "4h",
      population: 50,
      generations: 20,
      numSeeds: 5,
      crossWindowMultiplier: 1,
      gridCount: 1,
    });
    expect(fiveSeeds).toStrictEqual({ totalEvals: 5000, parallelSec: 4000 });
    expect(evalBudget(fiveSeeds.totalEvals, fiveSeeds.parallelSec, "4h")).toBe("danger");
  });

  it("leaves the 1m 20x15 numbers and verdict unchanged (U5)", () => {
    const estimate = discoveryEvalEstimate({
      timeframe: "1m",
      population: 20,
      generations: 15,
      numSeeds: 1,
      crossWindowMultiplier: 1,
      gridCount: 1,
    });
    expect(estimate).toStrictEqual({ totalEvals: 300, parallelSec: 570 });
    expect(evalBudget(estimate.totalEvals, estimate.parallelSec, "1m")).toBe("warn");
  });
});

describe("dangerAdvice (U6)", () => {
  it("recommends the 4h budget 50x20", () => {
    expect(dangerAdvice("4h")).toBe("Recommended: pop=50 x gen=20 = 1000 evals.");
  });

  it("recommends the default budget 20x15 on every other timeframe", () => {
    expect(dangerAdvice("1m")).toBe("Recommended: pop=20 x gen=15 = 300 evals.");
    expect(dangerAdvice("1d")).toBe("Recommended: pop=20 x gen=15 = 300 evals.");
  });
});
