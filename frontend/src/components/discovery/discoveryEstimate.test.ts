/**
 * Tests for the discovery runtime estimate (vibe-quant-fq0pi).
 *
 * The Discovery form's "Total evaluations" / "Parallel (8 cores)" hint must count
 * the bankroll grid: a grid of N setups runs N times the search, so the number the
 * operator sees before launching is population * generations * seeds * windows * N.
 * These are pure-function tests (this repo's vitest is node-only, no jsdom).
 */
import { describe, expect, it } from "vitest";
import {
  discoveryEvalEstimate,
  EVAL_CORES,
  evalBudget,
  gridSetupCount,
  SEC_PER_EVAL,
} from "@/components/discovery/discoveryEstimate";
import type { BankrollGridParse } from "@/components/discovery/discoveryLaunchBody";

describe("discoveryEvalEstimate", () => {
  it("counts a grid-OFF search as one setup (E1)", () => {
    expect(
      discoveryEvalEstimate({
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
        population: 1,
        generations: 1,
        numSeeds: 1,
        crossWindowMultiplier: 1,
        gridCount: 9,
      }),
    ).toStrictEqual({ totalEvals: 9, parallelSec: 30 });
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
  it("is 'ok' at and below 200 evaluations", () => {
    expect(evalBudget(0)).toBe("ok");
    expect(evalBudget(200)).toBe("ok");
  });

  it("is 'warn' just above 200 and up to 500", () => {
    expect(evalBudget(201)).toBe("warn");
    expect(evalBudget(500)).toBe("warn");
  });

  it("is 'danger' just above 500", () => {
    expect(evalBudget(501)).toBe("danger");
    expect(evalBudget(2400)).toBe("danger");
  });
});
