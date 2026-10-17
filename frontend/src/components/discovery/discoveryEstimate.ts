/**
 * The discovery runtime estimate (bead vibe-quant-fq0pi): the arithmetic behind the
 * form's "Total evaluations" / "Parallel (N cores)" hint, extracted so a test can pin
 * it without a DOM (this repo has no jsdom).
 *
 * A bankroll grid runs the whole search once per cell, so an ON grid multiplies the
 * evaluation count by its setup count; an OFF or REFUSED grid is one setup (a refused
 * grid never launches, and its own message is what the form shows instead).
 */
import type { BankrollGridParse } from "./discoveryLaunchBody";

/** The cores the estimate assumes (the hint says "Parallel (8 cores)"). */
export const EVAL_CORES = 8;

/** The assumed seconds one evaluation takes. */
export const SEC_PER_EVAL = 15;

/** The warn / danger thresholds of the runtime hint, in evaluations. */
export const WARN_EVALS = 200;
export const DANGER_EVALS = 500;

/** The inputs of the form's runtime hint. */
export interface DiscoveryEvalEstimateInput {
  population: number;
  generations: number;
  numSeeds: number;
  crossWindowMultiplier: number;
  gridCount: number;
}

/** The hint's two numbers: total evaluations and their wall-clock seconds across the cores. */
export interface DiscoveryEvalEstimate {
  totalEvals: number;
  parallelSec: number;
}

/**
 * The form's runtime hint. The grid count multiplies the search (each setup is its own
 * run), and the parallel seconds spread the evaluations over {@link EVAL_CORES} cores,
 * rounded up, at {@link SEC_PER_EVAL} seconds each.
 */
export function discoveryEvalEstimate(input: DiscoveryEvalEstimateInput): DiscoveryEvalEstimate {
  const totalEvals =
    input.population *
    input.generations *
    input.numSeeds *
    input.crossWindowMultiplier *
    input.gridCount;
  return { totalEvals, parallelSec: Math.ceil(totalEvals / EVAL_CORES) * SEC_PER_EVAL };
}

/**
 * The number of times a bankroll grid runs the search: its cell count when the grid is
 * on, and one when it is off or refused.
 */
export function gridSetupCount(grid: BankrollGridParse): number {
  return grid.kind === "on" ? grid.count : 1;
}

/** The form's verdict on an evaluation count: comfortable, worth a warning, or heavy. */
export type EvalBudget = "ok" | "warn" | "danger";

/** The verdict of a multiplied total: danger above {@link DANGER_EVALS}, warn above {@link WARN_EVALS}. */
export function evalBudget(totalEvals: number): EvalBudget {
  if (totalEvals > DANGER_EVALS) return "danger";
  if (totalEvals > WARN_EVALS) return "warn";
  return "ok";
}
