/**
 * The discovery runtime estimate (bead vibe-quant-fq0pi): the arithmetic behind the
 * form's "Total evaluations" / "Parallel (N cores)" hint, extracted so a test can pin
 * it without a DOM (this repo has no jsdom).
 *
 * A bankroll grid runs the whole search once per cell, so an ON grid multiplies the
 * evaluation count by its setup count; an OFF or REFUSED grid is one setup (a refused
 * grid never launches, and its own message is what the form shows instead).
 *
 * P-2 (docs/discovery-journal.md, 2026-10-19) measured the whole-run wall seconds per
 * evaluation at 0.673-0.810 on 4h and 0.227-0.362 on 1d, and only 50x20 recovered the
 * planted S 4-5 edges as champions on 4h. So those two timeframes estimate from the
 * measured figure and judge a run by wall seconds; every other timeframe keeps the
 * core-spread evaluation thresholds. The 4h form default is that P-2 rung (bead
 * vibe-quant-wnqea).
 */
import type { BankrollGridParse } from "./discoveryLaunchBody";

/** The cores the estimate assumes (the hint says "Parallel (8 cores)"). */
export const EVAL_CORES = 8;

/** The assumed seconds one evaluation takes. */
export const SEC_PER_EVAL = 15;

/** The warn / danger thresholds of the runtime hint, in evaluations. */
export const WARN_EVALS = 200;
export const DANGER_EVALS = 500;

/** The warn / danger thresholds of the runtime hint, in wall seconds (MEASURED timeframes). */
const WARN_SEC = 1200;
const DANGER_SEC = 3600;

/** A discovery search budget: population x generations, plus the convergence window. */
export interface DiscoveryBudget {
  population: number;
  generations: number;
  convergenceWindow: number;
}

/** The budget every timeframe except 4h starts from (the CLI/API default). */
export const DEFAULT_BUDGET: DiscoveryBudget = {
  population: 20,
  generations: 15,
  convergenceWindow: 5,
};

/**
 * The P-2 result: on 4h only 50x20 recovered planted S 4-5 edges as champions, and
 * every run used convergence 10 and reached generation 20 (user-approved 2026-10-19).
 */
export const RECOMMENDED_BUDGET: Readonly<Record<string, DiscoveryBudget>> = {
  "4h": { population: 50, generations: 20, convergenceWindow: 10 },
};

/**
 * The measured whole-run wall seconds one evaluation takes on a timeframe (P-2):
 * 4h 0.673-0.810, 1d 0.227-0.362. Absent means the core-spread estimate applies.
 */
export const MEASURED_SEC_PER_EVAL: Readonly<Record<string, number>> = {
  "4h": 0.8,
  "1d": 0.4,
};

/** The recommended budget of a timeframe, or {@link DEFAULT_BUDGET}. */
export function budgetFor(timeframe: string): DiscoveryBudget {
  return RECOMMENDED_BUDGET[timeframe] ?? DEFAULT_BUDGET;
}

/**
 * The budget of the timeframe a form switches TO, unless the user edited the current
 * one. "Untouched" means all three fields still equal the previous timeframe's budget;
 * any edit survives the switch.
 */
export function budgetOnTimeframeChange(
  prevTimeframe: string,
  nextTimeframe: string,
  current: DiscoveryBudget,
): DiscoveryBudget {
  const prev = budgetFor(prevTimeframe);
  const untouched =
    current.population === prev.population &&
    current.generations === prev.generations &&
    current.convergenceWindow === prev.convergenceWindow;
  return untouched ? budgetFor(nextTimeframe) : current;
}

/** The seconds per evaluation the hint's "~Xs/eval" label shows for a timeframe. */
export function secPerEvalFor(timeframe: string): number {
  return MEASURED_SEC_PER_EVAL[timeframe] ?? SEC_PER_EVAL;
}

/** The inputs of the form's runtime hint. */
export interface DiscoveryEvalEstimateInput {
  timeframe: string;
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
 * run). A timeframe with a MEASURED wall seconds per evaluation multiplies by it,
 * rounded up; every other spreads the evaluations over {@link EVAL_CORES} cores,
 * rounded up, at {@link SEC_PER_EVAL} seconds each.
 */
export function discoveryEvalEstimate(input: DiscoveryEvalEstimateInput): DiscoveryEvalEstimate {
  const totalEvals =
    input.population *
    input.generations *
    input.numSeeds *
    input.crossWindowMultiplier *
    input.gridCount;
  const measured = MEASURED_SEC_PER_EVAL[input.timeframe];
  return {
    totalEvals,
    parallelSec:
      measured === undefined
        ? Math.ceil(totalEvals / EVAL_CORES) * SEC_PER_EVAL
        : Math.ceil(totalEvals * measured),
  };
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

/**
 * The verdict of a run. A MEASURED timeframe is judged by its wall seconds: danger
 * above {@link DANGER_SEC}, warn above {@link WARN_SEC}. Every other timeframe keeps
 * the evaluation thresholds: danger above {@link DANGER_EVALS}, warn above
 * {@link WARN_EVALS}.
 */
export function evalBudget(totalEvals: number, parallelSec: number, timeframe: string): EvalBudget {
  if (MEASURED_SEC_PER_EVAL[timeframe] !== undefined) {
    if (parallelSec > DANGER_SEC) return "danger";
    if (parallelSec > WARN_SEC) return "warn";
    return "ok";
  }
  if (totalEvals > DANGER_EVALS) return "danger";
  if (totalEvals > WARN_EVALS) return "warn";
  return "ok";
}

/** The heavy-run hint's recommendation: the timeframe's own budget, never a fixed one. */
export function dangerAdvice(timeframe: string): string {
  const budget = budgetFor(timeframe);
  const evals = budget.population * budget.generations;
  return `Recommended: pop=${budget.population} x gen=${budget.generations} = ${evals} evals.`;
}
