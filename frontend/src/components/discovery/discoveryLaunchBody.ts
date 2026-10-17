/**
 * The discovery launch body (vibe-quant-kdy5i).
 *
 * The exact object the launch form submits, extracted from DiscoveryConfig's
 * submit handler so a test can pin it without a DOM (this repo has no jsdom) —
 * including the liquidation/leverage spread (vibe-quant-6wq6w, contract C6):
 * a default form sends neither key, so stored run parameters stay unchanged.
 */
import type { DiscoveryLaunchRequest } from "@/api/generated/models";
import { parseMaxMarginPct } from "@/components/backtest/marginLimit";
import { type LiquidationMode, leveragePayload, liquidationPayload } from "@/lib/liquidation";
import { riskPctToFraction, sizingPayload } from "@/lib/sizing";

export type DiscoveryDirection = "long" | "short" | "both" | "random";
export type DiscoverySymbolAgg = "portfolio" | "worst";

/** Raw form state of DiscoveryConfig, one field per input the launch reads. */
export interface DiscoveryLaunchForm {
  population: number;
  generations: number;
  mutationRate: number;
  crossoverRate: number;
  eliteCount: number;
  tournamentSize: number;
  convergenceWindow: number;
  symbols: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  direction: DiscoveryDirection;
  evalWindows: number;
  trainTestSplit: number;
  numSeeds: number;
  symbolAgg: DiscoverySymbolAgg;
  wfaOosStepDays: number;
  wfaMinConsistency: number;
  crossWindowMonths: string;
  crossWindowMinSharpe: number;
  immigrantFraction: number;
  entropyThreshold: number;
  crowdingEnabled: boolean;
  seedRunId: string;
  rngSeed: string;
  noBootstrapCi: boolean;
  bootstrapMinSharpe: string;
  bnbDiscount: boolean;
  liquidationMode: LiquidationMode;
  leverage: number;
  /** Max exposure in multiples of equity (validated by sizingProblems). */
  maxPositionPct: number;
  /** Risk per trade in PERCENT (validated by sizingProblems). */
  riskPct: number;
  /**
   * Max margin in use as a PERCENT of equity ("50" is half of equity), or "" for no
   * limit at all (validated by parseMaxMarginPct, T6). Discovery takes the limit as a
   * top-level request field, not a run parameter.
   */
  maxMarginPct: string;
  selectedIndicators: string[];
}

/** Positive whole month offsets from the comma-separated cross-window input. */
export function parseCrossWindowMonths(raw: string): number[] {
  return raw
    .split(",")
    .map((s) => Number.parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n) && n > 0);
}

/** The launch request body for a form; optional keys appear only when set. */
export function buildDiscoveryLaunchBody(form: DiscoveryLaunchForm): DiscoveryLaunchRequest {
  const symbols = form.symbols
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const crossWindowMonths = parseCrossWindowMonths(form.crossWindowMonths);
  const margin = parseMaxMarginPct(form.maxMarginPct);

  return {
    population: form.population,
    generations: form.generations,
    mutation_rate: form.mutationRate,
    crossover_rate: form.crossoverRate,
    elite_count: form.eliteCount,
    tournament_size: form.tournamentSize,
    convergence_generations: form.convergenceWindow,
    symbols,
    timeframes: [form.timeframe],
    indicator_pool: form.selectedIndicators.length > 0 ? form.selectedIndicators : null,
    ...(form.startDate && { start_date: form.startDate }),
    ...(form.endDate && { end_date: form.endDate }),
    ...(form.direction !== "random" && { direction: form.direction }),
    eval_windows: form.evalWindows,
    train_test_split: form.trainTestSplit,
    num_seeds: form.numSeeds,
    ...(form.symbolAgg === "worst" && { symbol_agg: form.symbolAgg }),
    wfa_oos_step_days: form.wfaOosStepDays,
    wfa_min_consistency: form.wfaMinConsistency,
    ...(crossWindowMonths.length > 0 && {
      cross_window_months: crossWindowMonths,
      cross_window_min_sharpe: form.crossWindowMinSharpe,
    }),
    immigrant_fraction: form.immigrantFraction,
    entropy_threshold: form.entropyThreshold,
    crowding_enabled: form.crowdingEnabled,
    ...(form.seedRunId.trim() !== "" &&
      Number.isFinite(Number(form.seedRunId)) && {
        seed_run_id: Number(form.seedRunId),
      }),
    ...(form.rngSeed.trim() !== "" &&
      Number.isInteger(Number(form.rngSeed)) && { seed: Number(form.rngSeed) }),
    ...(form.noBootstrapCi && { no_bootstrap_ci: true }),
    ...(form.bootstrapMinSharpe.trim() !== "" &&
      Number.isFinite(Number(form.bootstrapMinSharpe)) && {
        bootstrap_min_sharpe: Number(form.bootstrapMinSharpe),
      }),
    ...(form.bnbDiscount && { bnb_discount: true }),
    ...liquidationPayload(form.liquidationMode),
    ...leveragePayload(form.liquidationMode, form.leverage),
    // Non-default sizing keys only, so a default launch body is unchanged.
    ...sizingPayload(form.maxPositionPct, riskPctToFraction(form.riskPct)),
    // The margin limit (T6) is a top-level discovery field, sent only when set.
    ...(margin.kind === "on" && { max_margin_pct: margin.fraction }),
  } as DiscoveryLaunchRequest;
}
