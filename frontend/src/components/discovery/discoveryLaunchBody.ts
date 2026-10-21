/**
 * The discovery launch body (vibe-quant-kdy5i).
 *
 * The exact object the launch form submits, extracted from DiscoveryConfig's
 * submit handler so a test can pin it without a DOM (this repo has no jsdom) —
 * including the liquidation/leverage spread (vibe-quant-6wq6w, contract C6):
 * a default form sends neither key, so stored run parameters stay unchanged.
 */
import type { DiscoveryLaunchRequest } from "@/api/generated/models";
import { MAX_MARGIN_PCT, parseMaxMarginPct } from "@/components/backtest/marginLimit";
import {
  type LiquidationMode,
  leveragePayload,
  liquidationPayload,
  MAX_LEVERAGE,
  MIN_LEVERAGE,
} from "@/lib/liquidation";
import { riskPctToFraction, sizingPayload } from "@/lib/sizing";

export type DiscoveryDirection = "long" | "short" | "both" | "random";
export type DiscoverySymbolAgg = "portfolio" | "worst";

/** The modes a bankroll grid may run, in cell order (mirrors `bankroll_grid.GRID_MODES`). */
export const BANKROLL_GRID_MODES: ReadonlyArray<BankrollGridMode> = ["isolated", "cross"];

/** The most cells one bankroll grid may have (mirrors `bankroll_grid.MAX_CELLS`). */
export const MAX_BANKROLL_CELLS = 12;

export type BankrollGridMode = "isolated" | "cross";

/**
 * The opt-in bankroll-grid section of the launch form (T8/T9, bead vibe-quant-t8nrv): the
 * margin modes to search, and comma-separated leverage / cushion lists. Empty lists are a
 * problem when the section is on, never a silent skip.
 */
export interface BankrollGridForm {
  enabled: boolean;
  modes: BankrollGridMode[];
  /** Comma-separated whole leverages, e.g. "1, 5, 10". */
  leverages: string;
  /** Comma-separated cushions in PERCENT, e.g. "25, 50"; a fractional one uses a dot ("25.5"). */
  maxMarginPcts: string;
}

/** The `bankroll_grid` request payload: the lists the API re-validates and expands. */
export interface BankrollGridPayload {
  modes: BankrollGridMode[];
  leverages: number[];
  max_margin_pcts: number[];
}

/** A parsed grid section: off, a valid grid (with its cell count), or the reason it is refused. */
export type BankrollGridParse =
  | { kind: "off" }
  | { kind: "on"; payload: BankrollGridPayload; count: number }
  | { kind: "invalid"; message: string };

/** The one message for a bare comma inside a grid list token (chief ruling T9-R1). */
function ambiguousComma(token: string): string {
  return `Ambiguous value '${token}': write 25.5 for one value or '25, 50' for two`;
}

/** The tokens of a grid list, plus the first token that carries an ambiguous comma. */
interface GridListTokens {
  tokens: string[];
  ambiguous: string | null;
}

/**
 * The list tokens of a grid input. Items are separated by a comma followed by whitespace, a
 * semicolon, or a whitespace run, so "25, 50", "25;50" and "25 50" each read as two items. A
 * comma BETWEEN TWO DIGITS with no space ("25,50", "25,5") is AMBIGUOUS — it could be one
 * decimal-comma value or two items — and is named rather than guessed (chief ruling T9-R1);
 * a fractional value in a list is written with a dot ("25.5"). A token with a comma at either
 * END ("25 ,50" -> ",50", "25,") is the same mistake written with the separator in the wrong
 * place, and is refused with the same message rather than read as a bare decimal (zeqit).
 */
function splitGridList(raw: string): GridListTokens {
  const tokens = raw
    .split(/,\s+|;|\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
  return {
    tokens,
    ambiguous: tokens.find((token) => /\d,\d/.test(token) || /^,|,$/.test(token)) ?? null,
  };
}

/**
 * The one rule for the grid section: no section -> off; otherwise the modes, whole leverages
 * (1..125) and cushions (0..100% of equity) must each name at least one value, deduplicate to
 * the API's sorted order, and make at most {@link MAX_BANKROLL_CELLS} cells. Mirrors
 * `vibe_quant.bankroll_grid.build_cells` so the form refuses what the API would 422.
 */
export function parseBankrollGrid(form: BankrollGridForm): BankrollGridParse {
  if (!form.enabled) return { kind: "off" };

  const modes = BANKROLL_GRID_MODES.filter((mode) => form.modes.includes(mode));
  if (modes.length === 0) {
    return {
      kind: "invalid",
      message: "Bankroll grid needs at least one margin mode (isolated or cross)",
    };
  }

  const leverageList = splitGridList(form.leverages);
  if (leverageList.ambiguous !== null) {
    return { kind: "invalid", message: ambiguousComma(leverageList.ambiguous) };
  }
  const leverageTokens = leverageList.tokens;
  if (leverageTokens.length === 0) {
    return { kind: "invalid", message: "Bankroll grid needs at least one leverage" };
  }
  const leverages: number[] = [];
  for (const token of leverageTokens) {
    const value = Number(token);
    if (!/^\d+$/.test(token) || value < MIN_LEVERAGE || value > MAX_LEVERAGE) {
      return {
        kind: "invalid",
        message: `Bankroll leverage must be a whole number between ${MIN_LEVERAGE} and ${MAX_LEVERAGE} (you entered ${token})`,
      };
    }
    if (!leverages.includes(value)) leverages.push(value);
  }
  leverages.sort((a, b) => a - b);

  const cushionList = splitGridList(form.maxMarginPcts);
  if (cushionList.ambiguous !== null) {
    return { kind: "invalid", message: ambiguousComma(cushionList.ambiguous) };
  }
  const cushionTokens = cushionList.tokens;
  if (cushionTokens.length === 0) {
    return { kind: "invalid", message: "Bankroll grid needs at least one cushion" };
  }
  const maxMarginPcts: number[] = [];
  for (const token of cushionTokens) {
    const parsed = parseMaxMarginPct(token);
    if (parsed.kind !== "on") {
      return {
        kind: "invalid",
        message: `Bankroll cushion must be more than 0% and at most ${MAX_MARGIN_PCT}% of equity (you entered ${token})`,
      };
    }
    if (!maxMarginPcts.includes(parsed.fraction)) maxMarginPcts.push(parsed.fraction);
  }
  maxMarginPcts.sort((a, b) => a - b);

  const count = modes.length * leverages.length * maxMarginPcts.length;
  if (count > MAX_BANKROLL_CELLS) {
    return {
      kind: "invalid",
      message: `Bankroll grid has ${count} setups; at most ${MAX_BANKROLL_CELLS} are allowed`,
    };
  }
  return { kind: "on", payload: { modes, leverages, max_margin_pcts: maxMarginPcts }, count };
}

/**
 * The leverage the launch form's run-level exposure-cap cross-check uses against a grid (T9,
 * bead vibe-quant-t8nrv): the run-level leverage with the grid OFF; the SMALLEST grid leverage
 * with the grid ON, because a grid checks the cap at every cell so the smallest cell binds
 * first; and `NaN` with the grid ON but invalid, so no cross-check fires and the grid's own
 * message is the one shown.
 */
export function gridCrossCheckLeverage(grid: BankrollGridParse, leverage: number): number {
  if (grid.kind === "off") return leverage;
  return grid.kind === "on" ? Math.min(...grid.payload.leverages) : Number.NaN;
}

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
  /** Max exposure in multiples of balance (validated by sizingProblems). */
  maxPositionPct: number;
  /** Risk per trade in PERCENT (validated by sizingProblems). */
  riskPct: number;
  /**
   * Max margin in use as a PERCENT of equity ("50" is half of equity), or "" for no
   * limit at all (validated by parseMaxMarginPct, T6). Discovery takes the limit as a
   * top-level request field, not a run parameter.
   */
  maxMarginPct: string;
  /** The opt-in bankroll-grid section (T8/T9): off unless the user turns it on. */
  bankrollGrid: BankrollGridForm;
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
  // A grid OWNS the margin mode, leverage and cushion: while the section is on, the run-level
  // fields are exclusive (the API 422s a mixture), so they are dropped from the body. The grid
  // key itself appears only when the section holds a valid grid.
  const grid = parseBankrollGrid(form.bankrollGrid);
  const gridOn = grid.kind === "on";
  const gridEnabled = form.bankrollGrid.enabled;

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
    ...(gridOn ? { bankroll_grid: grid.payload } : {}),
    ...(gridEnabled ? {} : liquidationPayload(form.liquidationMode)),
    ...(gridEnabled ? {} : leveragePayload(form.liquidationMode, form.leverage)),
    // Non-default sizing keys only, so a default launch body is unchanged.
    ...sizingPayload(form.maxPositionPct, riskPctToFraction(form.riskPct)),
    // The margin limit (T6) is a top-level discovery field, sent only when set; a grid
    // replaces it with the per-cell cushion, so it is dropped while the grid is on.
    ...(margin.kind === "on" && !gridEnabled ? { max_margin_pct: margin.fraction } : {}),
  } as DiscoveryLaunchRequest;
}
