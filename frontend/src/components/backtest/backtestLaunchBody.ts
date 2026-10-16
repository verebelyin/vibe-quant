/**
 * The backtest launch body (vibe-quant-2gyk1.2, S7a).
 *
 * The exact object BacktestLaunchForm submits, extracted so a test can pin it
 * without a DOM (this repo has no jsdom). Sizing keys ride in `parameters` and
 * only when moved off the default, so a default launch body is unchanged.
 */
import type {
  BacktestLaunchRequest,
  BacktestLaunchRequestEntryOrder,
} from "@/api/generated/models";
import { type LiquidationMode, liquidationPayload } from "@/lib/liquidation";
import { riskPctToFraction, sizingPayload } from "@/lib/sizing";

export type BacktestLaunchMode = "screening" | "validation";

/** Raw form state of BacktestLaunchForm, one field per input the launch reads. */
export interface BacktestLaunchForm {
  strategyId: number;
  symbols: string[];
  timeframe: string;
  startDate: string;
  endDate: string;
  initialBalance: number;
  leverage: number;
  /** Max exposure in multiples of equity (validated by sizingProblems). */
  maxPositionPct: number;
  /** Risk per trade in PERCENT (validated by sizingProblems). */
  riskPct: number;
  liquidationMode: LiquidationMode;
  /** The already-expanded sweep, or null when the sweep is off/empty. */
  sweep: Record<string, number[]> | null;
  mode: BacktestLaunchMode;
  latencyPreset: string;
  bnbDiscount: boolean;
  entryOrder: BacktestLaunchRequestEntryOrder;
  makerOffsetTicks: number;
  makerEntryTimeoutBars: number;
}

/** The launch request body for a form; optional keys appear only when set. */
export function buildBacktestLaunchBody(form: BacktestLaunchForm): BacktestLaunchRequest {
  return {
    strategy_id: form.strategyId,
    symbols: form.symbols,
    timeframe: form.timeframe,
    start_date: form.startDate,
    end_date: form.endDate,
    parameters: {
      initial_balance: form.initialBalance,
      leverage: form.leverage,
      ...sizingPayload(form.maxPositionPct, riskPctToFraction(form.riskPct)),
      ...(form.sweep && { sweep: form.sweep }),
    },
    // Off sends no liquidation_mode, so stored parameters stay identical.
    ...liquidationPayload(form.liquidationMode),
    // The BNB fee discount applies to screening and validation alike (t1ee0);
    // unchecked sends no key, so stored screening parameters stay identical.
    ...(form.bnbDiscount && { bnb_discount: true }),
    ...(form.mode === "validation" && {
      latency_preset:
        form.latencyPreset && form.latencyPreset !== "__none__" ? form.latencyPreset : null,
      // Post-only maker entries are validation-only and opt-in; the three
      // fields reach the run only when post_only is chosen (D3-6).
      ...(form.entryOrder === "post_only" && {
        entry_order: form.entryOrder,
        maker_offset_ticks: form.makerOffsetTicks,
        maker_entry_timeout_bars: form.makerEntryTimeoutBars,
      }),
    }),
  };
}
