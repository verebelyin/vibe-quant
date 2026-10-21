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
import { parseMaxMarginPct } from "./marginLimit";

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
  /** Max exposure in multiples of balance (validated by sizingProblems). */
  maxPositionPct: number;
  /** Risk per trade in PERCENT (validated by sizingProblems). */
  riskPct: number;
  /**
   * Max margin in use as a PERCENT of equity ("50" is half of equity), or "" for no
   * limit at all (validated by parseMaxMarginPct, T6).
   */
  maxMarginPct: string;
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
  const margin = parseMaxMarginPct(form.maxMarginPct);
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
      // The margin limit (T6) is stored like the other run-level sizing keys: only
      // when the field holds a value, so an empty field leaves the run unchanged.
      ...(margin.kind === "on" && { max_margin_pct: margin.fraction }),
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
