/**
 * The Validate-a-sweep-row body (bead vibe-quant-2gyk1.2 + vibe-quant-w2nge,
 * S7b): the ONE pure builder behind the results-page Validate button.
 *
 * A sweep row holds the swept strategy parameters; the PARENT run holds the
 * run-level environment (initial balance, leverage, liquidation mode, bnb
 * discount, fee tier, sizing). A validation run must reproduce that
 * environment, so the body stacks the parent's settings on top of the swept
 * parameters. It never carries the parent's `sweep` grid spec: the validation
 * runner must not re-sweep (U9). A screening parent that used a DSL override
 * cannot be reproduced and is refused (U9c); unreadable parent sizing is
 * refused with the same Python text the chip shows (U12), so the caller can
 * toast it and send no request.
 */
import type { BacktestLaunchRequest } from "@/api/generated/models/backtestLaunchRequest";
import type { BacktestRunResponse } from "@/api/generated/models/backtestRunResponse";
import type { SweepResultResponse } from "@/api/generated/models/sweepResultResponse";
import { runSizing, sizingPayload } from "@/lib/sizing";

export type SweepValidationBody =
  | { ok: true; body: BacktestLaunchRequest }
  | { ok: false; error: string };

const ON_MODES = ["isolated", "cross"] as const;

/** The parent's liquidation mode when it is an on mode ("isolated"/"cross"). */
function onMode(value: unknown): "isolated" | "cross" | null {
  return typeof value === "string" && (ON_MODES as readonly string[]).includes(value)
    ? (value as "isolated" | "cross")
    : null;
}

/** The run-level keys forwarded verbatim when the parent recorded them (U9b). */
const CARRIED_KEYS = ["initial_balance", "leverage", "fee_tier"] as const;

export function sweepValidationBody(
  run: BacktestRunResponse,
  sweep: SweepResultResponse,
): SweepValidationBody {
  const parent = run.parameters ?? null;

  if (parent?.dsl_override != null) {
    return {
      ok: false,
      error:
        "This screening run used a DSL override; validation cannot reproduce it. Save the override as a strategy and validate that.",
    };
  }

  const sizing = runSizing(parent);
  if ("invalid" in sizing) {
    return { ok: false, error: `Parent run sizing is invalid: ${sizing.invalid}` };
  }

  const carried: Record<string, unknown> = {};
  if (parent != null) {
    for (const key of CARRIED_KEYS) {
      if (Object.hasOwn(parent, key)) carried[key] = parent[key];
    }
  }
  // Only the keys a run moved off the sizing default, so a default parent
  // carries nothing (U10/U11).
  Object.assign(carried, sizingPayload(sizing.maxPositionPct, sizing.riskPerTrade));

  const body: BacktestLaunchRequest = {
    strategy_id: run.strategy_id ?? 0,
    symbols: run.symbols,
    timeframe: run.timeframe ?? "",
    start_date: run.start_date ?? "",
    end_date: run.end_date ?? "",
    parameters: { ...sweep.parameters, ...carried },
  };

  const liquidationMode = onMode(parent?.liquidation_mode);
  if (liquidationMode != null) body.liquidation_mode = liquidationMode;
  if (parent?.bnb_discount === true) body.bnb_discount = true;

  return { ok: true, body };
}
