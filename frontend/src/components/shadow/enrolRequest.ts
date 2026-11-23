import type { BacktestRunResponse } from "@/api/generated/models/backtestRunResponse";
import type { FrozenDslResponse } from "@/api/generated/models/frozenDslResponse";
import type { ShadowEnrolRequest } from "@/api/generated/models/shadowEnrolRequest";
import type { ShadowStrategy } from "@/api/generated/models/shadowStrategy";

/** The API's minimum (after strip); the form only counts, the API refuses (422). */
export const PREREG_MIN_CHARS = 40;
export const PREREG_HINT =
  "What would make you stop believing this strategy? Written before any forward day is seen.";

/** The form as typed; numbers stay strings until the request is built. */
export interface EnrolForm {
  name: string;
  preRegistration: string;
  maxDdPercent: string;
  warmupDays: string;
}

/** Only a completed validation run has a frozen DSL to track. */
export function shadowEligible(run: Pick<BacktestRunResponse, "run_mode" | "status">): boolean {
  return run.run_mode === "validation" && run.status === "completed";
}

export function defaultEnrolForm(strategyName: string, runId: number): EnrolForm {
  return {
    name: `${strategyName} (run ${runId})`,
    preRegistration: "",
    maxDdPercent: "20",
    warmupDays: "180",
  };
}

/** Characters still missing before the API accepts the pre-registration (0 = enough). */
export function preRegistrationShortBy(text: string): number {
  return Math.max(0, PREREG_MIN_CHARS - text.trim().length);
}

/**
 * The enrol body for a validation run: the DSL the run executed (frozen-dsl endpoint),
 * the run's own symbols and parameters, the drawdown limit as a fraction.
 */
export function buildEnrolRequest(
  run: Pick<BacktestRunResponse, "id" | "symbols" | "parameters">,
  frozen: Pick<FrozenDslResponse, "strategy_id" | "dsl_json">,
  form: EnrolForm,
): ShadowEnrolRequest {
  return {
    name: form.name,
    source_kind: "strategy",
    dsl_json: frozen.dsl_json,
    symbols: [...run.symbols],
    parameters: run.parameters ?? {},
    pre_registration: form.preRegistration,
    max_dd_limit: Number(form.maxDdPercent) / 100,
    warmup_days: Number(form.warmupDays),
    source_ref: { run_id: run.id },
    strategy_id: frozen.strategy_id,
  };
}

/** The live (not retired), real row enrolled from this run, if any. */
export function trackedRowForRun<
  R extends Pick<ShadowStrategy, "state" | "is_test" | "source_ref">,
>(rows: readonly R[], runId: number): R | undefined {
  return rows.find((r) => r.source_ref.run_id === runId && r.state !== "retired" && !r.is_test);
}
