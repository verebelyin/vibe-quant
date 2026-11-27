/**
 * Results-page line for the opt-in timing null (`--null-gate`, vibe-quant-clmmc.14): did the
 * champion's as-traded return beat random re-timings of its own trades? The pass bar and
 * exposure cap come from the run's `notes.timing_null_gate`, never from UI constants; the
 * verdict mirrors the backend rule (`timing_null_decision`: percentile >= pct; a record with
 * exposure above the cap is refused: timing untestable, not certified).
 */

function finite(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function pct2(fraction: number): string {
  return `${(fraction * 100).toFixed(2)}%`;
}

/** A policy bar: 0.95 -> "95%", 0.975 -> "97.5%" (no padded zeros). */
function pctBar(fraction: number): string {
  return `${Number((fraction * 100).toFixed(2))}%`;
}

const UNREADABLE = "record unreadable";

/** The backend's `{unreadable: detail}` marker (a policy promote refuses), else null. */
export function timingNullPolicyProblem(policy: unknown): string | null {
  if (policy == null || typeof policy !== "object") return null;
  const detail = (policy as Record<string, unknown>).unreadable;
  return typeof detail === "string" ? `policy unreadable (${detail})` : null;
}

/** One champion's line, shown under a "Timing null" header (so no prefix here);
 * null when the run had no `--null-gate` (no policy, or not enforced). */
export function timingNullLine(record: unknown, policy: unknown): string | null {
  if (policy == null || typeof policy !== "object") return null;
  const gate = policy as Record<string, unknown>;
  if (gate.enforced !== true) return null;
  const bar = finite(gate.pct);
  const cap = finite(gate.exposure_max);
  if (record == null || typeof record !== "object" || bar == null || cap == null) return UNREADABLE;
  const r = record as Record<string, unknown>;
  if (r.fallback != null) {
    const exposure = finite(r.exposure);
    const shown = exposure != null ? pct2(exposure) : "--";
    return (
      `timing untestable, not certified - in a position ${shown} of the time ` +
      `(over ${pctBar(cap)}), so re-timing its trades changes almost nothing`
    );
  }
  const percentile = finite(r.percentile);
  if (percentile == null || percentile < 0 || percentile > 1) return UNREADABLE;
  const draws = finite(r.draws);
  const verdict = percentile >= bar ? "passed" : "failed";
  return (
    `${verdict} - real return beat ${pct2(percentile)} of ` +
    `${draws ?? "?"} random re-timings of its own trades (needs ${pctBar(bar)})`
  );
}
