import type { ShadowStrategy } from "@/api/generated/models/shadowStrategy";

// Mirrors vibe_quant/shadow/estat.py: E_PASS = 1 / (ALPHA / SLOTS) = 100, family F1 = 5 slots.
export const E_PASS = 100;
export const SLOTS = 5;

export const HONEST_LIMIT =
  "A strategy with no edge can stay 'tracking' for years: at a forward Sharpe of 0 it is dropped only after 3,951 days (10.8 years), unless it loses clearly (Sharpe below −1.73 after 200 days) or breaks its drawdown limit.";
export const POWER_LINE =
  "A true Sharpe of 2 passes within 2 years about 44 % of the time (3 years: 67 %).";
// Mirrors vibe_quant/shadow/tracker.py: PROPOSALS_ENABLED (on since 2026-11-23); test rows never propose.
export const PROPOSED =
  "evidence bar met — proposed for paper; the overfit check, the risk check and your approval decide, nothing starts by itself";
export const TEST_NOT_PROPOSED = "evidence bar met — not proposed (test row)";

export type ShadowSentenceRow = Pick<
  ShadowStrategy,
  | "state"
  | "state_reason"
  | "state_changed_at"
  | "verdict_day"
  | "first_day"
  | "slot"
  | "is_test"
  | "max_dd_limit"
  | "proposal_stale"
  | "engine_epoch_last"
  | "stats"
>;

type Stats = NonNullable<ShadowStrategy["stats"]>;

function pct(x: number): string {
  const v = Number((x * 100).toFixed(1)) + 0; // + 0 turns -0 into 0: never "-0.0 %"
  return `${v.toFixed(1)} %`;
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function weeks(blocks: number): string {
  return blocks === 0 ? "no full week yet" : plural(blocks, "full week");
}

function evidence(e: number): string {
  if (e >= 1e6) return e.toExponential(1);
  // Below the bar, round down: 99.96 must not read as "100.0 of 100".
  return e < E_PASS ? (Math.floor(e * 10) / 10).toFixed(1) : e.toFixed(1);
}

function slotText(row: ShadowSentenceRow): string {
  if (row.is_test) return "test row, no slot";
  if (row.slot === null) return "no evidence slot left; a new family needs your OK";
  return `slot ${row.slot} of ${SLOTS}`;
}

function numbers(row: ShadowSentenceRow, s: Stats): string {
  const ret = `${s.total_return >= 0 ? "+" : ""}${pct(s.total_return)}`;
  const sharpe =
    s.sharpe_ann === null ? "n/a (no variation)" : `${s.sharpe_ann.toFixed(1)} (365-day basis)`;
  const limit = `${Number((row.max_dd_limit * 100).toFixed(1))} %`;
  return (
    `Forward return ${ret}, Sharpe ${sharpe}, worst drawdown ${pct(s.max_dd)} on daily closes (limit ${limit}). ` +
    `Evidence ${evidence(s.e_value)} of ${E_PASS} needed (${slotText(row)}).`
  );
}

function headline(row: ShadowSentenceRow): string {
  const day = row.verdict_day ?? row.state_changed_at?.slice(0, 10) ?? "?";
  const reason = row.state_reason ?? "no reason recorded";
  switch (row.state) {
    case "passed":
      return `Passed on ${day}: ${row.is_test ? TEST_NOT_PROPOSED : PROPOSED}.`;
    case "futile":
      return `Dropped on ${day} by the futility rule: ${reason}.`;
    case "retired":
      return `Retired on ${day}: ${reason}. Record kept; re-enrol to start a fresh clock.`;
    case "stalled":
      return `Stalled: ${reason}.`;
    default:
      return "";
  }
}

/** One plain-language status paragraph for a shadow row; numbers come from the API as-is. */
export function shadowSentence(row: ShadowSentenceRow): string {
  const prefix = row.is_test ? "[TEST] " : "";
  const s = row.stats;
  const head = headline(row);
  if (s === null || s.days === 0) {
    const wait = head
      ? `${head} No forward day scored.`
      : `Tracking starts on ${row.first_day}; no forward day scored yet.`;
    return prefix + wait;
  }
  const span = `${plural(s.days, "day")} (${weeks(s.blocks)})`;
  const parts = [head ? `${head} Tracked ${span}.` : `Tracking ${span}.`, numbers(row, s)];
  if (row.proposal_stale) {
    parts.push(
      `Evidence measured on engine epoch ${row.engine_epoch_last}; a paper decision must re-check it under the current engine.`,
    );
  }
  return prefix + parts.join(" ");
}
