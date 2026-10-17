import type { SystemStatusResponse } from "@/api/generated/models";

/**
 * The resolved `/api/system/status` payload the header widget renders from, or
 * `undefined` while the query has no response (loading / error).
 */
export interface KillSwitchStatusPayload {
  status: number;
  data?: SystemStatusResponse;
}

/**
 * How the header should present the system kill switch, and which controls it
 * shows.
 *
 * `unknown` is load-bearing: a failed or non-200 status query must never be
 * rendered as the disengaged (unlocked) look, or a broken backend would look
 * like a healthy trading system. It must also keep the Kill trigger: when the
 * state cannot be read, the operator still has to be able to halt trading
 * (POST /kill still cascades to live sessions).
 *
 * `KillSwitch.tsx` renders controls ONLY from `showKill` / `showUnlock` /
 * `showUnknownLabel`, so what is pinned here is what is on screen.
 */
export interface KillSwitchView {
  state: "engaged" | "disengaged" | "unknown";
  /** Operator halt reason; only set while `engaged`. */
  reason: string | null;
  /** Render the Kill dialog trigger (POST /kill). */
  showKill: boolean;
  /** Render the Unlock dialog trigger. */
  showUnlock: boolean;
  /** Render the red "Kill switch state unknown" label. */
  showUnknownLabel: boolean;
}

/** Header text for the fail-closed `unknown` state. */
export const KILL_SWITCH_UNKNOWN_LABEL = "Kill switch state unknown";

/**
 * Map a status query result to the widget view.
 *
 * `unknown` when the query is erroring, produced no response (initial load),
 * answered with a non-200 status, or carried no body; `engaged` only when the
 * backend says so.
 */
export function killSwitchView(
  payload: KillSwitchStatusPayload | undefined,
  isError: boolean,
): KillSwitchView {
  if (isError || payload === undefined || payload.status !== 200 || payload.data === undefined) {
    return {
      state: "unknown",
      reason: null,
      showKill: true,
      showUnlock: false,
      showUnknownLabel: true,
    };
  }
  if (payload.data.kill_switch === true) {
    return {
      state: "engaged",
      reason: payload.data.reason ?? null,
      showKill: false,
      showUnlock: true,
      showUnknownLabel: false,
    };
  }
  return {
    state: "disengaged",
    reason: null,
    showKill: true,
    showUnlock: false,
    showUnknownLabel: false,
  };
}
