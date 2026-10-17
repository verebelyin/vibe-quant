import { describe, expect, it } from "vitest";
import type { SystemStatusResponse } from "@/api/generated/models";
import {
  KILL_SWITCH_UNKNOWN_LABEL,
  type KillSwitchStatusPayload,
  killSwitchView,
} from "@/components/layout/killSwitchState";

function status(over: Partial<SystemStatusResponse> = {}): SystemStatusResponse {
  return {
    kill_switch: false,
    reason: null,
    killed_at: null,
    killed_by: null,
    updated_at: null,
    ...over,
  };
}

function payload(over: Partial<SystemStatusResponse> = {}, code = 200): KillSwitchStatusPayload {
  return { status: code, data: status(over) };
}

const UNKNOWN = {
  state: "unknown",
  reason: null,
  showKill: true,
  showUnlock: false,
  showUnknownLabel: true,
} as const;

describe("killSwitchView", () => {
  it("disengaged: Kill trigger only", () => {
    expect(killSwitchView(payload(), false)).toEqual({
      state: "disengaged",
      reason: null,
      showKill: true,
      showUnlock: false,
      showUnknownLabel: false,
    });
  });

  it("engaged: Unlock only, with the operator reason", () => {
    expect(
      killSwitchView(payload({ kill_switch: true, reason: "validation↔paper divergence" }), false),
    ).toEqual({
      state: "engaged",
      reason: "validation↔paper divergence",
      showKill: false,
      showUnlock: true,
      showUnknownLabel: false,
    });
  });

  it("engaged with a null reason", () => {
    const view = killSwitchView(payload({ kill_switch: true, reason: null }), false);
    expect(view.state).toBe("engaged");
    expect(view.reason).toBeNull();
    expect(view.showUnlock).toBe(true);
  });

  it("503 /status: unknown AND the Kill trigger stays", () => {
    expect(killSwitchView({ status: 503 }, false)).toEqual(UNKNOWN);
    expect(killSwitchView(payload({}, 500), false)).toEqual(UNKNOWN);
  });

  it("isError: unknown AND the Kill trigger stays, even with stale data", () => {
    expect(killSwitchView(payload({ kill_switch: true, reason: "ops" }), true)).toEqual(UNKNOWN);
    expect(killSwitchView(payload(), true)).toEqual(UNKNOWN);
  });

  it("initial load (no response yet): unknown AND the Kill trigger shows", () => {
    expect(killSwitchView(undefined, false)).toEqual(UNKNOWN);
  });

  it("a 200 response carrying no body is unknown", () => {
    expect(killSwitchView({ status: 200 }, false)).toEqual(UNKNOWN);
  });

  it("names the unknown state without leaking the disengaged look", () => {
    expect(KILL_SWITCH_UNKNOWN_LABEL).toBe("Kill switch state unknown");
  });
});
