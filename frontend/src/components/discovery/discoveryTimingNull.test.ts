import { describe, expect, it } from "vitest";
import { timingNullLine, timingNullPolicyProblem } from "./discoveryTimingNull";

const policy = { enforced: true, draws: 500, pct: 0.95, exposure_max: 0.9, base_seed: 7 };

function rec(extra: Record<string, unknown>): Record<string, unknown> {
  return {
    percentile: null,
    draws: 500,
    exposure: 0.42,
    fallback: null,
    fallback_passed: null,
    ...extra,
  };
}

describe("timingNullLine", () => {
  it("states the verdict against the run's own bar", () => {
    expect(timingNullLine(rec({ percentile: 0.972 }), policy)).toBe(
      "passed - real return beat 97.20% of 500 random re-timings of its own trades (needs 95%)",
    );
    expect(timingNullLine(rec({ percentile: 0.9499 }), policy)).toBe(
      "failed - real return beat 94.99% of 500 random re-timings of its own trades (needs 95%)",
    );
  });

  it("the pass bar is the policy's, never a UI constant (0.95 edge passes)", () => {
    expect(timingNullLine(rec({ percentile: 0.95 }), policy)).toMatch(/^passed - /);
    expect(timingNullLine(rec({ percentile: 0.95, draws: 200 }), { ...policy, pct: 0.99 })).toBe(
      "failed - real return beat 95.00% of 200 random re-timings of its own trades (needs 99%)",
    );
  });

  it("says in plain words the book is untestable and not certified (exposure above the cap)", () => {
    // a legacy stored buy-and-hold pass never reads as "passed"
    const legacy = timingNullLine(
      rec({ exposure: 0.9401, fallback: "passed", fallback_passed: true }),
      policy,
    );
    expect(legacy).toBe(
      "timing untestable, not certified - in a position 94.01% of the time (over 90%), so re-timing its trades changes almost nothing",
    );
    expect(legacy).not.toMatch(/passed/);
    expect(
      timingNullLine(
        rec({
          exposure: 0.95,
          fallback: "Timing null: exposure 0.9500 > 0.80, timing untestable (not certified)",
          fallback_passed: false,
        }),
        { ...policy, exposure_max: 0.8 },
      ),
    ).toBe(
      "timing untestable, not certified - in a position 95.00% of the time (over 80%), so re-timing its trades changes almost nothing",
    );
  });

  it("no line for a run without the gate; an unreadable record says so", () => {
    expect(timingNullLine(rec({ percentile: 0.99 }), null)).toBeNull();
    expect(timingNullLine(rec({ percentile: 0.99 }), undefined)).toBeNull();
    expect(timingNullLine(rec({ percentile: 0.99 }), { ...policy, enforced: false })).toBeNull();
    expect(timingNullLine(undefined, policy)).toBe("record unreadable");
    expect(timingNullLine("x", policy)).toBe("record unreadable");
    expect(timingNullLine(rec({ percentile: Number.NaN }), policy)).toBe("record unreadable");
  });
});

describe("timingNullPolicyProblem", () => {
  it("shows the backend's unreadable-policy marker, nothing for a readable or absent policy", () => {
    expect(timingNullPolicyProblem({ unreadable: "timing_null_gate is not an object: 'x'" })).toBe(
      "policy unreadable (timing_null_gate is not an object: 'x')",
    );
    expect(timingNullPolicyProblem(policy)).toBeNull();
    expect(timingNullPolicyProblem(null)).toBeNull();
    expect(timingNullPolicyProblem({ unreadable: 3 })).toBeNull();
  });

  it("a marker policy gives no per-champion line", () => {
    expect(timingNullLine(rec({ percentile: 0.99 }), { unreadable: "x" })).toBeNull();
  });
});
