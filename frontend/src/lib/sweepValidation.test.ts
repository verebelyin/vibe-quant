/**
 * Tests for the sweep-row Validate body builder (bead vibe-quant-2gyk1.2 +
 * vibe-quant-w2nge, S7b; contract U9-U12, U9b, U9c).
 *
 * The builder stacks the parent run's run-level settings on the swept row
 * parameters; the parent's `sweep` grid spec must never appear. A DSL-override
 * parent and unreadable parent sizing are refused before any request.
 */
import { describe, expect, it } from "vitest";
import type { BacktestRunResponse } from "@/api/generated/models/backtestRunResponse";
import type { SweepResultResponse } from "@/api/generated/models/sweepResultResponse";
import { sweepValidationBody } from "@/lib/sweepValidation";

function makeRun(over: Partial<BacktestRunResponse> = {}): BacktestRunResponse {
  return {
    id: 1,
    run_mode: "screening",
    symbols: ["BTCUSDT"],
    timeframe: "1h",
    start_date: "2026-01-01",
    end_date: "2026-02-01",
    strategy_id: 7,
    status: "completed",
    created_at: "2026-02-01T00:00:00Z",
    ...over,
  };
}

function makeSweep(parameters: Record<string, unknown>): SweepResultResponse {
  return {
    id: 11,
    run_id: 1,
    parameters,
    sharpe_ratio: null,
    sortino_ratio: null,
    max_drawdown: null,
    total_return: null,
    profit_factor: null,
    win_rate: null,
    total_trades: null,
    total_fees: null,
    total_funding: null,
    execution_time_seconds: null,
    is_pareto_optimal: false,
    passed_deflated_sharpe: null,
    passed_walk_forward: null,
    passed_purged_kfold: null,
  };
}

describe("sweepValidationBody", () => {
  it("forwards every run-level setting", () => {
    const run = makeRun({
      parameters: {
        initial_balance: 5000,
        leverage: 3,
        liquidation_mode: "isolated",
        bnb_discount: true,
        max_position_pct: 2,
        risk_per_trade: 0.01,
        sweep: { rsi_period: [1, 2] },
      },
    });
    const sweep = makeSweep({ rsi_period: 14 });

    const res = sweepValidationBody(run, sweep);

    expect(res).toStrictEqual({
      ok: true,
      body: {
        strategy_id: 7,
        symbols: ["BTCUSDT"],
        timeframe: "1h",
        start_date: "2026-01-01",
        end_date: "2026-02-01",
        parameters: {
          rsi_period: 14,
          initial_balance: 5000,
          leverage: 3,
          max_position_pct: 2,
          risk_per_trade: 0.01,
        },
        liquidation_mode: "isolated",
        bnb_discount: true,
      },
    });
    // The parent's sweep grid spec is never forwarded.
    expect(res.ok && "sweep" in res.body.parameters).toBe(false);
  });

  it("legacy parent", () => {
    const run = makeRun({
      parameters: { initial_balance: 1000, leverage: 10, sweep: { rsi_period: [1, 2] } },
    });
    const sweep = makeSweep({ rsi_period: 14 });

    const res = sweepValidationBody(run, sweep);

    expect(res.ok && res.body.parameters).toStrictEqual({
      rsi_period: 14,
      initial_balance: 1000,
      leverage: 10,
    });
    // No liquidation_mode, no bnb_discount on a legacy parent.
    expect(res.ok && "liquidation_mode" in res.body).toBe(false);
    expect(res.ok && "bnb_discount" in res.body).toBe(false);
  });

  it("null parent parameters", () => {
    const run = makeRun({ parameters: null });
    const sweep = makeSweep({ rsi_period: 14 });

    const res = sweepValidationBody(run, sweep);

    expect(res.ok && res.body.parameters).toStrictEqual({ rsi_period: 14 });
  });

  it("invalid parent sizing refuses", () => {
    const run = makeRun({ parameters: { max_position_pct: -1 } });
    const sweep = makeSweep({ rsi_period: 14 });

    expect(sweepValidationBody(run, sweep)).toStrictEqual({
      ok: false,
      error: "Parent run sizing is invalid: max_position_pct must be a number in (0, 125], got -1",
    });
  });

  it("forwards fee_tier", () => {
    const run = makeRun({
      parameters: { initial_balance: 5000, leverage: 3, fee_tier: "vip3" },
    });
    const sweep = makeSweep({ rsi_period: 14 });

    const res = sweepValidationBody(run, sweep);

    expect(res.ok && res.body.parameters).toStrictEqual({
      rsi_period: 14,
      initial_balance: 5000,
      leverage: 3,
      fee_tier: "vip3",
    });
  });

  it("dsl override refuses", () => {
    const run = makeRun({ parameters: { dsl_override: "src: custom rule" } });
    const sweep = makeSweep({ rsi_period: 14 });

    expect(sweepValidationBody(run, sweep)).toStrictEqual({
      ok: false,
      error:
        "This screening run used a DSL override; validation cannot reproduce it. Save the override as a strategy and validate that.",
    });
  });
});
