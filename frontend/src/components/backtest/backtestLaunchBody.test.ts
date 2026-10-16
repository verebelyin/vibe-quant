/**
 * Tests for the backtest launch body (vibe-quant-2gyk1.2, S7a).
 *
 * A default form must produce exactly the body the form built inline before
 * sizing existed: no sizing keys in `parameters`.
 */
import { describe, expect, it } from "vitest";
import {
  type BacktestLaunchForm,
  buildBacktestLaunchBody,
} from "@/components/backtest/backtestLaunchBody";
import { DEFAULT_MAX_POSITION_PCT, DEFAULT_RISK_PCT } from "@/lib/sizing";

function defaultForm(overrides: Partial<BacktestLaunchForm> = {}): BacktestLaunchForm {
  return {
    strategyId: 7,
    symbols: ["BTCUSDT"],
    timeframe: "1m",
    startDate: "2025-01-01",
    endDate: "2025-04-01",
    initialBalance: 10000,
    leverage: 10,
    maxPositionPct: DEFAULT_MAX_POSITION_PCT,
    riskPct: DEFAULT_RISK_PCT,
    liquidationMode: "off",
    sweep: null,
    mode: "screening",
    latencyPreset: "",
    bnbDiscount: false,
    entryOrder: "market",
    makerOffsetTicks: 1,
    makerEntryTimeoutBars: 3,
    ...overrides,
  };
}

describe("buildBacktestLaunchBody", () => {
  it("equals the pre-sizing literal for a default form (no sizing keys)", () => {
    expect(buildBacktestLaunchBody(defaultForm())).toStrictEqual({
      strategy_id: 7,
      symbols: ["BTCUSDT"],
      timeframe: "1m",
      start_date: "2025-01-01",
      end_date: "2025-04-01",
      parameters: { initial_balance: 10000, leverage: 10 },
    });
  });

  it("stores max_position_pct 1 and risk_per_trade 0.01 for exposure 1 / risk 1%", () => {
    const body = buildBacktestLaunchBody(defaultForm({ maxPositionPct: 1, riskPct: 1 }));
    expect(body.parameters).toStrictEqual({
      initial_balance: 10000,
      leverage: 10,
      max_position_pct: 1,
      risk_per_trade: 0.01,
    });
  });

  it("stores no keys for an explicit 0.5 / 2% and only the moved key otherwise", () => {
    expect(
      buildBacktestLaunchBody(defaultForm({ maxPositionPct: 0.5, riskPct: 2 })).parameters,
    ).toStrictEqual({ initial_balance: 10000, leverage: 10 });
    expect(buildBacktestLaunchBody(defaultForm({ riskPct: 7 })).parameters).toStrictEqual({
      initial_balance: 10000,
      leverage: 10,
      risk_per_trade: 0.07,
    });
  });

  it("keeps sizing next to the sweep and the validation extras", () => {
    const body = buildBacktestLaunchBody(
      defaultForm({
        maxPositionPct: 2,
        sweep: { rsi_period: [10, 14] },
        mode: "validation",
        latencyPreset: "__none__",
        bnbDiscount: true,
        entryOrder: "post_only",
        liquidationMode: "isolated",
      }),
    );
    expect(body.parameters).toStrictEqual({
      initial_balance: 10000,
      leverage: 10,
      max_position_pct: 2,
      sweep: { rsi_period: [10, 14] },
    });
    expect(body.latency_preset).toBeNull();
    expect(body.bnb_discount).toBe(true);
    expect(body.entry_order).toBe("post_only");
    expect(body.liquidation_mode).toBe("isolated");
  });

  it("sends bnb_discount for a screening run when the discount is checked", () => {
    const body = buildBacktestLaunchBody(defaultForm({ mode: "screening", bnbDiscount: true }));
    expect(body.bnb_discount).toBe(true);
  });

  it("omits bnb_discount for a screening run when the discount is unchecked", () => {
    const body = buildBacktestLaunchBody(defaultForm({ mode: "screening", bnbDiscount: false }));
    expect(body).not.toHaveProperty("bnb_discount");
  });

  it("omits bnb_discount for a validation run when the discount is unchecked", () => {
    const body = buildBacktestLaunchBody(defaultForm({ mode: "validation", bnbDiscount: false }));
    expect(body).not.toHaveProperty("bnb_discount");
  });
});
