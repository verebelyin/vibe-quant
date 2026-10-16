/**
 * Tests for the discovery launch body (vibe-quant-kdy5i).
 *
 * The form has no jsdom/testing-library in this repo, so the body is pinned by
 * calling the SAME function the submit handler calls. A default form must
 * produce exactly the body the component used to build inline, byte for byte
 * (contract C6 / vibe-quant-6wq6w): no `leverage`, no `liquidation_mode`.
 */
import { describe, expect, it } from "vitest";
import {
  buildDiscoveryLaunchBody,
  type DiscoveryLaunchForm,
  parseCrossWindowMonths,
} from "@/components/discovery/discoveryLaunchBody";
import { DEFAULT_LEVERAGE } from "@/lib/liquidation";
import { DEFAULT_MAX_POSITION_PCT, DEFAULT_RISK_PCT } from "@/lib/sizing";

function defaultForm(overrides: Partial<DiscoveryLaunchForm> = {}): DiscoveryLaunchForm {
  return {
    population: 20,
    generations: 15,
    mutationRate: 0.1,
    crossoverRate: 0.8,
    eliteCount: 2,
    tournamentSize: 3,
    convergenceWindow: 5,
    symbols: "BTCUSDT",
    timeframe: "4h",
    startDate: "2025-01-01",
    endDate: "2025-04-01",
    direction: "random",
    evalWindows: 3,
    trainTestSplit: 0.8,
    numSeeds: 1,
    symbolAgg: "portfolio",
    wfaOosStepDays: 0,
    wfaMinConsistency: 0.75,
    crossWindowMonths: "",
    crossWindowMinSharpe: 0.5,
    immigrantFraction: 0.15,
    entropyThreshold: 0.4,
    crowdingEnabled: true,
    seedRunId: "",
    rngSeed: "",
    noBootstrapCi: false,
    bootstrapMinSharpe: "",
    bnbDiscount: false,
    liquidationMode: "off",
    leverage: DEFAULT_LEVERAGE,
    maxPositionPct: DEFAULT_MAX_POSITION_PCT,
    riskPct: DEFAULT_RISK_PCT,
    selectedIndicators: [],
    ...overrides,
  };
}

describe("buildDiscoveryLaunchBody", () => {
  it("equals the pre-refactor literal for a default form", () => {
    expect(buildDiscoveryLaunchBody(defaultForm())).toStrictEqual({
      population: 20,
      generations: 15,
      mutation_rate: 0.1,
      crossover_rate: 0.8,
      elite_count: 2,
      tournament_size: 3,
      convergence_generations: 5,
      symbols: ["BTCUSDT"],
      timeframes: ["4h"],
      indicator_pool: null,
      start_date: "2025-01-01",
      end_date: "2025-04-01",
      eval_windows: 3,
      train_test_split: 0.8,
      num_seeds: 1,
      wfa_oos_step_days: 0,
      wfa_min_consistency: 0.75,
      immigrant_fraction: 0.15,
      entropy_threshold: 0.4,
      crowding_enabled: true,
    });
  });

  it("sends no leverage and no liquidation_mode off at the default", () => {
    const body = buildDiscoveryLaunchBody(defaultForm());
    expect("leverage" in body).toBe(false);
    expect("liquidation_mode" in body).toBe(false);
  });

  it("sends leverage but no liquidation_mode off at a non-default leverage", () => {
    const body = buildDiscoveryLaunchBody(defaultForm({ leverage: 3 }));
    expect(body.leverage).toBe(3);
    expect("liquidation_mode" in body).toBe(false);
  });

  it("sends the mode and the leverage when isolated at 20", () => {
    const body = buildDiscoveryLaunchBody(
      defaultForm({ liquidationMode: "isolated", leverage: 20 }),
    );
    expect(body.liquidation_mode).toBe("isolated");
    expect(body.leverage).toBe(20);
  });

  it("sends the leverage when the mode is on at the default", () => {
    const body = buildDiscoveryLaunchBody(
      defaultForm({ liquidationMode: "cross", leverage: DEFAULT_LEVERAGE }),
    );
    expect(body.liquidation_mode).toBe("cross");
    expect(body.leverage).toBe(DEFAULT_LEVERAGE);
  });

  it("sends the selected indicator pool, and null when none is selected", () => {
    expect(buildDiscoveryLaunchBody(defaultForm()).indicator_pool).toBeNull();
    expect(
      buildDiscoveryLaunchBody(defaultForm({ selectedIndicators: ["RSI", "ADX"] })).indicator_pool,
    ).toStrictEqual(["RSI", "ADX"]);
  });

  it("omits the date and direction keys a default form leaves unset", () => {
    const body = buildDiscoveryLaunchBody(defaultForm({ startDate: "", endDate: "" }));
    expect("start_date" in body).toBe(false);
    expect("end_date" in body).toBe(false);
    expect("direction" in body).toBe(false);
    expect("symbol_agg" in body).toBe(false);
    expect("cross_window_months" in body).toBe(false);
    expect("cross_window_min_sharpe" in body).toBe(false);
    expect("seed_run_id" in body).toBe(false);
    expect("seed" in body).toBe(false);
    expect("no_bootstrap_ci" in body).toBe(false);
    expect("bootstrap_min_sharpe" in body).toBe(false);
    expect("bnb_discount" in body).toBe(false);
  });

  it("sends the optional keys a non-default form sets", () => {
    expect(
      buildDiscoveryLaunchBody(
        defaultForm({
          symbols: "BTCUSDT, ETHUSDT ,",
          startDate: "2025-02-01",
          endDate: "2025-03-01",
          direction: "long",
          symbolAgg: "worst",
          crossWindowMonths: "1, 2",
          crossWindowMinSharpe: 0.6,
          seedRunId: "7",
          rngSeed: "42",
          noBootstrapCi: true,
          bootstrapMinSharpe: "0.7",
          bnbDiscount: true,
        }),
      ),
    ).toStrictEqual({
      population: 20,
      generations: 15,
      mutation_rate: 0.1,
      crossover_rate: 0.8,
      elite_count: 2,
      tournament_size: 3,
      convergence_generations: 5,
      symbols: ["BTCUSDT", "ETHUSDT"],
      timeframes: ["4h"],
      indicator_pool: null,
      start_date: "2025-02-01",
      end_date: "2025-03-01",
      direction: "long",
      eval_windows: 3,
      train_test_split: 0.8,
      num_seeds: 1,
      symbol_agg: "worst",
      wfa_oos_step_days: 0,
      wfa_min_consistency: 0.75,
      cross_window_months: [1, 2],
      cross_window_min_sharpe: 0.6,
      immigrant_fraction: 0.15,
      entropy_threshold: 0.4,
      crowding_enabled: true,
      seed_run_id: 7,
      seed: 42,
      no_bootstrap_ci: true,
      bootstrap_min_sharpe: 0.7,
      bnb_discount: true,
    });
  });

  it("ignores a non-integer seed and a non-numeric warm-start run id", () => {
    const body = buildDiscoveryLaunchBody(defaultForm({ rngSeed: "1.5", seedRunId: "abc" }));
    expect("seed" in body).toBe(false);
    expect("seed_run_id" in body).toBe(false);
  });
});

describe("parseCrossWindowMonths", () => {
  it("drops empty, non-numeric, and non-positive offsets", () => {
    expect(parseCrossWindowMonths("")).toStrictEqual([]);
    expect(parseCrossWindowMonths("1, 2,3")).toStrictEqual([1, 2, 3]);
    expect(parseCrossWindowMonths("0,-2,x,")).toStrictEqual([]);
  });
});

describe("buildDiscoveryLaunchBody sizing (vibe-quant-2gyk1.2)", () => {
  it("sends no sizing keys at the defaults (0.5 x equity, 2%)", () => {
    const body = buildDiscoveryLaunchBody(defaultForm());
    expect("max_position_pct" in body).toBe(false);
    expect("risk_per_trade" in body).toBe(false);
  });

  it("sends max_position_pct 1 and risk_per_trade 0.01 for exposure 1 / risk 1%", () => {
    const body = buildDiscoveryLaunchBody(defaultForm({ maxPositionPct: 1, riskPct: 1 }));
    expect(body.max_position_pct).toBe(1);
    expect(body.risk_per_trade).toBe(0.01);
  });

  it("sends only the key that moved off the default", () => {
    const cap = buildDiscoveryLaunchBody(defaultForm({ maxPositionPct: 2 }));
    expect(cap.max_position_pct).toBe(2);
    expect("risk_per_trade" in cap).toBe(false);
    const risk = buildDiscoveryLaunchBody(defaultForm({ riskPct: 7 }));
    expect(risk.risk_per_trade).toBe(0.07);
    expect("max_position_pct" in risk).toBe(false);
  });
});
