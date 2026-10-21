/**
 * Tests for the discovery launch body (vibe-quant-kdy5i).
 *
 * The form has no jsdom/testing-library in this repo, so the body is pinned by
 * calling the SAME function the submit handler calls. A default form must
 * produce exactly the body the component used to build inline, byte for byte
 * (contract C6 / vibe-quant-6wq6w): no `leverage`, no `liquidation_mode`.
 */
import { describe, expect, it } from "vitest";
import { budgetFor } from "@/components/discovery/discoveryEstimate";
import {
  type BankrollGridForm,
  buildDiscoveryLaunchBody,
  type DiscoveryLaunchForm,
  gridCrossCheckLeverage,
  MAX_BANKROLL_CELLS,
  parseBankrollGrid,
  parseCrossWindowMonths,
} from "@/components/discovery/discoveryLaunchBody";
import { DEFAULT_LEVERAGE } from "@/lib/liquidation";
import { DEFAULT_MAX_POSITION_PCT, DEFAULT_RISK_PCT, sizingProblems } from "@/lib/sizing";

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
    // Margin limit (T6): "" is the empty field = OFF.
    maxMarginPct: "",
    // Bankroll grid (T8): disabled = the section is off and no key is sent.
    bankrollGrid: { enabled: false, modes: [], leverages: "", maxMarginPcts: "" },
    selectedIndicators: [],
    ...overrides,
  };
}

/** An enabled grid form, overridable per case (T8 input table). */
function grid(over: Partial<BankrollGridForm> = {}): BankrollGridForm {
  return {
    enabled: true,
    modes: ["isolated", "cross"],
    leverages: "5, 10",
    maxMarginPcts: "25, 50",
    ...over,
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

/**
 * U2 (bead vibe-quant-wnqea): an untouched form on the default 4h timeframe starts at
 * the P-2 rung 50x20 with convergence 10, and the launch body carries exactly that.
 */
describe("buildDiscoveryLaunchBody 4h budget default", () => {
  it("launches 50x20 with convergence 10 from an untouched 4h form", () => {
    const budget = budgetFor("4h");
    const body = buildDiscoveryLaunchBody(
      defaultForm({
        population: budget.population,
        generations: budget.generations,
        convergenceWindow: budget.convergenceWindow,
        timeframe: "4h",
      }),
    );
    expect(body.population).toBe(50);
    expect(body.generations).toBe(20);
    expect(body.convergence_generations).toBe(10);
    expect(body.timeframes).toStrictEqual(["4h"]);
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

/**
 * T6 (bead vibe-quant-t8nrv): the discovery margin limit is a TOP-LEVEL request field
 * (not a run parameter), and only when the field holds a value.
 */
describe("buildDiscoveryLaunchBody margin limit", () => {
  it("sends no max_margin_pct for an empty field", () => {
    expect("max_margin_pct" in buildDiscoveryLaunchBody(defaultForm())).toBe(false);
  });

  it("sends the top-level max_margin_pct 0.5 for a limit of 50% of equity", () => {
    const body = buildDiscoveryLaunchBody(defaultForm({ maxMarginPct: "50" }));
    expect(body.max_margin_pct).toBe(0.5);
  });

  it("sends the inclusive 100% edge as 1", () => {
    expect(buildDiscoveryLaunchBody(defaultForm({ maxMarginPct: "100" })).max_margin_pct).toBe(1);
  });
});

/**
 * T8/T9 (bead vibe-quant-t8nrv): the discovery bankroll grid. The section is opt-in; a
 * disabled section parses to `off` and the body carries NO `bankroll_grid` key, so an old
 * launch stays byte-identical. This is the frontend mirror of `vibe_quant.bankroll_grid`
 * (mode x leverage x cushion, whole leverages from 1x, at most 12 cells).
 */
describe("parseBankrollGrid", () => {
  it("is off when the section is disabled, whatever the fields hold", () => {
    expect(parseBankrollGrid({ ...grid(), enabled: false })).toStrictEqual({ kind: "off" });
    expect(
      parseBankrollGrid({ enabled: false, modes: [], leverages: "x", maxMarginPcts: "" }),
    ).toStrictEqual({ kind: "off" });
  });

  it("needs at least one margin mode", () => {
    const parsed = parseBankrollGrid(grid({ modes: [] }));
    expect(parsed.kind).toBe("invalid");
  });

  it("accepts whole leverages 1..125, de-duplicated and sorted", () => {
    expect(
      parseBankrollGrid(
        grid({ modes: ["isolated"], leverages: "10, 1, 5, 5", maxMarginPcts: "50" }),
      ),
    ).toStrictEqual({
      kind: "on",
      count: 3,
      payload: { modes: ["isolated"], leverages: [1, 5, 10], max_margin_pcts: [0.5] },
    });
    expect(
      parseBankrollGrid(grid({ modes: ["cross"], leverages: "1", maxMarginPcts: "50" })),
    ).toMatchObject({ kind: "on" });
  });

  it("rejects a zero, fractional, non-numeric or empty leverage list", () => {
    for (const bad of ["0", "1.5", "abc", ""]) {
      expect(parseBankrollGrid(grid({ leverages: bad })).kind).toBe("invalid");
    }
  });

  it("parses a fractional cushion written with a dot", () => {
    expect(
      parseBankrollGrid(grid({ modes: ["cross"], leverages: "5", maxMarginPcts: "25.5" })),
    ).toStrictEqual({
      kind: "on",
      count: 1,
      payload: { modes: ["cross"], leverages: [5], max_margin_pcts: [0.255] },
    });
    expect(
      parseBankrollGrid(grid({ modes: ["cross"], leverages: "5", maxMarginPcts: "50" })),
    ).toMatchObject({ payload: { max_margin_pcts: [0.5] } });
  });

  it("splits a list on comma+space, a semicolon, or a whitespace run", () => {
    for (const list of ["25, 50", "25;50", "25 50"]) {
      expect(
        parseBankrollGrid(grid({ modes: ["cross"], leverages: "5", maxMarginPcts: list })),
      ).toStrictEqual({
        kind: "on",
        count: 2,
        payload: { modes: ["cross"], leverages: [5], max_margin_pcts: [0.25, 0.5] },
      });
    }
  });

  it("refuses a bare comma between digits in a list as ambiguous (chief ruling T9-R1)", () => {
    const ambiguous = (token: string) =>
      `Ambiguous value '${token}': write 25.5 for one value or '25, 50' for two`;
    for (const bad of ["25,50", "25,5"]) {
      expect(parseBankrollGrid(grid({ maxMarginPcts: bad }))).toStrictEqual({
        kind: "invalid",
        message: ambiguous(bad),
      });
    }
    expect(parseBankrollGrid(grid({ leverages: "5,10" }))).toStrictEqual({
      kind: "invalid",
      message: ambiguous("5,10"),
    });
    expect(
      parseBankrollGrid(grid({ modes: ["cross"], leverages: "5, 10", maxMarginPcts: "50" })),
    ).toMatchObject({ kind: "on", payload: { leverages: [5, 10] } });
  });

  it("refuses a token that starts or ends with a comma (vibe-quant-zeqit)", () => {
    const ambiguous = (token: string) =>
      `Ambiguous value '${token}': write 25.5 for one value or '25, 50' for two`;
    expect(parseBankrollGrid(grid({ maxMarginPcts: "25 ,50" }))).toStrictEqual({
      kind: "invalid",
      message: ambiguous(",50"),
    });
    expect(parseBankrollGrid(grid({ leverages: "5 ,10", maxMarginPcts: "50" }))).toStrictEqual({
      kind: "invalid",
      message: ambiguous(",10"),
    });
    expect(parseBankrollGrid(grid({ maxMarginPcts: "25," }))).toStrictEqual({
      kind: "invalid",
      message: ambiguous("25,"),
    });
    expect(parseBankrollGrid(grid({ maxMarginPcts: "," }))).toStrictEqual({
      kind: "invalid",
      message: ambiguous(","),
    });
  });

  it("still reads a trailing separator as a list separator, not a token (vibe-quant-zeqit)", () => {
    expect(
      parseBankrollGrid(grid({ modes: ["cross"], leverages: "5", maxMarginPcts: "25, " })),
    ).toMatchObject({ kind: "on", payload: { max_margin_pcts: [0.25] } });
  });

  it("rejects a cushion of 0, above 100% or non-numeric", () => {
    for (const bad of ["0", "101", "x"]) {
      expect(parseBankrollGrid(grid({ maxMarginPcts: bad })).kind).toBe("invalid");
    }
  });

  it("orders the modes isolated before cross", () => {
    const parsed = parseBankrollGrid(
      grid({ modes: ["cross", "isolated"], leverages: "1", maxMarginPcts: "50" }),
    );
    expect(parsed.kind === "on" && parsed.payload.modes).toStrictEqual(["isolated", "cross"]);
  });

  it("accepts exactly the 12-cell cap", () => {
    const parsed = parseBankrollGrid(
      grid({ modes: ["isolated", "cross"], leverages: "1, 2, 3", maxMarginPcts: "10, 20" }),
    );
    expect(parsed).toMatchObject({ kind: "on", count: MAX_BANKROLL_CELLS });
  });

  it("rejects more than 12 cells, naming the count and the cap", () => {
    const parsed = parseBankrollGrid(
      grid({
        modes: ["isolated", "cross"],
        leverages: "1, 2, 3, 4, 5",
        maxMarginPcts: "10, 20, 30, 40, 50, 60",
      }),
    );
    expect(parsed.kind).toBe("invalid");
    expect(parsed.kind === "invalid" && parsed.message).toContain("60");
    expect(parsed.kind === "invalid" && parsed.message).toContain("12");
  });
});

describe("buildDiscoveryLaunchBody bankroll grid", () => {
  it("sends no bankroll_grid key for a disabled grid (old launches unchanged)", () => {
    expect("bankroll_grid" in buildDiscoveryLaunchBody(defaultForm())).toBe(false);
  });

  it("sends the grid lists when the grid is on", () => {
    const body = buildDiscoveryLaunchBody(defaultForm({ bankrollGrid: grid() }));
    expect(body.bankroll_grid).toStrictEqual({
      modes: ["isolated", "cross"],
      leverages: [5, 10],
      max_margin_pcts: [0.25, 0.5],
    });
  });

  it("drops the run-level liquidation, leverage and margin keys a grid excludes", () => {
    const body = buildDiscoveryLaunchBody(
      defaultForm({
        liquidationMode: "isolated",
        leverage: 20,
        maxMarginPct: "50",
        bankrollGrid: grid({ modes: ["isolated"] }),
      }),
    );
    expect(body.bankroll_grid).toBeDefined();
    expect("liquidation_mode" in body).toBe(false);
    expect("leverage" in body).toBe(false);
    expect("max_margin_pct" in body).toBe(false);
  });
});

/**
 * T9 (bead vibe-quant-t8nrv): the run-level exposure-cap cross-check against the grid.
 * A grid checks the cap at EVERY cell, so the binding leverage is the SMALLEST grid
 * leverage; with the grid off it is the run-level leverage, and an invalid grid is
 * checked against nothing (its own message is the one shown).
 */
describe("gridCrossCheckLeverage", () => {
  it("uses the run-level leverage while the grid is off", () => {
    const off = parseBankrollGrid({ ...grid(), enabled: false });
    expect(gridCrossCheckLeverage(off, 20)).toBe(20);
  });

  it("uses the smallest grid leverage while the grid is on", () => {
    const on = parseBankrollGrid(
      grid({ modes: ["isolated"], leverages: "5, 10", maxMarginPcts: "50" }),
    );
    expect(gridCrossCheckLeverage(on, 20)).toBe(5);
  });

  it("fires the cap cross-check at the equality edge of the smallest grid leverage", () => {
    const on = parseBankrollGrid(
      grid({ modes: ["isolated"], leverages: "5, 10", maxMarginPcts: "50" }),
    );
    const leverage = gridCrossCheckLeverage(on, 20);
    expect(sizingProblems(5, 2, leverage)).toStrictEqual([]);
    expect(sizingProblems(6, 2, leverage)).toStrictEqual([
      "Max exposure (6 × balance) must not exceed leverage (5)",
    ]);
  });

  it("cross-checks nothing while the grid is on but invalid", () => {
    const invalid = parseBankrollGrid(grid({ leverages: "" }));
    expect(Number.isNaN(gridCrossCheckLeverage(invalid, 20))).toBe(true);
    expect(sizingProblems(10, 2, gridCrossCheckLeverage(invalid, 20))).toStrictEqual([]);
  });
});
