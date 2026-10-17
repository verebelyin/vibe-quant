/**
 * Tests for the results-page SizingChip label (bead vibe-quant-2gyk1.2 +
 * vibe-quant-w2nge, S7b; contract U6-U8, plus S6's explicit-default nit, plus
 * Z1's notes.sizing fallback for a CLI-launched discovery run).
 *
 * Pure: the chip's label rule is `sizingChipForRun`, so no render is needed.
 * An explicit value that happens to equal the default (`{max_position_pct: 0.5}`
 * alone) is NOT legacy and must render without the "(default)" suffix.
 */
import { describe, expect, it } from "vitest";
import { sizingChipForRun } from "@/components/results/SizingChip";

type ChipRun = Parameters<typeof sizingChipForRun>[0];

function run(over: Partial<NonNullable<ChipRun>> = {}): ChipRun {
  return { run_mode: "screening", symbols: ["BTCUSDT"], parameters: {}, ...over };
}

describe("sizingChipForRun", () => {
  it("single symbol", () => {
    // A legacy row (no stored sizing keys) is marked "(default)".
    expect(sizingChipForRun(run())).toBe("Max exposure 0.5× · Risk 2% (default)");
    expect(
      sizingChipForRun(run({ parameters: { max_position_pct: 1, risk_per_trade: 0.01 } })),
    ).toBe("Max exposure 1× · Risk 1%");
    // One non-default key: no "(default)".
    expect(sizingChipForRun(run({ parameters: { max_position_pct: 1 } }))).toBe(
      "Max exposure 1× · Risk 2%",
    );
  });

  it("explicit defaults are not legacy", () => {
    // S6 nit: a stored value EQUAL to the default is explicit, not legacy.
    expect(sizingChipForRun(run({ parameters: { max_position_pct: 0.5 } }))).toBe(
      "Max exposure 0.5× · Risk 2%",
    );
    expect(sizingChipForRun(run({ parameters: { risk_per_trade: 0.02 } }))).toBe(
      "Max exposure 0.5× · Risk 2%",
    );
  });

  it("spans symbols", () => {
    expect(sizingChipForRun(run({ symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"] }))).toBe(
      "Max exposure 0.5× per symbol, up to 3 × 0.5× total · Risk 2% (default)",
    );
  });

  it("invalid sizing", () => {
    expect(sizingChipForRun(run({ parameters: { max_position_pct: -1 } }))).toBe(
      "Sizing: invalid (max_position_pct must be a number in (0, 125], got -1)",
    );
  });

  it("hidden for paper and loading", () => {
    expect(sizingChipForRun(run({ run_mode: "paper" }))).toBeNull();
    expect(sizingChipForRun(undefined)).toBeNull();
    expect(sizingChipForRun(null)).toBeNull();
  });
});

/**
 * Z1 (bead vibe-quant-zrh23): a discovery run launched from the CLI records its
 * sizing only in the machine `notes.sizing` — its run parameters carry no sizing
 * key — so the chip falls back to the notes. The run's own parameters win when
 * present; an absent `notes.sizing` keeps today's legacy-default behaviour; a
 * present but unreadable value is refused rather than silently defaulted.
 */
describe("sizingChipForRun notes.sizing fallback", () => {
  const notes = (payload: unknown) => JSON.stringify(payload);
  const discovery = run({ run_mode: "discovery" });

  it("reads the discovery run's recorded sizing", () => {
    expect(sizingChipForRun(discovery, notes({ sizing: { max_position_pct: 1 } }))).toBe(
      "Max exposure 1× · Risk 2%",
    );
    expect(
      sizingChipForRun(discovery, notes({ sizing: { max_position_pct: 2, risk_per_trade: 0.01 } })),
    ).toBe("Max exposure 2× · Risk 1%");
  });

  it("the run's parameters win when present", () => {
    expect(
      sizingChipForRun(
        run({ run_mode: "discovery", parameters: { max_position_pct: 3 } }),
        notes({ sizing: { max_position_pct: 1 } }),
      ),
    ).toBe("Max exposure 3× · Risk 2%");
  });

  it("an absent notes.sizing keeps today's legacy default", () => {
    expect(sizingChipForRun(discovery, notes({ seed: 1 }))).toBe(
      "Max exposure 0.5× · Risk 2% (default)",
    );
    expect(sizingChipForRun(discovery, null)).toBe("Max exposure 0.5× · Risk 2% (default)");
    expect(sizingChipForRun(discovery, undefined)).toBe("Max exposure 0.5× · Risk 2% (default)");
    // Notes that are not a JSON mapping carry no sizing key.
    expect(sizingChipForRun(discovery, notes([1, 2]))).toBe(
      "Max exposure 0.5× · Risk 2% (default)",
    );
  });

  it("a present but unreadable notes.sizing is invalid, never a default", () => {
    expect(sizingChipForRun(discovery, notes({ sizing: null }))).toBe(
      "Sizing: invalid (notes.sizing must be an object, got null)",
    );
    expect(sizingChipForRun(discovery, notes({ sizing: [1] }))).toBe(
      "Sizing: invalid (notes.sizing must be an object, got [1])",
    );
    expect(sizingChipForRun(discovery, notes({ sizing: { max_position_pct: -1 } }))).toBe(
      "Sizing: invalid (max_position_pct must be a number in (0, 125], got -1)",
    );
  });

  it("only a discovery run reads notes.sizing", () => {
    expect(sizingChipForRun(run(), notes({ sizing: { max_position_pct: 1 } }))).toBe(
      "Max exposure 0.5× · Risk 2% (default)",
    );
  });

  it("paper still renders nothing", () => {
    expect(
      sizingChipForRun(run({ run_mode: "paper" }), notes({ sizing: { max_position_pct: 1 } })),
    ).toBeNull();
  });
});

/**
 * T6 (bead vibe-quant-t8nrv): the run's margin limit and its refused entries.
 *
 * The validation runner records both in the machine `notes.margin_limit` (and only
 * when the limit was on, so an off limit adds nothing to the chip); the chip appends
 * them after the sizing label.
 */
describe("sizingChipForRun notes.margin_limit", () => {
  const notes = (payload: unknown) => JSON.stringify(payload);
  const validation = run({ run_mode: "validation" });

  it("appends the limit and the refused count of a validation run", () => {
    expect(
      sizingChipForRun(
        validation,
        notes({ margin_limit: { max_margin_pct: 0.5, refused_entries: 3 } }),
      ),
    ).toBe("Max exposure 0.5× · Risk 2% (default) · Max margin 50% · 3 entries refused");
    expect(
      sizingChipForRun(
        run({ run_mode: "validation", parameters: { max_position_pct: 1 } }),
        notes({ margin_limit: { max_margin_pct: 0.255, refused_entries: 1 } }),
      ),
    ).toBe("Max exposure 1× · Risk 2% · Max margin 25.5% · 1 entries refused");
  });

  it("shows a limit that refused nothing", () => {
    expect(
      sizingChipForRun(
        validation,
        notes({ margin_limit: { max_margin_pct: 0.5, refused_entries: 0 } }),
      ),
    ).toBe("Max exposure 0.5× · Risk 2% (default) · Max margin 50% · 0 entries refused");
  });

  it("shows the limit alone when no refused count was recorded", () => {
    expect(sizingChipForRun(validation, notes({ margin_limit: { max_margin_pct: 1 } }))).toBe(
      "Max exposure 0.5× · Risk 2% (default) · Max margin 100%",
    );
  });

  it("adds nothing without a recorded margin limit", () => {
    const plain = "Max exposure 0.5× · Risk 2% (default)";
    const moved = run({ run_mode: "validation", parameters: { max_position_pct: 1 } });
    expect(sizingChipForRun(moved, notes({ margin_limit: null }))).toBe(
      "Max exposure 1× · Risk 2%",
    );
    expect(sizingChipForRun(validation, notes({ margin_limit: [1, 2] }))).toBe(plain);
    expect(sizingChipForRun(validation, notes({}))).toBe(plain);
    expect(sizingChipForRun(validation, "not json")).toBe(plain);
    expect(sizingChipForRun(validation, null)).toBe(plain);
    expect(sizingChipForRun(validation, undefined)).toBe(plain);
  });

  it("renders only the parts that are numbers", () => {
    expect(
      sizingChipForRun(
        validation,
        notes({ margin_limit: { max_margin_pct: 0.5, refused_entries: "3" } }),
      ),
    ).toBe("Max exposure 0.5× · Risk 2% (default) · Max margin 50%");
    expect(
      sizingChipForRun(
        validation,
        notes({ margin_limit: { max_margin_pct: null, refused_entries: 3 } }),
      ),
    ).toBe("Max exposure 0.5× · Risk 2% (default) · 3 entries refused");
  });

  it("paper still renders nothing", () => {
    expect(
      sizingChipForRun(
        run({ run_mode: "paper" }),
        notes({ margin_limit: { max_margin_pct: 0.5, refused_entries: 3 } }),
      ),
    ).toBeNull();
  });
});

/**
 * T6 review B1: the margin limit shows for EVERY run type — a screening run has only
 * `parameters.max_margin_pct`, a CLI discovery run only `notes.sizing.max_margin_pct`.
 * The refused count stays validation-only.
 */
describe("sizingChipForRun margin limit by run type", () => {
  const notes = (payload: unknown) => JSON.stringify(payload);

  it("screening: reads parameters.max_margin_pct with no notes", () => {
    expect(sizingChipForRun(run({ parameters: { max_margin_pct: 0.5 } }), null)).toBe(
      "Max exposure 0.5× · Risk 2% (default) · Max margin 50%",
    );
    expect(sizingChipForRun(run({ parameters: { max_margin_pct: 0.255 } }))).toBe(
      "Max exposure 0.5× · Risk 2% (default) · Max margin 25.5%",
    );
  });

  it("screening: a margin key alone does not make the sizing explicit", () => {
    expect(sizingChipForRun(run({ parameters: { max_position_pct: 1, max_margin_pct: 1 } }))).toBe(
      "Max exposure 1× · Risk 2% · Max margin 100%",
    );
  });

  it("discovery: reads notes.sizing.max_margin_pct", () => {
    const discovery = run({ run_mode: "discovery" });
    expect(
      sizingChipForRun(discovery, notes({ sizing: { max_position_pct: 1, max_margin_pct: 0.4 } })),
    ).toBe("Max exposure 1× · Risk 2% · Max margin 40%");
  });

  it("parameters win over the notes", () => {
    expect(
      sizingChipForRun(
        run({ run_mode: "discovery", parameters: { max_margin_pct: 0.3 } }),
        notes({ sizing: { max_margin_pct: 0.4 } }),
      ),
    ).toBe("Max exposure 0.5× · Risk 2% (default) · Max margin 30%");
  });

  it("the refused count is validation-only", () => {
    const margin = notes({ margin_limit: { max_margin_pct: 0.5, refused_entries: 3 } });
    expect(sizingChipForRun(run({ parameters: { max_margin_pct: 0.5 } }), margin)).toBe(
      "Max exposure 0.5× · Risk 2% (default) · Max margin 50%",
    );
    expect(
      sizingChipForRun(
        run({ run_mode: "validation", parameters: { max_margin_pct: 0.5 } }),
        margin,
      ),
    ).toBe("Max exposure 0.5× · Risk 2% (default) · Max margin 50% · 3 entries refused");
  });

  it("an unreadable stored limit adds nothing", () => {
    for (const bad of [0, -1, 2, "50", null]) {
      expect(sizingChipForRun(run({ parameters: { max_margin_pct: bad } }))).toBe(
        "Max exposure 0.5× · Risk 2% (default)",
      );
    }
  });
});
