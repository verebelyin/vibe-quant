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
