/**
 * Tests for the discovery champion bankroll chip (T9, bead vibe-quant-t8nrv).
 *
 * A grid run records, on every champion entry, the cell it was found under:
 * `bankroll = {"liquidation_mode": str, "leverage": float, "max_margin_pct": float}`
 * (design B5b / contract R51). The results page shows that cell as
 * "Isolated 5x · cushion 50%"; a run without a grid records no `bankroll`, so the
 * chip renders nothing. Pure: the label rule is `bankrollCellLabel`.
 */
import { describe, expect, it } from "vitest";
import { bankrollCellLabel } from "@/lib/bankrollGrid";

describe("bankrollCellLabel", () => {
  it("labels an isolated cell", () => {
    expect(
      bankrollCellLabel({ liquidation_mode: "isolated", leverage: 5, max_margin_pct: 0.5 }),
    ).toBe("Isolated 5x · cushion 50%");
  });

  it("labels a cross cell", () => {
    expect(bankrollCellLabel({ liquidation_mode: "cross", leverage: 10, max_margin_pct: 1 })).toBe(
      "Cross 10x · cushion 100%",
    );
  });

  it("rounds the cushion like the margin field does", () => {
    expect(
      bankrollCellLabel({ liquidation_mode: "isolated", leverage: 1, max_margin_pct: 0.255 }),
    ).toBe("Isolated 1x · cushion 25.5%");
  });

  it("renders nothing when the cell is absent", () => {
    expect(bankrollCellLabel(undefined)).toBeNull();
    expect(bankrollCellLabel(null)).toBeNull();
  });

  it("renders nothing for a malformed cell", () => {
    expect(bankrollCellLabel("x")).toBeNull();
    expect(bankrollCellLabel([1, 2])).toBeNull();
    expect(bankrollCellLabel({})).toBeNull();
    expect(
      bankrollCellLabel({ liquidation_mode: "off", leverage: 5, max_margin_pct: 0.5 }),
    ).toBeNull();
    expect(
      bankrollCellLabel({ liquidation_mode: "isolated", leverage: 0, max_margin_pct: 0.5 }),
    ).toBeNull();
    expect(
      bankrollCellLabel({ liquidation_mode: "isolated", leverage: 5, max_margin_pct: 0 }),
    ).toBeNull();
    expect(
      bankrollCellLabel({ liquidation_mode: "isolated", leverage: 5, max_margin_pct: 2 }),
    ).toBeNull();
  });
});
