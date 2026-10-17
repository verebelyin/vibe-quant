/**
 * The champion bankroll chip of the discovery results page (T9, bead vibe-quant-t8nrv).
 *
 * A discovery launched with a bankroll grid records, on every champion entry, the cell it
 * was found under: `bankroll = {"liquidation_mode": str, "leverage": float,
 * "max_margin_pct": float}` (design B5b, contract R51). A run without a grid records no
 * `bankroll`, so the chip renders nothing. Pure so every stored shape — absent, off,
 * malformed — is named by a test.
 */
import { marginPctText } from "@/components/backtest/marginLimit";

/** The two modes a grid cell may carry, as the chip spells them. */
const MODE_LABELS: Record<string, string> = { isolated: "Isolated", cross: "Cross" };

/**
 * The chip label of one champion's recorded cell, e.g. "Isolated 5x · cushion 50%", or null
 * when the entry carries none: only a complete, in-domain cell (`isolated`/`cross`, a
 * leverage ≥ 1, a cushion in (0, 1]) is shown.
 */
export function bankrollCellLabel(raw: unknown): string | null {
  if (raw == null || typeof raw !== "object" || Array.isArray(raw)) return null;
  const cell = raw as Record<string, unknown>;

  const mode = cell.liquidation_mode;
  if (typeof mode !== "string" || MODE_LABELS[mode] === undefined) return null;

  const leverage = cell.leverage;
  if (typeof leverage !== "number" || !Number.isFinite(leverage) || leverage < 1) return null;

  const cushion = cell.max_margin_pct;
  if (typeof cushion !== "number" || !Number.isFinite(cushion) || cushion <= 0 || cushion > 1) {
    return null;
  }

  return `${MODE_LABELS[mode]} ${leverage}x · cushion ${marginPctText(cushion)}%`;
}
