/**
 * The margin-limit field of the launch forms (bead vibe-quant-t8nrv, T6): the ONE
 * frontend rule mirroring `SizingSettings.max_margin_pct`
 * (`vibe_quant/sizing_settings.py`).
 *
 * The forms enter the limit in PERCENT of equity and the request carries the fraction
 * the backend validates, so this is where "50" becomes 0.5. The field is OPTIONAL —
 * empty means the limit is off (no key anywhere, so the run's numbers stay as today) —
 * but a value outside (0, 100] is REFUSED, never silently read as off.
 *
 * It lives in the backtest components (both launch forms import it) rather than beside
 * the results chip: `results/sizingChip.ts` would differ from `results/SizingChip.tsx`
 * only by case, which macOS folds onto one file and TypeScript rejects (TS1261).
 */

/** The limit's domain, in percent: more than 0 and at most 100% of equity. */
export const MAX_MARGIN_PCT = 100;

/** A parsed margin-limit field: off (empty), on (a validated fraction), or refused. */
export type MaxMarginPct =
  | { kind: "off" }
  | { kind: "on"; fraction: number }
  | { kind: "invalid"; message: string };

const DOMAIN = `Max margin in use must be more than 0% and at most ${MAX_MARGIN_PCT}% of equity`;

/**
 * The one parse rule for the field. An empty (or blank) field is OFF; otherwise the
 * text must read as a number in (0, 100], and its fraction is the run's
 * `max_margin_pct`.
 */
export function parseMaxMarginPct(raw: string): MaxMarginPct {
  const text = raw.trim();
  if (text === "") return { kind: "off" };

  // A single comma with no dot is a decimal comma ("25,5", Hungarian locale).
  const decimal = /^[^,.]*,[^,.]*$/.test(text) ? text.replace(",", ".") : text;

  const pct = Number(decimal);
  if (!Number.isFinite(pct) || pct <= 0 || pct > MAX_MARGIN_PCT) {
    return { kind: "invalid", message: `${DOMAIN} (you entered ${text})` };
  }
  return { kind: "on", fraction: Number((pct / MAX_MARGIN_PCT).toPrecision(12)) };
}

/** A stored fraction rendered as the percentage the forms and the chip show. */
export function marginPctText(fraction: number): string {
  return `${Number((fraction * MAX_MARGIN_PCT).toPrecision(12))}`;
}
