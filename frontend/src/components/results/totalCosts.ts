/**
 * Total trading costs for a result row (vibe-quant-nre1o).
 *
 * Legacy results carry null fees/slippage/funding. When ALL three are missing
 * the total is genuinely unavailable: returning null lets the UI show "N/A"
 * instead of a measured-looking $0.00. Any non-null component (including an
 * explicit 0) makes the missing ones count as 0, matching the previous inline
 * `(fees ?? 0) + (slippage ?? 0) + (funding ?? 0)` sum.
 */
export function totalCosts(
  fees: number | null | undefined,
  slippage: number | null | undefined,
  funding: number | null | undefined,
): number | null {
  if (fees == null && slippage == null && funding == null) return null;
  return (fees ?? 0) + (slippage ?? 0) + (funding ?? 0);
}
