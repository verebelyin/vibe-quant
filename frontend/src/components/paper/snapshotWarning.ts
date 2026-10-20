/** Title of the warning rendered when the node could not snapshot positions. */
export const SNAPSHOT_WARNING_TITLE = "Position snapshot incomplete";

const NO_OPEN_TEXT = "No open positions.";
const INCOMPLETE_TEXT = "Positions may be incomplete (checkpoint snapshot failed).";

export interface PositionsEmptyView {
  /** Non-blank snapshot errors, verbatim and in checkpoint order. */
  warnings: string[];
  /** Text for the zero-row state; never claims there are none after a failure. */
  emptyText: string;
}

/**
 * Map the `/api/paper/status` error list to what the Positions panels show.
 *
 * The node omits `snapshot_errors` when the checkpoint was clean, so the API
 * sends `[]`; the plain "No open positions." text is only safe in that case.
 * Any reported entry (even a blank one) means the positions listing may have
 * been truncated, so the empty state says so instead.
 */
export function positionsEmptyView(
  snapshotErrors: readonly string[] | null | undefined,
): PositionsEmptyView {
  const reported = snapshotErrors ?? [];
  return {
    warnings: reported.map((error) => error.trim()).filter((error) => error.length > 0),
    emptyText: reported.length > 0 ? INCOMPLETE_TEXT : NO_OPEN_TEXT,
  };
}
