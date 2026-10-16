import type { BacktestRunResponse } from "@/api/generated/models/backtestRunResponse";
import {
  useGetRunMetaApiResultsRunsRunIdMetaGet,
  useGetRunSummaryApiResultsRunsRunIdGet,
} from "@/api/generated/results/results";
import { Badge } from "@/components/ui/badge";
import { notesSizing, runSizing, sizingChipLabel } from "@/lib/sizing";

/** The run fields the chip reads (RunMeta's stored parameters). */
type SizingRun = Pick<BacktestRunResponse, "run_mode" | "symbols" | "parameters">;

/**
 * U6-U8: the SizingChip label for a run, or null when the chip renders nothing.
 *
 * Paper runs size through the validated run's `parameters.sizing`, not the
 * run-level sizing keys, so they have no chip. Every other mode reads its stored
 * sizing the one documented way (S6 `runSizing`): a legacy row shows the
 * defaults with "(default)", a row with a stored but unreadable value shows
 * "Sizing: invalid (...)". An absent run (still loading) renders nothing.
 *
 * Z1 (bead vibe-quant-zrh23): a discovery run launched from the CLI records its
 * sizing only in the machine `notes.sizing` — its run parameters carry no sizing
 * key — so when the parameters hold no sizing key the chip falls back to the
 * notes (`notesSizing`). The run's own parameters win; an absent or unreadable
 * `notes` keeps the legacy default; a present but unreadable `notes.sizing` is
 * refused with the "Sizing: invalid" text rather than silently defaulted.
 */
export function sizingChipForRun(
  run: SizingRun | null | undefined,
  notes?: string | null,
): string | null {
  if (run == null) return null;
  if (run.run_mode === "paper") return null;

  const fromParams = runSizing(run.parameters);
  // `legacy` means the parameters carry no sizing key (or are not a mapping), the
  // one case a discovery run's recorded notes.sizing has to supply.
  if (run.run_mode === "discovery" && "legacy" in fromParams && fromParams.legacy) {
    const recorded = notesSizing(notes);
    if (recorded != null) return sizingChipLabel(recorded, run.symbols.length);
  }
  return sizingChipLabel(fromParams, run.symbols.length);
}

interface SizingChipProps {
  runId: number;
}

/**
 * U6-U8: the run-level sizing chip, beside LiquidationChip on the results page.
 * The run comes from RunMeta, already fetched elsewhere on the page; the machine
 * `notes` (a discovery run's recorded sizing, Z1) come from the run summary,
 * likewise already fetched by the page's other panels.
 */
export function SizingChip({ runId }: SizingChipProps) {
  const metaQuery = useGetRunMetaApiResultsRunsRunIdMetaGet(runId);
  const run = metaQuery.data?.status === 200 ? metaQuery.data.data : undefined;
  const summaryQuery = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const notes = summaryQuery.data?.status === 200 ? summaryQuery.data.data.notes : undefined;

  const label = sizingChipForRun(run, notes);
  if (label == null) return null;

  return (
    <Badge variant="outline" className="font-mono text-[11px]">
      {label}
    </Badge>
  );
}
