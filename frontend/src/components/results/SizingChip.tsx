import type { BacktestRunResponse } from "@/api/generated/models/backtestRunResponse";
import { useGetRunMetaApiResultsRunsRunIdMetaGet } from "@/api/generated/results/results";
import { Badge } from "@/components/ui/badge";
import { runSizing, sizingChipLabel } from "@/lib/sizing";

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
 */
export function sizingChipForRun(run: SizingRun | null | undefined): string | null {
  if (run == null) return null;
  if (run.run_mode === "paper") return null;
  return sizingChipLabel(runSizing(run.parameters), run.symbols.length);
}

interface SizingChipProps {
  runId: number;
}

/**
 * U6-U8: the run-level sizing chip, beside LiquidationChip on the results page.
 * The run comes from RunMeta, already fetched elsewhere on the page.
 */
export function SizingChip({ runId }: SizingChipProps) {
  const metaQuery = useGetRunMetaApiResultsRunsRunIdMetaGet(runId);
  const run = metaQuery.data?.status === 200 ? metaQuery.data.data : undefined;

  const label = sizingChipForRun(run);
  if (label == null) return null;

  return (
    <Badge variant="outline" className="font-mono text-[11px]">
      {label}
    </Badge>
  );
}
