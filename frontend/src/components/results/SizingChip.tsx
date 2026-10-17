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
 * T6 (bead vibe-quant-t8nrv): the validation runner's machine `notes.margin_limit`
 * (`{"max_margin_pct", "refused_entries"}`), written only when the limit was ON.
 * `refused` is the validation-only count; `maxMarginPct` is the recorded limit, the
 * last fallback when the run's parameters and notes.sizing carry none.
 */
function marginLimitNotes(notes: string | null | undefined): {
  maxMarginPct?: number;
  refused?: number;
} {
  if (notes == null) return {};

  let payload: unknown;
  try {
    payload = JSON.parse(notes);
  } catch {
    return {};
  }
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return {};

  const limit = (payload as Record<string, unknown>).margin_limit;
  if (typeof limit !== "object" || limit === null || Array.isArray(limit)) return {};

  const recorded = limit as Record<string, unknown>;
  const out: { maxMarginPct?: number; refused?: number } = {};
  if (typeof recorded.max_margin_pct === "number" && Number.isFinite(recorded.max_margin_pct)) {
    out.maxMarginPct = recorded.max_margin_pct;
  }
  if (typeof recorded.refused_entries === "number" && Number.isFinite(recorded.refused_entries)) {
    out.refused = recorded.refused_entries;
  }
  return out;
}

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
 *
 * T6 (bead vibe-quant-t8nrv): the run-level margin limit ("Max margin 50%") comes from
 * the run's `parameters.max_margin_pct` (every run type), else a discovery run's
 * `notes.sizing`, else the validation runner's `notes.margin_limit`; all read through
 * `lib/sizing`. The "N entries refused" count is validation-only (`notes.margin_limit`).
 */
export function sizingChipForRun(
  run: SizingRun | null | undefined,
  notes?: string | null,
): string | null {
  if (run == null) return null;
  if (run.run_mode === "paper") return null;

  const fromParams = runSizing(run.parameters);
  const limit = marginLimitNotes(notes);
  const refused =
    run.run_mode === "validation" && limit.refused !== undefined
      ? ` · ${limit.refused} entries refused`
      : "";
  // `legacy` means the parameters carry no sizing key (or are not a mapping), the
  // one case a discovery run's recorded notes.sizing has to supply.
  const recorded = run.run_mode === "discovery" ? notesSizing(notes) : null;
  const useNotes =
    recorded != null && "legacy" in fromParams && fromParams.legacy ? recorded : fromParams;
  if ("invalid" in useNotes) return sizingChipLabel(useNotes, run.symbols.length);

  const maxMarginPct =
    ("maxMarginPct" in fromParams ? fromParams.maxMarginPct : undefined) ??
    (recorded != null && "maxMarginPct" in recorded ? recorded.maxMarginPct : undefined) ??
    limit.maxMarginPct;
  const sizing = maxMarginPct === undefined ? useNotes : { ...useNotes, maxMarginPct };
  return `${sizingChipLabel(sizing, run.symbols.length)}${refused}`;
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
