import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import type { IndicatorScaffoldRow } from "@/api/generated/models";
import {
  getGetItemApiResearchItemsItemIdGetQueryKey,
  useScaffoldProposedIndicatorApiResearchExtractionsExtractionIdIndicatorsIdxScaffoldPost,
} from "@/api/generated/research/research";
import { Button } from "@/components/ui/button";

interface Proposal {
  name?: unknown;
  formula?: unknown;
}

interface Props {
  missingTypes: string[];
  proposalsJson: string | null | undefined;
  scaffolds: IndicatorScaffoldRow[];
  extractionId: number;
  itemId: number;
  onCancel: () => void;
}

function parseProposals(json: string | null | undefined): Proposal[] {
  if (!json) return [];
  try {
    const parsed: unknown = JSON.parse(json);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((x): x is Proposal => typeof x === "object" && x !== null);
  } catch {
    return [];
  }
}

interface Row {
  type: string;
  proposalIdx: number | null;
  hasFormula: boolean;
  scaffold: IndicatorScaffoldRow | undefined;
}

function buildRows(
  missing: string[],
  proposals: Proposal[],
  scaffolds: IndicatorScaffoldRow[],
): Row[] {
  const byIdx = new Map<number, IndicatorScaffoldRow>(scaffolds.map((s) => [s.idx, s]));
  return missing.map((type) => {
    const target = type.toLowerCase();
    const idx = proposals.findIndex(
      (p) => typeof p.name === "string" && p.name.toLowerCase() === target,
    );
    if (idx < 0) return { type, proposalIdx: null, hasFormula: false, scaffold: undefined };
    const formula = proposals[idx]?.formula;
    return {
      type,
      proposalIdx: idx,
      hasFormula: typeof formula === "string" && formula.trim().length > 0,
      scaffold: byIdx.get(idx),
    };
  });
}

export function MissingIndicatorsBanner({
  missingTypes,
  proposalsJson,
  scaffolds,
  extractionId,
  itemId,
  onCancel,
}: Props) {
  const queryClient = useQueryClient();
  const itemKey = getGetItemApiResearchItemsItemIdGetQueryKey(itemId);
  const scaffoldMut =
    useScaffoldProposedIndicatorApiResearchExtractionsExtractionIdIndicatorsIdxScaffoldPost();
  const proposals = parseProposals(proposalsJson);
  const rows = buildRows(missingTypes, proposals, scaffolds);

  const handleScaffold = (idx: number, type: string) => {
    scaffoldMut.mutate(
      { extractionId, idx },
      {
        onSuccess: (resp) => {
          const body = resp.data;
          if (body.status === "ok") {
            toast.success(`Scaffolded ${type}`);
          } else if (body.status === "already_scaffolded") {
            toast.info(`${type} already scaffolded`);
          } else {
            toast.error(`Scaffold ${type} failed: ${body.error ?? body.status}`);
          }
          queryClient.invalidateQueries({ queryKey: itemKey });
          queryClient.invalidateQueries({ queryKey: ["indicators", "catalog"] });
        },
        onError: () => toast.error(`Scaffold ${type} request failed`),
      },
    );
  };

  return (
    <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-amber-200">
          Cannot promote: {missingTypes.length} indicator
          {missingTypes.length === 1 ? "" : "s"} not registered
        </span>
        <Button size="sm" variant="ghost" onClick={onCancel} className="h-6 text-[10px]">
          Cancel
        </Button>
      </div>
      <ul className="flex flex-col gap-1.5 text-xs">
        {rows.map((r) => {
          const isScaffolded = r.scaffold?.status === "ok";
          const pending = scaffoldMut.isPending && scaffoldMut.variables?.idx === r.proposalIdx;
          return (
            <li key={r.type} className="flex items-center gap-2">
              <code className="font-mono text-amber-300/95">{r.type}</code>
              {r.proposalIdx === null ? (
                <span className="text-muted-foreground">
                  no proposal — cannot auto-scaffold; edit DSL or extract again
                </span>
              ) : !r.hasFormula ? (
                <span className="text-muted-foreground">
                  proposal missing formula — cannot auto-scaffold
                </span>
              ) : isScaffolded ? (
                <span className="text-emerald-300/90">scaffolded · catalog refreshing…</span>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleScaffold(r.proposalIdx as number, r.type)}
                  disabled={pending}
                  className="h-6 text-[10px]"
                >
                  {pending ? "Scaffolding…" : "Scaffold plugin"}
                </Button>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
