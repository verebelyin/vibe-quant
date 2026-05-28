import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import type { IndicatorScaffoldRow } from "@/api/generated/models";
import {
  getGetItemApiResearchItemsItemIdGetQueryKey,
  useScaffoldProposedIndicatorApiResearchExtractionsExtractionIdIndicatorsIdxScaffoldPost,
} from "@/api/generated/research/research";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface ProposedIndicator {
  name?: unknown;
  display_name?: unknown;
  description?: unknown;
  formula?: unknown;
  parameters?: unknown;
  output_range?: unknown;
  source_quote?: unknown;
}

interface Props {
  json: string | null | undefined;
  extractionId: number;
  itemId: number;
  scaffolds: IndicatorScaffoldRow[];
}

function asString(v: unknown): string | null {
  return typeof v === "string" && v.trim().length > 0 ? v : null;
}

function parseProposals(json: string | null | undefined): ProposedIndicator[] {
  if (!json) return [];
  try {
    const parsed: unknown = JSON.parse(json);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((x): x is ProposedIndicator => typeof x === "object" && x !== null);
  } catch {
    return [];
  }
}

function ScaffoldStatusPill({
  scaffold,
  pending,
  suggestedName,
}: {
  scaffold: IndicatorScaffoldRow | undefined;
  pending: boolean;
  suggestedName: string | null;
}) {
  // Reserve a min-width so the pill text swaps don't shift layout.
  const base = "min-w-[8rem] justify-center text-[10px]";
  if (pending) {
    return (
      <Badge variant="secondary" className={base}>
        scaffolding…
      </Badge>
    );
  }
  if (suggestedName) {
    return (
      <Badge variant="outline" className={`${base} text-amber-300 border-amber-500/40`}>
        collision: try {suggestedName}
      </Badge>
    );
  }
  if (!scaffold) {
    return (
      <Badge variant="outline" className={`${base} text-muted-foreground`}>
        not scaffolded
      </Badge>
    );
  }
  if (scaffold.status === "ok") {
    return (
      <Badge
        variant="outline"
        className={`${base} text-emerald-300 border-emerald-500/40 bg-emerald-500/10`}
      >
        scaffolded
      </Badge>
    );
  }
  if (
    scaffold.status === "codegen_failed" ||
    scaffold.status === "test_failed" ||
    scaffold.status === "invalid_input"
  ) {
    return (
      <Badge variant="outline" className={`${base} text-red-300 border-red-500/40 bg-red-500/10`}>
        failed: {scaffold.status.replace("_", " ")}
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className={base}>
      {scaffold.status}
    </Badge>
  );
}

function ProposalRow({
  proposal,
  idx,
  extractionId,
  itemId,
  scaffold,
}: {
  proposal: ProposedIndicator;
  idx: number;
  extractionId: number;
  itemId: number;
  scaffold: IndicatorScaffoldRow | undefined;
}) {
  const queryClient = useQueryClient();
  const itemKey = getGetItemApiResearchItemsItemIdGetQueryKey(itemId);
  const scaffoldMut =
    useScaffoldProposedIndicatorApiResearchExtractionsExtractionIdIndicatorsIdxScaffoldPost();
  const [suggestedName, setSuggestedName] = useState<string | null>(null);

  const name = asString(proposal.name);
  const display = asString(proposal.display_name);
  const desc = asString(proposal.description);
  const formula = asString(proposal.formula);
  const range = asString(proposal.output_range);
  const quote = asString(proposal.source_quote);
  const params =
    proposal.parameters && typeof proposal.parameters === "object"
      ? (proposal.parameters as Record<string, unknown>)
      : null;

  const hasFormula = !!formula;
  const isScaffolded = scaffold?.status === "ok";
  const pending = scaffoldMut.isPending;
  const isFailed = scaffold?.status === "codegen_failed" || scaffold?.status === "test_failed";

  const handleScaffold = () => {
    setSuggestedName(null);
    scaffoldMut.mutate(
      { extractionId, idx },
      {
        onSuccess: (resp) => {
          // Non-2xx throws in customInstance → onError; only the 200
          // IndicatorScaffoldResponse reaches here. Guard narrows the union.
          if (resp.status !== 200) return;
          const body = resp.data;
          if (body.status === "ok") {
            toast.success(`Scaffolded ${body.name ?? "indicator"}`);
            queryClient.invalidateQueries({ queryKey: itemKey });
            queryClient.invalidateQueries({ queryKey: ["indicators", "catalog"] });
          } else if (body.status === "name_collision") {
            setSuggestedName(body.suggested_name ?? null);
            toast.error(`Name collision: try ${body.suggested_name ?? "another name"}`);
          } else if (body.status === "already_scaffolded") {
            toast.info("Already scaffolded — no work to do");
            queryClient.invalidateQueries({ queryKey: itemKey });
            queryClient.invalidateQueries({ queryKey: ["indicators", "catalog"] });
          } else if (body.status === "invalid_input") {
            toast.error(body.error ?? "Invalid proposal");
            queryClient.invalidateQueries({ queryKey: itemKey });
            queryClient.invalidateQueries({ queryKey: ["indicators", "catalog"] });
          } else {
            // codegen_failed | test_failed — surface error inline via pill +
            // refetch so the cached row drives state on next reload.
            toast.error(`Scaffold failed: ${body.error ?? body.status}`);
            queryClient.invalidateQueries({ queryKey: itemKey });
            queryClient.invalidateQueries({ queryKey: ["indicators", "catalog"] });
          }
        },
        onError: () => toast.error("Scaffold request failed"),
      },
    );
  };

  const copySuggestedName = () => {
    if (!suggestedName) return;
    void navigator.clipboard.writeText(suggestedName);
    toast.success(`Copied "${suggestedName}"`);
  };

  // vscode://file/ link only works for absolute paths — server returns a
  // workspace-relative path, so prefix the project root if we can guess.
  const pluginPath = scaffold?.plugin_path ?? null;
  const vscodeUrl = pluginPath
    ? `vscode://file/${pluginPath.startsWith("/") ? pluginPath : `${window.location.pathname.startsWith("/") ? "" : ""}${pluginPath}`}`
    : null;

  return (
    <li className="flex flex-col gap-2 text-xs border-t border-amber-500/20 first:border-t-0 first:pt-0 pt-3">
      <div className="flex items-baseline flex-wrap gap-2">
        <code className="font-mono text-[11px] text-amber-300/95">{name ?? "(unnamed)"}</code>
        {display && <span className="text-foreground/80">{display}</span>}
        {range && (
          <Badge variant="outline" className="font-mono text-[10px] text-muted-foreground">
            {range}
          </Badge>
        )}
        <div className="ml-auto flex items-center gap-2">
          <ScaffoldStatusPill scaffold={scaffold} pending={pending} suggestedName={suggestedName} />
          {suggestedName && (
            <Button
              size="sm"
              variant="outline"
              onClick={copySuggestedName}
              className="h-6 text-[10px]"
            >
              Copy name
            </Button>
          )}
          <Button
            size="sm"
            variant="outline"
            onClick={handleScaffold}
            disabled={!hasFormula || pending || isScaffolded}
            title={
              !hasFormula
                ? "no formula — won't synthesize"
                : isScaffolded
                  ? "Already scaffolded — edit the plugin file directly"
                  : "Generate plugin via LLM + contract test + auto-commit"
            }
            className="h-6 text-[10px]"
          >
            {pending ? "Scaffolding…" : "Scaffold plugin"}
          </Button>
        </div>
      </div>
      {desc && <p className="text-muted-foreground leading-relaxed">{desc}</p>}
      {formula && (
        <pre className="overflow-auto rounded border border-border/40 bg-muted/30 p-2 font-mono text-[10px] leading-relaxed text-foreground/85 whitespace-pre">
          {formula}
        </pre>
      )}
      {params && Object.keys(params).length > 0 && (
        <div className="flex flex-col gap-0.5 text-[10px]">
          <span className="uppercase tracking-wide text-muted-foreground">Parameters</span>
          <ul className="font-mono text-foreground/80">
            {Object.entries(params).map(([k, v]) => (
              <li key={k}>
                {k}: {typeof v === "object" ? JSON.stringify(v) : String(v)}
              </li>
            ))}
          </ul>
        </div>
      )}
      {quote && (
        <blockquote className="border-l-2 border-amber-500/30 pl-2 italic text-muted-foreground">
          “{quote}”
        </blockquote>
      )}
      {isScaffolded && pluginPath && (
        <div className="flex items-center gap-2 text-[10px] text-emerald-300/90">
          <span className="font-mono">{pluginPath}</span>
          {vscodeUrl && (
            <a
              href={vscodeUrl}
              className="underline hover:text-emerald-200"
              title="Open in VS Code"
            >
              open
            </a>
          )}
          {scaffold?.commit_sha && (
            <code className="text-muted-foreground" title="Local commit SHA">
              {scaffold.commit_sha.slice(0, 7)}
            </code>
          )}
        </div>
      )}
      {isFailed && (scaffold?.error || scaffold?.test_output) && (
        <details className="group">
          <summary className="cursor-pointer text-[10px] uppercase tracking-wide text-red-300/80 hover:text-red-200 select-none">
            Failure details
          </summary>
          {scaffold?.error && (
            <div className="mt-1 text-[10px] text-red-300/90 font-mono">
              error: {scaffold.error}
            </div>
          )}
          {scaffold?.test_output && (
            <pre className="mt-1 max-h-64 overflow-auto rounded border border-red-500/30 bg-red-500/5 p-2 font-mono text-[10px] leading-relaxed text-red-200/90 whitespace-pre">
              {scaffold.test_output}
            </pre>
          )}
        </details>
      )}
    </li>
  );
}

export function ProposedIndicatorsList({ json, extractionId, itemId, scaffolds }: Props) {
  const proposals = parseProposals(json);
  if (proposals.length === 0) return null;
  // Index the scaffold rows by idx for O(1) lookup.
  const byIdx = new Map<number, IndicatorScaffoldRow>(scaffolds.map((s) => [s.idx, s]));
  return (
    <div className="rounded-md border border-amber-500/30 bg-amber-500/5 p-3 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <span className="text-[10px] uppercase tracking-wide text-amber-300/90 font-medium">
          Proposed indicators ({proposals.length})
        </span>
        <span className="text-[10px] text-muted-foreground">
          not yet implemented · scaffold to register as a plugin
        </span>
      </div>
      <ul className="flex flex-col gap-3">
        {proposals.map((p, idx) => (
          <ProposalRow
            key={`${asString(p.name) ?? "anon"}-${idx}`}
            proposal={p}
            idx={idx}
            extractionId={extractionId}
            itemId={itemId}
            scaffold={byIdx.get(idx)}
          />
        ))}
      </ul>
    </div>
  );
}
