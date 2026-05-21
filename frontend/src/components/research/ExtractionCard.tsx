import { useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";
import type { ExtractionResponse } from "@/api/generated/models";
import {
  getGetItemApiResearchItemsItemIdGetQueryKey,
  useExtractItemApiResearchItemsItemIdExtractPost,
  usePromoteExtractionApiResearchExtractionsExtractionIdPromotePost,
  useRejectExtractionApiResearchExtractionsExtractionIdRejectPost,
  useRescreenExtractionApiResearchExtractionsExtractionIdRescreenPost,
} from "@/api/generated/research/research";
import { ConfidenceBar } from "@/components/research/ConfidenceBar";
import { ProposedIndicatorsList } from "@/components/research/ProposedIndicatorsList";
import { ScreenBadge, type ScreenSummary } from "@/components/research/ScreenBadge";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { YamlViewer } from "@/components/research/YamlViewer";
import { Button } from "@/components/ui/button";

interface Props {
  extraction: ExtractionResponse;
  itemId: number;
  index: number;
}

function formatTimestamp(iso: string | null): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  return new Date(t).toLocaleString();
}

export function ExtractionCard({ extraction, itemId, index }: Props) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const itemKey = getGetItemApiResearchItemsItemIdGetQueryKey(itemId);

  const promoteMut = usePromoteExtractionApiResearchExtractionsExtractionIdPromotePost();
  const rejectMut = useRejectExtractionApiResearchExtractionsExtractionIdRejectPost();
  const reExtractMut = useExtractItemApiResearchItemsItemIdExtractPost();
  const rescreenMut = useRescreenExtractionApiResearchExtractionsExtractionIdRescreenPost();

  const [promoting, setPromoting] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [reExtracting, setReExtracting] = useState(false);
  const [rescreening, setRescreening] = useState(false);

  const status = extraction.status;
  const isParsed = status === "parsed";
  const isFailed = status === "failed";
  const isPromoted = status === "promoted";
  const isRejected = status === "rejected";
  const canPromote = isParsed;
  const canReject = isParsed || isFailed;
  const canReExtract = !isPromoted && !isRejected;
  const canRescreen = isParsed && !!extraction.parsed_dsl_json;

  const handlePromote = () => {
    setPromoting(true);
    promoteMut.mutate(
      { extractionId: extraction.id },
      {
        onSuccess: (resp) => {
          if (resp.status === 200) {
            const strategyId = resp.data.strategy_id;
            toast.success("Strategy created");
            queryClient.invalidateQueries({ queryKey: itemKey });
            navigate({
              to: "/strategies/$strategyId",
              params: { strategyId: String(strategyId) },
            });
          } else {
            const detail = (resp.data as { detail?: unknown })?.detail;
            toast.error(typeof detail === "string" ? detail : "Promote failed");
          }
        },
        onError: () => toast.error("Promote failed"),
        onSettled: () => setPromoting(false),
      },
    );
  };

  const handleReject = () => {
    setRejecting(true);
    rejectMut.mutate(
      { extractionId: extraction.id },
      {
        onSuccess: (resp) => {
          if (resp.status === 200) {
            toast.success("Extraction rejected");
            queryClient.invalidateQueries({ queryKey: itemKey });
          } else {
            const detail = (resp.data as { detail?: unknown })?.detail;
            toast.error(typeof detail === "string" ? detail : "Reject failed");
          }
        },
        onError: () => toast.error("Reject failed"),
        onSettled: () => setRejecting(false),
      },
    );
  };

  const handleReExtract = () => {
    setReExtracting(true);
    reExtractMut.mutate(
      { itemId },
      {
        onSuccess: (resp) => {
          if (resp.status === 202) {
            toast.success("Extraction queued");
            queryClient.invalidateQueries({ queryKey: itemKey });
          } else {
            const detail = (resp.data as { detail?: unknown })?.detail;
            toast.error(typeof detail === "string" ? detail : "Re-extract failed");
          }
        },
        onError: () => toast.error("Re-extract failed"),
        onSettled: () => setReExtracting(false),
      },
    );
  };

  const handleRescreen = () => {
    setRescreening(true);
    rescreenMut.mutate(
      { extractionId: extraction.id },
      {
        onSuccess: (resp) => {
          if (resp.status === 200) {
            toast.success("Re-screen complete");
            queryClient.invalidateQueries({ queryKey: itemKey });
          } else {
            const detail = (resp.data as { detail?: unknown })?.detail;
            toast.error(typeof detail === "string" ? detail : "Re-screen failed");
          }
        },
        onError: () => toast.error("Re-screen failed"),
        onSettled: () => setRescreening(false),
      },
    );
  };

  return (
    <div className="rounded-md border border-border/40 bg-card/40 p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs">
          <span className="font-mono text-muted-foreground">#{index}</span>
          <StatusBadge status={status} />
          {extraction.llm_model && (
            <span className="text-muted-foreground">· {extraction.llm_model}</span>
          )}
          <span className="text-muted-foreground">
            · {formatTimestamp(extraction.extracted_at)}
          </span>
        </div>
        <ScreenBadge summary={extraction as unknown as ScreenSummary} />
      </div>

      {typeof extraction.confidence === "number" && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
            Confidence
          </span>
          <ConfidenceBar value={extraction.confidence} />
        </div>
      )}

      {extraction.rationale && (
        <div className="text-xs text-muted-foreground whitespace-pre-wrap">
          {extraction.rationale}
        </div>
      )}

      {isFailed && extraction.parse_error && (
        <div className="rounded-md border border-red-500/30 bg-red-500/5 p-3 text-xs text-red-300 whitespace-pre-wrap">
          {extraction.parse_error}
        </div>
      )}

      {isParsed && extraction.dsl_yaml && <YamlViewer yaml={extraction.dsl_yaml} />}

      <ProposedIndicatorsList json={extraction.proposed_indicators_json} />

      {isPromoted && extraction.strategy_id != null && (
        <div className="text-xs">
          <Link
            to="/strategies/$strategyId"
            params={{ strategyId: String(extraction.strategy_id) }}
            className="text-primary hover:underline"
          >
            View promoted strategy #{extraction.strategy_id} →
          </Link>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 pt-1">
        <Button
          size="sm"
          variant="default"
          disabled={!canPromote || promoting || isRejected}
          onClick={handlePromote}
          title={
            canPromote
              ? "Create strategy from extracted DSL"
              : "Only parsed extractions can be promoted"
          }
        >
          {promoting ? "Promoting…" : "Promote → Backtest"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={!canReject || rejecting || isRejected || isPromoted}
          onClick={handleReject}
        >
          {rejecting ? "Rejecting…" : "Reject"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={!canReExtract || reExtracting}
          onClick={handleReExtract}
        >
          {reExtracting ? "Re-extracting…" : "Re-extract"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={!canRescreen || rescreening}
          onClick={handleRescreen}
          title={
            canRescreen
              ? "Re-run the screening backtest for this extraction"
              : "Only parsed extractions with a valid DSL can be re-screened"
          }
        >
          {rescreening ? "Re-screening…" : "Re-screen"}
        </Button>
      </div>

      {extraction.prompt && (
        <details className="group">
          <summary className="cursor-pointer text-[10px] uppercase tracking-wide text-muted-foreground hover:text-foreground select-none">
            Prompt
          </summary>
          <pre className="mt-2 max-h-96 overflow-auto rounded-md border border-border/40 bg-muted/30 p-3 font-mono text-[10px] leading-relaxed text-foreground/80 whitespace-pre-wrap break-words">
            {extraction.prompt}
          </pre>
        </details>
      )}

      {extraction.raw_response && (
        <details className="group">
          <summary className="cursor-pointer text-[10px] uppercase tracking-wide text-muted-foreground hover:text-foreground select-none">
            Raw LLM response
          </summary>
          <pre className="mt-2 max-h-96 overflow-auto rounded-md border border-border/40 bg-muted/30 p-3 font-mono text-[10px] leading-relaxed text-foreground/80 whitespace-pre">
            {extraction.raw_response}
          </pre>
        </details>
      )}
    </div>
  );
}
