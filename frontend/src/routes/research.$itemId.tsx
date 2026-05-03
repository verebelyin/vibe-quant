import { useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { ChevronLeft, ExternalLink } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import {
  getGetItemApiResearchItemsItemIdGetQueryKey,
  useExtractItemApiResearchItemsItemIdExtractPost,
  useGetItemApiResearchItemsItemIdGet,
} from "@/api/generated/research/research";
import { ExtractionCard } from "@/components/research/ExtractionCard";
import { PostContent } from "@/components/research/PostContent";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatRelative } from "@/lib/time";

interface Props {
  itemId: number;
}

interface Comment {
  author?: string | null;
  body?: string | null;
  score?: number | null;
}

function getComments(extras: { [k: string]: unknown } | null | undefined): Comment[] {
  if (!extras) return [];
  const c = extras.comments;
  if (!Array.isArray(c)) return [];
  return c.map((row) => {
    if (typeof row !== "object" || row === null) return {};
    const r = row as Record<string, unknown>;
    return {
      author: typeof r.author === "string" ? r.author : null,
      body: typeof r.body === "string" ? r.body : null,
      score: typeof r.score === "number" ? r.score : null,
    };
  });
}

export function ResearchItemPage({ itemId }: Props) {
  const queryClient = useQueryClient();
  const itemKey = getGetItemApiResearchItemsItemIdGetQueryKey(itemId);
  const { data, isLoading, isError } = useGetItemApiResearchItemsItemIdGet(itemId, {
    query: {
      refetchInterval: (q) => {
        const r = q.state.data;
        if (r?.status !== 200) return false;
        return r.data.extraction_status === "running" ? 2_000 : false;
      },
    },
  });
  const reExtractMut = useExtractItemApiResearchItemsItemIdExtractPost();
  const [running, setRunning] = useState(false);

  if (isLoading) {
    return <div className="py-16 text-center text-sm text-muted-foreground">Loading item…</div>;
  }

  if (isError || !data || data.status !== 200) {
    return (
      <div className="flex flex-col items-center gap-4 py-16">
        <div className="text-sm text-red-400">Failed to load item.</div>
        <Button asChild variant="outline" size="sm">
          <Link to="/research">Back to research</Link>
        </Button>
      </div>
    );
  }

  const item = data.data;
  const comments = getComments(item.extras);
  const extractions = [...item.extractions].sort((a, b) => {
    const ta = a.extracted_at ? Date.parse(a.extracted_at) : 0;
    const tb = b.extracted_at ? Date.parse(b.extracted_at) : 0;
    return tb - ta;
  });

  const handleRunExtraction = () => {
    setRunning(true);
    reExtractMut.mutate(
      { itemId },
      {
        onSuccess: (resp) => {
          if (resp.status === 202) {
            toast.success("Extraction queued");
            queryClient.invalidateQueries({ queryKey: itemKey });
          } else {
            const detail = (resp.data as { detail?: unknown })?.detail;
            toast.error(typeof detail === "string" ? detail : "Extraction failed");
          }
        },
        onError: () => toast.error("Extraction failed"),
        onSettled: () => setRunning(false),
      },
    );
  };

  return (
    <div className="flex flex-col h-full gap-4 overflow-auto">
      <div className="flex items-center justify-between">
        <Button asChild variant="ghost" size="sm" className="gap-1.5 h-8 text-xs">
          <Link to="/research">
            <ChevronLeft className="h-3 w-3" />
            Back to research
          </Link>
        </Button>
        <StatusBadge status={item.extraction_status} />
      </div>

      <div className="flex flex-col gap-2">
        <h1 className="text-lg font-semibold text-foreground">{item.title || "(untitled)"}</h1>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          {item.author && <span>by {item.author}</span>}
          <Badge variant="outline" className="font-mono text-[10px] text-muted-foreground">
            {item.source}
          </Badge>
          <span>· {formatRelative(item.posted_at ?? item.fetched_at)}</span>
          {typeof item.score === "number" && <span className="tabular-nums">· {item.score} ↑</span>}
          <a
            href={item.url}
            target="_blank"
            rel="noreferrer"
            className="ml-1 inline-flex items-center gap-1 text-primary hover:underline"
          >
            open <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="flex flex-col gap-3">
          <h2 className="text-[10px] uppercase tracking-wide text-muted-foreground">Post</h2>
          <PostContent body={item.body} comments={comments} />
        </div>

        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h2 className="text-[10px] uppercase tracking-wide text-muted-foreground">
              Extractions ({extractions.length})
            </h2>
            {extractions.length > 0 && (
              <Button
                size="sm"
                variant="outline"
                disabled={running}
                onClick={handleRunExtraction}
                className="h-7 px-2 text-xs"
              >
                {running ? "Running…" : "+ Re-run extraction"}
              </Button>
            )}
          </div>

          {extractions.length === 0 ? (
            <div className="flex flex-col items-center gap-3 rounded-md border border-border/40 bg-card/40 p-8 text-center">
              <p className="text-sm text-muted-foreground">No extractions yet</p>
              <Button size="sm" variant="default" disabled={running} onClick={handleRunExtraction}>
                {running ? "Running…" : "Run extraction"}
              </Button>
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              {extractions.map((e, idx) => (
                <ExtractionCard
                  key={e.id}
                  extraction={e}
                  itemId={itemId}
                  index={extractions.length - idx}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
