import type { ScrapeRunResponse } from "@/api/generated/models";
import { useGetLatestScrapeApiResearchScrapeLatestGet } from "@/api/generated/research/research";
import { formatRelative } from "@/lib/time";

interface Props {
  source: string;
}

export function ScrapeRunBanner({ source }: Props) {
  const { data, isLoading } = useGetLatestScrapeApiResearchScrapeLatestGet({ source });

  if (isLoading) {
    return <div className="text-xs text-muted-foreground/60">Loading scrape status…</div>;
  }

  const run: ScrapeRunResponse | null = data && data.status === 200 ? data.data : null;

  if (!run) {
    return (
      <div className="text-xs text-muted-foreground/60">
        No scrapes yet for <span className="font-mono text-foreground/80">{source}</span>
      </div>
    );
  }

  const ts = run.completed_at ?? run.heartbeat_at ?? run.started_at;
  const isFailed = run.status === "failed";

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
      <span>
        Last scrape: <span className="text-foreground/80">{formatRelative(ts, "never")}</span>
      </span>
      <span className="text-muted-foreground/30">·</span>
      <span className="font-mono tabular-nums">
        <span className="text-emerald-300">{run.items_new}</span> new
      </span>
      <span className="text-muted-foreground/30">·</span>
      <span className="font-mono tabular-nums">
        <span className="text-foreground/80">{run.items_extracted}</span> parsed
      </span>
      <span className="text-muted-foreground/30">·</span>
      <span className="font-mono tabular-nums">
        <span className="text-amber-300">{run.items_failed}</span> failed
      </span>
      <span className="text-muted-foreground/30">·</span>
      <span className="font-mono">
        status:{" "}
        <span className={isFailed ? "text-red-400" : "text-foreground/70"}>{run.status}</span>
      </span>
      {isFailed && run.error_message && (
        <span className="text-red-400/80 truncate max-w-md" title={run.error_message}>
          ({run.error_message})
        </span>
      )}
    </div>
  );
}
