import { Link } from "@tanstack/react-router";
import { ListChecks } from "lucide-react";
import { useExtractionQueueStatusApiResearchExtractionQueueStatusGet } from "@/api/generated/research/research";

export function ExtractionQueueBadge() {
  const { data } = useExtractionQueueStatusApiResearchExtractionQueueStatusGet({
    query: { refetchInterval: 5_000, refetchOnWindowFocus: true },
  });

  if (!data || data.status !== 200) return null;
  const { active_count, queued_count, running_count } = data.data;
  if (active_count === 0) return null;

  return (
    <Link
      to="/research/queue"
      className="flex items-center gap-1.5 rounded-md border border-border/60 bg-card/60 px-2 py-1 text-[11px] text-foreground/80 hover:bg-card hover:text-foreground"
      title={`${queued_count} queued · ${running_count} running`}
    >
      <ListChecks className="h-3.5 w-3.5 text-primary" />
      <span className="font-mono tabular-nums">{active_count}</span>
      <span className="text-muted-foreground">in queue</span>
    </Link>
  );
}
