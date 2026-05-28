import { useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { ChevronLeft, ListChecks } from "lucide-react";
import { toast } from "sonner";
import {
  getListExtractionQueueApiResearchExtractionQueueGetQueryKey,
  getExtractionQueueStatusApiResearchExtractionQueueStatusGetQueryKey,
  useCancelExtractionJobApiResearchExtractionJobsJobIdCancelPost,
  useListExtractionQueueApiResearchExtractionQueueGet,
} from "@/api/generated/research/research";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { Button } from "@/components/ui/button";
import { formatRelative } from "@/lib/time";

export function ResearchQueuePage() {
  const queryClient = useQueryClient();
  const { data, isLoading } = useListExtractionQueueApiResearchExtractionQueueGet(undefined, {
    query: { refetchInterval: 3_000, refetchOnWindowFocus: true },
  });

  const cancelMutation = useCancelExtractionJobApiResearchExtractionJobsJobIdCancelPost({
    mutation: {
      onSuccess: (resp) => {
        // customInstance throws on non-2xx, so 409 (already running/finished)
        // and 404 (job gone) land in onError, not here — only 200 reaches us.
        if (resp.status === 200) {
          toast.success(`Job ${resp.data.id} cancelled`);
        }
        queryClient.invalidateQueries({
          queryKey: getListExtractionQueueApiResearchExtractionQueueGetQueryKey(),
        });
        queryClient.invalidateQueries({
          queryKey: getExtractionQueueStatusApiResearchExtractionQueueStatusGetQueryKey(),
        });
      },
      onError: (error) => {
        // The thrown Error message embeds the HTTP status ("API error: 409 …").
        const msg = error instanceof Error ? error.message : "";
        if (msg.includes("409")) {
          toast.error("Job already running or finished — cannot cancel");
        } else if (msg.includes("404")) {
          toast.error("Job not found");
        } else {
          toast.error("Cancel request failed");
        }
        // A 409/404 means our view is stale — refetch so the row updates.
        queryClient.invalidateQueries({
          queryKey: getListExtractionQueueApiResearchExtractionQueueGetQueryKey(),
        });
      },
    },
  });

  if (isLoading || !data) {
    return <div className="p-6 text-sm text-muted-foreground">Loading queue…</div>;
  }
  if (data.status !== 200) {
    return <div className="p-6 text-sm text-destructive">Failed to load queue.</div>;
  }

  const { jobs, active_count } = data.data;

  return (
    <div className="flex flex-col h-full gap-4">
      <div className="flex items-center gap-2 pb-2">
        <Link
          to="/research"
          className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="h-3.5 w-3.5" />
          Back to Research
        </Link>
        <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
          <ListChecks className="h-3.5 w-3.5 text-primary" />
          <span className="font-mono tabular-nums">{active_count}</span>
          <span>active</span>
        </div>
      </div>

      {jobs.length === 0 ? (
        <div className="flex flex-1 items-center justify-center rounded-md border border-dashed border-border/60 bg-card/30 p-12 text-center">
          <div className="space-y-2">
            <ListChecks className="mx-auto h-8 w-8 text-muted-foreground/40" />
            <p className="text-sm text-muted-foreground">Queue is empty.</p>
            <Link to="/research" className="text-xs text-primary hover:underline">
              Go to Research
            </Link>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-auto rounded-md border border-border/40 bg-card/40">
          <table className="w-full text-xs">
            <thead className="sticky top-0 z-10 bg-card/95 backdrop-blur border-b border-border/40">
              <tr className="text-left text-muted-foreground">
                <th className="px-3 py-2 font-medium w-16">Job</th>
                <th className="px-3 py-2 font-medium">Item</th>
                <th className="px-3 py-2 font-medium w-24">Status</th>
                <th className="px-3 py-2 font-medium w-20">Attempts</th>
                <th className="px-3 py-2 font-medium w-32">Queued</th>
                <th className="px-3 py-2 font-medium w-32">Started</th>
                <th className="px-3 py-2 font-medium w-24">Source</th>
                <th className="px-3 py-2 font-medium w-24" />
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => {
                const isQueued = job.status === "queued";
                return (
                  <tr key={job.id} className="border-b border-border/30 hover:bg-card/60">
                    <td className="px-3 py-2 font-mono tabular-nums text-muted-foreground">
                      {job.id}
                    </td>
                    <td className="px-3 py-2">
                      <Link
                        to="/research/$itemId"
                        params={{ itemId: String(job.research_item_id) }}
                        className="text-foreground hover:text-primary hover:underline"
                      >
                        {job.item_title ?? `Item #${job.research_item_id}`}
                      </Link>
                      {job.last_error && (
                        <div
                          className="mt-1 truncate text-[10px] text-destructive/80"
                          title={job.last_error}
                        >
                          {job.last_error}
                        </div>
                      )}
                    </td>
                    <td className="px-3 py-2">
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="px-3 py-2 font-mono tabular-nums text-muted-foreground">
                      {job.attempts}/{job.max_attempts}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {formatRelative(job.queued_at)}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {formatRelative(job.started_at)}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">{job.item_source}</td>
                    <td className="px-3 py-2 text-right">
                      {isQueued && (
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 text-[11px]"
                          disabled={cancelMutation.isPending}
                          onClick={() => cancelMutation.mutate({ jobId: job.id })}
                        >
                          Cancel
                        </Button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
