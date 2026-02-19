import { useMemo } from "react";
import { toast } from "sonner";
import {
  getListDiscoveryJobsApiDiscoveryJobsGetQueryKey,
  useKillDiscoveryJobApiDiscoveryJobsRunIdDelete,
  useListDiscoveryJobsApiDiscoveryJobsGet,
} from "@/api/generated/discovery/discovery";
import type { DiscoveryJobResponse } from "@/api/generated/models";
import { queryClient } from "@/api/query-client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

const STATUS_BADGE: Record<string, string> = {
  running: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 animate-pulse",
  completed: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
  failed: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  queued: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  cancelled: "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
};
const FALLBACK_BADGE = "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300";

function formatDate(iso: string | null): string {
  if (!iso) return "--";
  return new Date(iso).toLocaleString();
}

function progressInfo(job: DiscoveryJobResponse): { generation: string; bestFitness: string } {
  const p = job.progress as Record<string, unknown> | null | undefined;
  if (!p) return { generation: "--", bestFitness: "--" };
  const gen = p.generation ?? p.current_generation;
  const fit = p.best_fitness ?? p.fitness;
  return {
    generation: gen != null ? String(gen) : "--",
    bestFitness: fit != null ? Number(fit).toFixed(4) : "--",
  };
}

interface DiscoveryJobListProps {
  selectedRunId?: number | null;
  onSelectRun?: (runId: number) => void;
}

export function DiscoveryJobList({ selectedRunId, onSelectRun }: DiscoveryJobListProps = {}) {
  const { data: jobsResp, isLoading } = useListDiscoveryJobsApiDiscoveryJobsGet({
    query: { refetchInterval: 10_000 },
  });

  const killMutation = useKillDiscoveryJobApiDiscoveryJobsRunIdDelete({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getListDiscoveryJobsApiDiscoveryJobsGetQueryKey(),
        });
        toast.success("Job killed");
      },
      onError: (err: unknown) => {
        const message = err instanceof Error ? err.message : "Kill failed";
        toast.error("Failed to kill job", { description: message });
      },
    },
  });

  const jobs: DiscoveryJobResponse[] = useMemo(() => {
    if (!jobsResp) return [];
    if (jobsResp.status === 200) return jobsResp.data;
    return [];
  }, [jobsResp]);

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-foreground">Discovery Jobs</h2>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading jobs...</p>
      ) : jobs.length === 0 ? (
        <p className="text-xs text-muted-foreground">No discovery jobs yet.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-xs">Run ID</TableHead>
              <TableHead className="text-xs">Status</TableHead>
              <TableHead className="text-xs">Started</TableHead>
              <TableHead className="text-xs">Generation</TableHead>
              <TableHead className="text-xs">Best Fitness</TableHead>
              <TableHead className="text-xs">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {jobs.map((job) => {
              const normalized = job.status.toLowerCase();
              const badgeCls = STATUS_BADGE[normalized] ?? FALLBACK_BADGE;
              const info = progressInfo(job);
              const isSelected = selectedRunId === job.run_id;
              const isClickable = normalized === "completed";
              return (
                <TableRow
                  key={job.run_id}
                  className={cn(
                    isClickable && "cursor-pointer hover:bg-muted/50",
                    isSelected && "bg-muted/40",
                  )}
                  onClick={isClickable && onSelectRun ? () => onSelectRun(job.run_id) : undefined}
                >
                  <TableCell className="font-mono text-xs text-foreground">{job.run_id}</TableCell>
                  <TableCell className="text-xs">
                    <Badge variant="outline" className={cn("border-transparent", badgeCls)}>
                      {job.status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs text-foreground">
                    {formatDate(job.started_at)}
                  </TableCell>
                  <TableCell className="font-mono text-xs text-foreground">
                    {info.generation}
                  </TableCell>
                  <TableCell className="font-mono text-xs text-foreground">
                    {info.bestFitness}
                  </TableCell>
                  <TableCell className="text-xs">
                    {(normalized === "running" || normalized === "queued") && (
                      <Button
                        type="button"
                        variant="destructive"
                        size="xs"
                        disabled={killMutation.isPending}
                        onClick={() => killMutation.mutate({ runId: job.run_id })}
                      >
                        Kill
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
