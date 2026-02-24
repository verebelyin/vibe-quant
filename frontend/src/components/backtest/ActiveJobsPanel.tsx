import { useEffect, useMemo, useRef } from "react";
import {
  getListJobsApiBacktestJobsGetQueryKey,
  useCleanupStaleJobsApiBacktestJobsCleanupStalePost,
  useKillJobApiBacktestJobsRunIdDelete,
  useListJobsApiBacktestJobsGet,
} from "@/api/generated/backtest/backtest";
import type { JobStatusResponse } from "@/api/generated/models";
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
import { useJobsWS } from "@/hooks/useJobsWS";
import type { WsStatus } from "@/hooks/useWebSocket";
import { cn } from "@/lib/utils";

const STALE_THRESHOLD_MS = 60_000;

const WS_INDICATOR: Record<WsStatus, { className: string; label: string }> = {
  connected: { className: "bg-green-500", label: "Connected" },
  connecting: { className: "bg-yellow-500", label: "Connecting" },
  disconnected: { className: "bg-red-500", label: "Disconnected" },
};

const STATUS_BADGE_VARIANTS: Record<string, { className: string }> = {
  queued: { className: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300" },
  running: {
    className: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 animate-pulse",
  },
  completed: { className: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300" },
  failed: { className: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300" },
  cancelled: { className: "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300" },
};

const FALLBACK_BADGE = "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300";

function formatElapsed(startedAt: string | null): string {
  if (!startedAt) return "--";
  const diffMs = Date.now() - new Date(startedAt).getTime();
  if (diffMs < 0) return "0s";
  const totalSec = Math.floor(diffMs / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function isStaleJob(job: JobStatusResponse): boolean {
  if (job.is_stale) return true;
  if (job.status !== "running" && job.status !== "queued") return false;
  if (!job.heartbeat_at) return false;
  return Date.now() - new Date(job.heartbeat_at).getTime() > STALE_THRESHOLD_MS;
}

export function ActiveJobsPanel() {
  const { status: wsStatus } = useJobsWS();
  const { data: jobsResp, isLoading } = useListJobsApiBacktestJobsGet({
    query: { refetchInterval: 15_000 },
  });

  const killMutation = useKillJobApiBacktestJobsRunIdDelete({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getListJobsApiBacktestJobsGetQueryKey(),
        });
      },
    },
  });

  const cleanupMutation = useCleanupStaleJobsApiBacktestJobsCleanupStalePost({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getListJobsApiBacktestJobsGetQueryKey(),
        });
      },
    },
  });

  const jobs: JobStatusResponse[] = useMemo(() => {
    if (!jobsResp) return [];
    if (jobsResp.status === 200) return jobsResp.data;
    return [];
  }, [jobsResp]);

  const hasStale = useMemo(() => jobs.some(isStaleJob), [jobs]);
  const indicator = WS_INDICATOR[wsStatus];

  // Auto-cleanup stale jobs
  const cleanupTriggered = useRef(false);
  useEffect(() => {
    if (hasStale && !cleanupMutation.isPending && !cleanupTriggered.current) {
      cleanupTriggered.current = true;
      cleanupMutation.mutate();
    }
    if (!hasStale) {
      cleanupTriggered.current = false;
    }
  }, [hasStale, cleanupMutation]);

  return (
    <div className="rounded-lg border border-border bg-background p-4">
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-foreground">Active Jobs</h2>
          <span className="flex items-center gap-1 text-xs text-muted-foreground">
            <span className={cn("inline-block h-2 w-2 rounded-full", indicator.className)} />
            {indicator.label}
          </span>
        </div>

        {hasStale && (
          <Button
            type="button"
            variant="destructive"
            size="xs"
            disabled={cleanupMutation.isPending}
            onClick={() => cleanupMutation.mutate()}
          >
            {cleanupMutation.isPending ? "Cleaning..." : "Cleanup Stale"}
          </Button>
        )}
      </div>

      {/* Table */}
      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading jobs...</p>
      ) : jobs.length === 0 ? (
        <p className="text-xs text-muted-foreground">No active jobs.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-xs">Run ID</TableHead>
              <TableHead className="text-xs">Type</TableHead>
              <TableHead className="text-xs">Status</TableHead>
              <TableHead className="text-xs">Elapsed</TableHead>
              <TableHead className="text-xs">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {jobs.map((job) => {
              const stale = isStaleJob(job);
              const normalized = job.status.toLowerCase();
              const badgeCls = STATUS_BADGE_VARIANTS[normalized]?.className ?? FALLBACK_BADGE;
              return (
                <TableRow key={job.run_id}>
                  <TableCell className="font-mono text-xs text-foreground">{job.run_id}</TableCell>
                  <TableCell className="text-xs text-foreground">{job.job_type}</TableCell>
                  <TableCell className="text-xs">
                    <div className="flex items-center gap-1.5">
                      <Badge variant="outline" className={cn("border-transparent", badgeCls)}>
                        {job.status}
                      </Badge>
                      {stale && (
                        <Badge
                          variant="outline"
                          className="border-transparent bg-amber-100 text-[10px] text-amber-700"
                        >
                          stale
                        </Badge>
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="font-mono text-xs text-foreground">
                    {formatElapsed(job.started_at)}
                  </TableCell>
                  <TableCell className="text-xs">
                    {(job.status === "running" || job.status === "queued") && (
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
