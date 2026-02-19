import { useMemo } from "react";
import {
  getListJobsApiBacktestJobsGetQueryKey,
  useCleanupStaleJobsApiBacktestJobsCleanupStalePost,
  useKillJobApiBacktestJobsRunIdDelete,
  useListJobsApiBacktestJobsGet,
} from "@/api/generated/backtest/backtest";
import type { JobStatusResponse } from "@/api/generated/models";
import { queryClient } from "@/api/query-client";
import { useJobsWS } from "@/hooks/useJobsWS";
import type { WsStatus } from "@/hooks/useWebSocket";
import { JobStatusBadge } from "./JobStatusBadge";

const STALE_THRESHOLD_MS = 60_000;

const WS_INDICATOR: Record<WsStatus, { color: string; label: string }> = {
  connected: { color: "#22c55e", label: "Connected" },
  connecting: { color: "#eab308", label: "Connecting" },
  disconnected: { color: "#ef4444", label: "Disconnected" },
};

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
  const { data: jobsResp, isLoading } = useListJobsApiBacktestJobsGet();

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

  return (
    <div
      className="rounded-lg border p-4"
      style={{ borderColor: "var(--border)", backgroundColor: "var(--background)" }}
    >
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
            Active Jobs
          </h2>
          <span className="flex items-center gap-1 text-xs" style={{ color: "var(--muted)" }}>
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ backgroundColor: indicator.color }}
            />
            {indicator.label}
          </span>
        </div>

        {hasStale && (
          <button
            type="button"
            className="rounded px-2 py-1 text-xs font-medium text-white"
            style={{ backgroundColor: "var(--destructive)" }}
            disabled={cleanupMutation.isPending}
            onClick={() => cleanupMutation.mutate()}
          >
            {cleanupMutation.isPending ? "Cleaning..." : "Cleanup Stale"}
          </button>
        )}
      </div>

      {/* Table */}
      {isLoading ? (
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          Loading jobs...
        </p>
      ) : jobs.length === 0 ? (
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          No active jobs.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr
                className="border-b text-left"
                style={{ borderColor: "var(--border)", color: "var(--muted)" }}
              >
                <th className="pb-1 pr-3 font-medium">Run ID</th>
                <th className="pb-1 pr-3 font-medium">Type</th>
                <th className="pb-1 pr-3 font-medium">Status</th>
                <th className="pb-1 pr-3 font-medium">Elapsed</th>
                <th className="pb-1 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => {
                const stale = isStaleJob(job);
                return (
                  <tr
                    key={job.run_id}
                    className="border-b"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <td className="py-1.5 pr-3 font-mono" style={{ color: "var(--foreground)" }}>
                      {job.run_id}
                    </td>
                    <td className="py-1.5 pr-3" style={{ color: "var(--foreground)" }}>
                      {job.job_type}
                    </td>
                    <td className="py-1.5 pr-3">
                      <div className="flex items-center gap-1.5">
                        <JobStatusBadge status={job.status} />
                        {stale && (
                          <span className="inline-flex items-center rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-700">
                            stale
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-1.5 pr-3 font-mono" style={{ color: "var(--foreground)" }}>
                      {formatElapsed(job.started_at)}
                    </td>
                    <td className="py-1.5">
                      {(job.status === "running" || job.status === "queued") && (
                        <button
                          type="button"
                          className="rounded px-1.5 py-0.5 text-[10px] font-medium text-white"
                          style={{ backgroundColor: "var(--destructive)" }}
                          disabled={killMutation.isPending}
                          onClick={() => killMutation.mutate({ runId: job.run_id })}
                        >
                          Kill
                        </button>
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
