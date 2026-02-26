import { useEffect, useMemo, useState } from "react";
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

interface ProgressInfo {
  generation: string;
  maxGenerations: string;
  bestFitness: string;
  bestReturn: string;
  bestTrades: string;
  eta: string;
  pct: number;
  genTime: string;
}

function formatDuration(seconds: number): string {
  if (seconds <= 0) return "--";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function progressInfo(job: DiscoveryJobResponse): ProgressInfo {
  const p = job.progress as Record<string, unknown> | null | undefined;
  const empty: ProgressInfo = {
    generation: "--", maxGenerations: "--", bestFitness: "--",
    bestReturn: "--", bestTrades: "--", eta: "--", pct: 0, genTime: "--",
  };
  if (!p) return empty;

  const gen = Number(p.generation ?? p.current_generation ?? 0);
  const maxGen = Number(p.max_generations ?? 0);
  const fit = p.best_fitness ?? p.fitness;
  const ret = p.best_return;
  const trades = p.best_trades;
  const eta = Number(p.eta_seconds ?? 0);
  const genTime = Number(p.gen_time ?? 0);
  const pct = maxGen > 0 ? Math.round((gen / maxGen) * 100) : 0;

  return {
    generation: gen > 0 ? String(gen) : "--",
    maxGenerations: maxGen > 0 ? String(maxGen) : "--",
    bestFitness: fit != null ? Number(fit).toFixed(4) : "--",
    bestReturn: typeof ret === "number" ? `${(ret * 100).toFixed(1)}%` : "--",
    bestTrades: trades != null ? String(trades) : "--",
    eta: eta > 0 ? formatDuration(eta) : "--",
    pct,
    genTime: genTime > 0 ? `${genTime.toFixed(1)}s` : "--",
  };
}

interface DiscoveryJobListProps {
  selectedRunId?: number | null;
  onSelectRun?: (runId: number) => void;
}

export function DiscoveryJobList({ selectedRunId, onSelectRun }: DiscoveryJobListProps = {}) {
  const [hasRunning, setHasRunning] = useState(false);
  const { data: jobsResp, isLoading } = useListDiscoveryJobsApiDiscoveryJobsGet({
    query: { refetchInterval: hasRunning ? 15_000 : false },
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

  useEffect(() => {
    setHasRunning(jobs.some((j) => j.status.toLowerCase() === "running"));
  }, [jobs]);

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-foreground">Discovery Jobs</h2>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading jobs...</p>
      ) : jobs.length === 0 ? (
        <p className="text-xs text-muted-foreground">No discovery jobs yet.</p>
      ) : (
        <div className="space-y-2">
          {jobs.map((job) => {
            const normalized = job.status.toLowerCase();
            const badgeCls = STATUS_BADGE[normalized] ?? FALLBACK_BADGE;
            const info = progressInfo(job);
            const isSelected = selectedRunId === job.run_id;
            const isClickable = normalized === "completed";
            const isRunning = normalized === "running";
            return (
              <div
                key={job.run_id}
                className={cn(
                  "rounded-lg border border-border bg-card p-3 transition-colors",
                  isClickable && "cursor-pointer hover:bg-muted/50",
                  isSelected && "bg-muted/40 border-primary/50",
                )}
                onClick={isClickable && onSelectRun ? () => onSelectRun(job.run_id) : undefined}
              >
                {/* Header row */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-muted-foreground">#{job.run_id}</span>
                    <Badge variant="outline" className={cn("border-transparent text-[10px]", badgeCls)}>
                      {job.status}
                    </Badge>
                    <span className="text-[10px] text-muted-foreground">{formatDate(job.started_at)}</span>
                  </div>
                  {(isRunning || normalized === "queued") && (
                    <Button
                      type="button"
                      variant="destructive"
                      size="xs"
                      disabled={killMutation.isPending}
                      onClick={(e) => { e.stopPropagation(); killMutation.mutate({ runId: job.run_id }); }}
                    >
                      Kill
                    </Button>
                  )}
                </div>

                {/* Progress bar + metrics (only when we have progress data) */}
                {info.generation !== "--" && (
                  <div className="mt-2 space-y-1.5">
                    {/* Progress bar */}
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-primary transition-all duration-500"
                          style={{ width: `${info.pct}%` }}
                        />
                      </div>
                      <span className="min-w-[3rem] text-right font-mono text-[10px] text-foreground">
                        {info.generation}/{info.maxGenerations}
                      </span>
                    </div>
                    {/* Metrics row */}
                    <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-[10px]">
                      <span className="text-muted-foreground">
                        Fitness: <span className="font-mono text-foreground">{info.bestFitness}</span>
                      </span>
                      <span className="text-muted-foreground">
                        Return: <span className="font-mono text-foreground">{info.bestReturn}</span>
                      </span>
                      <span className="text-muted-foreground">
                        Trades: <span className="font-mono text-foreground">{info.bestTrades}</span>
                      </span>
                      <span className="text-muted-foreground">
                        Gen time: <span className="font-mono text-foreground">{info.genTime}</span>
                      </span>
                      {isRunning && info.eta !== "--" && (
                        <span className="text-muted-foreground">
                          ETA: <span className="font-mono text-foreground">{info.eta}</span>
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* No progress yet */}
                {info.generation === "--" && isRunning && (
                  <p className="mt-1.5 text-[10px] text-muted-foreground animate-pulse">
                    Evaluating initial population...
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
