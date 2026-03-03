import { useMemo } from "react";
import { Link } from "@tanstack/react-router";
import { useListDiscoveryJobsApiDiscoveryJobsGet } from "@/api/generated/discovery/discovery";
import type { DiscoveryJobResponse } from "@/api/generated/models";
import { DiscoveryConfig } from "@/components/discovery/DiscoveryConfig";
import { Badge } from "@/components/ui/badge";

function formatDuration(start: string | null | undefined): string {
  if (!start) return "-";
  const secs = Math.max(0, Math.round((Date.now() - new Date(start).getTime()) / 1000));
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.round(secs / 60)}m`;
  const h = Math.floor(secs / 3600);
  const m = Math.round((secs % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

export function DiscoveryPage() {
  const { data: jobsResp } = useListDiscoveryJobsApiDiscoveryJobsGet({
    query: { refetchInterval: 5_000 },
  });

  const runningJobs: DiscoveryJobResponse[] = useMemo(() => {
    if (!jobsResp || jobsResp.status !== 200) return [];
    return jobsResp.data.filter((j) => j.status.toLowerCase() === "running");
  }, [jobsResp]);

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <DiscoveryConfig />

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-foreground">
            Running Discoveries
          </h3>
          <Link
            to="/discovery/results"
            className="text-xs text-blue-400 hover:text-blue-300 hover:underline"
          >
            All results &rarr;
          </Link>
        </div>

        {runningJobs.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No discoveries running.
          </p>
        ) : (
          <div className="space-y-2">
            {runningJobs.map((job) => {
              const p = job.progress as Record<string, unknown> | null;
              const gen = p ? Number(p.generation ?? p.current_generation ?? 0) : 0;
              const maxGen = p ? Number(p.max_generations ?? 0) : 0;
              const pct = maxGen > 0 ? Math.round((gen / maxGen) * 100) : 0;
              const bestFitness = p?.best_fitness ?? p?.fitness;
              const bestReturn = p?.best_return;
              const genTime = p?.gen_time;

              return (
                <div
                  key={job.run_id}
                  className="rounded-lg border border-blue-500/20 bg-blue-950/15 p-3"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 animate-pulse rounded-full bg-blue-400" />
                      <span className="font-mono text-xs font-medium text-foreground">
                        Run #{job.run_id}
                      </span>
                      {job.symbols && (
                        <Badge variant="outline" className="text-[10px]">
                          {job.symbols.join(", ")}
                        </Badge>
                      )}
                      {job.timeframe && (
                        <Badge variant="outline" className="text-[10px]">
                          {job.timeframe}
                        </Badge>
                      )}
                    </div>
                    <span className="text-[10px] text-muted-foreground">
                      {formatDuration(job.started_at)}
                    </span>
                  </div>

                  {/* Progress bar */}
                  <div className="mt-2 flex items-center gap-2">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-blue-500 transition-all duration-500"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="min-w-[3.5rem] text-right font-mono text-[10px] text-foreground">
                      {gen}/{maxGen || "?"}
                    </span>
                  </div>

                  {/* Metrics */}
                  <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-[10px]">
                    {bestFitness != null && (
                      <span className="text-muted-foreground">
                        Fitness:{" "}
                        <span className="font-mono text-foreground">
                          {Number(bestFitness).toFixed(4)}
                        </span>
                      </span>
                    )}
                    {bestReturn != null && (
                      <span className="text-muted-foreground">
                        Return:{" "}
                        <span className="font-mono text-foreground">
                          {typeof bestReturn === "number"
                            ? `${(bestReturn * 100).toFixed(1)}%`
                            : String(bestReturn)}
                        </span>
                      </span>
                    )}
                    {genTime != null && (
                      <span className="text-muted-foreground">
                        Gen time:{" "}
                        <span className="font-mono text-foreground">
                          {Number(genTime).toFixed(1)}s
                        </span>
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
