import { useMemo } from "react";
import { Link } from "@tanstack/react-router";
import { useListDiscoveryJobsApiDiscoveryJobsGet } from "@/api/generated/discovery/discovery";
import type { DiscoveryJobResponse } from "@/api/generated/models";
import { DiscoveryConfig } from "@/components/discovery/DiscoveryConfig";
import { Badge } from "@/components/ui/badge";

export function DiscoveryPage() {
  const { data: jobsResp } = useListDiscoveryJobsApiDiscoveryJobsGet({
    query: { refetchInterval: 10_000 },
  });

  const runningJobs: DiscoveryJobResponse[] = useMemo(() => {
    if (!jobsResp || jobsResp.status !== 200) return [];
    return jobsResp.data.filter((j) => j.status.toLowerCase() === "running");
  }, [jobsResp]);

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      {runningJobs.length > 0 && (
        <div className="rounded-lg border border-blue-500/30 bg-blue-950/20 p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 animate-pulse rounded-full bg-blue-400" />
              <span className="text-sm font-medium text-foreground">
                {runningJobs.length} discovery{" "}
                {runningJobs.length === 1 ? "run" : "runs"} active
              </span>
            </div>
            <Link
              to="/discovery/results"
              className="text-xs text-blue-400 hover:text-blue-300 hover:underline"
            >
              View results &rarr;
            </Link>
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {runningJobs.map((job) => {
              const p = job.progress as Record<string, unknown> | null;
              const gen = p ? Number(p.generation ?? 0) : 0;
              const maxGen = p ? Number(p.max_generations ?? 0) : 0;
              return (
                <Badge
                  key={job.run_id}
                  variant="outline"
                  className="font-mono text-[10px]"
                >
                  #{job.run_id} — Gen {gen}/{maxGen || "?"}
                </Badge>
              );
            })}
          </div>
        </div>
      )}

      <DiscoveryConfig />
    </div>
  );
}
