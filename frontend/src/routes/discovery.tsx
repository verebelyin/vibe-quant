import { useMemo } from "react";
import { useListDiscoveryJobsApiDiscoveryJobsGet } from "@/api/generated/discovery/discovery";
import type { DiscoveryJobResponse } from "@/api/generated/models";
import { DiscoveryConfig } from "@/components/discovery/DiscoveryConfig";
import { DiscoveryJobList } from "@/components/discovery/DiscoveryJobList";
import { DiscoveryProgress } from "@/components/discovery/DiscoveryProgress";

export function DiscoveryPage() {
  const { data: jobsResp } = useListDiscoveryJobsApiDiscoveryJobsGet({
    query: { refetchInterval: 10_000 },
  });

  const jobs: DiscoveryJobResponse[] = useMemo(() => {
    if (!jobsResp) return [];
    if (jobsResp.status === 200) return jobsResp.data;
    return [];
  }, [jobsResp]);

  const runningJob = jobs.find((j) => j.status.toLowerCase() === "running");

  // Extract total generations from progress or fallback to 100
  const totalGens = runningJob
    ? Number(
        (runningJob.progress as Record<string, unknown> | null)?.total_generations ??
          (runningJob.progress as Record<string, unknown> | null)?.generations ??
          100,
      )
    : 100;

  return (
    <div className="mx-auto max-w-5xl space-y-8 p-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Discovery</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Use genetic algorithms to discover new trading strategies.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[1fr_1fr]">
        <DiscoveryConfig />
        <div className="rounded-lg border border-border bg-background p-4">
          <DiscoveryJobList />
        </div>
      </div>

      {runningJob && <DiscoveryProgress runId={runningJob.run_id} totalGenerations={totalGens} />}
    </div>
  );
}
