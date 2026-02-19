import { useMemo, useState } from "react";
import { useListDiscoveryJobsApiDiscoveryJobsGet } from "@/api/generated/discovery/discovery";
import type { DiscoveryJobResponse } from "@/api/generated/models";
import { DiscoveryConfig, type DiscoveryConvergenceConfig } from "@/components/discovery/DiscoveryConfig";
import { DiscoveryJobList } from "@/components/discovery/DiscoveryJobList";
import { DiscoveryProgress } from "@/components/discovery/DiscoveryProgress";
import { DiscoveryResults } from "@/components/discovery/DiscoveryResults";

export function DiscoveryPage() {
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [convergence, setConvergence] = useState<DiscoveryConvergenceConfig>({
    convergenceWindow: 5,
    convergenceThreshold: 0.001,
  });

  const { data: jobsResp } = useListDiscoveryJobsApiDiscoveryJobsGet({
    query: { refetchInterval: 10_000 },
  });

  const jobs: DiscoveryJobResponse[] = useMemo(() => {
    if (!jobsResp) return [];
    if (jobsResp.status === 200) return jobsResp.data;
    return [];
  }, [jobsResp]);

  const runningJob = jobs.find((j) => j.status.toLowerCase() === "running");

  // Auto-select latest completed job if none selected
  const completedJobs = jobs.filter((j) => j.status.toLowerCase() === "completed");
  const effectiveRunId =
    selectedRunId ?? (completedJobs.length > 0 ? completedJobs[0].run_id : null);

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

      <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
        <DiscoveryConfig onConvergenceChange={setConvergence} />
        <div className="rounded-xl border border-border/60 bg-card/40 p-5 backdrop-blur-sm">
          <DiscoveryJobList selectedRunId={effectiveRunId} onSelectRun={setSelectedRunId} />
        </div>
      </div>

      {runningJob && (
        <DiscoveryProgress
          runId={runningJob.run_id}
          totalGenerations={totalGens}
          convergenceWindow={convergence.convergenceWindow}
          convergenceThreshold={convergence.convergenceThreshold}
        />
      )}

      <DiscoveryResults runId={effectiveRunId} />
    </div>
  );
}
