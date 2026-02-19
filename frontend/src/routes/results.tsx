import { useState } from "react";
import { CostBreakdown } from "@/components/results/CostBreakdown";
import { MetricsPanel } from "@/components/results/MetricsPanel";
import { OverfittingBadges } from "@/components/results/OverfittingBadges";
import { RunSelector } from "@/components/results/RunSelector";
import { EmptyState } from "@/components/ui";

export function ResultsPage() {
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-2xl font-bold text-foreground">Results Analysis</h1>

      <RunSelector selectedRunId={selectedRunId} onSelectRun={setSelectedRunId} />

      {selectedRunId == null ? (
        <EmptyState
          title="No run selected"
          description="Select a backtest run above to view performance metrics, cost breakdown, and overfitting analysis."
        />
      ) : (
        <>
          <MetricsPanel runId={selectedRunId} />

          <div className="grid gap-6 lg:grid-cols-2">
            <CostBreakdown runId={selectedRunId} />
            <OverfittingBadges runId={selectedRunId} />
          </div>
        </>
      )}
    </div>
  );
}
