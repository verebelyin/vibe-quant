import { useState } from "react";
import { useGetSweepsApiResultsRunsRunIdSweepsGet } from "@/api/generated/results/results";
import { ChartsPanel } from "@/components/results/ChartsPanel";
import { CostBreakdown } from "@/components/results/CostBreakdown";
import { MetricsPanel } from "@/components/results/MetricsPanel";
import { OverfittingBadges } from "@/components/results/OverfittingBadges";
import { RunSelector } from "@/components/results/RunSelector";
import { SweepAnalysis } from "@/components/results/SweepAnalysis";
import { TradeLog } from "@/components/results/TradeLog";
import { EmptyState } from "@/components/ui";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

function SweepTab({ runId }: { runId: number }) {
  const query = useGetSweepsApiResultsRunsRunIdSweepsGet(runId);
  const sweeps = query.data?.data;

  // Hide tab content until we know sweep data exists
  if (query.isLoading || query.isError || !sweeps || sweeps.length === 0) return null;

  return <SweepAnalysis runId={runId} />;
}

function useSweepAvailable(runId: number | null) {
  const query = useGetSweepsApiResultsRunsRunIdSweepsGet(runId ?? 0, {
    query: { enabled: runId != null },
  });
  const sweeps = query.data?.data;
  return !query.isLoading && sweeps != null && sweeps.length > 0;
}

export function ResultsPage() {
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const hasSweeps = useSweepAvailable(selectedRunId);

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

          <Tabs defaultValue="charts">
            <TabsList>
              <TabsTrigger value="charts">Charts</TabsTrigger>
              <TabsTrigger value="trades">Trades</TabsTrigger>
              {hasSweeps && <TabsTrigger value="sweep">Sweep</TabsTrigger>}
            </TabsList>

            <TabsContent value="charts" className="pt-4">
              <ChartsPanel runId={selectedRunId} />
            </TabsContent>

            <TabsContent value="trades" className="pt-4">
              <TradeLog runId={selectedRunId} />
            </TabsContent>

            {hasSweeps && (
              <TabsContent value="sweep" className="pt-4">
                <SweepTab runId={selectedRunId} />
              </TabsContent>
            )}
          </Tabs>
        </>
      )}
    </div>
  );
}
