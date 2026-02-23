import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  getGetRunSummaryApiResultsRunsRunIdGetQueryKey,
  getGetSweepsApiResultsRunsRunIdSweepsGetQueryKey,
  getGetTradesApiResultsRunsRunIdTradesGetQueryKey,
  useGetSweepsApiResultsRunsRunIdSweepsGet,
  useListRunsApiResultsRunsGet,
} from "@/api/generated/results/results";
import { ChartsPanel } from "@/components/results/ChartsPanel";
import { ComparisonView } from "@/components/results/ComparisonView";
import { CostBreakdown } from "@/components/results/CostBreakdown";
import { ExportPanel } from "@/components/results/ExportPanel";
import { FuturesAnalytics } from "@/components/results/FuturesAnalytics";
import { LiquidationSummary } from "@/components/results/LiquidationSummary";
import { LongShortSplit } from "@/components/results/LongShortSplit";
import { MetricsPanel } from "@/components/results/MetricsPanel";
import { NotesPanel } from "@/components/results/NotesPanel";
import { OverfittingBadges } from "@/components/results/OverfittingBadges";
import { RawStatsPanel } from "@/components/results/RawStatsPanel";
import { RunDetailsExpander } from "@/components/results/RunDetailsExpander";
import { RunSelector } from "@/components/results/RunSelector";
import { SweepAnalysis } from "@/components/results/SweepAnalysis";
import { TradeLog } from "@/components/results/TradeLog";
import { WinLossPanel } from "@/components/results/WinLossPanel";
import { EmptyState } from "@/components/ui";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

type ViewMode = "single" | "compare";

const POLL_INTERVAL_MS = 3000;

/** Poll runs list while selected run is "running"; invalidate result queries on completion. */
function useRunPolling(selectedRunId: number | null) {
  const queryClient = useQueryClient();
  const prevStatus = useRef<string | null>(null);

  const runsQuery = useListRunsApiResultsRunsGet(undefined, {
    query: {
      enabled: selectedRunId != null,
      refetchInterval: (query) => {
        const resp = query.state.data;
        const runs = resp && resp.status === 200 ? resp.data.runs : [];
        const run = runs.find((r: { id: number }) => r.id === selectedRunId);
        return run?.status === "running" ? POLL_INTERVAL_MS : false;
      },
    },
  });

  useEffect(() => {
    if (selectedRunId == null) return;
    const resp = runsQuery.data;
    const runs = resp && resp.status === 200 ? resp.data.runs : [];
    const run = runs.find((r: { id: number; status: string }) => r.id === selectedRunId);
    const status = run?.status ?? null;

    if (
      prevStatus.current === "running" &&
      (status === "completed" || status === "failed")
    ) {
      // Invalidate all result-related queries for this run
      queryClient.invalidateQueries({
        queryKey: getGetRunSummaryApiResultsRunsRunIdGetQueryKey(selectedRunId),
      });
      queryClient.invalidateQueries({
        queryKey: getGetSweepsApiResultsRunsRunIdSweepsGetQueryKey(selectedRunId),
      });
      queryClient.invalidateQueries({
        queryKey: getGetTradesApiResultsRunsRunIdTradesGetQueryKey(selectedRunId),
      });
    }
    prevStatus.current = status;
  }, [runsQuery.data, selectedRunId, queryClient]);
}

function SweepTab({ runId }: { runId: number }) {
  const query = useGetSweepsApiResultsRunsRunIdSweepsGet(runId);
  const resp = query.data;
  const sweeps = resp && resp.status === 200 ? resp.data : null;

  if (query.isLoading || query.isError || !sweeps || sweeps.length === 0) return null;

  return <SweepAnalysis runId={runId} />;
}

function useSweepAvailable(runId: number | null) {
  const query = useGetSweepsApiResultsRunsRunIdSweepsGet(runId ?? 0, undefined, {
    query: { enabled: runId != null },
  });
  const resp = query.data;
  const sweeps = resp && resp.status === 200 ? resp.data : null;
  return !query.isLoading && sweeps != null && sweeps.length > 0;
}

export function ResultsPage() {
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("single");
  const hasSweeps = useSweepAvailable(selectedRunId);

  useRunPolling(selectedRunId);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-foreground">Results Analysis</h1>
        <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as ViewMode)}>
          <TabsList>
            <TabsTrigger value="single">Single Run</TabsTrigger>
            <TabsTrigger value="compare">Compare</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {viewMode === "compare" ? (
        <ComparisonView />
      ) : (
        <>
          <RunSelector selectedRunId={selectedRunId} onSelectRun={setSelectedRunId} />

          {selectedRunId == null ? (
            <EmptyState
              title="No run selected"
              description="Select a backtest run above to view performance metrics, cost breakdown, and overfitting analysis."
            />
          ) : (
            <>
              <RunDetailsExpander runId={selectedRunId} />

              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                  Run #{selectedRunId}
                </h3>
                <ExportPanel runId={selectedRunId} />
              </div>

              <MetricsPanel runId={selectedRunId} />

              <div className="grid gap-6 lg:grid-cols-2">
                <CostBreakdown runId={selectedRunId} />
                <OverfittingBadges runId={selectedRunId} />
              </div>

              <WinLossPanel runId={selectedRunId} />

              <div className="grid gap-6 lg:grid-cols-2">
                <FuturesAnalytics runId={selectedRunId} />
                <LongShortSplit runId={selectedRunId} />
              </div>

              <LiquidationSummary runId={selectedRunId} />

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

              <RawStatsPanel runId={selectedRunId} />

              <NotesPanel runId={selectedRunId} />
            </>
          )}
        </>
      )}
    </div>
  );
}
