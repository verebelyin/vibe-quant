import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import {
  getGetRunSummaryApiResultsRunsRunIdGetQueryKey,
  getGetSweepsApiResultsRunsRunIdSweepsGetQueryKey,
  getGetTradesApiResultsRunsRunIdTradesGetQueryKey,
  useGetSweepsApiResultsRunsRunIdSweepsGet,
  useListRunsApiResultsRunsGet,
} from "@/api/generated/results/results";
import { ArrowLeft } from "lucide-react";
import { ChartsPanel } from "@/components/results/ChartsPanel";
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
import { SweepAnalysis } from "@/components/results/SweepAnalysis";
import { TradeChart } from "@/components/results/TradeChart";
import { TradeLog } from "@/components/results/TradeLog";
import { WinLossPanel } from "@/components/results/WinLossPanel";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const POLL_INTERVAL_MS = 3000;

/** Poll runs list while selected run is "running"; invalidate result queries on completion. */
function useRunPolling(runId: number) {
  const queryClient = useQueryClient();
  const prevStatus = useRef<string | null>(null);

  const runsQuery = useListRunsApiResultsRunsGet(undefined, {
    query: {
      refetchInterval: (query) => {
        const resp = query.state.data;
        const runs = resp && resp.status === 200 ? resp.data.runs : [];
        const run = runs.find((r: { id: number }) => r.id === runId);
        return run?.status === "running" ? POLL_INTERVAL_MS : false;
      },
    },
  });

  useEffect(() => {
    const resp = runsQuery.data;
    const runs = resp && resp.status === 200 ? resp.data.runs : [];
    const run = runs.find((r: { id: number; status: string }) => r.id === runId);
    const status = run?.status ?? null;

    if (
      prevStatus.current === "running" &&
      (status === "completed" || status === "failed")
    ) {
      queryClient.invalidateQueries({
        queryKey: getGetRunSummaryApiResultsRunsRunIdGetQueryKey(runId),
      });
      queryClient.invalidateQueries({
        queryKey: getGetSweepsApiResultsRunsRunIdSweepsGetQueryKey(runId),
      });
      queryClient.invalidateQueries({
        queryKey: getGetTradesApiResultsRunsRunIdTradesGetQueryKey(runId),
      });
    }
    prevStatus.current = status;
  }, [runsQuery.data, runId, queryClient]);
}

function SweepTab({ runId }: { runId: number }) {
  const query = useGetSweepsApiResultsRunsRunIdSweepsGet(runId);
  const resp = query.data;
  const sweeps = resp && resp.status === 200 ? resp.data : null;

  if (query.isLoading || query.isError || !sweeps || sweeps.length === 0) return null;

  return <SweepAnalysis runId={runId} />;
}

function useSweepAvailable(runId: number) {
  const query = useGetSweepsApiResultsRunsRunIdSweepsGet(runId);
  const resp = query.data;
  const sweeps = resp && resp.status === 200 ? resp.data : null;
  return !query.isLoading && sweeps != null && sweeps.length > 0;
}

interface ResultsDetailPageProps {
  runId: number;
}

export function ResultsDetailPage({ runId }: ResultsDetailPageProps) {
  const hasSweeps = useSweepAvailable(runId);
  useRunPolling(runId);
  const [hoveredTradeId, setHoveredTradeId] = useState<number | null>(null);
  const [pinnedTradeId, setPinnedTradeId] = useState<number | null>(null);
  const highlightedTradeId = pinnedTradeId ?? hoveredTradeId;
  const handleTradeHover = useCallback((tradeId: number | null) => {
    setHoveredTradeId(tradeId);
  }, []);
  const handleTradeClick = useCallback((tradeId: number) => {
    setPinnedTradeId((prev) => (prev === tradeId ? null : tradeId));
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="sm" asChild>
          <Link to="/results">
            <ArrowLeft className="mr-1 h-4 w-4" />
            Back
          </Link>
        </Button>
        <h1 className="text-2xl font-bold text-foreground">Run #{runId}</h1>
      </div>

      <RunDetailsExpander runId={runId} />

      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Run #{runId}
        </h3>
        <ExportPanel runId={runId} />
      </div>

      <MetricsPanel runId={runId} />

      <TradeChart runId={runId} highlightedTradeId={highlightedTradeId} />

      <TradeLog runId={runId} onTradeHover={handleTradeHover} onTradeClick={handleTradeClick} highlightedTradeId={highlightedTradeId} />

      <div className="grid gap-6 lg:grid-cols-2">
        <CostBreakdown runId={runId} />
        <OverfittingBadges runId={runId} />
      </div>

      <WinLossPanel runId={runId} />

      <div className="grid gap-6 lg:grid-cols-2">
        <FuturesAnalytics runId={runId} />
        <LongShortSplit runId={runId} />
      </div>

      <LiquidationSummary runId={runId} />

      <Tabs defaultValue="charts">
        <TabsList>
          <TabsTrigger value="charts">Charts</TabsTrigger>
          {hasSweeps && <TabsTrigger value="sweep">Sweep</TabsTrigger>}
        </TabsList>

        <TabsContent value="charts" className="pt-4">
          <ChartsPanel runId={runId} />
        </TabsContent>

        {hasSweeps && (
          <TabsContent value="sweep" className="pt-4">
            <SweepTab runId={runId} />
          </TabsContent>
        )}
      </Tabs>

      <RawStatsPanel runId={runId} />

      <NotesPanel runId={runId} />
    </div>
  );
}
