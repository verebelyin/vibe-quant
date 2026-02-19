import { useMemo } from "react";
import {
  useGetDrawdownApiResultsRunsRunIdDrawdownGet,
  useGetEquityCurveApiResultsRunsRunIdEquityCurveGet,
  useGetRunSummaryApiResultsRunsRunIdGet,
  useGetTradesApiResultsRunsRunIdTradesGet,
} from "@/api/generated/results/results";
import DrawdownChart from "@/components/charts/DrawdownChart";
import EquityCurveChart from "@/components/charts/EquityCurveChart";
import type { PerformanceMetric } from "@/components/charts/PerformanceRadar";
import PerformanceRadar from "@/components/charts/PerformanceRadar";
import TradeDistributionChart from "@/components/charts/TradeDistributionChart";
import { DailyReturnsChart } from "@/components/results/DailyReturnsChart";
import { MonthlyReturnsHeatmap } from "@/components/results/MonthlyReturnsHeatmap";
import { RollingSharpeChart } from "@/components/results/RollingSharpeChart";
import { TradeScatterPlots } from "@/components/results/TradeScatterPlots";
import { YearlyReturnsChart } from "@/components/results/YearlyReturnsChart";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ChartsPanelProps {
  runId: number;
}

function ChartSkeleton({ height = 300 }: { height?: number }) {
  return <Skeleton className="w-full rounded-xl" style={{ height }} />;
}

export function ChartsPanel({ runId }: ChartsPanelProps) {
  const equityQuery = useGetEquityCurveApiResultsRunsRunIdEquityCurveGet(runId);
  const drawdownQuery = useGetDrawdownApiResultsRunsRunIdDrawdownGet(runId);
  const summaryQuery = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const tradesQuery = useGetTradesApiResultsRunsRunIdTradesGet(runId);

  const equityData = equityQuery.data?.data ?? [];
  const drawdownData = drawdownQuery.data?.data ?? [];
  const summary = summaryQuery.data?.data;
  const trades = tradesQuery.data?.data;

  const distributionTrades = useMemo(() => {
    if (!trades) return [];
    return trades
      .filter((t) => t.roi_percent != null)
      .map((t) => ({
        roi_percent: t.roi_percent as number,
        direction: t.direction,
      }));
  }, [trades]);

  const radarMetrics: PerformanceMetric[] = useMemo(() => {
    if (!summary) return [];
    return [
      { label: "Sharpe", value: summary.sharpe_ratio ?? 0, max: 4 },
      { label: "Sortino", value: summary.sortino_ratio ?? 0, max: 6 },
      { label: "Win Rate", value: summary.win_rate ?? 0, max: 100 },
      { label: "Profit Factor", value: summary.profit_factor ?? 0, max: 5 },
      { label: "Calmar", value: summary.calmar_ratio ?? 0, max: 5 },
      { label: "Return", value: Math.min(summary.total_return ?? 0, 200), max: 200 },
    ];
  }, [summary]);

  return (
    <Tabs defaultValue="equity">
      <TabsList className="flex-wrap">
        <TabsTrigger value="equity">Equity</TabsTrigger>
        <TabsTrigger value="drawdown">Drawdown</TabsTrigger>
        <TabsTrigger value="distribution">Distribution</TabsTrigger>
        <TabsTrigger value="performance">Performance</TabsTrigger>
        <TabsTrigger value="rolling-sharpe">Rolling Sharpe</TabsTrigger>
        <TabsTrigger value="yearly">Yearly</TabsTrigger>
        <TabsTrigger value="monthly">Monthly</TabsTrigger>
        <TabsTrigger value="daily">Daily</TabsTrigger>
        <TabsTrigger value="scatter">Scatter</TabsTrigger>
      </TabsList>

      <TabsContent value="equity" className="pt-4">
        {equityQuery.isLoading ? (
          <ChartSkeleton />
        ) : equityData.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No equity curve data available.
          </p>
        ) : (
          <EquityCurveChart data={equityData} height={350} />
        )}
      </TabsContent>

      <TabsContent value="drawdown" className="pt-4">
        {drawdownQuery.isLoading ? (
          <ChartSkeleton height={250} />
        ) : drawdownData.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No drawdown data available.
          </p>
        ) : (
          <DrawdownChart data={drawdownData} height={300} />
        )}
      </TabsContent>

      <TabsContent value="distribution" className="pt-4">
        {tradesQuery.isLoading ? (
          <ChartSkeleton />
        ) : distributionTrades.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No trade distribution data available.
          </p>
        ) : (
          <TradeDistributionChart trades={distributionTrades} height={350} />
        )}
      </TabsContent>

      <TabsContent value="performance" className="pt-4">
        {summaryQuery.isLoading ? (
          <ChartSkeleton />
        ) : radarMetrics.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No performance data available.
          </p>
        ) : (
          <PerformanceRadar metrics={radarMetrics} height={350} />
        )}
      </TabsContent>

      <TabsContent value="rolling-sharpe" className="pt-4">
        {equityQuery.isLoading ? (
          <ChartSkeleton />
        ) : equityData.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No equity data for rolling Sharpe.
          </p>
        ) : (
          <RollingSharpeChart data={equityData} height={350} />
        )}
      </TabsContent>

      <TabsContent value="yearly" className="pt-4">
        {equityQuery.isLoading ? (
          <ChartSkeleton />
        ) : equityData.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No equity data for yearly returns.
          </p>
        ) : (
          <YearlyReturnsChart data={equityData} height={350} />
        )}
      </TabsContent>

      <TabsContent value="monthly" className="pt-4">
        <MonthlyReturnsHeatmap runId={runId} height={400} />
      </TabsContent>

      <TabsContent value="daily" className="pt-4">
        {equityQuery.isLoading ? (
          <ChartSkeleton />
        ) : equityData.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No equity data for daily returns.
          </p>
        ) : (
          <DailyReturnsChart data={equityData} height={350} />
        )}
      </TabsContent>

      <TabsContent value="scatter" className="pt-4">
        {tradesQuery.isLoading ? (
          <ChartSkeleton />
        ) : !trades || trades.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No trade data for scatter plots.
          </p>
        ) : (
          <TradeScatterPlots trades={trades} height={300} />
        )}
      </TabsContent>
    </Tabs>
  );
}
