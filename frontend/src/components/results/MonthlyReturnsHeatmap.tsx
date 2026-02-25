import { useMemo } from "react";
import type { MonthlyReturn } from "@/api/generated/models/monthlyReturn";
import { useGetMonthlyReturnsApiResultsRunsRunIdMonthlyReturnsGet } from "@/api/generated/results/results";
import HeatmapChart from "@/components/charts/HeatmapChart";
import { Skeleton } from "@/components/ui/skeleton";

interface MonthlyReturnsHeatmapProps {
  runId: number;
  height?: number;
}

const MONTH_LABELS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

function buildHeatmapData(returns: MonthlyReturn[]): {
  x: string[];
  y: string[];
  z: number[][];
} {
  if (returns.length === 0) return { x: [], y: [], z: [] };

  const years = [...new Set(returns.map((r) => r.year))].sort((a, b) => a - b);
  const yearLabels = years.map(String);

  const lookup = new Map<string, number>();
  for (const r of returns) {
    lookup.set(`${r.year}-${r.month}`, Number.parseFloat(r.return_pct.toFixed(2)));
  }

  // z[yearIndex][monthIndex]
  const z: number[][] = years.map((year) =>
    MONTH_LABELS.map((_, mi) => lookup.get(`${year}-${mi + 1}`) ?? 0),
  );

  return { x: MONTH_LABELS, y: yearLabels, z };
}

export function MonthlyReturnsHeatmap({ runId, height = 400 }: MonthlyReturnsHeatmapProps) {
  const query = useGetMonthlyReturnsApiResultsRunsRunIdMonthlyReturnsGet(runId);
  const monthlyData = query.data?.data as MonthlyReturn[] | undefined;

  const heatmapData = useMemo(() => {
    if (!monthlyData) return null;
    return buildHeatmapData(monthlyData);
  }, [monthlyData]);

  if (query.isLoading) {
    return <Skeleton className="h-[400px] w-full rounded-xl" />;
  }

  if (query.isError || !heatmapData || heatmapData.y.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No monthly return data available.
      </p>
    );
  }

  return <HeatmapChart data={heatmapData} title="Monthly Returns (%)" height={height} />;
}
