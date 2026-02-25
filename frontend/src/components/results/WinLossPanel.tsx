import type { BacktestResultResponse } from "@/api/generated/models";
import { useGetRunSummaryApiResultsRunsRunIdGet } from "@/api/generated/results/results";
import { MetricCard } from "@/components/ui";
import { Skeleton } from "@/components/ui/skeleton";

interface WinLossPanelProps {
  runId: number;
}

function fmt(value: number | null | undefined, decimals = 2, suffix = ""): string {
  if (value == null) return "N/A";
  return `${value.toFixed(decimals)}${suffix}`;
}

function trend(value: number | null | undefined): "up" | "down" | "neutral" {
  if (value == null || value === 0) return "neutral";
  return value > 0 ? "up" : "down";
}

export function WinLossPanel({ runId }: WinLossPanelProps) {
  const query = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const data = query.data?.data as BacktestResultResponse | undefined;

  if (query.isLoading) {
    return (
      <div>
        <Skeleton className="mb-3 h-4 w-40" />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {Array.from({ length: 6 }).map((_, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: skeleton placeholders
            <Skeleton key={i} className="h-24 rounded-xl" />
          ))}
        </div>
      </div>
    );
  }

  if (query.isError || !data) return null;

  const payoffRatio =
    data.avg_win != null && data.avg_loss != null && data.avg_loss !== 0
      ? Math.abs(data.avg_win / data.avg_loss)
      : null;

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
        Win / Loss Analysis
      </h3>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        <MetricCard label="Avg Win" value={fmt(data.avg_win)} trend={trend(data.avg_win)} />
        <MetricCard label="Avg Loss" value={fmt(data.avg_loss)} trend="down" />
        <MetricCard label="Largest Win" value={fmt(data.largest_win)} trend="up" />
        <MetricCard label="Largest Loss" value={fmt(data.largest_loss)} trend="down" />
        <MetricCard label="Max Consec. Wins" value={data.max_consecutive_wins ?? "N/A"} />
        <MetricCard label="Max Consec. Losses" value={data.max_consecutive_losses ?? "N/A"} />
        <MetricCard
          label="Payoff Ratio"
          value={fmt(payoffRatio)}
          trend={payoffRatio != null ? (payoffRatio >= 1 ? "up" : "down") : "neutral"}
          subtitle="Avg Win / Avg Loss"
        />
      </div>
    </div>
  );
}
