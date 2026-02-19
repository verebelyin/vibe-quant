import { useGetRunSummaryApiResultsRunsRunIdGet } from "@/api/generated/results/results";
import { MetricCard } from "@/components/ui";
import { Skeleton } from "@/components/ui/skeleton";

interface MetricsPanelProps {
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

function MetricsSkeleton() {
  return (
    <div>
      <Skeleton className="mb-3 h-4 w-40" />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {Array.from({ length: 12 }).map((_, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: skeleton placeholders have no stable key
          <Skeleton key={i} className="h-24 rounded-xl" />
        ))}
      </div>
    </div>
  );
}

export function MetricsPanel({ runId }: MetricsPanelProps) {
  const query = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const data = query.data?.data;

  if (query.isLoading) {
    return <MetricsSkeleton />;
  }

  if (query.isError || !data) {
    return <p className="py-4 text-sm text-destructive">Failed to load run metrics.</p>;
  }

  const expectancy =
    data.win_rate != null && data.avg_win != null && data.avg_loss != null
      ? (data.win_rate / 100) * data.avg_win - (1 - data.win_rate / 100) * Math.abs(data.avg_loss)
      : null;

  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
        Performance Metrics
      </h3>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        <MetricCard
          label="Total Return"
          value={fmt(data.total_return, 2, "%")}
          trend={trend(data.total_return)}
        />
        <MetricCard
          label="Sharpe Ratio"
          value={fmt(data.sharpe_ratio)}
          trend={trend(data.sharpe_ratio)}
        />
        <MetricCard
          label="Max Drawdown"
          value={fmt(data.max_drawdown, 2, "%")}
          trend={
            data.max_drawdown != null ? (data.max_drawdown < -10 ? "down" : "neutral") : "neutral"
          }
          subtitle={
            data.max_drawdown_duration_days != null
              ? `${data.max_drawdown_duration_days}d duration`
              : undefined
          }
        />
        <MetricCard
          label="Win Rate"
          value={fmt(data.win_rate, 1, "%")}
          trend={data.win_rate != null ? (data.win_rate >= 50 ? "up" : "down") : "neutral"}
        />
        <MetricCard
          label="Profit Factor"
          value={fmt(data.profit_factor)}
          trend={data.profit_factor != null ? (data.profit_factor >= 1 ? "up" : "down") : "neutral"}
        />
        <MetricCard
          label="Total Trades"
          value={data.total_trades ?? "N/A"}
          subtitle={
            data.winning_trades != null && data.losing_trades != null
              ? `${data.winning_trades}W / ${data.losing_trades}L`
              : undefined
          }
        />
        <MetricCard
          label="Avg Trade Duration"
          value={
            data.avg_trade_duration_hours != null
              ? `${data.avg_trade_duration_hours.toFixed(1)}h`
              : "N/A"
          }
        />
        <MetricCard label="Expectancy" value={fmt(expectancy, 2)} trend={trend(expectancy)} />
        <MetricCard
          label="Sortino Ratio"
          value={fmt(data.sortino_ratio)}
          trend={trend(data.sortino_ratio)}
        />
        <MetricCard
          label="Calmar Ratio"
          value={fmt(data.calmar_ratio)}
          trend={trend(data.calmar_ratio)}
        />
        <MetricCard label="CAGR" value={fmt(data.cagr, 2, "%")} trend={trend(data.cagr)} />
        <MetricCard
          label="Annual Volatility"
          value={fmt(data.volatility_annual, 2, "%")}
          trend={
            data.volatility_annual != null
              ? data.volatility_annual > 50
                ? "down"
                : "neutral"
              : "neutral"
          }
        />
      </div>
    </div>
  );
}
