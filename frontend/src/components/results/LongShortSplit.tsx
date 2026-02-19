import { useMemo } from "react";
import type { TradeResponse } from "@/api/generated/models/tradeResponse";
import { useGetTradesApiResultsRunsRunIdTradesGet } from "@/api/generated/results/results";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface LongShortSplitProps {
  runId: number;
}

interface DirectionStats {
  count: number;
  winRate: number;
  totalPnl: number;
  avgPnl: number;
}

function computeDirectionStats(trades: TradeResponse[], direction: string): DirectionStats {
  const filtered = trades.filter((t) => t.direction === direction);
  const count = filtered.length;
  if (count === 0) return { count: 0, winRate: 0, totalPnl: 0, avgPnl: 0 };

  const wins = filtered.filter((t) => (t.net_pnl ?? 0) > 0).length;
  const totalPnl = filtered.reduce((sum, t) => sum + (t.net_pnl ?? 0), 0);
  return {
    count,
    winRate: (wins / count) * 100,
    totalPnl,
    avgPnl: totalPnl / count,
  };
}

function StatColumn({
  label,
  stats,
  color,
}: {
  label: string;
  stats: DirectionStats;
  color: string;
}) {
  return (
    <div className="space-y-3 text-center">
      <p className={cn("text-sm font-bold uppercase", color)}>{label}</p>
      <div className="space-y-2 text-sm">
        <div>
          <p className="text-xs text-muted-foreground">Trades</p>
          <p className="font-bold tabular-nums">{stats.count}</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Win Rate</p>
          <p className="font-bold tabular-nums">{stats.winRate.toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Total PnL</p>
          <p
            className={cn(
              "font-bold tabular-nums",
              stats.totalPnl >= 0 ? "text-green-500" : "text-red-500",
            )}
          >
            ${stats.totalPnl.toFixed(2)}
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Avg PnL</p>
          <p
            className={cn(
              "font-bold tabular-nums",
              stats.avgPnl >= 0 ? "text-green-500" : "text-red-500",
            )}
          >
            ${stats.avgPnl.toFixed(2)}
          </p>
        </div>
      </div>
    </div>
  );
}

export function LongShortSplit({ runId }: LongShortSplitProps) {
  const query = useGetTradesApiResultsRunsRunIdTradesGet(runId);
  const trades = query.data?.data;

  const longStats = useMemo(
    () => (trades ? computeDirectionStats(trades, "long") : null),
    [trades],
  );
  const shortStats = useMemo(
    () => (trades ? computeDirectionStats(trades, "short") : null),
    [trades],
  );

  if (query.isLoading) {
    return <Skeleton className="h-48 w-full rounded-xl" />;
  }

  if (!trades || trades.length === 0 || !longStats || !shortStats) return null;
  if (longStats.count === 0 && shortStats.count === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Long vs Short
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 divide-x divide-border">
          <StatColumn label="Long" stats={longStats} color="text-green-500" />
          <StatColumn label="Short" stats={shortStats} color="text-red-500" />
        </div>
      </CardContent>
    </Card>
  );
}
