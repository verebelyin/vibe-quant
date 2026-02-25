import { useMemo } from "react";
import type { TradeResponse } from "@/api/generated/models";
import { useGetTradesApiResultsRunsRunIdTradesGet } from "@/api/generated/results/results";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface LiquidationSummaryProps {
  runId: number;
}

export function LiquidationSummary({ runId }: LiquidationSummaryProps) {
  const query = useGetTradesApiResultsRunsRunIdTradesGet(runId);
  const trades = query.data?.data as TradeResponse[] | undefined;

  const stats = useMemo(() => {
    if (!trades || trades.length === 0) return null;
    const liquidations = trades.filter((t) => t.exit_reason === "liquidation");
    const count = liquidations.length;
    if (count === 0) return null;
    const totalLoss = liquidations.reduce((sum, t) => sum + (t.net_pnl ?? 0), 0);
    const pctOfTrades = (count / trades.length) * 100;
    return { count, totalLoss, pctOfTrades };
  }, [trades]);

  if (query.isLoading || !stats) return null;

  return (
    <Card className="border-destructive/30">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-destructive">
          Liquidations
          <Badge variant="destructive" className="text-[10px]">
            {stats.count}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-6 text-sm">
          <div>
            <span className="text-muted-foreground">Total Loss: </span>
            <span className="font-bold tabular-nums text-destructive">
              ${Math.abs(stats.totalLoss).toFixed(2)}
            </span>
          </div>
          <div>
            <span className="text-muted-foreground">% of Trades: </span>
            <span className="font-bold tabular-nums">{stats.pctOfTrades.toFixed(1)}%</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
