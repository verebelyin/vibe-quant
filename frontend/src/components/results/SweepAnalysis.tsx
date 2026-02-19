import { useMemo } from "react";
import type { SweepResultResponse } from "@/api/generated/models/sweepResultResponse";
import { useGetSweepsApiResultsRunsRunIdSweepsGet } from "@/api/generated/results/results";
import SweepScatterChart from "@/components/charts/SweepScatterChart";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

interface SweepAnalysisProps {
  runId: number;
}

function fmt(value: number | null | undefined, decimals = 2, suffix = ""): string {
  if (value == null) return "-";
  return `${value.toFixed(decimals)}${suffix}`;
}

function pct(value: number | null | undefined): string {
  if (value == null) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatParams(params: Record<string, unknown>): string {
  return Object.entries(params)
    .map(([k, v]) => `${k}=${String(v)}`)
    .join(", ");
}

function SweepSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-[300px] w-full rounded-xl" />
      <Skeleton className="h-9 w-full" />
      {Array.from({ length: 5 }).map((_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: skeleton placeholders
        <Skeleton key={i} className="h-10 w-full" />
      ))}
    </div>
  );
}

export function SweepAnalysis({ runId }: SweepAnalysisProps) {
  const query = useGetSweepsApiResultsRunsRunIdSweepsGet(runId);
  const sweeps = query.data?.data;

  const scatterData = useMemo(() => {
    if (!sweeps) return [];
    return sweeps
      .filter(
        (
          s,
        ): s is SweepResultResponse & {
          sharpe_ratio: number;
          max_drawdown: number;
          total_return: number;
        } => s.sharpe_ratio != null && s.max_drawdown != null && s.total_return != null,
      )
      .map((s) => ({
        sharpe_ratio: s.sharpe_ratio,
        max_drawdown: s.max_drawdown,
        total_return: s.total_return,
        is_pareto_optimal: s.is_pareto_optimal,
      }));
  }, [sweeps]);

  if (query.isLoading) return <SweepSkeleton />;

  if (query.isError || !sweeps) {
    return <p className="py-4 text-sm text-destructive">Failed to load sweep data.</p>;
  }

  if (sweeps.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">No sweep results available.</p>
    );
  }

  const paretoCount = sweeps.filter((s) => s.is_pareto_optimal).length;

  return (
    <div className="space-y-6">
      {scatterData.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-medium text-muted-foreground">
            Sharpe vs Drawdown ({scatterData.length} combos, {paretoCount} Pareto optimal)
          </h4>
          <SweepScatterChart data={scatterData} height={350} />
        </div>
      )}

      <div className="max-h-[500px] overflow-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Parameters</TableHead>
              <TableHead className="text-right">Sharpe</TableHead>
              <TableHead className="text-right">Sortino</TableHead>
              <TableHead className="text-right">Max DD</TableHead>
              <TableHead className="text-right">Return</TableHead>
              <TableHead className="text-right">Win Rate</TableHead>
              <TableHead className="text-right">Trades</TableHead>
              <TableHead>Pareto</TableHead>
              <TableHead>Filters</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sweeps.map((sweep) => (
              <TableRow key={sweep.id} className={cn(sweep.is_pareto_optimal && "bg-blue-500/5")}>
                <TableCell className="max-w-[240px] truncate text-xs font-mono">
                  {formatParams(sweep.parameters)}
                </TableCell>
                <TableCell className="text-right tabular-nums">{fmt(sweep.sharpe_ratio)}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {fmt(sweep.sortino_ratio)}
                </TableCell>
                <TableCell className="text-right tabular-nums">{pct(sweep.max_drawdown)}</TableCell>
                <TableCell className="text-right tabular-nums">{pct(sweep.total_return)}</TableCell>
                <TableCell className="text-right tabular-nums">{pct(sweep.win_rate)}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {sweep.total_trades ?? "-"}
                </TableCell>
                <TableCell>
                  {sweep.is_pareto_optimal ? (
                    <Badge variant="default">Pareto</Badge>
                  ) : (
                    <span className="text-xs text-muted-foreground">-</span>
                  )}
                </TableCell>
                <TableCell>
                  <div className="flex gap-1">
                    {sweep.passed_deflated_sharpe != null && (
                      <Badge
                        variant={sweep.passed_deflated_sharpe ? "default" : "destructive"}
                        className="text-[10px]"
                      >
                        DSR
                      </Badge>
                    )}
                    {sweep.passed_walk_forward != null && (
                      <Badge
                        variant={sweep.passed_walk_forward ? "default" : "destructive"}
                        className="text-[10px]"
                      >
                        WF
                      </Badge>
                    )}
                    {sweep.passed_purged_kfold != null && (
                      <Badge
                        variant={sweep.passed_purged_kfold ? "default" : "destructive"}
                        className="text-[10px]"
                      >
                        KF
                      </Badge>
                    )}
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
