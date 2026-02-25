import { useMemo, useState } from "react";
import { toast } from "sonner";
import { useLaunchValidationApiBacktestValidationPost } from "@/api/generated/backtest/backtest";
import type { BacktestRunResponse, RunListResponse } from "@/api/generated/models";
import type { SweepResultResponse } from "@/api/generated/models/sweepResultResponse";
import {
  useGetSweepsApiResultsRunsRunIdSweepsGet,
  useListRunsApiResultsRunsGet,
} from "@/api/generated/results/results";
import ParetoSurface3D from "@/components/charts/ParetoSurface3D";
import SweepScatterChart from "@/components/charts/SweepScatterChart";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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

function exportSweepCsv(sweeps: SweepResultResponse[], runId: number) {
  const headers = [
    "Parameters",
    "Sharpe",
    "Sortino",
    "Max DD",
    "Return",
    "Win Rate",
    "Trades",
    "Fees",
    "Funding",
    "Pareto",
    "DSR",
    "WF",
    "KF",
  ];
  const rows = sweeps.map((s) => [
    `"${formatParams(s.parameters)}"`,
    s.sharpe_ratio ?? "",
    s.sortino_ratio ?? "",
    s.max_drawdown ?? "",
    s.total_return ?? "",
    s.win_rate ?? "",
    s.total_trades ?? "",
    s.total_fees ?? "",
    s.total_funding ?? "",
    s.is_pareto_optimal ? "yes" : "no",
    s.passed_deflated_sharpe ?? "",
    s.passed_walk_forward ?? "",
    s.passed_purged_kfold ?? "",
  ]);

  const csvContent = [headers, ...rows].map((row) => row.join(",")).join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `sweep_run_${runId}.csv`;
  link.click();
  URL.revokeObjectURL(url);
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

function ValidateButton({
  sweep,
  run,
}: {
  sweep: SweepResultResponse;
  run: BacktestRunResponse | undefined;
}) {
  const mutation = useLaunchValidationApiBacktestValidationPost();

  function handleValidate() {
    if (!run) {
      toast.error("Run metadata not available");
      return;
    }
    mutation.mutate(
      {
        data: {
          strategy_id: run.strategy_id,
          symbols: run.symbols,
          timeframe: run.timeframe,
          start_date: run.start_date,
          end_date: run.end_date,
          parameters: sweep.parameters,
        },
      },
      {
        onSuccess: () => toast.success("Validation launched"),
        onError: () => toast.error("Failed to launch validation"),
      },
    );
  }

  return (
    <Button
      variant="outline"
      size="sm"
      className="h-6 text-[10px]"
      onClick={handleValidate}
      disabled={mutation.isPending || !run}
    >
      {mutation.isPending ? "..." : "Validate"}
    </Button>
  );
}

export function SweepAnalysis({ runId }: SweepAnalysisProps) {
  const query = useGetSweepsApiResultsRunsRunIdSweepsGet(runId);
  const sweeps = query.data?.data as SweepResultResponse[] | undefined;
  const runsQuery = useListRunsApiResultsRunsGet();
  const run = (runsQuery.data?.data as RunListResponse | undefined)?.runs?.find((r: BacktestRunResponse) => r.id === runId);

  // Filter state
  const [minSharpe, setMinSharpe] = useState("");
  const [maxDrawdown, setMaxDrawdown] = useState("");
  const [paretoOnly, setParetoOnly] = useState(false);
  const [show3D, setShow3D] = useState(false);

  const chartData = useMemo(() => {
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

  const filteredSweeps = useMemo(() => {
    if (!sweeps) return [];
    return sweeps.filter((s) => {
      if (paretoOnly && !s.is_pareto_optimal) return false;
      if (minSharpe !== "" && (s.sharpe_ratio ?? 0) < Number(minSharpe)) return false;
      if (maxDrawdown !== "" && (s.max_drawdown ?? 0) > Number(maxDrawdown)) return false;
      return true;
    });
  }, [sweeps, paretoOnly, minSharpe, maxDrawdown]);

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
      {/* Chart toggle */}
      <div className="flex items-center gap-2">
        <Button
          variant={show3D ? "outline" : "secondary"}
          size="sm"
          onClick={() => setShow3D(false)}
        >
          2D Scatter
        </Button>
        <Button
          variant={show3D ? "secondary" : "outline"}
          size="sm"
          onClick={() => setShow3D(true)}
        >
          3D Pareto
        </Button>
      </div>

      {chartData.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-medium text-muted-foreground">
            {show3D
              ? `3D Pareto Surface (${chartData.length} combos, ${paretoCount} Pareto optimal)`
              : `Sharpe vs Drawdown (${chartData.length} combos, ${paretoCount} Pareto optimal)`}
          </h4>
          {show3D ? (
            <ParetoSurface3D data={chartData} height={450} />
          ) : (
            <SweepScatterChart data={chartData} height={350} />
          )}
        </div>
      )}

      {/* Sweep filters */}
      <div className="rounded-lg border border-border bg-card/50 p-3">
        <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Filters
        </h4>
        <div className="flex flex-wrap items-end gap-4">
          <div className="space-y-1">
            <Label className="text-xs">Min Sharpe</Label>
            <Input
              type="number"
              step="0.1"
              placeholder="e.g. 0.5"
              value={minSharpe}
              onChange={(e) => setMinSharpe(e.target.value)}
              className="h-7 w-28 text-xs"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Max Drawdown</Label>
            <Input
              type="number"
              step="0.01"
              placeholder="e.g. 0.20"
              value={maxDrawdown}
              onChange={(e) => setMaxDrawdown(e.target.value)}
              className="h-7 w-28 text-xs"
            />
          </div>
          <Label className="flex cursor-pointer items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={paretoOnly}
              onChange={(e) => setParetoOnly(e.target.checked)}
            />
            Pareto optimal only
          </Label>
          <span className="ml-auto text-xs text-muted-foreground">
            {filteredSweeps.length} / {sweeps.length} shown
          </span>
        </div>
      </div>

      <div className="flex justify-end">
        <Button variant="outline" size="sm" onClick={() => exportSweepCsv(filteredSweeps, runId)}>
          Export CSV
        </Button>
      </div>

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
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredSweeps.map((sweep) => (
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
                <TableCell>
                  <ValidateButton sweep={sweep} run={run} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
