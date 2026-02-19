import { useCallback, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { BacktestResultResponse, BacktestRunResponse } from "@/api/generated/models";
import {
  useCompareRunsApiResultsCompareGet,
  useGetEquityCurveApiResultsRunsRunIdEquityCurveGet,
  useListRunsApiResultsRunsGet,
} from "@/api/generated/results/results";
import { LoadingSpinner } from "@/components/ui";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";

const MAX_COMPARE = 5;
const MIN_COMPARE = 2;

const COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7"] as const;

interface MetricDef {
  key: keyof BacktestResultResponse;
  label: string;
  format: (v: number) => string;
  /** true = higher is better, false = lower (less negative) is better */
  higherIsBetter: boolean;
}

const METRICS: MetricDef[] = [
  {
    key: "total_return",
    label: "Total Return",
    format: (v) => `${v.toFixed(2)}%`,
    higherIsBetter: true,
  },
  { key: "sharpe_ratio", label: "Sharpe Ratio", format: (v) => v.toFixed(2), higherIsBetter: true },
  {
    key: "max_drawdown",
    label: "Max Drawdown",
    format: (v) => `${v.toFixed(2)}%`,
    higherIsBetter: false,
  },
  { key: "win_rate", label: "Win Rate", format: (v) => `${v.toFixed(1)}%`, higherIsBetter: true },
  {
    key: "profit_factor",
    label: "Profit Factor",
    format: (v) => v.toFixed(2),
    higherIsBetter: true,
  },
  {
    key: "sortino_ratio",
    label: "Sortino Ratio",
    format: (v) => v.toFixed(2),
    higherIsBetter: true,
  },
  { key: "calmar_ratio", label: "Calmar Ratio", format: (v) => v.toFixed(2), higherIsBetter: true },
];

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatChartDate(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function RunPicker({
  selectedIds,
  onToggle,
}: {
  selectedIds: Set<number>;
  onToggle: (id: number) => void;
}) {
  const query = useListRunsApiResultsRunsGet({ status: "completed" });
  const runs = query.data?.data?.runs ?? [];

  const sorted = useMemo(
    () =>
      [...runs].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()),
    [runs],
  );

  if (query.isLoading) return <LoadingSpinner size="sm" />;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Select Runs ({selectedIds.size}/{MAX_COMPARE})
        </CardTitle>
      </CardHeader>
      <CardContent className="max-h-64 space-y-1 overflow-y-auto">
        {sorted.map((run: BacktestRunResponse) => {
          const checked = selectedIds.has(run.id);
          const disabled = !checked && selectedIds.size >= MAX_COMPARE;
          return (
            <button
              key={run.id}
              type="button"
              className={cn(
                "flex w-full cursor-pointer items-center gap-3 rounded-md border bg-transparent px-3 py-2 text-left transition-colors hover:bg-accent/50",
                checked && "border-primary bg-primary/5",
                disabled && "cursor-not-allowed opacity-50",
              )}
              disabled={disabled}
              onClick={() => onToggle(run.id)}
            >
              <Checkbox
                checked={checked}
                disabled={disabled}
                tabIndex={-1}
                onCheckedChange={() => onToggle(run.id)}
              />
              <span className="text-xs font-medium">#{run.id}</span>
              <span className="text-xs text-muted-foreground">{run.run_mode}</span>
              <span className="text-xs text-muted-foreground">{run.symbols.join(", ")}</span>
              <Badge variant="outline" className="ml-auto text-[10px]">
                {formatDate(run.created_at)}
              </Badge>
            </button>
          );
        })}
        {sorted.length === 0 && (
          <p className="py-4 text-center text-sm text-muted-foreground">No completed runs found.</p>
        )}
      </CardContent>
    </Card>
  );
}

function getBestWorst(
  runs: BacktestResultResponse[],
  key: keyof BacktestResultResponse,
  higherIsBetter: boolean,
): { bestId: number | null; worstId: number | null } {
  let bestId: number | null = null;
  let worstId: number | null = null;
  let bestVal = higherIsBetter ? -Infinity : Infinity;
  let worstVal = higherIsBetter ? Infinity : -Infinity;

  for (const run of runs) {
    const val = run[key];
    if (typeof val !== "number") continue;

    if (higherIsBetter) {
      if (val > bestVal) {
        bestVal = val;
        bestId = run.run_id;
      }
      if (val < worstVal) {
        worstVal = val;
        worstId = run.run_id;
      }
    } else {
      // For max_drawdown: less negative is better
      if (val > bestVal) {
        bestVal = val;
        bestId = run.run_id;
      }
      if (val < worstVal) {
        worstVal = val;
        worstId = run.run_id;
      }
    }
  }

  return { bestId, worstId };
}

function MetricsTable({ runs }: { runs: BacktestResultResponse[] }) {
  if (runs.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Metrics Comparison
        </CardTitle>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="py-2 pr-4 text-left text-xs font-medium text-muted-foreground">
                Metric
              </th>
              {runs.map((run, i) => (
                <th
                  key={run.run_id}
                  className="px-3 py-2 text-right text-xs font-medium"
                  style={{ color: COLORS[i % COLORS.length] }}
                >
                  Run #{run.run_id}
                </th>
              ))}
              {runs.length === 2 && (
                <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground">
                  Delta
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {METRICS.map((metric) => {
              const { bestId, worstId } = getBestWorst(runs, metric.key, metric.higherIsBetter);
              const values = runs.map((r) => r[metric.key]);

              // Delta between first two runs
              let delta: string | null = null;
              if (runs.length === 2) {
                const v0 = values[0];
                const v1 = values[1];
                if (typeof v0 === "number" && typeof v1 === "number" && v0 !== 0) {
                  const pctDiff = ((v1 - v0) / Math.abs(v0)) * 100;
                  delta = `${pctDiff > 0 ? "+" : ""}${pctDiff.toFixed(1)}%`;
                }
              }

              return (
                <tr key={metric.key} className="border-b last:border-b-0">
                  <td className="py-2 pr-4 text-xs font-medium text-foreground">{metric.label}</td>
                  {runs.map((run) => {
                    const val = run[metric.key];
                    const isBest = run.run_id === bestId;
                    const isWorst = run.run_id === worstId && runs.length > 1;
                    return (
                      <td
                        key={run.run_id}
                        className={cn(
                          "px-3 py-2 text-right text-xs tabular-nums",
                          isBest && "font-bold text-green-500",
                          isWorst && !isBest && "text-red-400 opacity-70",
                        )}
                      >
                        {typeof val === "number" ? metric.format(val) : "N/A"}
                      </td>
                    );
                  })}
                  {runs.length === 2 && (
                    <td className="px-3 py-2 text-right text-xs tabular-nums text-muted-foreground">
                      {delta ?? "N/A"}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function EquityCurveOverlay({ runIds }: { runIds: number[] }) {
  // Fetch equity curves for each run
  const q0 = useGetEquityCurveApiResultsRunsRunIdEquityCurveGet(runIds[0] ?? 0, {
    query: { enabled: (runIds[0] ?? 0) > 0 },
  });
  const q1 = useGetEquityCurveApiResultsRunsRunIdEquityCurveGet(runIds[1] ?? 0, {
    query: { enabled: (runIds[1] ?? 0) > 0 },
  });
  const q2 = useGetEquityCurveApiResultsRunsRunIdEquityCurveGet(runIds[2] ?? 0, {
    query: { enabled: (runIds[2] ?? 0) > 0 },
  });
  const q3 = useGetEquityCurveApiResultsRunsRunIdEquityCurveGet(runIds[3] ?? 0, {
    query: { enabled: (runIds[3] ?? 0) > 0 },
  });
  const q4 = useGetEquityCurveApiResultsRunsRunIdEquityCurveGet(runIds[4] ?? 0, {
    query: { enabled: (runIds[4] ?? 0) > 0 },
  });

  const queries = [q0, q1, q2, q3, q4].slice(0, runIds.length);
  const isLoading = queries.some((q) => q.isLoading);

  // Merge all equity curves into one dataset keyed by timestamp
  const mergedData = useMemo(() => {
    const map = new Map<string, Record<string, number>>();

    for (let i = 0; i < runIds.length; i++) {
      const points = queries[i]?.data?.data ?? [];
      for (const pt of points) {
        const existing = map.get(pt.timestamp) ?? {};
        existing[`run_${runIds[i]}`] = pt.equity;
        map.set(pt.timestamp, existing);
      }
    }

    return Array.from(map.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([timestamp, values]) => ({ timestamp, ...values }));
  }, [runIds, queries]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner size="sm" />
      </div>
    );
  }

  if (mergedData.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No equity curve data available.
      </p>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Equity Curves Overlay
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={mergedData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatChartDate}
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tickFormatter={(v: number) => formatCurrency(v)}
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              width={80}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--background))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
                fontSize: "12px",
              }}
              labelFormatter={formatChartDate}
              formatter={(value: number) => [formatCurrency(value), ""]}
            />
            <Legend />
            {runIds.map((id, i) => (
              <Line
                key={id}
                type="monotone"
                dataKey={`run_${id}`}
                name={`Run #${id}`}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

export function ComparisonView() {
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [comparing, setComparing] = useState(false);

  const toggleRun = useCallback((id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else if (next.size < MAX_COMPARE) {
        next.add(id);
      }
      return next;
    });
  }, []);

  const runIdsStr = useMemo(() => Array.from(selectedIds).join(","), [selectedIds]);

  const compareQuery = useCompareRunsApiResultsCompareGet(
    { run_ids: runIdsStr },
    { query: { enabled: comparing && selectedIds.size >= MIN_COMPARE } },
  );

  const comparedRuns = compareQuery.data?.data?.runs ?? [];
  const runIdsArray = useMemo(() => Array.from(selectedIds), [selectedIds]);

  const handleCompare = () => {
    setComparing(true);
  };

  const handleReset = () => {
    setComparing(false);
    setSelectedIds(new Set());
  };

  if (comparing && selectedIds.size >= MIN_COMPARE) {
    return (
      <div className="flex flex-col gap-6">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-semibold text-foreground">
            Comparing {selectedIds.size} Runs
          </h3>
          <Button variant="outline" size="sm" onClick={handleReset}>
            Reset
          </Button>
        </div>

        {compareQuery.isLoading ? (
          <div className="flex items-center justify-center py-12">
            <LoadingSpinner size="sm" />
          </div>
        ) : compareQuery.isError ? (
          <p className="py-4 text-sm text-destructive">Failed to load comparison data.</p>
        ) : (
          <>
            <MetricsTable runs={comparedRuns} />
            <EquityCurveOverlay runIds={runIdsArray} />
          </>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <RunPicker selectedIds={selectedIds} onToggle={toggleRun} />
      <Button
        onClick={handleCompare}
        disabled={selectedIds.size < MIN_COMPARE}
        className="self-start"
      >
        Compare {selectedIds.size} Run{selectedIds.size !== 1 ? "s" : ""}
      </Button>
    </div>
  );
}
