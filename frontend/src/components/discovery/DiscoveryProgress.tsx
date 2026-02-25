import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { Badge } from "@/components/ui/badge";
import { useWebSocket, type WsMessage } from "@/hooks/useWebSocket";
import { cn } from "@/lib/utils";

interface GenerationData {
  generation: number;
  best: number;
  avg: number;
  worst: number;
}

interface BestStrategy {
  sharpe: number | null;
  returnPct: number | null;
  maxDrawdown: number | null;
  winRate: number | null;
}

interface DiscoveryProgressProps {
  runId: number;
  totalGenerations: number;
  convergenceWindow?: number;
  convergenceThreshold?: number;
}

const DEFAULT_CONVERGENCE_WINDOW = 5;
const DEFAULT_CONVERGENCE_THRESHOLD = 0.001;

interface FitnessTooltipPayload {
  name: string;
  value: number;
  color: string;
}

function FitnessTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: FitnessTooltipPayload[];
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="mb-1 font-medium text-muted-foreground">Gen {label}</p>
      {payload.map((entry) => (
        <p key={entry.name} style={{ color: entry.color }}>
          {entry.name}: {entry.value.toFixed(4)}
        </p>
      ))}
    </div>
  );
}

export function DiscoveryProgress({
  runId,
  totalGenerations,
  convergenceWindow = DEFAULT_CONVERGENCE_WINDOW,
  convergenceThreshold = DEFAULT_CONVERGENCE_THRESHOLD,
}: DiscoveryProgressProps) {
  const ws = useWebSocket("discovery");
  const [history, setHistory] = useState<GenerationData[]>([]);
  const [bestStrategy, setBestStrategy] = useState<BestStrategy>({
    sharpe: null,
    returnPct: null,
    maxDrawdown: null,
    winRate: null,
  });
  const chartRef = useRef<HTMLDivElement>(null);

  const handleMessage = useCallback(
    (msg: WsMessage) => {
      if (msg.type !== "discovery_progress") return;
      if (msg.run_id !== runId) return;

      const gen = Number(msg.generation ?? 0);
      const best = Number(msg.best_fitness ?? 0);
      const avg = Number(msg.avg_fitness ?? 0);
      const worst = Number(msg.worst_fitness ?? 0);

      setHistory((prev) => {
        // avoid duplicates
        if (prev.length > 0 && prev[prev.length - 1]!.generation === gen) return prev;
        return [...prev, { generation: gen, best, avg, worst }];
      });

      // Update best strategy preview from WS payload
      const metrics = msg.best_metrics as Record<string, unknown> | undefined;
      if (metrics) {
        setBestStrategy({
          sharpe: metrics.sharpe != null ? Number(metrics.sharpe) : null,
          returnPct: metrics.return_pct != null ? Number(metrics.return_pct) : null,
          maxDrawdown: metrics.max_drawdown != null ? Number(metrics.max_drawdown) : null,
          winRate: metrics.win_rate != null ? Number(metrics.win_rate) : null,
        });
      }
    },
    [runId],
  );

  useEffect(() => {
    if (ws.lastMessage) {
      handleMessage(ws.lastMessage);
    }
  }, [ws.lastMessage, handleMessage]);

  const currentGen = history.length > 0 ? history[history.length - 1]!.generation : 0;
  const progressPct = totalGenerations > 0 ? Math.round((currentGen / totalGenerations) * 100) : 0;

  const isConverging = useMemo(() => {
    if (history.length < convergenceWindow + 1) return false;
    const recent = history.slice(-convergenceWindow);
    const firstBest = recent[0]!.best;
    const lastBest = recent[recent.length - 1]!.best;
    return Math.abs(lastBest - firstBest) < convergenceThreshold;
  }, [history, convergenceWindow, convergenceThreshold]);

  const currentBestFitness = history.length > 0 ? history[history.length - 1]!.best : null;

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
          Discovery Progress
        </h3>
        <div className="flex items-center gap-2">
          {isConverging && (
            <Badge className="bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300 border-transparent">
              Converging
            </Badge>
          )}
          <Badge variant="outline" className="font-mono text-xs">
            {ws.status}
          </Badge>
        </div>
      </div>

      {/* Generation counter */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">
            Generation {currentGen} / {totalGenerations}
          </span>
          <span className="font-mono text-foreground">{progressPct}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-blue-500 transition-all duration-300"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {/* Fitness chart */}
      <div ref={chartRef} className="h-64">
        {history.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <p className="text-xs text-muted-foreground">Waiting for generation data...</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
              <XAxis
                dataKey="generation"
                stroke="hsl(var(--muted-foreground))"
                fontSize={11}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                stroke="hsl(var(--muted-foreground))"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                width={60}
                tickFormatter={(v: number) => v.toFixed(2)}
              />
              <Tooltip content={<FitnessTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line
                type="monotone"
                dataKey="best"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
                name="Best"
              />
              <Line
                type="monotone"
                dataKey="avg"
                stroke="#3b82f6"
                strokeWidth={1.5}
                dot={false}
                name="Average"
              />
              <Line
                type="monotone"
                dataKey="worst"
                stroke="#ef4444"
                strokeWidth={1}
                dot={false}
                name="Worst"
                strokeDasharray="4 2"
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Best strategy preview */}
      {currentBestFitness != null && (
        <div className="space-y-2 rounded-md border border-border bg-background p-3">
          <h4 className="text-xs font-semibold text-muted-foreground">Best Strategy</h4>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <MetricItem label="Fitness" value={currentBestFitness.toFixed(4)} />
            <MetricItem
              label="Sharpe"
              value={bestStrategy.sharpe != null ? bestStrategy.sharpe.toFixed(2) : "--"}
            />
            <MetricItem
              label="Return"
              value={
                bestStrategy.returnPct != null ? `${bestStrategy.returnPct.toFixed(1)}%` : "--"
              }
              positive={bestStrategy.returnPct != null ? bestStrategy.returnPct >= 0 : undefined}
            />
            <MetricItem
              label="Max DD"
              value={
                bestStrategy.maxDrawdown != null ? `${bestStrategy.maxDrawdown.toFixed(1)}%` : "--"
              }
              positive={
                bestStrategy.maxDrawdown != null ? bestStrategy.maxDrawdown >= 0 : undefined
              }
            />
          </div>
        </div>
      )}
    </div>
  );
}

function MetricItem({
  label,
  value,
  positive,
}: {
  label: string;
  value: string;
  positive?: boolean | undefined;
}) {
  return (
    <div>
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p
        className={cn(
          "font-mono text-sm font-medium",
          positive === true && "text-green-600",
          positive === false && "text-red-600",
          positive === undefined && "text-foreground",
        )}
      >
        {value}
      </p>
    </div>
  );
}
