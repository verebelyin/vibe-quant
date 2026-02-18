import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

export interface SweepScatterPoint {
  sharpe_ratio: number;
  max_drawdown: number;
  total_return: number;
  is_pareto_optimal: boolean;
}

export interface SweepScatterChartProps {
  data: SweepScatterPoint[];
  height?: number;
  className?: string;
}

interface TooltipPayloadEntry {
  payload: SweepScatterPoint;
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--background))] px-3 py-2 text-sm shadow-md">
      <p className="text-[hsl(var(--foreground))]">
        Sharpe: <span className="font-medium">{p.sharpe_ratio.toFixed(2)}</span>
      </p>
      <p className="text-[hsl(var(--foreground))]">
        Max DD: <span className="font-medium">{(p.max_drawdown * 100).toFixed(1)}%</span>
      </p>
      <p className="text-[hsl(var(--foreground))]">
        Return: <span className="font-medium">{(p.total_return * 100).toFixed(1)}%</span>
      </p>
      {p.is_pareto_optimal && <p className="mt-1 font-medium text-[#3b82f6]">Pareto Optimal</p>}
    </div>
  );
}

export default function SweepScatterChart({
  data,
  height = 400,
  className,
}: SweepScatterChartProps) {
  const paretoPoints = data.filter((d) => d.is_pareto_optimal);
  const otherPoints = data.filter((d) => !d.is_pareto_optimal);

  const returnValues = data.map((d) => Math.abs(d.total_return));
  const maxReturn = Math.max(...returnValues, 0.01);

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey="max_drawdown"
            name="Max Drawdown"
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            label={{
              value: "Max Drawdown",
              position: "insideBottom",
              offset: -4,
              fill: "hsl(var(--muted-foreground))",
              fontSize: 12,
            }}
          />
          <YAxis
            dataKey="sharpe_ratio"
            name="Sharpe Ratio"
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            label={{
              value: "Sharpe Ratio",
              angle: -90,
              position: "insideLeft",
              fill: "hsl(var(--muted-foreground))",
              fontSize: 12,
            }}
          />
          <ZAxis dataKey="total_return" range={[40, 400]} domain={[0, maxReturn]} />
          <Tooltip content={<CustomTooltip />} />
          <Scatter
            name="Other"
            data={otherPoints}
            fill="hsl(var(--muted-foreground))"
            fillOpacity={0.4}
          />
          <Scatter
            name="Pareto Optimal"
            data={paretoPoints}
            fill="#3b82f6"
            fillOpacity={0.9}
            stroke="#3b82f6"
            strokeWidth={1}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
