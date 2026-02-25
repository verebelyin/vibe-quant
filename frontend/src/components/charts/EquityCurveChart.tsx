import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EquityCurvePoint } from "../../api/generated/models/equityCurvePoint";

export interface EquityCurveChartProps {
  data: EquityCurvePoint[];
  height?: number;
  className?: string;
}

function formatDate(ts: string): string {
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

interface TooltipPayloadEntry {
  value: number;
  payload: EquityCurvePoint;
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) {
  if (!active || !payload?.length) return null;
  const point = payload[0]!;
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="text-muted-foreground">
        {new Date(point.payload.timestamp).toLocaleDateString()}
      </p>
      <p className="font-medium text-foreground">{formatCurrency(point.value)}</p>
    </div>
  );
}

export default function EquityCurveChart({ data, height = 300, className }: EquityCurveChartProps) {
  // Find max drawdown trough: track running peak and find min equity/peak ratio
  const maxDrawdownPoint = useMemo(() => {
    if (data.length < 2) return null;
    let peak = data[0]!.equity;
    let maxDd = 0;
    let troughIdx = -1;
    for (let i = 1; i < data.length; i++) {
      if (data[i]!.equity > peak) peak = data[i]!.equity;
      const dd = (peak - data[i]!.equity) / peak;
      if (dd > maxDd) {
        maxDd = dd;
        troughIdx = i;
      }
    }
    return troughIdx >= 0 ? data[troughIdx]! : null;
  }, [data]);

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22c55e" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#22c55e" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatDate}
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
          <Tooltip content={<CustomTooltip />} />
          {maxDrawdownPoint && (
            <ReferenceLine
              x={maxDrawdownPoint.timestamp}
              stroke="#ef4444"
              strokeDasharray="4 3"
              strokeWidth={1.5}
              label={{
                value: "Max DD",
                position: "insideTopRight",
                fontSize: 10,
                fill: "#ef4444",
              }}
            />
          )}
          <Area
            type="monotone"
            dataKey="equity"
            stroke="#22c55e"
            strokeWidth={2}
            fill="url(#equityGradient)"
            dot={false}
            activeDot={{ r: 4, fill: "#22c55e" }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
