import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EquityCurvePoint } from "@/api/generated/models/equityCurvePoint";
import { computeRollingSharpe, type SharpePoint } from "@/lib/equity";

// Re-export so existing imports (tests) keep working; the implementation
// now resamples per-trade equity points to calendar days first and
// annualizes with sqrt(365) — see lib/equity.ts (vibe-quant-pazbr).
export { computeRollingSharpe };

interface RollingSharpeChartProps {
  data: EquityCurvePoint[];
  height?: number;
  window?: number;
}

interface TooltipPayloadEntry {
  value: number;
  payload: SharpePoint;
}

function formatDate(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) {
  if (!active || !payload?.length) return null;
  const point = payload[0]!;
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="text-muted-foreground">
        {new Date(point.payload.timestamp).toLocaleDateString()}
      </p>
      <p className="font-medium text-foreground">Sharpe: {point.value.toFixed(3)}</p>
    </div>
  );
}

export function RollingSharpeChart({ data, height = 300, window = 30 }: RollingSharpeChartProps) {
  const sharpeData = useMemo(() => computeRollingSharpe(data, window), [data, window]);

  if (sharpeData.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        Not enough data for {window}-day rolling Sharpe.
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={sharpeData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
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
          stroke="hsl(var(--muted-foreground))"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          width={50}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
        <Line
          type="monotone"
          dataKey="sharpe"
          stroke="#8b5cf6"
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3, fill: "#8b5cf6" }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
