import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EquityCurvePoint } from "@/api/generated/models/equityCurvePoint";

interface DailyReturnsChartProps {
  data: EquityCurvePoint[];
  height?: number;
}

interface DailyReturn {
  timestamp: string;
  returnPct: number;
}

interface TooltipPayloadEntry {
  value: number;
  payload: DailyReturn;
}

function computeDailyReturns(data: EquityCurvePoint[]): DailyReturn[] {
  if (data.length < 2) return [];

  const result: DailyReturn[] = [];
  for (let i = 1; i < data.length; i++) {
    const prev = data[i - 1].equity;
    const curr = data[i].equity;
    if (prev !== 0) {
      result.push({
        timestamp: data[i].timestamp,
        returnPct: Number.parseFloat((((curr - prev) / prev) * 100).toFixed(4)),
      });
    }
  }
  return result;
}

function formatDate(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) {
  if (!active || !payload?.length) return null;
  const point = payload[0];
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="text-muted-foreground">
        {new Date(point.payload.timestamp).toLocaleDateString()}
      </p>
      <p className="font-medium text-foreground">{point.value.toFixed(4)}%</p>
    </div>
  );
}

export function DailyReturnsChart({ data, height = 300 }: DailyReturnsChartProps) {
  const dailyData = useMemo(() => computeDailyReturns(data), [data]);

  if (dailyData.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No daily return data available.
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={dailyData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
        <XAxis
          dataKey="timestamp"
          tickFormatter={formatDate}
          stroke="hsl(var(--muted-foreground))"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tickFormatter={(v: number) => `${v}%`}
          stroke="hsl(var(--muted-foreground))"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          width={60}
        />
        <Tooltip content={<CustomTooltip />} />
        <Bar dataKey="returnPct" maxBarSize={3}>
          {dailyData.map((entry) => (
            <Cell key={entry.timestamp} fill={entry.returnPct >= 0 ? "#22c55e" : "#ef4444"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
