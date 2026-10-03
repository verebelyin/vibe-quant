import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { drawdownToNegativePercent } from "@/lib/metrics";
import type { DrawdownPoint } from "../../api/generated/models/drawdownPoint";

export interface DrawdownChartProps {
  data: DrawdownPoint[];
  height?: number;
  className?: string;
}

function formatDate(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

interface UnderwaterPoint {
  timestamp: string;
  /** Percent below the running peak, as a negative number (-3.12 = 3.12% under water). */
  drawdown_pct: number;
}

interface TooltipPayloadEntry {
  value: number;
  payload: UnderwaterPoint;
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) {
  if (!active || !payload?.length) return null;
  const point = payload[0]!;
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="text-muted-foreground">
        {new Date(point.payload.timestamp).toLocaleDateString()}
      </p>
      <p className="font-medium text-red-500">{point.value.toFixed(2)}%</p>
    </div>
  );
}

export default function DrawdownChart({ data, height = 200, className }: DrawdownChartProps) {
  // The API sends positive fractions ((peak - equity) / peak); plot them under the zero line.
  const underwater: UnderwaterPoint[] = data.map((p) => ({
    timestamp: p.timestamp,
    drawdown_pct: drawdownToNegativePercent(p.drawdown),
  }));
  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={underwater} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="drawdownGradient" x1="0" y1="1" x2="0" y2="0">
              <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#ef4444" stopOpacity={0.02} />
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
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            width={56}
            domain={["dataMin", 0]}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="drawdown_pct"
            stroke="#ef4444"
            strokeWidth={2}
            fill="url(#drawdownGradient)"
            dot={false}
            activeDot={{ r: 4, fill: "#ef4444" }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
