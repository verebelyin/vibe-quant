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

interface YearlyReturnsChartProps {
  data: EquityCurvePoint[];
  height?: number;
}

interface YearReturn {
  year: string;
  returnPct: number;
  partial: boolean;
  dateRange: string; // e.g. "Jan 1 – Dec 31" or "Mar 15 – Dec 31"
}

interface TooltipPayloadEntry {
  value: number;
  payload: YearReturn;
}

function formatMonthDay(date: Date): string {
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function computeYearlyReturns(data: EquityCurvePoint[]): YearReturn[] {
  if (data.length < 2) return [];

  const byYear = new Map<number, { first: number; last: number; firstDate: Date; lastDate: Date }>();

  for (const point of data) {
    const d = new Date(point.timestamp);
    const year = d.getFullYear();
    const existing = byYear.get(year);
    if (!existing) {
      byYear.set(year, { first: point.equity, last: point.equity, firstDate: d, lastDate: d });
    } else {
      existing.last = point.equity;
      existing.lastDate = d;
    }
  }

  const result: YearReturn[] = [];
  for (const [year, { first, last, firstDate, lastDate }] of byYear) {
    if (first !== 0) {
      // Partial if first data point is after Jan 7 or last data point is before Dec 24
      const startsLate = firstDate.getMonth() > 0 || firstDate.getDate() > 7;
      const endsEarly = lastDate.getMonth() < 11 || lastDate.getDate() < 24;
      const partial = startsLate || endsEarly;
      const dateRange = `${formatMonthDay(firstDate)} – ${formatMonthDay(lastDate)}`;

      result.push({
        year: String(year),
        returnPct: Number.parseFloat((((last - first) / first) * 100).toFixed(2)),
        partial,
        dateRange,
      });
    }
  }

  return result.sort((a, b) => Number(a.year) - Number(b.year));
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) {
  if (!active || !payload?.length) return null;
  const point = payload[0]!;
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="text-muted-foreground">
        {point.payload.year}
        {point.payload.partial && " (partial)"}
      </p>
      <p className="font-medium text-foreground">{(point.value as number).toFixed(2)}%</p>
      {point.payload.partial && (
        <p className="text-xs text-muted-foreground">{point.payload.dateRange}</p>
      )}
    </div>
  );
}

export function YearlyReturnsChart({ data, height = 300 }: YearlyReturnsChartProps) {
  const yearlyData = useMemo(() => computeYearlyReturns(data), [data]);

  if (yearlyData.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No yearly return data available.
      </p>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={yearlyData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
        <XAxis
          dataKey="year"
          stroke="hsl(var(--muted-foreground))"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          tickFormatter={(year: string) => {
            const entry = yearlyData.find((d) => d.year === year);
            return entry?.partial ? `${year}*` : year;
          }}
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
        <Bar dataKey="returnPct" radius={[4, 4, 0, 0]}>
          {yearlyData.map((entry) => (
            <Cell
              key={entry.year}
              fill={entry.returnPct >= 0 ? "#22c55e" : "#ef4444"}
              fillOpacity={entry.partial ? 0.55 : 1}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
