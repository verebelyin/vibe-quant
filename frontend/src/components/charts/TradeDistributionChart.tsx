import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface TradeDistributionChartProps {
  trades: { roi_percent: number; direction: string }[];
  height?: number;
  className?: string;
}

interface Bin {
  range: string;
  center: number;
  count: number;
}

const BIN_COUNT = 20;

/**
 * Bin range from the data. roi_percent is a percent of notional (typically
 * well under ±5%), so the old fixed ±50% / 5% bins put nearly every trade
 * into the two bins around zero.
 */
export function histogramRange(values: number[]): { min: number; max: number; step: number } {
  if (!values.length) return { min: -1, max: 1, step: 0.1 };
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const step = (hi - lo) / BIN_COUNT;
  return { min: lo, max: hi, step };
}

export function buildHistogram(
  trades: { roi_percent: number }[],
  min: number,
  max: number,
  step: number,
): Bin[] {
  const bins: Bin[] = [];
  const n = Math.max(1, Math.round((max - min) / step));
  for (let i = 0; i < n; i++) {
    const lo = min + i * step;
    const hi = lo + step;
    bins.push({
      range: `${lo.toFixed(2)}% to ${hi.toFixed(2)}%`,
      center: lo + step / 2,
      count: 0,
    });
  }
  for (const t of trades) {
    const roi = t.roi_percent;
    const idx = Math.floor((roi - min) / step);
    const clamped = Math.max(0, Math.min(bins.length - 1, idx));
    bins[clamped]!.count += 1;
  }
  return bins;
}

function computeMean(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function computeMedian(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
}

interface TooltipPayloadEntry {
  payload: Bin;
  value: number;
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) {
  if (!active || !payload?.length) return null;
  const bin = payload[0]!.payload;
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="text-muted-foreground">{bin.range}</p>
      <p className="font-medium text-foreground">
        {bin.count} trade{bin.count !== 1 ? "s" : ""}
      </p>
    </div>
  );
}

export default function TradeDistributionChart({
  trades,
  height = 300,
  className,
}: TradeDistributionChartProps) {
  const roiValues = trades.map((t) => t.roi_percent);
  const { min, max, step } = histogramRange(roiValues);
  const bins = buildHistogram(trades, min, max, step);
  const mean = computeMean(roiValues);
  const median = computeMedian(roiValues);

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={bins} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
          {/* Numeric axis so the mean/median reference lines land at their values
              (on a category axis they only rendered on an exact bin centre). */}
          <XAxis
            type="number"
            dataKey="center"
            domain={[min, max]}
            tickFormatter={(v: number) => `${v.toFixed(2)}%`}
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
            allowDecimals={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            x={mean}
            stroke="#3b82f6"
            strokeDasharray="4 4"
            strokeWidth={2}
            label={{ value: "Mean", position: "top", fill: "#3b82f6", fontSize: 11 }}
          />
          <ReferenceLine
            x={median}
            stroke="#a855f7"
            strokeDasharray="4 4"
            strokeWidth={2}
            label={{ value: "Median", position: "top", fill: "#a855f7", fontSize: 11 }}
          />
          <Bar dataKey="count" radius={[2, 2, 0, 0]}>
            {bins.map((bin) => (
              <Cell
                key={bin.range}
                fill={bin.center >= 0 ? "#22c55e" : "#ef4444"}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
