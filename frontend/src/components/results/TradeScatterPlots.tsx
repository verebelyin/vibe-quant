import { useMemo } from "react";
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
import type { TradeResponse } from "@/api/generated/models/tradeResponse";

interface TradeScatterPlotsProps {
  trades: TradeResponse[];
  height?: number;
}

interface RoiDurationPoint {
  duration: number;
  roi: number;
}

interface SizePnlPoint {
  size: number;
  pnl: number;
}

function getDurationHours(entry: string, exit: string | null): number | null {
  if (!exit) return null;
  return (new Date(exit).getTime() - new Date(entry).getTime()) / (1000 * 60 * 60);
}

function RoiTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { value: number; name: string; payload: RoiDurationPoint }[];
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="text-muted-foreground">Duration: {p.duration.toFixed(1)}h</p>
      <p className="font-medium text-foreground">ROI: {p.roi.toFixed(2)}%</p>
    </div>
  );
}

function SizePnlTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { value: number; name: string; payload: SizePnlPoint }[];
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="text-muted-foreground">Size: {p.size.toFixed(4)}</p>
      <p className="font-medium text-foreground">PnL: ${p.pnl.toFixed(2)}</p>
    </div>
  );
}

export function TradeScatterPlots({ trades, height = 300 }: TradeScatterPlotsProps) {
  const roiDuration = useMemo(() => {
    const points: RoiDurationPoint[] = [];
    for (const t of trades) {
      const dur = getDurationHours(t.entry_time, t.exit_time);
      if (dur != null && t.roi_percent != null) {
        points.push({ duration: Number.parseFloat(dur.toFixed(2)), roi: t.roi_percent });
      }
    }
    return points;
  }, [trades]);

  const sizePnl = useMemo(() => {
    const points: SizePnlPoint[] = [];
    for (const t of trades) {
      if (t.net_pnl != null) {
        points.push({ size: t.quantity, pnl: t.net_pnl });
      }
    }
    return points;
  }, [trades]);

  if (roiDuration.length === 0 && sizePnl.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">No scatter data available.</p>
    );
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {roiDuration.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-medium text-muted-foreground">ROI vs Duration</h4>
          <ResponsiveContainer width="100%" height={height}>
            <ScatterChart margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="duration"
                name="Duration (h)"
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
                unit="h"
              />
              <YAxis
                dataKey="roi"
                name="ROI %"
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
                unit="%"
                width={60}
              />
              <ZAxis range={[20, 20]} />
              <Tooltip content={<RoiTooltip />} />
              <Scatter data={roiDuration} fill="#8b5cf6" fillOpacity={0.6} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {sizePnl.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-medium text-muted-foreground">Size vs PnL</h4>
          <ResponsiveContainer width="100%" height={height}>
            <ScatterChart margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="size"
                name="Size"
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
              />
              <YAxis
                dataKey="pnl"
                name="PnL"
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
                width={70}
                tickFormatter={(v: number) => `$${v}`}
              />
              <ZAxis range={[20, 20]} />
              <Tooltip content={<SizePnlTooltip />} />
              <Scatter data={sizePnl} fill="#3b82f6" fillOpacity={0.6} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
