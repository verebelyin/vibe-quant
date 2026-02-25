import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

export interface PerformanceMetric {
  label: string;
  value: number;
  max: number;
}

export interface PerformanceRadarProps {
  metrics: PerformanceMetric[];
  height?: number;
  className?: string;
}

interface NormalizedMetric {
  label: string;
  normalized: number;
  raw: number;
}

interface TooltipPayloadEntry {
  payload: NormalizedMetric;
  value: number;
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) {
  if (!active || !payload?.length) return null;
  const metric = payload[0]!.payload;
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="text-muted-foreground">{metric.label}</p>
      <p className="font-medium text-foreground">{metric.raw.toFixed(2)}</p>
    </div>
  );
}

export default function PerformanceRadar({
  metrics,
  height = 300,
  className,
}: PerformanceRadarProps) {
  const normalized: NormalizedMetric[] = metrics.map((m) => ({
    label: m.label,
    normalized: m.max !== 0 ? Math.min(m.value / m.max, 1) * 100 : 0,
    raw: m.value,
  }));

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={height}>
        <RadarChart cx="50%" cy="50%" outerRadius="75%" data={normalized}>
          <PolarGrid stroke="hsl(var(--border))" />
          <PolarAngleAxis
            dataKey="label"
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          />
          <PolarRadiusAxis angle={90} domain={[0, 100]} tick={false} axisLine={false} />
          <Tooltip content={<CustomTooltip />} />
          <Radar
            name="Performance"
            dataKey="normalized"
            stroke="hsl(var(--accent-foreground))"
            fill="hsl(var(--accent))"
            fillOpacity={0.5}
            strokeWidth={2}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
