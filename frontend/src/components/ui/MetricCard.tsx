import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string | undefined;
  trend?: "up" | "down" | "neutral" | undefined;
  className?: string | undefined;
}

const trendConfig = {
  up: {
    symbol: "\u25B2",
    color: "text-profit",
    border: "border-l-emerald-500/60",
    bg: "bg-emerald-500/[0.04]",
  },
  down: {
    symbol: "\u25BC",
    color: "text-loss",
    border: "border-l-red-500/60",
    bg: "bg-red-500/[0.04]",
  },
  neutral: {
    symbol: "\u2014",
    color: "text-muted-foreground",
    border: "border-l-transparent",
    bg: "",
  },
} as const;

export function MetricCard({ label, value, subtitle, trend, className }: MetricCardProps) {
  const t = trend ? trendConfig[trend] : null;
  return (
    <Card
      className={cn(
        "gap-0 py-4 group border-l-2 transition-all",
        t?.border ?? "border-l-transparent",
        t?.bg,
        className,
      )}
    >
      <CardContent>
        <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </p>
        <div className="mt-2 flex items-baseline gap-2">
          <span className="text-2xl font-bold tracking-tight font-mono">{value}</span>
          {trend && trend !== "neutral" && (
            <span className={cn("text-[10px] font-semibold", t?.color)}>
              {t?.symbol}
            </span>
          )}
        </div>
        {subtitle && <p className="mt-1 text-xs text-muted-foreground/70">{subtitle}</p>}
      </CardContent>
    </Card>
  );
}
