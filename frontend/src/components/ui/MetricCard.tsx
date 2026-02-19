interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
  className?: string;
}

const trendConfig = {
  up: { symbol: "\u2191", color: "text-green-500" },
  down: { symbol: "\u2193", color: "text-red-500" },
  neutral: { symbol: "\u2014", color: "text-gray-400" },
} as const;

export function MetricCard({ label, value, subtitle, trend, className = "" }: MetricCardProps) {
  return (
    <div
      className={`rounded-lg border p-4 ${className}`}
      style={{
        backgroundColor: "hsl(var(--card))",
        color: "hsl(var(--card-foreground))",
        borderColor: "hsl(var(--border))",
      }}
    >
      <p
        className="text-xs font-medium uppercase tracking-wide"
        style={{ color: "hsl(var(--muted-foreground))" }}
      >
        {label}
      </p>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="text-2xl font-bold">{value}</span>
        {trend && (
          <span className={`text-sm font-semibold ${trendConfig[trend].color}`}>
            {trendConfig[trend].symbol}
          </span>
        )}
      </div>
      {subtitle && (
        <p className="mt-1 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
          {subtitle}
        </p>
      )}
    </div>
  );
}
