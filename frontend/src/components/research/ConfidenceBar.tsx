import { cn } from "@/lib/utils";

interface Props {
  value: number | null | undefined;
  className?: string;
}

export function ConfidenceBar({ value, className }: Props) {
  if (value == null) {
    return <span className="text-xs text-muted-foreground">—</span>;
  }
  const clamped = Math.max(0, Math.min(1, value));
  const pct = Math.round(clamped * 100);
  const tone = clamped >= 0.7 ? "bg-emerald-500" : clamped >= 0.4 ? "bg-amber-400" : "bg-red-500";
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full transition-all", tone)}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-[10px] tabular-nums text-muted-foreground">
        {clamped.toFixed(2)}
      </span>
    </div>
  );
}
