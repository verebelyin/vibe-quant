import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const STYLES: Record<string, string> = {
  queued: "bg-sky-500/15 text-sky-300 border-sky-500/25",
  pending: "bg-sky-500/15 text-sky-300 border-sky-500/25",
  running: "bg-blue-500/15 text-blue-300 border-blue-500/25 animate-pulse",
  completed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/25",
  parsed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/25",
  extracted: "bg-emerald-500/15 text-emerald-300 border-emerald-500/25",
  promoted: "bg-violet-500/15 text-violet-300 border-violet-500/25",
  failed: "bg-amber-500/15 text-amber-300 border-amber-500/25",
  cancelled: "bg-amber-500/15 text-amber-300 border-amber-500/25",
  killed: "bg-amber-500/15 text-amber-300 border-amber-500/25",
  rejected: "bg-red-500/15 text-red-300 border-red-500/25",
  skipped: "bg-muted text-muted-foreground border-border",
};

const FALLBACK = "bg-muted text-muted-foreground border-border";

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const cls = STYLES[status.toLowerCase()] ?? FALLBACK;
  return (
    <Badge variant="outline" className={cn(cls, "font-mono text-[10px] capitalize", className)}>
      {status}
    </Badge>
  );
}
