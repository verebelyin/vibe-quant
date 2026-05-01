import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type Status = "pending" | "extracted" | "parsed" | "failed" | "promoted" | "rejected" | "skipped";

const STYLES: Record<Status, string> = {
  parsed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/25",
  extracted: "bg-emerald-500/15 text-emerald-300 border-emerald-500/25",
  failed: "bg-amber-500/15 text-amber-300 border-amber-500/25",
  pending: "bg-sky-500/15 text-sky-300 border-sky-500/25",
  skipped: "bg-muted text-muted-foreground border-border",
  promoted: "bg-violet-500/15 text-violet-300 border-violet-500/25",
  rejected: "bg-red-500/15 text-red-300 border-red-500/25",
};

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const key = (status in STYLES ? status : "pending") as Status;
  return (
    <Badge variant="outline" className={cn(STYLES[key], "font-mono text-[10px]", className)}>
      {status}
    </Badge>
  );
}
