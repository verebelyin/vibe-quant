import type { ShadowStrategy } from "@/api/generated/models/shadowStrategy";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { shadowSentence } from "./shadowSentence";

const STATE_BADGE: Record<ShadowStrategy["state"], string> = {
  tracking: "bg-sky-100 text-sky-700 dark:bg-sky-900 dark:text-sky-300",
  stalled: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300",
  passed: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
  futile: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  retired: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
};

export function ShadowRowCard({ row }: { row: ShadowStrategy }) {
  return (
    <div className="space-y-3 rounded-xl border border-border/60 bg-card/40 p-5">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold text-foreground">{row.name}</h3>
        <Badge className={cn("capitalize", STATE_BADGE[row.state])}>{row.state}</Badge>
        {row.is_test && <Badge variant="outline">test</Badge>}
        <span className="text-xs text-muted-foreground">
          {row.symbols.join(", ")} · {row.timeframe} · enrolled {row.enrolled_at.slice(0, 10)} ·
          first forward day {row.first_day}
        </span>
      </div>
      <p className="text-sm leading-relaxed text-foreground/90">{shadowSentence(row)}</p>
      {row.status_note && row.status_note !== row.state_reason && (
        <p className="text-xs text-muted-foreground">Last update: {row.status_note}</p>
      )}
      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer select-none">Pre-registration and frozen inputs</summary>
        <div className="mt-2 space-y-1">
          <p className="whitespace-pre-wrap text-foreground/80">{row.pre_registration}</p>
          <p>
            Scored through {row.committed_through ?? "—"} · last run {row.last_update_at ?? "never"}
            {" · "}engine epoch {row.engine_epoch_last} · DSL {row.dsl_sha256.slice(0, 12)}
          </p>
        </div>
      </details>
    </div>
  );
}
