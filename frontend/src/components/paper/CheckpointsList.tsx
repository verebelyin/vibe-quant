import { useMemo } from "react";
import type { CheckpointResponse } from "@/api/generated/models";
import { useGetCheckpointsApiPaperCheckpointsGet } from "@/api/generated/paper/paper";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const STATE_BADGE: Record<string, string> = {
  running: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
  halted: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300",
  stopped: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  error: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
};
const FALLBACK = "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300";

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString();
}

export function CheckpointsList() {
  const { data: cpResp, isLoading } = useGetCheckpointsApiPaperCheckpointsGet();

  const checkpoints: CheckpointResponse[] = useMemo(() => {
    if (!cpResp) return [];
    if (cpResp.status === 200) return cpResp.data;
    return [];
  }, [cpResp]);

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-foreground">Checkpoints</h2>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading checkpoints...</p>
      ) : checkpoints.length === 0 ? (
        <p className="text-xs text-muted-foreground">No checkpoints recorded.</p>
      ) : (
        <div className="space-y-2">
          {checkpoints.map((cp, idx) => {
            const normalized = cp.state.toLowerCase();
            const badgeCls = STATE_BADGE[normalized] ?? FALLBACK;
            return (
              <div
                key={`${cp.timestamp}-${idx}`}
                className="flex items-start justify-between rounded-md border border-border bg-card px-4 py-3"
              >
                <div className="space-y-1">
                  <p className="font-mono text-xs text-foreground">
                    {formatTimestamp(cp.timestamp)}
                  </p>
                  {cp.halt_reason && (
                    <p className="text-xs text-yellow-600">Halt: {cp.halt_reason}</p>
                  )}
                  {cp.error_message && (
                    <p className="text-xs text-red-600">Error: {cp.error_message}</p>
                  )}
                </div>
                <Badge variant="outline" className={cn("border-transparent", badgeCls)}>
                  {cp.state}
                </Badge>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
