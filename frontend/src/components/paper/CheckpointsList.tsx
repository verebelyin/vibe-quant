import { useMemo } from "react";
import { toast } from "sonner";
import type { CheckpointResponse } from "@/api/generated/models";
import {
  getGetStatusApiPaperStatusGetQueryKey,
  useGetCheckpointsApiPaperCheckpointsGet,
  useRestorePaperApiPaperRestorePost,
} from "@/api/generated/paper/paper";
import { queryClient } from "@/api/query-client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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

interface CheckpointsListProps {
  traderId?: string | undefined;
  sessionActive?: boolean;
}

export function CheckpointsList({ traderId, sessionActive = false }: CheckpointsListProps = {}) {
  const { data: cpResp, isLoading } = useGetCheckpointsApiPaperCheckpointsGet(
    traderId ? { trader_id: traderId } : undefined,
    { query: { enabled: !!traderId } },
  );

  const restoreMutation = useRestorePaperApiPaperRestorePost();

  const checkpoints: CheckpointResponse[] = useMemo(() => {
    if (!cpResp) return [];
    if (cpResp.status === 200) return cpResp.data;
    return [];
  }, [cpResp]);

  function handleRestore() {
    if (!traderId) return;
    if (sessionActive) {
      toast.error("Stop the active session before restoring");
      return;
    }
    restoreMutation.mutate(
      { data: { trader_id: traderId } },
      {
        onSuccess: (resp) => {
          if (resp.status === 201) {
            toast.success("Paper session restored", {
              description: `Trader: ${traderId}`,
            });
            queryClient.invalidateQueries({
              queryKey: getGetStatusApiPaperStatusGetQueryKey(),
            });
          }
        },
        onError: (err: unknown) => {
          let message = "Restore failed";
          const axiosErr = err as { response?: { data?: { detail?: string }; status?: number } };
          if (axiosErr.response?.data?.detail) {
            message = axiosErr.response.data.detail;
          } else if (err instanceof Error) {
            message = err.message;
          }
          toast.error("Restore failed", { description: message });
        },
      },
    );
  }

  const restoreDisabled = sessionActive || restoreMutation.isPending || !traderId;

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-foreground">Checkpoints</h2>

      {!traderId ? (
        <p className="text-xs text-muted-foreground">Active session required.</p>
      ) : isLoading ? (
        <p className="text-xs text-muted-foreground">Loading checkpoints...</p>
      ) : checkpoints.length === 0 ? (
        <p className="text-xs text-muted-foreground">No checkpoints recorded.</p>
      ) : (
        <div className="space-y-2">
          {checkpoints.map((cp, idx) => {
            const normalized = cp.state.toLowerCase();
            const badgeCls = STATE_BADGE[normalized] ?? FALLBACK;
            const isLatest = idx === 0;
            return (
              <div
                key={`${cp.timestamp}-${idx}`}
                className="flex items-start justify-between gap-3 rounded-md border border-border bg-card px-4 py-3"
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
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className={cn("border-transparent", badgeCls)}>
                    {cp.state}
                  </Badge>
                  {isLatest && (
                    <Button
                      type="button"
                      size="xs"
                      variant="outline"
                      disabled={restoreDisabled}
                      onClick={handleRestore}
                      title={
                        sessionActive
                          ? "Stop the active session first"
                          : "Start a new session reusing this trader's last config + latest checkpoint"
                      }
                    >
                      {restoreMutation.isPending ? "Restoring..." : "Restore"}
                    </Button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
