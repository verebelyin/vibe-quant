import { useEffect } from "react";
import { toast } from "sonner";
import { apiErrorDetail } from "@/api/client";
import {
  getListShadowApiShadowGetQueryKey,
  useListShadowApiShadowGet,
  useMarkSeenApiShadowEventsSeenPost,
  useStartShadowUpdateApiShadowUpdatePost,
} from "@/api/generated/shadow/shadow";
import { queryClient } from "@/api/query-client";
import { ShadowRowCard } from "@/components/shadow/ShadowRowCard";
import { HONEST_LIMIT, POWER_LINE } from "@/components/shadow/shadowSentence";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/EmptyState";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

export function ShadowPage() {
  const listQuery = useListShadowApiShadowGet({ query: { refetchInterval: 30_000 } });
  const list = listQuery.data?.status === 200 ? listQuery.data.data : null;

  const { mutate: markSeen } = useMarkSeenApiShadowEventsSeenPost({
    mutation: {
      onSuccess: () =>
        queryClient.invalidateQueries({ queryKey: getListShadowApiShadowGetQueryKey() }),
    },
  });
  useEffect(() => {
    markSeen();
  }, [markSeen]);

  const update = useStartShadowUpdateApiShadowUpdatePost({
    mutation: {
      onSuccess: () =>
        toast.success("Shadow update started", {
          description: "It scores data already downloaded; rows refresh when it finishes.",
        }),
      onError: (err) =>
        toast.error("Shadow update not started", { description: apiErrorDetail(err) }),
    },
  });

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <p className="max-w-2xl text-sm text-muted-foreground">
            Frozen strategies scored on days they never saw. No orders are placed. The tracker runs
            only after a data update you request (Data page or{" "}
            <code>python -m vibe_quant.data update</code>), or when you press Update now; nothing
            runs on a schedule.
          </p>
        </div>
        <Button size="sm" onClick={() => update.mutate()} disabled={update.isPending}>
          {update.isPending ? "Starting…" : "Update now"}
        </Button>
      </div>

      <div className="space-y-1 rounded-lg border border-border/60 bg-muted/30 p-4 text-xs text-muted-foreground">
        <p>{HONEST_LIMIT}</p>
        <p>{POWER_LINE}</p>
        <p>
          Evidence stays 1.0 until 26 full weeks are tracked. A pass is a statistical bar, not a
          recommendation: paper trading stays your decision after the overfit and risk checks.
        </p>
        {list?.kill_switch_engaged && (
          <p>The kill switch is engaged (shown for information; the tracker places no orders).</p>
        )}
      </div>

      {listQuery.isLoading && <LoadingSpinner />}
      {list && list.strategies.length === 0 && (
        <EmptyState
          title="No shadow strategies yet"
          description="Enrol one via POST /api/shadow/enrol with a pre-registration paragraph and a drawdown limit."
        />
      )}
      {list?.strategies.map((row) => (
        <ShadowRowCard key={row.id} row={row} />
      ))}
    </div>
  );
}
