import { useGetStatusApiPaperStatusGet } from "@/api/generated/paper/paper";
import { CheckpointsList } from "@/components/paper/CheckpointsList";
import { LiveDashboard } from "@/components/paper/LiveDashboard";
import { PositionsTable } from "@/components/paper/PositionsTable";
import { SessionControl } from "@/components/paper/SessionControl";
import { TraderInfo } from "@/components/paper/TraderInfo";

export function PaperTradingPage() {
  const statusQuery = useGetStatusApiPaperStatusGet({
    query: { refetchInterval: 5_000 },
  });

  const status = statusQuery.data?.status === 200 ? statusQuery.data.data : null;
  const currentState = status?.state?.toLowerCase() ?? "unknown";
  const isActive =
    currentState === "running" || currentState === "halted" || currentState === "starting";

  // Extract session metadata from status (untyped fields)
  const statusRecord = status as Record<string, unknown> | null;
  const traderId = String(statusRecord?.trader_id ?? statusRecord?.session_id ?? "");
  const strategyName = String(statusRecord?.strategy_name ?? statusRecord?.strategy ?? "");
  const startedAt = statusRecord?.started_at ? String(statusRecord.started_at) : null;

  return (
    <div className="mx-auto max-w-5xl space-y-8 p-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Paper Trading</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Run strategies against live market data with simulated execution.
        </p>
      </div>

      {isActive && traderId && (
        <TraderInfo
          traderId={traderId}
          state={status?.state ?? "unknown"}
          strategyName={strategyName}
          startedAt={startedAt}
        />
      )}

      <SessionControl />

      {isActive && <LiveDashboard />}

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-border/60 bg-card/40 p-5 backdrop-blur-sm">
          <PositionsTable />
        </div>
        <div className="rounded-xl border border-border/60 bg-card/40 p-5 backdrop-blur-sm">
          <CheckpointsList />
        </div>
      </div>
    </div>
  );
}
