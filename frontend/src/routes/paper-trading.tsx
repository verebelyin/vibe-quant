import { useGetStatusApiPaperStatusGet } from "@/api/generated/paper/paper";
import { CheckpointsList } from "@/components/paper/CheckpointsList";
import { LiveDashboard } from "@/components/paper/LiveDashboard";
import { PositionsTable } from "@/components/paper/PositionsTable";
import { ReconciliationPanel } from "@/components/paper/ReconciliationPanel";
import { SessionControl } from "@/components/paper/SessionControl";
import { TraderInfo } from "@/components/paper/TraderInfo";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export function PaperTradingPage() {
  const statusQuery = useGetStatusApiPaperStatusGet({
    query: { refetchInterval: 5_000 },
  });

  const status = statusQuery.data?.status === 200 ? statusQuery.data.data : null;
  const currentState = status?.state?.toLowerCase() ?? "unknown";
  const isActive =
    currentState === "running" ||
    currentState === "paused" ||
    currentState === "halted" ||
    currentState === "starting";

  const traderId = status?.trader_id ?? "";
  const strategyName = status?.run_id != null ? `paper run ${status.run_id}` : "";
  const startedAt: string | null = null;

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      {isActive && traderId && (
        <TraderInfo
          traderId={traderId}
          state={status?.state ?? "unknown"}
          strategyName={strategyName}
          startedAt={startedAt}
        />
      )}

      <SessionControl />

      {isActive && <LiveDashboard traderId={traderId || undefined} />}

      <Tabs defaultValue="session">
        <TabsList>
          <TabsTrigger value="session">Session</TabsTrigger>
          <TabsTrigger value="reconciliation">Reconciliation</TabsTrigger>
        </TabsList>

        <TabsContent value="session" className="mt-4">
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-border/60 bg-card/40 p-5 backdrop-blur-sm">
              <PositionsTable traderId={traderId || undefined} />
            </div>
            <div className="rounded-xl border border-border/60 bg-card/40 p-5 backdrop-blur-sm">
              <CheckpointsList traderId={traderId || undefined} sessionActive={isActive} />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="reconciliation" className="mt-4">
          <ReconciliationPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
