import { CheckpointsList } from "@/components/paper/CheckpointsList";
import { PositionsTable } from "@/components/paper/PositionsTable";
import { SessionControl } from "@/components/paper/SessionControl";

export function PaperTradingPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-8 p-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Paper Trading</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Run strategies against live market data with simulated execution.
        </p>
      </div>

      <SessionControl />

      <div className="grid gap-8 lg:grid-cols-2">
        <div className="rounded-lg border border-border bg-background p-4">
          <PositionsTable />
        </div>
        <div className="rounded-lg border border-border bg-background p-4">
          <CheckpointsList />
        </div>
      </div>
    </div>
  );
}
