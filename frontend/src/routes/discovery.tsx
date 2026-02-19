import { DiscoveryConfig } from "@/components/discovery/DiscoveryConfig";
import { DiscoveryJobList } from "@/components/discovery/DiscoveryJobList";

export function DiscoveryPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-8 p-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Discovery</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Use genetic algorithms to discover new trading strategies.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[1fr_1fr]">
        <DiscoveryConfig />
        <div className="rounded-lg border border-border bg-background p-4">
          <DiscoveryJobList />
        </div>
      </div>
    </div>
  );
}
