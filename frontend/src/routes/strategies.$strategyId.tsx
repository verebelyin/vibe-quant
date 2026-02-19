import { useGetStrategyApiStrategiesStrategyIdGet } from "@/api/generated/strategies/strategies";
import { StrategyEditor } from "@/components/strategies/StrategyEditor";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

export function StrategyEditPage({ strategyId }: { strategyId: number }) {
  const query = useGetStrategyApiStrategiesStrategyIdGet(strategyId);

  if (query.isLoading) {
    return (
      <div className="flex items-center justify-center p-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (query.isError || !query.data) {
    return (
      <div className="p-6">
        <div className="rounded-lg border border-destructive bg-destructive/10 p-6 text-destructive">
          <p className="font-medium">Failed to load strategy</p>
          <p className="mt-1 text-sm opacity-80">
            {query.error instanceof Error ? query.error.message : "Strategy not found"}
          </p>
        </div>
      </div>
    );
  }

  return <StrategyEditor strategy={query.data.data} />;
}
