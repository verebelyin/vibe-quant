import { useState } from "react";
import type { StrategyResponse } from "@/api/generated/models";
import { StrategyCreateDialog } from "@/components/strategies/StrategyCreateDialog";
import { StrategyDeleteDialog } from "@/components/strategies/StrategyDeleteDialog";
import { StrategyList } from "@/components/strategies/StrategyList";

export function StrategiesPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<StrategyResponse | null>(null);

  const handleSelect = (_strategy: StrategyResponse) => {
    // TODO: navigate to edit view (future task)
  };

  const handleCreated = (_strategyId: number) => {
    setCreateOpen(false);
    // TODO: navigate to edit view (future task)
  };

  return (
    <div className="p-6">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold">Strategy Management</h1>
        <button
          type="button"
          onClick={() => setCreateOpen(true)}
          className="cursor-pointer rounded-lg px-4 py-2 text-sm font-medium transition-colors hover:brightness-95"
          style={{
            backgroundColor: "hsl(var(--primary))",
            color: "hsl(var(--primary-foreground))",
          }}
        >
          + Create New
        </button>
      </div>

      <StrategyList onSelect={handleSelect} onDelete={setDeleteTarget} />

      <StrategyCreateDialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={handleCreated}
      />

      {deleteTarget && (
        <StrategyDeleteDialog
          open={!!deleteTarget}
          strategyId={deleteTarget.id}
          strategyName={deleteTarget.name}
          onClose={() => setDeleteTarget(null)}
        />
      )}
    </div>
  );
}
