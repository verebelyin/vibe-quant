import { useState } from "react";
import type { StrategyResponse } from "@/api/generated/models";
import { StrategyCreateDialog } from "@/components/strategies/StrategyCreateDialog";
import { StrategyDeleteDialog } from "@/components/strategies/StrategyDeleteDialog";
import { StrategyList } from "@/components/strategies/StrategyList";
import { Button } from "@/components/ui/button";

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
        <Button onClick={() => setCreateOpen(true)}>+ Create New</Button>
      </div>

      <StrategyList onSelect={handleSelect} onDelete={setDeleteTarget} />

      <StrategyCreateDialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={handleCreated}
      />

      <StrategyDeleteDialog
        open={!!deleteTarget}
        strategyId={deleteTarget?.id ?? 0}
        strategyName={deleteTarget?.name ?? ""}
        onClose={() => setDeleteTarget(null)}
      />
    </div>
  );
}
