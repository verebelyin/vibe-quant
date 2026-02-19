import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import type { StrategyResponse } from "@/api/generated/models";
import { StrategyCreateDialog } from "@/components/strategies/StrategyCreateDialog";
import { StrategyDeleteDialog } from "@/components/strategies/StrategyDeleteDialog";
import { StrategyList } from "@/components/strategies/StrategyList";
import { Button } from "@/components/ui/button";

export function StrategiesPage() {
  const navigate = useNavigate();
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<StrategyResponse | null>(null);

  const handleSelect = (strategy: StrategyResponse) => {
    navigate({ to: "/strategies/$strategyId", params: { strategyId: String(strategy.id) } });
  };

  const handleCreated = (strategyId: number) => {
    setCreateOpen(false);
    navigate({ to: "/strategies/$strategyId", params: { strategyId: String(strategyId) } });
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
