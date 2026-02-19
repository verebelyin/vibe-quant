import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import type { StrategyResponse } from "@/api/generated/models";
import { StrategyCreateDialog } from "@/components/strategies/StrategyCreateDialog";
import { StrategyDeleteDialog } from "@/components/strategies/StrategyDeleteDialog";
import { StrategyList } from "@/components/strategies/StrategyList";
import { StrategyWizard } from "@/components/strategies/StrategyWizard";
import { Button } from "@/components/ui/button";

export function StrategiesPage() {
  const navigate = useNavigate();
  const [createOpen, setCreateOpen] = useState(false);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<StrategyResponse | null>(null);

  const handleSelect = (strategy: StrategyResponse) => {
    navigate({ to: "/strategies/$strategyId", params: { strategyId: String(strategy.id) } });
  };

  const handleCreated = (strategyId: number) => {
    setCreateOpen(false);
    navigate({ to: "/strategies/$strategyId", params: { strategyId: String(strategyId) } });
  };

  if (wizardOpen) {
    return <StrategyWizard onCancel={() => setWizardOpen(false)} />;
  }

  return (
    <div className="p-6">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold">Strategy Management</h1>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => setWizardOpen(true)}>
            Wizard
          </Button>
          <Button onClick={() => setCreateOpen(true)}>+ Create New</Button>
        </div>
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
