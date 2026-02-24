import { useNavigate } from "@tanstack/react-router";
import { Wand2, Plus } from "lucide-react";
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
    <div className="flex flex-col h-full">
      {/* Action buttons */}
      <div className="flex items-center justify-start gap-2 pb-4">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setWizardOpen(true)}
          className="gap-1.5 h-8 text-xs border-white/[0.09] bg-white/[0.03] hover:bg-white/[0.06] text-muted-foreground hover:text-foreground"
        >
          <Wand2 className="h-3 w-3" />
          Wizard
        </Button>
        <Button
          size="sm"
          onClick={() => setCreateOpen(true)}
          className="gap-1.5 h-8 text-xs"
        >
          <Plus className="h-3 w-3" />
          New Strategy
        </Button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-auto">
        <StrategyList onSelect={handleSelect} onDelete={setDeleteTarget} />
      </div>

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
