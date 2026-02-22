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
      {/* Page header */}
      <div className="px-6 pt-6 pb-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2.5 mb-1">
              <h1 className="text-lg font-bold tracking-tight text-foreground">
                Strategy Management
              </h1>
            </div>
            <p className="text-[11px] text-muted-foreground/50 tracking-wide">
              Build, configure, and manage your algorithmic trading strategies
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0 mt-0.5">
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
        </div>
      </div>

      {/* Hairline separator */}
      <div className="mx-6 h-px bg-gradient-to-r from-transparent via-white/[0.07] to-transparent shrink-0" />

      {/* Scrollable content */}
      <div className="flex-1 overflow-auto px-6 py-5">
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
