import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import {
  getListStrategiesApiStrategiesGetQueryKey,
  useCreateStrategyApiStrategiesPost,
  useListTemplatesApiStrategiesTemplatesGet,
} from "@/api/generated/strategies/strategies";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface StrategyCreateDialogProps {
  open: boolean;
  onClose: () => void;
  onCreated: (strategyId: number) => void;
}

interface TemplateItem {
  name?: string;
  description?: string;
  strategy_type?: string;
  dsl_config?: Record<string, unknown>;
}

export function StrategyCreateDialog({ open, onClose, onCreated }: StrategyCreateDialogProps) {
  const [selected, setSelected] = useState<number | "blank">("blank");
  const queryClient = useQueryClient();
  const templatesQuery = useListTemplatesApiStrategiesTemplatesGet();
  const templates = (templatesQuery.data?.data ?? []) as TemplateItem[];
  const createMutation = useCreateStrategyApiStrategiesPost();

  const handleCreate = () => {
    if (selected === "blank") {
      createMutation.mutate(
        {
          data: {
            name: "Untitled Strategy",
            dsl_config: {},
          },
        },
        {
          onSuccess: (res) => {
            queryClient.invalidateQueries({
              queryKey: getListStrategiesApiStrategiesGetQueryKey(),
            });
            toast.success("Strategy created");
            onCreated(res.data.id);
          },
        },
      );
    } else {
      const tmpl = templates[selected];
      createMutation.mutate(
        {
          data: {
            name: tmpl.name ?? "Untitled Strategy",
            description: tmpl.description,
            strategy_type: tmpl.strategy_type,
            dsl_config: tmpl.dsl_config ?? {},
          },
        },
        {
          onSuccess: (res) => {
            queryClient.invalidateQueries({
              queryKey: getListStrategiesApiStrategiesGetQueryKey(),
            });
            toast.success("Strategy created");
            onCreated(res.data.id);
          },
        },
      );
    }
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Create Strategy</DialogTitle>
          <DialogDescription>Choose a template or start from scratch.</DialogDescription>
        </DialogHeader>

        <div className="max-h-64 space-y-2 overflow-y-auto">
          {/* Blank option */}
          <button
            type="button"
            onClick={() => setSelected("blank")}
            className={cn(
              "w-full cursor-pointer rounded-lg border p-3 text-left transition-colors",
              selected === "blank" ? "border-primary bg-primary/10" : "border-border bg-card",
            )}
          >
            <p className="text-sm font-medium">Blank Strategy</p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Start with an empty DSL configuration.
            </p>
          </button>

          {/* Templates */}
          {templatesQuery.isLoading && <div className="h-16 animate-pulse rounded-lg bg-muted" />}
          {templates.map((tmpl, idx) => (
            <button
              key={tmpl.name ?? idx}
              type="button"
              onClick={() => setSelected(idx)}
              className={cn(
                "w-full cursor-pointer rounded-lg border p-3 text-left transition-colors",
                selected === idx ? "border-primary bg-primary/10" : "border-border bg-card",
              )}
            >
              <p className="text-sm font-medium">{tmpl.name ?? `Template ${idx + 1}`}</p>
              {tmpl.description && (
                <p className="mt-0.5 text-xs text-muted-foreground">{tmpl.description}</p>
              )}
            </button>
          ))}
        </div>

        {createMutation.isError && (
          <p className="text-sm text-destructive">Failed to create strategy. Please try again.</p>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleCreate} disabled={createMutation.isPending}>
            {createMutation.isPending ? "Creating..." : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
