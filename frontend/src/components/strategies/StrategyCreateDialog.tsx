import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import {
  getListStrategiesApiStrategiesGetQueryKey,
  useCreateStrategyApiStrategiesPost,
  useListTemplatesApiStrategiesTemplatesGet,
} from "@/api/generated/strategies/strategies";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
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

  const uniqueName = (base: string) =>
    `${base}_${Date.now().toString(36)}`;

  const handleCreate = () => {
    if (selected === "blank") {
      createMutation.mutate(
        {
          data: {
            name: uniqueName("untitled_strategy"),
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
          onError: () => toast.error("Failed to create strategy"),
        },
      );
    } else {
      const tmpl = templates[selected];
      createMutation.mutate(
        {
          data: {
            name: uniqueName(tmpl.name ?? "strategy"),
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
          onError: () => toast.error("Failed to create strategy"),
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
          <Card
            className={cn(
              "cursor-pointer gap-0 py-0 transition-colors hover:bg-muted/50",
              selected === "blank" ? "border-primary bg-primary/10" : "",
            )}
            onClick={() => setSelected("blank")}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                setSelected("blank");
              }
            }}
          >
            <CardContent className="p-3">
              <p className="text-sm font-medium">Blank Strategy</p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Start with an empty DSL configuration.
              </p>
            </CardContent>
          </Card>

          {/* Templates */}
          {templatesQuery.isLoading && <Skeleton className="h-16 rounded-lg" />}
          {templates.map((tmpl, idx) => (
            <Card
              key={tmpl.name ?? idx}
              className={cn(
                "cursor-pointer gap-0 py-0 transition-colors hover:bg-muted/50",
                selected === idx ? "border-primary bg-primary/10" : "",
              )}
              onClick={() => setSelected(idx)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setSelected(idx);
                }
              }}
            >
              <CardContent className="p-3">
                <p className="text-sm font-medium">{tmpl.name ?? `Template ${idx + 1}`}</p>
                {tmpl.description && (
                  <p className="mt-0.5 text-xs text-muted-foreground">{tmpl.description}</p>
                )}
              </CardContent>
            </Card>
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
