import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getListStrategiesApiStrategiesGetQueryKey,
  useCreateStrategyApiStrategiesPost,
  useListTemplatesApiStrategiesTemplatesGet,
} from "@/api/generated/strategies/strategies";

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

  if (!open) return null;

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
            onCreated(res.data.id);
          },
        },
      );
    }
  };

  return (
    // biome-ignore lint/a11y/useSemanticElements: backdrop overlay
    <div
      role="button"
      tabIndex={-1}
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "hsl(0 0% 0% / 0.5)" }}
      onClick={onClose}
      onKeyDown={(e) => e.key === "Escape" && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        tabIndex={-1}
        className="mx-4 w-full max-w-lg rounded-xl border p-6 shadow-lg"
        style={{
          backgroundColor: "hsl(var(--card))",
          borderColor: "hsl(var(--border))",
          color: "hsl(var(--card-foreground))",
        }}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-bold">Create Strategy</h2>
        <p className="mt-1 text-sm" style={{ color: "hsl(var(--muted-foreground))" }}>
          Choose a template or start from scratch.
        </p>

        <div className="mt-4 max-h-64 space-y-2 overflow-y-auto">
          {/* Blank option */}
          <button
            type="button"
            onClick={() => setSelected("blank")}
            className="w-full cursor-pointer rounded-lg border p-3 text-left transition-colors"
            style={{
              borderColor: selected === "blank" ? "hsl(var(--primary))" : "hsl(var(--border))",
              backgroundColor:
                selected === "blank" ? "hsl(var(--primary) / 0.1)" : "hsl(var(--card))",
            }}
          >
            <p className="text-sm font-medium">Blank Strategy</p>
            <p className="mt-0.5 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
              Start with an empty DSL configuration.
            </p>
          </button>

          {/* Templates */}
          {templatesQuery.isLoading && (
            <div
              className="h-16 animate-pulse rounded-lg"
              style={{ backgroundColor: "hsl(var(--muted))" }}
            />
          )}
          {templates.map((tmpl, idx) => (
            <button
              key={tmpl.name ?? idx}
              type="button"
              onClick={() => setSelected(idx)}
              className="w-full cursor-pointer rounded-lg border p-3 text-left transition-colors"
              style={{
                borderColor: selected === idx ? "hsl(var(--primary))" : "hsl(var(--border))",
                backgroundColor:
                  selected === idx ? "hsl(var(--primary) / 0.1)" : "hsl(var(--card))",
              }}
            >
              <p className="text-sm font-medium">{tmpl.name ?? `Template ${idx + 1}`}</p>
              {tmpl.description && (
                <p className="mt-0.5 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
                  {tmpl.description}
                </p>
              )}
            </button>
          ))}
        </div>

        {createMutation.isError && (
          <p className="mt-3 text-sm" style={{ color: "hsl(0 84% 60%)" }}>
            Failed to create strategy. Please try again.
          </p>
        )}

        <div className="mt-5 flex justify-end gap-3">
          <button
            type="button"
            onClick={onClose}
            className="cursor-pointer rounded-lg border px-4 py-2 text-sm font-medium transition-colors hover:brightness-95"
            style={{
              borderColor: "hsl(var(--border))",
              color: "hsl(var(--foreground))",
            }}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleCreate}
            disabled={createMutation.isPending}
            className="cursor-pointer rounded-lg px-4 py-2 text-sm font-medium transition-colors hover:brightness-95 disabled:opacity-50"
            style={{
              backgroundColor: "hsl(var(--primary))",
              color: "hsl(var(--primary-foreground))",
            }}
          >
            {createMutation.isPending ? "Creating..." : "Create"}
          </button>
        </div>
      </div>
    </div>
  );
}
