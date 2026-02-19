import { useQueryClient } from "@tanstack/react-query";
import {
  getListStrategiesApiStrategiesGetQueryKey,
  useDeleteStrategyApiStrategiesStrategyIdDelete,
} from "@/api/generated/strategies/strategies";

interface StrategyDeleteDialogProps {
  open: boolean;
  strategyId: number;
  strategyName: string;
  onClose: () => void;
}

export function StrategyDeleteDialog({
  open,
  strategyId,
  strategyName,
  onClose,
}: StrategyDeleteDialogProps) {
  const queryClient = useQueryClient();
  const deleteMutation = useDeleteStrategyApiStrategiesStrategyIdDelete();

  if (!open) return null;

  const handleDelete = () => {
    deleteMutation.mutate(
      { strategyId },
      {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getListStrategiesApiStrategiesGetQueryKey(),
          });
          onClose();
        },
      },
    );
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
        className="mx-4 w-full max-w-sm rounded-xl border p-6 shadow-lg"
        style={{
          backgroundColor: "hsl(var(--card))",
          borderColor: "hsl(var(--border))",
          color: "hsl(var(--card-foreground))",
        }}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-bold">Delete Strategy</h2>
        <p className="mt-2 text-sm" style={{ color: "hsl(var(--muted-foreground))" }}>
          Are you sure you want to delete{" "}
          <span className="font-semibold" style={{ color: "hsl(var(--foreground))" }}>
            {strategyName}
          </span>
          ? This action cannot be undone.
        </p>

        {deleteMutation.isError && (
          <p className="mt-3 text-sm" style={{ color: "hsl(0 84% 60%)" }}>
            Failed to delete strategy. Please try again.
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
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
            className="cursor-pointer rounded-lg px-4 py-2 text-sm font-medium transition-colors hover:brightness-95 disabled:opacity-50"
            style={{
              backgroundColor: "hsl(0 84% 60%)",
              color: "white",
            }}
          >
            {deleteMutation.isPending ? "Deleting..." : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}
