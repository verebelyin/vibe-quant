import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  getListStrategiesApiStrategiesGetQueryKey,
  useDeleteStrategyApiStrategiesStrategyIdDelete,
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

  const handleDelete = () => {
    deleteMutation.mutate(
      { strategyId },
      {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getListStrategiesApiStrategiesGetQueryKey(),
          });
          toast.success(`Deleted "${strategyName}"`);
          onClose();
        },
      },
    );
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Delete Strategy</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete{" "}
            <span className="font-semibold text-foreground">{strategyName}</span>? This action
            cannot be undone.
          </DialogDescription>
        </DialogHeader>

        {deleteMutation.isError && (
          <p className="text-sm text-destructive">Failed to delete strategy. Please try again.</p>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={handleDelete} disabled={deleteMutation.isPending}>
            {deleteMutation.isPending ? "Deleting..." : "Delete"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
