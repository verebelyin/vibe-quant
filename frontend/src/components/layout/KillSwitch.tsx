import { useState } from "react";
import { toast } from "sonner";

import {
  getGetStatusApiSystemStatusGetQueryKey,
  useGetStatusApiSystemStatusGet,
  useKillApiSystemKillPost,
  useUnlockApiSystemUnlockPost,
} from "@/api/generated/system/system";
import { queryClient } from "@/api/query-client";
import { formatApiErrorDetail } from "@/components/discovery/apiErrorDetail";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { KILL_SWITCH_UNKNOWN_LABEL, killSwitchView } from "./killSwitchState";

// Status query polls every 5s so the UI catches kills engaged by other operators.
export function KillSwitch() {
  const statusQuery = useGetStatusApiSystemStatusGet({
    query: { refetchInterval: 5000, refetchOnWindowFocus: true },
  });
  const invalidateStatus = () =>
    queryClient.invalidateQueries({
      queryKey: getGetStatusApiSystemStatusGetQueryKey(),
    });
  // A failed kill/unlock (503 "NOT persisted", 409 "unlock again") says why and
  // refetches the status, so the header shows what the backend now holds.
  const onMutationError = (title: string) => (err: unknown) => {
    toast.error(title, {
      description: formatApiErrorDetail(err) || (err instanceof Error ? err.message : undefined),
    });
    invalidateStatus();
  };
  const kill = useKillApiSystemKillPost({
    mutation: { onSuccess: invalidateStatus, onError: onMutationError("Kill switch failed") },
  });
  const unlock = useUnlockApiSystemUnlockPost({
    mutation: { onSuccess: invalidateStatus, onError: onMutationError("Unlock failed") },
  });

  const [killOpen, setKillOpen] = useState(false);
  const [unlockOpen, setUnlockOpen] = useState(false);
  const [reason, setReason] = useState("");

  const view = killSwitchView(statusQuery.data, statusQuery.isError);

  // Controls come ONLY from the view: an unreadable status keeps the Kill
  // trigger (a broken status endpoint must never leave the operator without a halt).
  return (
    <>
      {view.showUnknownLabel && (
        <div className="flex items-center gap-2 rounded-md border border-red-500/60 bg-red-500/10 px-3 py-1.5">
          <span className="inline-flex size-2 rounded-full bg-red-500 shadow-[0_0_6px_1px] shadow-red-500/60" />
          <span className="text-xs font-semibold tracking-wide text-red-500">
            {KILL_SWITCH_UNKNOWN_LABEL}
          </span>
        </div>
      )}
      {view.state === "engaged" && (
        <div className="flex items-center gap-2 rounded-md border border-red-500/60 bg-red-500/10 px-3 py-1.5">
          <span className="inline-flex size-2 rounded-full bg-red-500 shadow-[0_0_6px_1px] shadow-red-500/60" />
          <span className="text-xs font-semibold tracking-wide text-red-500">HALTED</span>
          {view.reason && (
            <span
              className="max-w-[200px] truncate text-[11px] text-red-300/80"
              title={view.reason}
            >
              — {view.reason}
            </span>
          )}
        </div>
      )}
      {view.showUnlock && (
        <Dialog open={unlockOpen} onOpenChange={setUnlockOpen}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              Unlock
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Unlock the kill switch?</DialogTitle>
              <DialogDescription>
                This will re-enable starting paper and live trading sessions. Confirm you have
                investigated the halt cause.
              </DialogDescription>
            </DialogHeader>
            {view.reason && (
              <p className="rounded-md border bg-muted px-3 py-2 text-sm text-muted-foreground">
                <span className="font-medium">Halt reason:</span> {view.reason}
              </p>
            )}
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="ghost">Cancel</Button>
              </DialogClose>
              <Button
                variant="destructive"
                disabled={unlock.isPending}
                onClick={() =>
                  unlock.mutate(
                    { data: { acknowledge: true, cleared_by: "ui" } },
                    { onSuccess: () => setUnlockOpen(false) },
                  )
                }
              >
                {unlock.isPending ? "Unlocking..." : "Unlock"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
      {view.showKill && (
        <Dialog open={killOpen} onOpenChange={setKillOpen}>
          <DialogTrigger asChild>
            <Button variant="destructive" size="sm" title="Halt all trading sessions">
              Kill switch
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Engage portfolio kill switch?</DialogTitle>
              <DialogDescription>
                This halts any active paper trading session and prevents starting new sessions until
                explicitly unlocked.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-2">
              <Label htmlFor="kill-reason">Reason (required)</Label>
              <Input
                id="kill-reason"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="e.g. validation↔paper divergence"
                autoFocus
              />
            </div>
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="ghost">Cancel</Button>
              </DialogClose>
              <Button
                variant="destructive"
                disabled={!reason.trim() || kill.isPending}
                onClick={() =>
                  kill.mutate(
                    { data: { reason: reason.trim(), killed_by: "ui" } },
                    {
                      onSuccess: () => {
                        setKillOpen(false);
                        setReason("");
                      },
                    },
                  )
                }
              >
                {kill.isPending ? "Halting..." : "Halt all trading"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </>
  );
}
