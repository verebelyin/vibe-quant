import { useState } from "react";

import {
  getGetStatusApiSystemStatusGetQueryKey,
  useGetStatusApiSystemStatusGet,
  useKillApiSystemKillPost,
  useUnlockApiSystemUnlockPost,
} from "@/api/generated/system/system";
import { queryClient } from "@/api/query-client";
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

// Status query polls every 5s so the UI catches kills engaged by other operators.
export function KillSwitch() {
  const statusQuery = useGetStatusApiSystemStatusGet({
    query: { refetchInterval: 5000, refetchOnWindowFocus: true },
  });
  const status = statusQuery.data?.status === 200 ? statusQuery.data.data : undefined;
  const invalidateStatus = () =>
    queryClient.invalidateQueries({
      queryKey: getGetStatusApiSystemStatusGetQueryKey(),
    });
  const kill = useKillApiSystemKillPost({
    mutation: { onSuccess: invalidateStatus },
  });
  const unlock = useUnlockApiSystemUnlockPost({
    mutation: { onSuccess: invalidateStatus },
  });

  const [killOpen, setKillOpen] = useState(false);
  const [unlockOpen, setUnlockOpen] = useState(false);
  const [reason, setReason] = useState("");

  const killed = status?.kill_switch === true;

  if (killed) {
    return (
      <>
        <div className="flex items-center gap-2 rounded-md border border-red-500/60 bg-red-500/10 px-3 py-1.5">
          <span className="inline-flex size-2 rounded-full bg-red-500 shadow-[0_0_6px_1px] shadow-red-500/60" />
          <span className="text-xs font-semibold tracking-wide text-red-500">HALTED</span>
          {status?.reason && (
            <span
              className="max-w-[200px] truncate text-[11px] text-red-300/80"
              title={status.reason}
            >
              — {status.reason}
            </span>
          )}
        </div>
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
            {status?.reason && (
              <p className="rounded-md border bg-muted px-3 py-2 text-sm text-muted-foreground">
                <span className="font-medium">Halt reason:</span> {status.reason}
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
      </>
    );
  }

  return (
    <Dialog open={killOpen} onOpenChange={setKillOpen}>
      <DialogTrigger asChild>
        <Button variant="destructive" size="sm">
          KILL
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
  );
}
