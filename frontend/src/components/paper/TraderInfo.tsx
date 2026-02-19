import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface TraderInfoProps {
  traderId: string;
  state: string;
  strategyName: string;
  startedAt: string | null;
}

const STATE_BADGE: Record<string, string> = {
  running: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
  halted: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300",
  stopped: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  error: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  starting: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 animate-pulse",
};
const FALLBACK = "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300";

function formatUptime(startIso: string): string {
  const startMs = new Date(startIso).getTime();
  const now = Date.now();
  const diffSec = Math.floor((now - startMs) / 1000);
  if (diffSec < 0) return "0s";

  const hours = Math.floor(diffSec / 3600);
  const minutes = Math.floor((diffSec % 3600) / 60);
  const seconds = diffSec % 60;

  if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

export function TraderInfo({ traderId, state, strategyName, startedAt }: TraderInfoProps) {
  const [uptime, setUptime] = useState(() => (startedAt ? formatUptime(startedAt) : "--"));

  useEffect(() => {
    if (!startedAt) return;
    setUptime(formatUptime(startedAt));
    const interval = setInterval(() => {
      setUptime(formatUptime(startedAt));
    }, 1000);
    return () => clearInterval(interval);
  }, [startedAt]);

  const normalized = state.toLowerCase();

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex flex-wrap items-center gap-4">
        <div className="min-w-0 flex-1">
          <p className="text-xs text-muted-foreground">Trader ID</p>
          <p className="truncate font-mono text-sm font-medium text-foreground">{traderId}</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Strategy</p>
          <p className="text-sm font-medium text-foreground">{strategyName || "--"}</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Uptime</p>
          <p className="font-mono text-sm text-foreground">{uptime}</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">State</p>
          <Badge
            variant="outline"
            className={cn("mt-0.5 border-transparent", STATE_BADGE[normalized] ?? FALLBACK)}
          >
            {state}
          </Badge>
        </div>
      </div>
    </div>
  );
}
