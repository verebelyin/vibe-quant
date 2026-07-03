import { useRouterState } from "@tanstack/react-router";

import { useGetStatusApiSystemStatusGet } from "@/api/generated/system/system";
import { cn } from "@/lib/utils";
import { ExtractionQueueBadge } from "./ExtractionQueueBadge";
import { KillSwitch } from "./KillSwitch";

const routeMeta: Record<string, { title: string; description?: string }> = {
  "/strategies": {
    title: "Strategy Management",
    description: "Create, edit, and organize trading strategies",
  },
  "/discovery": {
    title: "Discovery",
    description: "Evolve new strategies with the genetic algorithm",
  },
  "/discovery/results": {
    title: "Discovery Results",
    description: "Review strategies found by past discovery runs",
  },
  "/backtest": {
    title: "Backtest Launch",
    description: "Run screening or full-fidelity validation backtests",
  },
  "/results": {
    title: "Results Analysis",
    description: "Compare metrics, charts, and trades across runs",
  },
  "/paper-trading": {
    title: "Paper Trading",
    description: "Trade live markets with simulated money",
  },
  "/browser": {
    title: "Data Browser",
    description: "Inspect candles and indicators in the catalog",
  },
  "/data": {
    title: "Data Management",
    description: "Download and archive exchange market data",
  },
  "/settings": { title: "Settings" },
  "/guide": { title: "Guide" },
  "/research": {
    title: "Research",
    description: "Collect and extract strategy ideas from external sources",
  },
  "/research/queue": { title: "Extraction Queue" },
};

function ConnectionStatus() {
  // Poll backend health; this is the single source of truth for the Live badge.
  const statusQuery = useGetStatusApiSystemStatusGet({
    query: { refetchInterval: 10_000, refetchOnWindowFocus: true },
  });

  const state = statusQuery.isLoading
    ? ("connecting" as const)
    : statusQuery.isError || statusQuery.data?.status !== 200
      ? ("offline" as const)
      : ("live" as const);

  const dot = {
    live: "bg-emerald-400 shadow-emerald-400/40",
    offline: "bg-red-500 shadow-red-500/40",
    connecting: "bg-amber-400 shadow-amber-400/40",
  }[state];

  const label = { live: "Live", offline: "Backend offline", connecting: "Connecting…" }[state];

  return (
    <div
      className="flex items-center gap-2"
      title={
        state === "offline" ? "The API is not responding — check the backend process" : undefined
      }
    >
      <div className="relative flex size-2">
        {state === "live" && (
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-400/60 opacity-40" />
        )}
        <span
          className={cn("relative inline-flex size-2 rounded-full shadow-[0_0_6px_1px]", dot)}
        />
      </div>
      <span
        className={cn(
          "text-[11px] font-medium tracking-wide",
          state === "offline" ? "text-red-400" : "text-muted-foreground/70",
        )}
      >
        {label}
      </span>
    </div>
  );
}

export function Header() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const meta = routeMeta[pathname] ?? { title: "vibe-quant" };

  return (
    <header className="relative flex h-14 items-center justify-between border-b px-6">
      <div className="flex min-w-0 items-baseline gap-3">
        <h1 className="shrink-0 text-lg font-semibold tracking-tight">{meta.title}</h1>
        {meta.description && (
          <p className="hidden truncate text-xs text-muted-foreground/60 md:block">
            {meta.description}
          </p>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-4">
        <ConnectionStatus />
        <ExtractionQueueBadge />
        <KillSwitch />
      </div>
    </header>
  );
}
