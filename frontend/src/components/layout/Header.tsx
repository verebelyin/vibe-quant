import { useRouterState } from "@tanstack/react-router";

const routeTitles: Record<string, string> = {
  "/strategies": "Strategy Management",
  "/discovery": "Discovery",
  "/backtest": "Backtest Launch",
  "/results": "Results Analysis",
  "/paper-trading": "Paper Trading",
  "/data": "Data Management",
  "/settings": "Settings",
};

export function Header() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const title = routeTitles[pathname] ?? "vibe-quant";

  return (
    <header className="relative flex h-14 items-center justify-between border-b px-6">
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        <span className="hidden text-xs font-medium text-muted-foreground/50 sm:inline">
          /
        </span>
        <span className="hidden text-xs text-muted-foreground/60 sm:inline">
          {pathname.slice(1) || "overview"}
        </span>
      </div>

      <div className="flex items-center gap-4">
        {/* Connection status */}
        <div className="flex items-center gap-2">
          <div className="relative flex size-2">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-400/60 opacity-40" />
            <span className="relative inline-flex size-2 rounded-full bg-emerald-400 shadow-[0_0_6px_1px] shadow-emerald-400/40" />
          </div>
          <span className="text-[11px] font-medium tracking-wide text-muted-foreground/70">
            Live
          </span>
        </div>
      </div>
    </header>
  );
}
