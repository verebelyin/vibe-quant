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
    <header className="flex h-14 items-center justify-between border-b px-6">
      <h1 className="text-lg font-semibold tracking-tight">{title}</h1>

      <div className="flex items-center gap-4">
        {/* Connection status */}
        <div className="flex items-center gap-2">
          <div className="relative flex size-2.5">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-green-400 opacity-50" />
            <span className="relative inline-flex size-2.5 rounded-full bg-green-500" />
          </div>
          <span className="text-xs text-muted-foreground">Connected</span>
        </div>
      </div>
    </header>
  );
}
