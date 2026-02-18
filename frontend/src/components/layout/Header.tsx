import { useRouterState } from "@tanstack/react-router";
import { useUIStore } from "@/stores/ui";

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
  const theme = useUIStore((s) => s.theme);
  const toggleTheme = useUIStore((s) => s.toggleTheme);

  const title = routeTitles[pathname] ?? "vibe-quant";

  return (
    <header
      className="flex h-14 items-center justify-between border-b px-6"
      style={{
        borderColor: "hsl(var(--border))",
        backgroundColor: "hsl(var(--background))",
      }}
    >
      <h1 className="text-lg font-semibold">{title}</h1>

      <div className="flex items-center gap-4">
        {/* Connection status indicator */}
        <div className="flex items-center gap-2">
          <div
            className="size-2.5 rounded-full"
            style={{ backgroundColor: "#22c55e" }}
            title="Connected"
          />
          <span className="text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
            Connected
          </span>
        </div>

        {/* Theme toggle */}
        <button
          type="button"
          onClick={toggleTheme}
          className="rounded-md p-2 transition-colors hover:opacity-80"
          style={{ backgroundColor: "hsl(var(--accent))" }}
          title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
        >
          {theme === "light" ? (
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="size-4"
            >
              <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
            </svg>
          ) : (
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="size-4"
            >
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2" />
              <path d="M12 20v2" />
              <path d="m4.93 4.93 1.41 1.41" />
              <path d="m17.66 17.66 1.41 1.41" />
              <path d="M2 12h2" />
              <path d="M20 12h2" />
              <path d="m6.34 17.66-1.41 1.41" />
              <path d="m19.07 4.93-1.41 1.41" />
            </svg>
          )}
        </button>
      </div>
    </header>
  );
}
