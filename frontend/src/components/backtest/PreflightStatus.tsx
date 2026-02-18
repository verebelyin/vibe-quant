import type { CoverageCheckResponseCoverage } from "@/api/generated/models";

interface SymbolCoverage {
  has_data: boolean;
  start_date?: string;
  end_date?: string;
  bars?: number;
  message?: string;
}

interface PreflightStatusProps {
  coverage: CoverageCheckResponseCoverage;
  requestedStart: string;
  requestedEnd: string;
}

export function PreflightStatus({ coverage, requestedStart, requestedEnd }: PreflightStatusProps) {
  const entries = Object.entries(coverage) as [string, SymbolCoverage][];

  if (entries.length === 0) {
    return (
      <p className="text-sm italic" style={{ color: "hsl(var(--muted-foreground))" }}>
        No coverage data returned.
      </p>
    );
  }

  const allGood = entries.every(([, v]) => v.has_data);

  return (
    <div
      className="rounded-lg border p-4"
      style={{
        borderColor: allGood ? "hsl(var(--accent))" : "hsl(var(--destructive, 0 84% 60%))",
        backgroundColor: "hsl(var(--card))",
      }}
    >
      <h3
        className="mb-3 text-sm font-semibold uppercase tracking-wider"
        style={{ color: "hsl(var(--foreground))" }}
      >
        Preflight Coverage Check
      </h3>
      <p className="mb-3 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
        Requested: {requestedStart} to {requestedEnd}
      </p>
      <div className="space-y-2">
        {entries.map(([symbol, info]) => (
          <div
            key={symbol}
            className="flex items-center justify-between rounded px-3 py-2 text-sm"
            style={{ backgroundColor: "hsl(var(--muted))" }}
          >
            <div className="flex items-center gap-2">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{
                  backgroundColor: info.has_data ? "hsl(142 76% 36%)" : "hsl(0 84% 60%)",
                }}
              />
              <span className="font-mono font-medium" style={{ color: "hsl(var(--foreground))" }}>
                {symbol}
              </span>
            </div>
            <span className="text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
              {info.has_data
                ? `${info.start_date ?? "?"} - ${info.end_date ?? "?"} (${info.bars ?? "?"} bars)`
                : (info.message ?? "No data available")}
            </span>
          </div>
        ))}
      </div>
      {allGood ? (
        <p className="mt-3 text-xs font-medium" style={{ color: "hsl(142 76% 36%)" }}>
          All symbols have sufficient coverage. Ready to launch.
        </p>
      ) : (
        <p className="mt-3 text-xs font-medium" style={{ color: "hsl(0 84% 60%)" }}>
          Some symbols lack coverage. Download data first via Data Management.
        </p>
      )}
    </div>
  );
}
