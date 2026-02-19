import type { CoverageCheckResponseCoverage } from "@/api/generated/models";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

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
    return <p className="text-sm italic text-muted-foreground">No coverage data returned.</p>;
  }

  const allGood = entries.every(([, v]) => v.has_data);

  return (
    <div
      className={cn(
        "rounded-lg border bg-card p-4",
        allGood ? "border-accent" : "border-destructive",
      )}
    >
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-foreground">
        Preflight Coverage Check
      </h3>
      <p className="mb-3 text-xs text-muted-foreground">
        Requested: {requestedStart} to {requestedEnd}
      </p>
      <div className="space-y-2">
        {entries.map(([symbol, info]) => (
          <div
            key={symbol}
            className="flex items-center justify-between rounded bg-muted px-3 py-2 text-sm"
          >
            <div className="flex items-center gap-2">
              <Badge
                variant="outline"
                className={cn(
                  "h-2.5 w-2.5 rounded-full border-transparent p-0",
                  info.has_data ? "bg-green-600" : "bg-red-500",
                )}
              />
              <span className="font-mono font-medium text-foreground">{symbol}</span>
            </div>
            <span className="text-xs text-muted-foreground">
              {info.has_data
                ? `${info.start_date ?? "?"} - ${info.end_date ?? "?"} (${info.bars ?? "?"} bars)`
                : (info.message ?? "No data available")}
            </span>
          </div>
        ))}
      </div>
      {allGood ? (
        <p className="mt-3 text-xs font-medium text-green-600">
          All symbols have sufficient coverage. Ready to launch.
        </p>
      ) : (
        <p className="mt-3 text-xs font-medium text-red-500">
          Some symbols lack coverage. Download data first via Data Management.
        </p>
      )}
    </div>
  );
}
