import { useMemo, useState } from "react";
import type { BacktestRunResponse } from "@/api/generated/models";
import { useListRunsApiResultsRunsGet } from "@/api/generated/results/results";
import { LoadingSpinner } from "@/components/ui";

type StatusFilter = "all" | "completed" | "failed" | "running" | "pending";

interface RunSelectorProps {
  selectedRunId: number | null;
  onSelectRun: (runId: number) => void;
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function shortId(id: number): string {
  return `#${id}`;
}

function statusColor(status: string): string {
  switch (status) {
    case "completed":
      return "text-green-500";
    case "failed":
      return "text-red-500";
    case "running":
      return "text-yellow-500";
    default:
      return "text-gray-400";
  }
}

export function RunSelector({ selectedRunId, onSelectRun }: RunSelectorProps) {
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");

  const query = useListRunsApiResultsRunsGet(
    statusFilter === "all" ? undefined : { status: statusFilter },
  );
  const data = query.data?.data;
  const runs = data?.runs ?? [];

  const sortedRuns = useMemo(
    () =>
      [...runs].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()),
    [runs],
  );

  return (
    <div
      className="rounded-lg border p-4"
      style={{
        backgroundColor: "hsl(var(--card))",
        borderColor: "hsl(var(--border))",
      }}
    >
      <div className="flex items-center gap-4">
        <span className="text-sm font-medium" style={{ color: "hsl(var(--foreground))" }}>
          Run
        </span>

        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
          className="rounded border px-2 py-1.5 text-xs"
          style={{
            backgroundColor: "hsl(var(--background))",
            color: "hsl(var(--foreground))",
            borderColor: "hsl(var(--border))",
          }}
        >
          <option value="all">All statuses</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="running">Running</option>
          <option value="pending">Pending</option>
        </select>

        {query.isLoading ? (
          <LoadingSpinner size="sm" />
        ) : (
          <select
            value={selectedRunId ?? ""}
            onChange={(e) => {
              const val = Number(e.target.value);
              if (val) onSelectRun(val);
            }}
            className="min-w-[320px] rounded border px-2 py-1.5 text-sm"
            style={{
              backgroundColor: "hsl(var(--background))",
              color: "hsl(var(--foreground))",
              borderColor: "hsl(var(--border))",
            }}
          >
            <option value="">Select a run...</option>
            {sortedRuns.map((run: BacktestRunResponse) => (
              <option key={run.id} value={run.id}>
                {shortId(run.id)} | {run.run_mode} | {run.symbols.join(", ")} | {run.status} |{" "}
                {formatDate(run.created_at)}
              </option>
            ))}
          </select>
        )}

        {selectedRunId != null && (
          <span className="text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
            {(() => {
              const run = runs.find((r: BacktestRunResponse) => r.id === selectedRunId);
              if (!run) return null;
              return <span className={statusColor(run.status)}>{run.status}</span>;
            })()}
          </span>
        )}
      </div>
    </div>
  );
}
