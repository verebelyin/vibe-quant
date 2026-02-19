import { useMemo, useState } from "react";
import type { BacktestRunResponse } from "@/api/generated/models";
import { useListRunsApiResultsRunsGet } from "@/api/generated/results/results";
import { LoadingSpinner } from "@/components/ui";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

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

function statusVariant(status: string): "default" | "destructive" | "secondary" | "outline" {
  switch (status) {
    case "completed":
      return "default";
    case "failed":
      return "destructive";
    case "running":
      return "secondary";
    default:
      return "outline";
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
    <Card className="flex-row items-center gap-4 px-4 py-3">
      <span className="text-sm font-medium text-foreground">Run</span>

      <Select value={statusFilter} onValueChange={(val) => setStatusFilter(val as StatusFilter)}>
        <SelectTrigger size="sm" className="w-[140px]">
          <SelectValue placeholder="All statuses" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All statuses</SelectItem>
          <SelectItem value="completed">Completed</SelectItem>
          <SelectItem value="failed">Failed</SelectItem>
          <SelectItem value="running">Running</SelectItem>
          <SelectItem value="pending">Pending</SelectItem>
        </SelectContent>
      </Select>

      {query.isLoading ? (
        <LoadingSpinner size="sm" />
      ) : (
        <Select
          value={selectedRunId != null ? String(selectedRunId) : ""}
          onValueChange={(val) => {
            const num = Number(val);
            if (num) onSelectRun(num);
          }}
        >
          <SelectTrigger className="min-w-[320px]">
            <SelectValue placeholder="Select a run..." />
          </SelectTrigger>
          <SelectContent>
            {sortedRuns.map((run: BacktestRunResponse) => (
              <SelectItem key={run.id} value={String(run.id)}>
                {shortId(run.id)} | {run.run_mode} | {run.symbols.join(", ")} | {run.status} |{" "}
                {formatDate(run.created_at)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}

      {selectedRunId != null &&
        (() => {
          const run = runs.find((r: BacktestRunResponse) => r.id === selectedRunId);
          if (!run) return null;
          return <Badge variant={statusVariant(run.status)}>{run.status}</Badge>;
        })()}
    </Card>
  );
}
