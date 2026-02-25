import { useState } from "react";
import type { BacktestRunResponse } from "@/api/generated/models";
import { useListRunsApiResultsRunsGet } from "@/api/generated/results/results";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

interface RunDetailsExpanderProps {
  runId: number;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return "N/A";
  return new Date(dateStr).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function DetailRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-xs font-medium uppercase text-muted-foreground min-w-[100px]">
        {label}
      </span>
      <span className="text-sm text-foreground">{children}</span>
    </div>
  );
}

export function RunDetailsExpander({ runId }: RunDetailsExpanderProps) {
  const [open, setOpen] = useState(false);
  const query = useListRunsApiResultsRunsGet();
  const runs = (query.data?.data as { runs?: BacktestRunResponse[] } | undefined)?.runs;
  const run = runs?.find((r: BacktestRunResponse) => r.id === runId);

  if (!run) return null;

  return (
    <Card className="overflow-hidden">
      <Button
        variant="ghost"
        className="w-full justify-between rounded-none px-4 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground"
        onClick={() => setOpen(!open)}
      >
        Run Details
        <span>{open ? "Collapse" : "Expand"}</span>
      </Button>
      {open && (
        <CardContent className="space-y-2 pt-2">
          <DetailRow label="Strategy">ID #{run.strategy_id}</DetailRow>
          <DetailRow label="Mode">
            <Badge variant="secondary" className="text-[10px]">
              {run.run_mode}
            </Badge>
          </DetailRow>
          <DetailRow label="Symbols">{run.symbols.join(", ")}</DetailRow>
          <DetailRow label="Timeframe">{run.timeframe}</DetailRow>
          <DetailRow label="Date Range">
            {run.start_date} to {run.end_date}
          </DetailRow>
          <DetailRow label="Created">{formatDate(run.created_at)}</DetailRow>
          {run.started_at && <DetailRow label="Started">{formatDate(run.started_at)}</DetailRow>}
          {run.completed_at && (
            <DetailRow label="Completed">{formatDate(run.completed_at)}</DetailRow>
          )}
          {run.error_message && (
            <DetailRow label="Error">
              <span className="text-destructive">{run.error_message}</span>
            </DetailRow>
          )}
        </CardContent>
      )}
    </Card>
  );
}
