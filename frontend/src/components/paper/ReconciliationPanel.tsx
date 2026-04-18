import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { LoadingSpinner, MetricCard } from "@/components/ui";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useReconcilePaperSessionApiReconciliationPaperSessionIdGet } from "@/api/generated/reconciliation/reconciliation";
import type { ReconciliationResponse } from "@/api/generated/models";

export function ReconciliationPanel() {
  const [runIdInput, setRunIdInput] = useState<string>("");
  const [submittedRunId, setSubmittedRunId] = useState<number | null>(null);
  const [validationRunId, setValidationRunId] = useState<string>("");

  const query = useReconcilePaperSessionApiReconciliationPaperSessionIdGet(
    submittedRunId ?? 0,
    {
      ...(validationRunId.trim() !== "" && {
        validation_run_id: Number(validationRunId),
      }),
    },
    {
      query: { enabled: submittedRunId != null },
    },
  );

  const submit = () => {
    const n = Number(runIdInput);
    if (!Number.isFinite(n)) return;
    setSubmittedRunId(n);
  };

  const data = query.data?.status === 200 ? (query.data.data as ReconciliationResponse) : null;
  let err: string | null = null;
  if (query.data && query.data.status !== 200) {
    const detail = (query.data as unknown as { data?: { detail?: unknown } }).data?.detail;
    err = typeof detail === "string" ? detail : "Reconciliation failed";
  } else if (query.error instanceof Error) {
    err = query.error.message;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Paper ↔ Validation Reconciliation
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-3 items-end">
          <div>
            <Label className="text-xs">Paper session run ID</Label>
            <Input
              type="number"
              value={runIdInput}
              onChange={(e) => setRunIdInput(e.target.value)}
              className="h-8 text-xs"
            />
          </div>
          <div>
            <Label className="text-xs">Validation run ID (optional)</Label>
            <Input
              type="number"
              value={validationRunId}
              onChange={(e) => setValidationRunId(e.target.value)}
              placeholder="auto-pick latest"
              className="h-8 text-xs"
            />
          </div>
          <Button onClick={submit} disabled={runIdInput === ""} size="sm">
            Reconcile
          </Button>
        </div>

        {submittedRunId == null && (
          <p className="text-xs text-muted-foreground">
            Enter a stopped paper session's run ID to reconcile against a validation run.
          </p>
        )}

        {submittedRunId != null && query.isLoading && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <LoadingSpinner size="sm" /> Reconciling…
          </div>
        )}

        {err && (
          <div className="rounded-md border border-destructive bg-destructive/10 p-3 text-xs text-destructive">
            {err}
          </div>
        )}

        {data && (
          <>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <MetricCard
                label="Parity rate"
                value={`${(data.divergence_summary.parity_rate * 100).toFixed(1)}%`}
              />
              <MetricCard label="Matched" value={data.divergence_summary.matched} />
              <MetricCard label="Paper only" value={data.divergence_summary.paper_only} />
              <MetricCard
                label="Validation only"
                value={data.divergence_summary.validation_only}
              />
              <MetricCard
                label="Mean PnL Δ"
                value={data.divergence_summary.mean_pnl_delta.toFixed(4)}
                trend={data.divergence_summary.mean_pnl_delta >= 0 ? "up" : "down"}
              />
              <MetricCard
                label="Mean entry slippage"
                value={data.divergence_summary.mean_entry_slippage.toFixed(6)}
              />
              <MetricCard
                label="Side disagreements"
                value={data.divergence_summary.side_disagreements}
                trend={data.divergence_summary.side_disagreements === 0 ? "up" : "down"}
              />
            </div>

            {data.paired_trades.length > 0 && (
              <div>
                <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">
                  Paired Trades
                </p>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Side</TableHead>
                      <TableHead className="text-right">Paper PnL</TableHead>
                      <TableHead className="text-right">Valid PnL</TableHead>
                      <TableHead className="text-right">PnL Δ</TableHead>
                      <TableHead className="text-right">Slippage</TableHead>
                      <TableHead>Agree</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.paired_trades.map((p, i) => (
                      <TableRow key={`${p.paper.position_id}-${i}`}>
                        <TableCell className="font-mono text-xs">{p.paper.symbol}</TableCell>
                        <TableCell className="text-xs">{p.paper.side}</TableCell>
                        <TableCell className="text-right font-mono text-xs">
                          {p.paper.net_pnl.toFixed(4)}
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs">
                          {p.validation.net_pnl.toFixed(4)}
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs">
                          {p.pnl_delta.toFixed(4)}
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs">
                          {p.entry_slippage.toFixed(6)}
                        </TableCell>
                        <TableCell>
                          {p.side_agrees ? (
                            <Badge>Yes</Badge>
                          ) : (
                            <Badge variant="destructive">No</Badge>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

