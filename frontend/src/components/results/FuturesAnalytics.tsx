import type { BacktestResultResponse } from "@/api/generated/models";
import { useGetRunSummaryApiResultsRunsRunIdGet } from "@/api/generated/results/results";
import { LoadingSpinner } from "@/components/ui";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatSignedUsd, grossPnlUsd, netPnlUsd } from "@/lib/metrics";
import { cn } from "@/lib/utils";

interface FuturesAnalyticsProps {
  runId: number;
}

function formatUsd(value: number): string {
  return `$${Math.abs(value).toFixed(2)}`;
}

interface CostSegment {
  label: string;
  value: number;
  color: string;
}

export function FuturesAnalytics({ runId }: FuturesAnalyticsProps) {
  const query = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const data = query.data?.data as BacktestResultResponse | undefined;

  if (query.isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <LoadingSpinner size="sm" />
      </div>
    );
  }

  if (query.isError || !data) return null;

  // Bar widths use magnitudes; PnL math keeps funding's sign (negative = received).
  const fees = Math.abs(data.total_fees ?? 0);
  const funding = Math.abs(data.total_funding ?? 0);
  const slippage = Math.abs(data.total_slippage ?? 0);
  const totalCosts = fees + funding + slippage;

  // total_return is a NET fraction (after fees, slippage, funding).
  const netPnl = netPnlUsd(data.total_return, data.starting_balance);
  const grossPnl = grossPnlUsd(netPnl, data.total_fees, data.total_slippage, data.total_funding);
  const fundingPctOfGross =
    grossPnl != null && grossPnl !== 0 ? (funding / Math.abs(grossPnl)) * 100 : null;

  const segments: CostSegment[] = [
    { label: "Fees", value: fees, color: "bg-red-500" },
    { label: "Funding", value: funding, color: "bg-yellow-500" },
    { label: "Slippage", value: slippage, color: "bg-orange-500" },
  ];

  const totalBar = totalCosts || 1;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Perpetual Futures Analytics
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Stacked cost bar */}
        <div>
          <p className="mb-1 text-xs text-muted-foreground">Cost Breakdown</p>
          <div className="flex h-5 w-full overflow-hidden rounded-full bg-muted">
            {segments.map((seg) => (
              <div
                key={seg.label}
                className={cn("h-full transition-all", seg.color)}
                style={{ width: `${(seg.value / totalBar) * 100}%` }}
                title={`${seg.label}: ${formatUsd(seg.value)}`}
              />
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-4 text-xs">
            {segments.map((seg) => (
              <div key={seg.label} className="flex items-center gap-1.5">
                <span className={cn("inline-block h-2.5 w-2.5 rounded-full", seg.color)} />
                <span className="text-muted-foreground">{seg.label}:</span>
                <span className="font-medium">{formatUsd(seg.value)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* PnL breakdown */}
        <div className="grid grid-cols-2 gap-4 border-t pt-4 sm:grid-cols-4">
          <div className="text-center">
            <p className="text-xs uppercase text-muted-foreground">Gross PnL</p>
            <p
              className={cn(
                "mt-1 text-sm font-bold tabular-nums",
                grossPnl != null && grossPnl >= 0 ? "text-green-500" : "text-red-500",
              )}
            >
              {formatSignedUsd(grossPnl)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs uppercase text-muted-foreground">Funding Costs</p>
            <p className="mt-1 text-sm font-bold tabular-nums text-yellow-500">
              {formatUsd(funding)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs uppercase text-muted-foreground">Net PnL</p>
            <p
              className={cn(
                "mt-1 text-sm font-bold tabular-nums",
                netPnl != null && netPnl >= 0 ? "text-green-500" : "text-red-500",
              )}
            >
              {formatSignedUsd(netPnl)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs uppercase text-muted-foreground">Funding % of Gross</p>
            <p className="mt-1 text-sm font-bold tabular-nums">
              {fundingPctOfGross != null ? `${fundingPctOfGross.toFixed(1)}%` : "N/A"}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
