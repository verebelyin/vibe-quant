import { useGetRunSummaryApiResultsRunsRunIdGet } from "@/api/generated/results/results";
import { LoadingSpinner } from "@/components/ui";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface CostBreakdownProps {
  runId: number;
}

interface CostRow {
  label: string;
  value: number;
  barClass: string;
}

function formatUsd(value: number | null | undefined): string {
  if (value == null) return "N/A";
  return `$${Math.abs(value).toFixed(2)}`;
}

export function CostBreakdown({ runId }: CostBreakdownProps) {
  const query = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const data = query.data?.data;

  if (query.isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <LoadingSpinner size="sm" />
      </div>
    );
  }

  if (query.isError || !data) {
    return <p className="py-4 text-sm text-destructive">Failed to load cost data.</p>;
  }

  const fees = data.total_fees ?? 0;
  const slippage = data.total_slippage ?? 0;
  const funding = data.total_funding ?? 0;
  const totalCosts = fees + slippage + funding;

  const grossPnl =
    data.total_return != null && data.starting_balance != null
      ? (data.total_return / 100) * data.starting_balance + totalCosts
      : null;

  const netPnl = grossPnl != null ? grossPnl - totalCosts : null;

  const costs: CostRow[] = [
    { label: "Fees / Commission", value: fees, barClass: "bg-red-500" },
    { label: "Slippage", value: slippage, barClass: "bg-orange-500" },
    { label: "Funding", value: funding, barClass: "bg-yellow-500" },
  ];

  const maxCost = Math.max(...costs.map((c) => Math.abs(c.value)), 1);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Cost Breakdown
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-3">
        {costs.map((cost) => (
          <div key={cost.label}>
            <div className="mb-1 flex items-center justify-between text-xs">
              <span className="text-foreground">{cost.label}</span>
              <span className="text-muted-foreground">{formatUsd(cost.value)}</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className={cn("h-full rounded-full transition-all", cost.barClass)}
                style={{
                  width: `${(Math.abs(cost.value) / maxCost) * 100}%`,
                }}
              />
            </div>
          </div>
        ))}

        <div className="border-t pt-3">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-xs uppercase text-muted-foreground">Gross PnL</p>
              <p
                className={cn(
                  "mt-1 text-sm font-bold",
                  grossPnl != null && grossPnl >= 0 ? "text-green-500" : "text-red-500",
                )}
              >
                {grossPnl != null ? `$${grossPnl.toFixed(2)}` : "N/A"}
              </p>
            </div>
            <div>
              <p className="text-xs uppercase text-muted-foreground">Total Costs</p>
              <p className="mt-1 text-sm font-bold text-red-500">{formatUsd(totalCosts)}</p>
            </div>
            <div>
              <p className="text-xs uppercase text-muted-foreground">Net PnL</p>
              <p
                className={cn(
                  "mt-1 text-sm font-bold",
                  netPnl != null && netPnl >= 0 ? "text-green-500" : "text-red-500",
                )}
              >
                {netPnl != null ? `$${netPnl.toFixed(2)}` : "N/A"}
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
