import { useGetRunSummaryApiResultsRunsRunIdGet } from "@/api/generated/results/results";
import { LoadingSpinner } from "@/components/ui";

interface CostBreakdownProps {
  runId: number;
}

interface CostRow {
  label: string;
  value: number;
  color: string;
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
    return (
      <p className="py-4 text-sm" style={{ color: "hsl(var(--destructive))" }}>
        Failed to load cost data.
      </p>
    );
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
    { label: "Fees / Commission", value: fees, color: "hsl(0, 70%, 55%)" },
    { label: "Slippage", value: slippage, color: "hsl(30, 70%, 55%)" },
    { label: "Funding", value: funding, color: "hsl(50, 70%, 55%)" },
  ];

  const maxCost = Math.max(...costs.map((c) => Math.abs(c.value)), 1);

  return (
    <div
      className="rounded-lg border p-4"
      style={{
        backgroundColor: "hsl(var(--card))",
        borderColor: "hsl(var(--border))",
      }}
    >
      <h3
        className="mb-3 text-sm font-semibold uppercase tracking-wide"
        style={{ color: "hsl(var(--muted-foreground))" }}
      >
        Cost Breakdown
      </h3>

      <div className="space-y-3">
        {costs.map((cost) => (
          <div key={cost.label}>
            <div className="mb-1 flex items-center justify-between text-xs">
              <span style={{ color: "hsl(var(--foreground))" }}>{cost.label}</span>
              <span style={{ color: "hsl(var(--muted-foreground))" }}>{formatUsd(cost.value)}</span>
            </div>
            <div
              className="h-2 w-full overflow-hidden rounded-full"
              style={{ backgroundColor: "hsl(var(--muted))" }}
            >
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${(Math.abs(cost.value) / maxCost) * 100}%`,
                  backgroundColor: cost.color,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 border-t pt-3" style={{ borderColor: "hsl(var(--border))" }}>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-xs uppercase" style={{ color: "hsl(var(--muted-foreground))" }}>
              Gross PnL
            </p>
            <p
              className="mt-1 text-sm font-bold"
              style={{
                color:
                  grossPnl != null && grossPnl >= 0 ? "hsl(142, 70%, 45%)" : "hsl(0, 70%, 55%)",
              }}
            >
              {grossPnl != null ? `$${grossPnl.toFixed(2)}` : "N/A"}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase" style={{ color: "hsl(var(--muted-foreground))" }}>
              Total Costs
            </p>
            <p className="mt-1 text-sm font-bold" style={{ color: "hsl(0, 70%, 55%)" }}>
              {formatUsd(totalCosts)}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase" style={{ color: "hsl(var(--muted-foreground))" }}>
              Net PnL
            </p>
            <p
              className="mt-1 text-sm font-bold"
              style={{
                color: netPnl != null && netPnl >= 0 ? "hsl(142, 70%, 45%)" : "hsl(0, 70%, 55%)",
              }}
            >
              {netPnl != null ? `$${netPnl.toFixed(2)}` : "N/A"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
