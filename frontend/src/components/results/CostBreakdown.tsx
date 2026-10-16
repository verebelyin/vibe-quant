import type { BacktestResultResponse } from "@/api/generated/models";
import {
  useGetRunMetaApiResultsRunsRunIdMetaGet,
  useGetRunSummaryApiResultsRunsRunIdGet,
} from "@/api/generated/results/results";
import { totalCosts } from "@/components/results/totalCosts";
import { LoadingSpinner } from "@/components/ui";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  costRatios,
  formatFiniteNumber,
  formatFractionPct,
  formatSignedUsd,
  parseResultNotes,
} from "@/lib/metrics";
import { cn } from "@/lib/utils";

interface CostBreakdownProps {
  runId: number;
}

interface CostRow {
  label: string;
  /** null = unavailable (shown N/A); the bar treats it as empty. */
  value: number | null;
  barClass: string;
}

function formatUsd(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  return `$${Math.abs(value).toFixed(2)}`;
}

/** Maker-entry statistics the D3-6 validation runner writes into result.notes. */
interface MakerEntryNotes {
  submitted?: number | null;
  filled?: number | null;
  partial?: number | null;
  expired?: number | null;
  rejected_post_only?: number | null;
  denied?: number | null;
  cancelled_by_exit?: number | null;
  fill_rate?: number | null;
  market_twin?: MarketTwinNotes | null;
}

interface MarketTwinNotes {
  sharpe_ratio?: number | null;
  total_return?: number | null;
  total_trades?: number | null;
  total_fees?: number | null;
  error?: string | null;
}

function makerNotes(notes: Record<string, unknown> | null): MakerEntryNotes | null {
  const raw = notes?.maker_entries;
  if (raw == null || typeof raw !== "object" || Array.isArray(raw)) return null;
  return raw as MakerEntryNotes;
}

function count(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "n/a" : String(value);
}

function MakerEntryPanel({
  maker,
  data,
}: {
  maker: MakerEntryNotes;
  data: BacktestResultResponse;
}) {
  const twin = maker.market_twin ?? null;
  const twinError = twin?.error ?? null;
  return (
    <div className="border-t pt-3">
      <p className="text-xs font-semibold uppercase text-muted-foreground">
        Maker Entries (post-only)
      </p>
      <div className="mt-1 grid grid-cols-2 gap-x-4 text-xs">
        <span className="text-muted-foreground">Fill rate</span>
        <span className="text-right font-mono text-foreground">
          {formatFractionPct(maker.fill_rate, 1)}
        </span>
        <span className="text-muted-foreground">Submitted</span>
        <span className="text-right font-mono text-foreground">{count(maker.submitted)}</span>
        <span className="text-muted-foreground">Filled</span>
        <span className="text-right font-mono text-foreground">{count(maker.filled)}</span>
        <span className="text-muted-foreground">Partial fills</span>
        <span className="text-right font-mono text-foreground">{count(maker.partial)}</span>
        <span className="text-muted-foreground">Expired / timed out</span>
        <span className="text-right font-mono text-foreground">{count(maker.expired)}</span>
        <span className="text-muted-foreground">Rejected (post-only)</span>
        <span className="text-right font-mono text-foreground">
          {count(maker.rejected_post_only)}
        </span>
        <span className="text-muted-foreground">Denied (risk)</span>
        <span className="text-right font-mono text-foreground">{count(maker.denied)}</span>
        <span className="text-muted-foreground">Cancelled by exit</span>
        <span className="text-right font-mono text-foreground">
          {count(maker.cancelled_by_exit)}
        </span>
      </div>

      {twin && twinError == null && (
        <div className="mt-3">
          <p className="text-xs font-semibold uppercase text-muted-foreground">Market Twin</p>
          <div className="mt-1 grid grid-cols-3 gap-x-3 text-xs">
            <span />
            <span className="text-right text-muted-foreground">This run</span>
            <span className="text-right text-muted-foreground">Twin</span>
            <span className="text-foreground">Sharpe</span>
            <span className="text-right font-mono text-foreground">
              {formatFiniteNumber(data.sharpe_ratio, 2)}
            </span>
            <span className="text-right font-mono text-foreground">
              {formatFiniteNumber(twin.sharpe_ratio, 2)}
            </span>
            <span className="text-foreground">Return</span>
            <span className="text-right font-mono text-foreground">
              {formatFractionPct(data.total_return, 2)}
            </span>
            <span className="text-right font-mono text-foreground">
              {formatFractionPct(twin.total_return, 2)}
            </span>
            <span className="text-foreground">Trades</span>
            <span className="text-right font-mono text-foreground">{count(data.total_trades)}</span>
            <span className="text-right font-mono text-foreground">{count(twin.total_trades)}</span>
            <span className="text-foreground">Fees</span>
            <span className="text-right font-mono text-foreground">
              {formatUsd(data.total_fees)}
            </span>
            <span className="text-right font-mono text-foreground">
              {formatUsd(twin.total_fees)}
            </span>
          </div>
        </div>
      )}

      {twinError != null && (
        <p className="mt-3 text-xs text-destructive">Market twin failed: {twinError}</p>
      )}

      <p className="mt-2 text-[11px] text-muted-foreground">
        Maker fills are modelled optimistically (full size when price trades through the limit);
        compare with the market twin.
      </p>
    </div>
  );
}

export function CostBreakdown({ runId }: CostBreakdownProps) {
  const query = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const metaQuery = useGetRunMetaApiResultsRunsRunIdMetaGet(runId);
  const data = query.data?.data as BacktestResultResponse | undefined;
  const run = metaQuery.data?.status === 200 ? metaQuery.data.data : null;

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

  // Raw fees stay nullable: unavailable fees must never become a measured 0% ratio.
  const fees = data.total_fees;
  const slippage = data.total_slippage;
  const funding = data.total_funding;
  // null when every component is missing (legacy rows) -> shown as N/A below.
  const costsTotal = totalCosts(fees, slippage, funding);

  // total_return is a NET fraction (after fees, slippage, funding). The window
  // comes from notes.data_window (validation clamps to 1m coverage) before the
  // requested run dates: the result row has no dates of its own.
  const {
    netPnl,
    grossPnl,
    feeDrag: drag,
    feesPctEquityYr: perYear,
  } = costRatios({
    totalReturn: data.total_return,
    startingBalance: data.starting_balance,
    fees,
    slippage,
    funding,
    notes: data.notes,
    runStart: run?.start_date,
    runEnd: run?.end_date,
  });

  const maker = makerNotes(parseResultNotes(data.notes));

  const costs: CostRow[] = [
    { label: "Fees / Commission", value: fees, barClass: "bg-red-500" },
    { label: "Slippage", value: slippage, barClass: "bg-orange-500" },
    { label: "Funding", value: funding, barClass: "bg-yellow-500" },
  ];

  const maxCost = Math.max(...costs.map((c) => Math.abs(c.value ?? 0)), 1);

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
                  width: `${(Math.abs(cost.value ?? 0) / maxCost) * 100}%`,
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
                {formatSignedUsd(grossPnl)}
              </p>
            </div>
            <div>
              <p className="text-xs uppercase text-muted-foreground">Total Costs</p>
              <p className="mt-1 text-sm font-bold text-red-500">{formatUsd(costsTotal)}</p>
            </div>
            <div>
              <p className="text-xs uppercase text-muted-foreground">Net PnL</p>
              <p
                className={cn(
                  "mt-1 text-sm font-bold",
                  netPnl != null && netPnl >= 0 ? "text-green-500" : "text-red-500",
                )}
              >
                {formatSignedUsd(netPnl)}
              </p>
            </div>
          </div>
        </div>

        <div className="border-t pt-3 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-foreground">
              Fee drag{" "}
              <span className="text-muted-foreground">(fees / gross PnL, n/a when gross ≤ 0)</span>
            </span>
            <span className="font-mono text-muted-foreground">{formatFractionPct(drag, 1)}</span>
          </div>
          <div className="mt-1 flex items-center justify-between">
            <span className="text-foreground">
              Fees / equity / yr{" "}
              <span className="text-muted-foreground">
                (fees / starting balance, no compounding)
              </span>
            </span>
            <span className="font-mono text-muted-foreground">{formatFractionPct(perYear, 2)}</span>
          </div>
        </div>

        {maker != null && <MakerEntryPanel maker={maker} data={data} />}
      </CardContent>
    </Card>
  );
}
