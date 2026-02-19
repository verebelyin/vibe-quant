import { useMemo, useState } from "react";
import type { TradeResponse } from "@/api/generated/models/tradeResponse";
import { useGetTradesApiResultsRunsRunIdTradesGet } from "@/api/generated/results/results";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

interface TradeLogProps {
  runId: number;
}

function formatPrice(value: number | null | undefined): string {
  if (value == null) return "-";
  return value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 });
}

function formatPnl(value: number | null | undefined): string {
  if (value == null) return "-";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}`;
}

function formatDuration(entry: string, exit: string | null): string {
  if (!exit) return "Open";
  const ms = new Date(exit).getTime() - new Date(entry).getTime();
  const hours = ms / (1000 * 60 * 60);
  if (hours < 1) return `${Math.round(hours * 60)}m`;
  if (hours < 24) return `${hours.toFixed(1)}h`;
  return `${(hours / 24).toFixed(1)}d`;
}

function formatTime(ts: string | null): string {
  if (!ts) return "-";
  return new Date(ts).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function isLiquidation(trade: TradeResponse): boolean {
  return trade.exit_reason === "liquidation";
}

function exportToCsv(trades: TradeResponse[], runId: number) {
  const headers = [
    "Symbol",
    "Side",
    "Entry Price",
    "Exit Price",
    "Net PnL",
    "ROI %",
    "Duration",
    "Entry Time",
    "Exit Time",
    "Exit Reason",
    "Leverage",
    "Quantity",
    "Entry Fee",
    "Exit Fee",
    "Funding Fees",
    "Slippage",
  ];
  const rows = trades.map((t) => [
    t.symbol,
    t.direction,
    t.entry_price,
    t.exit_price ?? "",
    t.net_pnl ?? "",
    t.roi_percent ?? "",
    formatDuration(t.entry_time, t.exit_time),
    t.entry_time,
    t.exit_time ?? "",
    t.exit_reason ?? "",
    t.leverage,
    t.quantity,
    t.entry_fee ?? "",
    t.exit_fee ?? "",
    t.funding_fees ?? "",
    t.slippage_cost ?? "",
  ]);

  const csvContent = [headers, ...rows].map((row) => row.join(",")).join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `trades_run_${runId}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

function TradeLogSkeleton() {
  return (
    <div className="space-y-2">
      <Skeleton className="h-9 w-full" />
      {Array.from({ length: 8 }).map((_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: skeleton placeholders
        <Skeleton key={i} className="h-10 w-full" />
      ))}
    </div>
  );
}

const ALL_SYMBOLS = "__all__";
const ALL_SIDES = "__all__";

export function TradeLog({ runId }: TradeLogProps) {
  const [symbolFilter, setSymbolFilter] = useState(ALL_SYMBOLS);
  const [sideFilter, setSideFilter] = useState(ALL_SIDES);

  const query = useGetTradesApiResultsRunsRunIdTradesGet(runId);
  const trades = query.data?.data;

  const symbols = useMemo(() => {
    if (!trades) return [];
    return [...new Set(trades.map((t) => t.symbol))].sort();
  }, [trades]);

  const filteredTrades = useMemo(() => {
    if (!trades) return [];
    return trades.filter((t) => {
      if (symbolFilter !== ALL_SYMBOLS && t.symbol !== symbolFilter) return false;
      if (sideFilter !== ALL_SIDES && t.direction !== sideFilter) return false;
      return true;
    });
  }, [trades, symbolFilter, sideFilter]);

  if (query.isLoading) return <TradeLogSkeleton />;

  if (query.isError || !trades) {
    return <p className="py-4 text-sm text-destructive">Failed to load trades.</p>;
  }

  if (trades.length === 0) {
    return <p className="py-8 text-center text-sm text-muted-foreground">No trades recorded.</p>;
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Select value={symbolFilter} onValueChange={setSymbolFilter}>
          <SelectTrigger className="w-44">
            <SelectValue placeholder="All Symbols" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL_SYMBOLS}>All Symbols</SelectItem>
            {symbols.map((s) => (
              <SelectItem key={s} value={s}>
                {s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={sideFilter} onValueChange={setSideFilter}>
          <SelectTrigger className="w-32">
            <SelectValue placeholder="All Sides" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL_SIDES}>All Sides</SelectItem>
            <SelectItem value="long">Long</SelectItem>
            <SelectItem value="short">Short</SelectItem>
          </SelectContent>
        </Select>

        <span className="ml-auto text-xs text-muted-foreground">
          {filteredTrades.length} of {trades.length} trades
        </span>

        <Button variant="outline" size="sm" onClick={() => exportToCsv(filteredTrades, runId)}>
          Export CSV
        </Button>
      </div>

      <div className="max-h-[600px] overflow-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Symbol</TableHead>
              <TableHead>Side</TableHead>
              <TableHead className="text-right">Entry Price</TableHead>
              <TableHead className="text-right">Exit Price</TableHead>
              <TableHead className="text-right">PnL</TableHead>
              <TableHead className="text-right">Duration</TableHead>
              <TableHead>Entry Time</TableHead>
              <TableHead>Exit Time</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredTrades.map((trade) => (
              <TableRow key={trade.id} className={cn(isLiquidation(trade) && "bg-destructive/10")}>
                <TableCell className="font-medium">
                  {trade.symbol}
                  {isLiquidation(trade) && (
                    <span className="ml-1.5 text-[10px] font-semibold uppercase text-destructive">
                      LIQ
                    </span>
                  )}
                </TableCell>
                <TableCell>
                  <span
                    className={cn(
                      "text-xs font-medium uppercase",
                      trade.direction === "long" ? "text-green-500" : "text-red-500",
                    )}
                  >
                    {trade.direction}
                  </span>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {formatPrice(trade.entry_price)}
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {formatPrice(trade.exit_price)}
                </TableCell>
                <TableCell
                  className={cn(
                    "text-right font-medium tabular-nums",
                    trade.net_pnl != null && trade.net_pnl >= 0 ? "text-green-500" : "text-red-500",
                  )}
                >
                  {formatPnl(trade.net_pnl)}
                </TableCell>
                <TableCell className="text-right text-xs text-muted-foreground">
                  {formatDuration(trade.entry_time, trade.exit_time)}
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {formatTime(trade.entry_time)}
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {formatTime(trade.exit_time)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
