import { useMemo, useState } from "react";
import {
  useBrowseDataApiDataBrowseSymbolGet,
  useListSymbolsApiDataSymbolsGet,
} from "@/api/generated/data/data";
import type { CandlestickData, VolumeData } from "@/components/charts/CandlestickChart";
import CandlestickChart from "@/components/charts/CandlestickChart";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
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

const INTERVALS = ["1m", "5m", "15m", "1h", "4h"] as const;

interface OhlcvRow {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

function parseRow(item: Record<string, unknown>): OhlcvRow {
  return {
    timestamp: String(item.timestamp ?? ""),
    open: Number(item.open ?? 0),
    high: Number(item.high ?? 0),
    low: Number(item.low ?? 0),
    close: Number(item.close ?? 0),
    volume: Number(item.volume ?? 0),
  };
}

function formatTimestamp(iso: string): string {
  if (!iso) return "--";
  const d = new Date(iso);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export function DataBrowser() {
  const [symbol, setSymbol] = useState("");
  const [interval, setInterval] = useState("1h");

  const symbolsQuery = useListSymbolsApiDataSymbolsGet();
  const symbols = symbolsQuery.data?.data ?? [];

  const browseQuery = useBrowseDataApiDataBrowseSymbolGet(
    symbol || "__none__",
    { interval },
    {
      query: { enabled: !!symbol },
    },
  );
  const browseData = browseQuery.data?.data;

  const rows = useMemo<OhlcvRow[]>(() => {
    if (!browseData?.data) return [];
    return browseData.data.map((item) => parseRow(item as Record<string, unknown>));
  }, [browseData]);

  const candlestickData = useMemo<CandlestickData[]>(
    () =>
      rows.map((r) => ({
        time: r.timestamp,
        open: r.open,
        high: r.high,
        low: r.low,
        close: r.close,
      })),
    [rows],
  );

  const volumeData = useMemo<VolumeData[]>(
    () =>
      rows.map((r) => ({
        time: r.timestamp,
        value: r.volume,
        color: r.close >= r.open ? "#26a69a80" : "#ef535080",
      })),
    [rows],
  );

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4">
        <div className="flex flex-col gap-1">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">Symbol</Label>
          <Select value={symbol} onValueChange={setSymbol}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="Select symbol..." />
            </SelectTrigger>
            <SelectContent>
              {symbols.map((s) => (
                <SelectItem key={s} value={s}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex flex-col gap-1">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">Interval</Label>
          <Select value={interval} onValueChange={setInterval}>
            <SelectTrigger className="w-24">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {INTERVALS.map((i) => (
                <SelectItem key={i} value={i}>
                  {i}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {!symbol && (
        <div className="rounded-lg border p-8 text-center text-sm text-muted-foreground">
          Select a symbol to browse data
        </div>
      )}

      {symbol && browseQuery.isLoading && <Skeleton className="h-[400px] rounded-lg" />}

      {symbol && browseQuery.isError && (
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4 text-destructive">
          Failed to load browse data
        </div>
      )}

      {/* Chart */}
      {symbol && rows.length > 0 && (
        <Card className="p-2">
          <CardContent className="p-0">
            <CandlestickChart data={candlestickData} volume={volumeData} height={400} />
          </CardContent>
        </Card>
      )}

      {/* OHLCV Table */}
      {symbol && rows.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-foreground">
            OHLCV Data ({rows.length} bars)
          </h4>
          <div className="max-h-[400px] overflow-auto rounded-lg border">
            <Table className="text-xs">
              <TableHeader className="sticky top-0 z-10 bg-muted">
                <TableRow className="hover:bg-muted">
                  {["Timestamp", "Open", "High", "Low", "Close", "Volume"].map((col) => (
                    <TableHead
                      key={col}
                      className={cn("px-3", col !== "Timestamp" && "text-right")}
                    >
                      {col}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => {
                  const isUp = row.close >= row.open;
                  return (
                    <TableRow key={row.timestamp}>
                      <TableCell className="whitespace-nowrap px-3 py-1.5 font-mono">
                        {formatTimestamp(row.timestamp)}
                      </TableCell>
                      <TableCell className="px-3 py-1.5 text-right font-mono">
                        {row.open.toFixed(2)}
                      </TableCell>
                      <TableCell className="px-3 py-1.5 text-right font-mono">
                        {row.high.toFixed(2)}
                      </TableCell>
                      <TableCell className="px-3 py-1.5 text-right font-mono">
                        {row.low.toFixed(2)}
                      </TableCell>
                      <TableCell
                        className={cn(
                          "px-3 py-1.5 text-right font-mono",
                          isUp ? "text-green-500" : "text-destructive",
                        )}
                      >
                        {row.close.toFixed(2)}
                      </TableCell>
                      <TableCell className="px-3 py-1.5 text-right font-mono text-muted-foreground">
                        {row.volume.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </div>
      )}

      {symbol && !browseQuery.isLoading && !browseQuery.isError && rows.length === 0 && (
        <div className="rounded-lg border p-8 text-center text-sm text-muted-foreground">
          No data available for {symbol} at {interval}
        </div>
      )}
    </div>
  );
}
