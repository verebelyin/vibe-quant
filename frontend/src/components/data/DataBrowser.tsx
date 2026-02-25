import { useEffect, useMemo, useState } from "react";
import type { BrowseDataResponse } from "@/api/generated/models";
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

const INTERVALS = ["1m", "5m", "15m", "1h", "4h"] as const;

function parseChartData(raw: unknown[]): { candles: CandlestickData[]; volume: VolumeData[] } {
  const candles: CandlestickData[] = [];
  const volume: VolumeData[] = [];
  for (const item of raw) {
    const r = item as Record<string, unknown>;
    const openTime = r.open_time;
    if (typeof openTime !== "number") continue;
    const time = Math.floor(openTime / 1000);
    const o = Number(r.open ?? 0);
    const h = Number(r.high ?? 0);
    const l = Number(r.low ?? 0);
    const c = Number(r.close ?? 0);
    const v = Number(r.volume ?? 0);
    candles.push({ time, open: o, high: h, low: l, close: c });
    volume.push({ time, value: v, color: c >= o ? "#26a69a80" : "#ef535080" });
  }
  return { candles, volume };
}

export function DataBrowser() {
  const [symbol, setSymbol] = useState("");
  const [interval, setInterval] = useState("1h");

  const symbolsQuery = useListSymbolsApiDataSymbolsGet();
  const symbols = symbolsQuery.data?.data ?? [];

  // Auto-select BTCUSDT (or first symbol) when symbols load
  useEffect(() => {
    if (symbols.length > 0 && !symbol) {
      const btc = symbols.find((s) => s === "BTCUSDT");
      setSymbol(btc ?? symbols[0]!);
    }
  }, [symbols, symbol]);

  const browseQuery = useBrowseDataApiDataBrowseSymbolGet(
    symbol || "__none__",
    { interval },
    {
      query: { enabled: !!symbol },
    },
  );
  const browseData = browseQuery.data?.data as BrowseDataResponse | undefined;

  const { candles, volume } = useMemo(() => {
    if (!browseData?.data) return { candles: [], volume: [] };
    return parseChartData(browseData.data);
  }, [browseData]);

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
        {candles.length > 0 && (
          <span className="pb-2 text-xs text-muted-foreground">
            {candles.length.toLocaleString()} bars
          </span>
        )}
      </div>

      {!symbol && (
        <div className="rounded-lg border p-8 text-center text-sm text-muted-foreground">
          Select a symbol to browse data
        </div>
      )}

      {symbol && browseQuery.isLoading && <Skeleton className="h-[500px] rounded-lg" />}

      {symbol && browseQuery.isError && (
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4 text-destructive">
          Failed to load browse data
        </div>
      )}

      {/* Chart */}
      {symbol && candles.length > 0 && (
        <Card className="p-2">
          <CardContent className="p-0">
            <CandlestickChart data={candles} volume={volume} height={600} />
          </CardContent>
        </Card>
      )}

      {symbol && !browseQuery.isLoading && !browseQuery.isError && candles.length === 0 && (
        <div className="rounded-lg border p-8 text-center text-sm text-muted-foreground">
          No data available for {symbol} at {interval}
        </div>
      )}
    </div>
  );
}
