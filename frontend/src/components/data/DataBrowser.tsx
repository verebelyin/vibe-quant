import { useId, useMemo, useState } from "react";
import {
  useBrowseDataApiDataBrowseSymbolGet,
  useListSymbolsApiDataSymbolsGet,
} from "@/api/generated/data/data";
import type { CandlestickData, VolumeData } from "@/components/charts/CandlestickChart";
import CandlestickChart from "@/components/charts/CandlestickChart";

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

function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  const id = useId();
  return (
    <div className="flex flex-col gap-1">
      <label
        htmlFor={id}
        className="text-xs font-medium uppercase tracking-wider"
        style={{ color: "hsl(var(--muted-foreground))" }}
      >
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border px-3 py-1.5 text-sm"
        style={{
          backgroundColor: "hsl(var(--card))",
          borderColor: "hsl(var(--border))",
          color: "hsl(var(--foreground))",
        }}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
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
        <SelectField
          label="Symbol"
          value={symbol}
          onChange={setSymbol}
          options={[
            { value: "", label: "Select symbol..." },
            ...symbols.map((s) => ({ value: s, label: s })),
          ]}
        />
        <SelectField
          label="Interval"
          value={interval}
          onChange={setInterval}
          options={INTERVALS.map((i) => ({ value: i, label: i }))}
        />
      </div>

      {!symbol && (
        <div
          className="rounded-lg border p-8 text-center text-sm"
          style={{
            borderColor: "hsl(var(--border))",
            color: "hsl(var(--muted-foreground))",
          }}
        >
          Select a symbol to browse data
        </div>
      )}

      {symbol && browseQuery.isLoading && (
        <div
          className="h-[400px] animate-pulse rounded-lg"
          style={{ backgroundColor: "hsl(var(--muted))" }}
        />
      )}

      {symbol && browseQuery.isError && (
        <div
          className="rounded-lg border p-4"
          style={{
            borderColor: "hsl(0 84% 60%)",
            backgroundColor: "hsl(0 84% 60% / 0.1)",
            color: "hsl(0 84% 60%)",
          }}
        >
          Failed to load browse data
        </div>
      )}

      {/* Chart */}
      {symbol && rows.length > 0 && (
        <div
          className="rounded-lg border p-2"
          style={{
            backgroundColor: "hsl(var(--card))",
            borderColor: "hsl(var(--border))",
          }}
        >
          <CandlestickChart data={candlestickData} volume={volumeData} height={400} />
        </div>
      )}

      {/* OHLCV Table */}
      {symbol && rows.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold" style={{ color: "hsl(var(--foreground))" }}>
            OHLCV Data ({rows.length} bars)
          </h4>
          <div
            className="overflow-auto rounded-lg border"
            style={{
              maxHeight: "400px",
              borderColor: "hsl(var(--border))",
            }}
          >
            <table className="w-full text-xs">
              <thead className="sticky top-0 z-10" style={{ backgroundColor: "hsl(var(--muted))" }}>
                <tr>
                  {["Timestamp", "Open", "High", "Low", "Close", "Volume"].map((col) => (
                    <th
                      key={col}
                      className={`px-3 py-2 font-medium ${
                        col === "Timestamp" ? "text-left" : "text-right"
                      }`}
                      style={{ color: "hsl(var(--muted-foreground))" }}
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => {
                  const isUp = row.close >= row.open;
                  return (
                    <tr
                      key={row.timestamp}
                      style={{
                        backgroundColor:
                          i % 2 === 0 ? "hsl(var(--card))" : "hsl(var(--muted) / 0.3)",
                        borderTop: i > 0 ? "1px solid hsl(var(--border))" : undefined,
                      }}
                    >
                      <td
                        className="whitespace-nowrap px-3 py-1.5 font-mono"
                        style={{ color: "hsl(var(--foreground))" }}
                      >
                        {formatTimestamp(row.timestamp)}
                      </td>
                      <td
                        className="px-3 py-1.5 text-right font-mono"
                        style={{ color: "hsl(var(--foreground))" }}
                      >
                        {row.open.toFixed(2)}
                      </td>
                      <td
                        className="px-3 py-1.5 text-right font-mono"
                        style={{ color: "hsl(var(--foreground))" }}
                      >
                        {row.high.toFixed(2)}
                      </td>
                      <td
                        className="px-3 py-1.5 text-right font-mono"
                        style={{ color: "hsl(var(--foreground))" }}
                      >
                        {row.low.toFixed(2)}
                      </td>
                      <td
                        className="px-3 py-1.5 text-right font-mono"
                        style={{ color: isUp ? "hsl(142 71% 45%)" : "hsl(0 84% 60%)" }}
                      >
                        {row.close.toFixed(2)}
                      </td>
                      <td
                        className="px-3 py-1.5 text-right font-mono"
                        style={{ color: "hsl(var(--muted-foreground))" }}
                      >
                        {row.volume.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {symbol && !browseQuery.isLoading && !browseQuery.isError && rows.length === 0 && (
        <div
          className="rounded-lg border p-8 text-center text-sm"
          style={{
            borderColor: "hsl(var(--border))",
            color: "hsl(var(--muted-foreground))",
          }}
        >
          No data available for {symbol} at {interval}
        </div>
      )}
    </div>
  );
}
