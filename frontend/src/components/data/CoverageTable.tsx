import { useMemo, useState } from "react";
import type { DataCoverageItem } from "@/api/generated/models";

type SortKey = keyof DataCoverageItem;
type SortDir = "asc" | "desc";

interface CoverageTableProps {
  coverage: DataCoverageItem[];
}

function formatDate(iso: string): string {
  if (!iso) return "--";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function coverageColor(barCount: number, klineCount: number): string {
  if (klineCount === 0) return "hsl(var(--muted-foreground))";
  const pct = (barCount / klineCount) * 100;
  if (pct > 90) return "hsl(142 71% 45%)";
  if (pct > 50) return "hsl(48 96% 53%)";
  return "hsl(0 84% 60%)";
}

function coveragePct(barCount: number, klineCount: number): string {
  if (klineCount === 0) return "--";
  return ((barCount / klineCount) * 100).toFixed(1) + "%";
}

const COLUMNS: { key: SortKey; label: string; align?: "right" }[] = [
  { key: "symbol", label: "Symbol" },
  { key: "start_date", label: "Start Date" },
  { key: "end_date", label: "End Date" },
  { key: "kline_count", label: "Klines", align: "right" },
  { key: "bar_count", label: "Bars", align: "right" },
  { key: "funding_rate_count", label: "Funding Rates", align: "right" },
];

export function CoverageTable({ coverage }: CoverageTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("symbol");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const sorted = useMemo(() => {
    const copy = [...coverage];
    copy.sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortDir === "asc" ? aVal - bVal : bVal - aVal;
      }
      const aStr = String(aVal);
      const bStr = String(bVal);
      return sortDir === "asc" ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
    });
    return copy;
  }, [coverage, sortKey, sortDir]);

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  function sortIndicator(key: SortKey): string {
    if (key !== sortKey) return "";
    return sortDir === "asc" ? " \u2191" : " \u2193";
  }

  if (coverage.length === 0) {
    return (
      <div
        className="rounded-lg border p-8 text-center"
        style={{
          borderColor: "hsl(var(--border))",
          color: "hsl(var(--muted-foreground))",
        }}
      >
        No coverage data available. Ingest data to see symbol coverage.
      </div>
    );
  }

  return (
    <div
      className="overflow-x-auto rounded-lg border"
      style={{ borderColor: "hsl(var(--border))" }}
    >
      <table className="w-full text-sm">
        <thead>
          <tr
            style={{
              backgroundColor: "hsl(var(--muted))",
              color: "hsl(var(--muted-foreground))",
            }}
          >
            {COLUMNS.map((col) => (
              <th
                key={col.key}
                className={`cursor-pointer select-none px-4 py-2.5 font-medium ${
                  col.align === "right" ? "text-right" : "text-left"
                }`}
                onClick={() => handleSort(col.key)}
              >
                {col.label}
                {sortIndicator(col.key)}
              </th>
            ))}
            <th className="px-4 py-2.5 text-right font-medium">Coverage</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((item, idx) => (
            <tr
              key={item.symbol}
              className="transition-colors hover:opacity-80"
              style={{
                backgroundColor: idx % 2 === 0 ? "hsl(var(--card))" : "hsl(var(--muted) / 0.3)",
                borderTop: idx > 0 ? "1px solid hsl(var(--border))" : undefined,
              }}
            >
              <td
                className="px-4 py-2 font-mono font-medium"
                style={{ color: "hsl(var(--foreground))" }}
              >
                {item.symbol}
              </td>
              <td className="px-4 py-2" style={{ color: "hsl(var(--foreground))" }}>
                {formatDate(item.start_date)}
              </td>
              <td className="px-4 py-2" style={{ color: "hsl(var(--foreground))" }}>
                {formatDate(item.end_date)}
              </td>
              <td
                className="px-4 py-2 text-right font-mono"
                style={{ color: "hsl(var(--foreground))" }}
              >
                {item.kline_count.toLocaleString()}
              </td>
              <td
                className="px-4 py-2 text-right font-mono"
                style={{ color: "hsl(var(--foreground))" }}
              >
                {item.bar_count.toLocaleString()}
              </td>
              <td
                className="px-4 py-2 text-right font-mono"
                style={{ color: "hsl(var(--foreground))" }}
              >
                {item.funding_rate_count.toLocaleString()}
              </td>
              <td className="px-4 py-2 text-right">
                <span
                  className="font-mono font-semibold"
                  style={{
                    color: coverageColor(item.bar_count, item.kline_count),
                  }}
                >
                  {coveragePct(item.bar_count, item.kline_count)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
