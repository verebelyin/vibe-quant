import { useMemo, useState } from "react";
import type { DataCoverageItem } from "@/api/generated/models";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

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

function coverageColorClass(barCount: number, klineCount: number): string {
  if (klineCount === 0) return "text-muted-foreground";
  const pct = (barCount / klineCount) * 100;
  if (pct > 90) return "text-green-500";
  if (pct > 50) return "text-yellow-500";
  return "text-destructive";
}

function coveragePct(barCount: number, klineCount: number): string {
  if (klineCount === 0) return "--";
  return `${((barCount / klineCount) * 100).toFixed(1)}%`;
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
      <div className="rounded-lg border p-8 text-center text-muted-foreground">
        No coverage data available. Ingest data to see symbol coverage.
      </div>
    );
  }

  return (
    <div className="rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow className="bg-muted hover:bg-muted">
            {COLUMNS.map((col) => (
              <TableHead
                key={col.key}
                className={cn(
                  "cursor-pointer select-none px-4 hover:bg-muted/70",
                  col.align === "right" && "text-right",
                )}
                onClick={() => handleSort(col.key)}
              >
                {col.label}
                {sortIndicator(col.key)}
              </TableHead>
            ))}
            <TableHead className="px-4 text-right">Coverage</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((item) => (
            <TableRow key={item.symbol}>
              <TableCell className="px-4 font-mono font-medium">{item.symbol}</TableCell>
              <TableCell className="px-4">{formatDate(item.start_date)}</TableCell>
              <TableCell className="px-4">{formatDate(item.end_date)}</TableCell>
              <TableCell className="px-4 text-right font-mono">
                {item.kline_count.toLocaleString()}
              </TableCell>
              <TableCell className="px-4 text-right font-mono">
                {item.bar_count.toLocaleString()}
              </TableCell>
              <TableCell className="px-4 text-right font-mono">
                {item.funding_rate_count.toLocaleString()}
              </TableCell>
              <TableCell className="px-4 text-right">
                <span
                  className={cn(
                    "font-mono font-semibold",
                    coverageColorClass(item.bar_count, item.kline_count),
                  )}
                >
                  {coveragePct(item.bar_count, item.kline_count)}
                </span>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
