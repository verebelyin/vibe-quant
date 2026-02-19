import { useMemo, useState } from "react";
import { useDownloadHistoryApiDataHistoryGet } from "@/api/generated/data/data";
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

type SortDir = "asc" | "desc";

function formatTimestamp(ts: unknown): string {
  if (!ts || typeof ts !== "string") return "--";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return String(ts);
  return d.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDate(val: unknown): string {
  if (!val || typeof val !== "string") return "--";
  const d = new Date(val);
  if (Number.isNaN(d.getTime())) return String(val);
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

const COLUMNS = [
  { key: "symbol", label: "Symbol" },
  { key: "interval", label: "Interval" },
  { key: "start_date", label: "Start Date" },
  { key: "end_date", label: "End Date" },
  { key: "rows", label: "Rows", align: "right" as const },
  { key: "timestamp", label: "Timestamp" },
] as const;

export function DownloadHistory() {
  const historyQuery = useDownloadHistoryApiDataHistoryGet({ limit: 50 });
  const items = historyQuery.data?.data ?? [];
  const [sortKey, setSortKey] = useState("timestamp");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sorted = useMemo(() => {
    const copy = [...items];
    copy.sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortDir === "asc" ? aVal - bVal : bVal - aVal;
      }
      const aStr = String(aVal ?? "");
      const bStr = String(bVal ?? "");
      return sortDir === "asc" ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
    });
    return copy;
  }, [items, sortKey, sortDir]);

  function handleSort(key: string) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  function sortIndicator(key: string): string {
    if (key !== sortKey) return "";
    return sortDir === "asc" ? " \u2191" : " \u2193";
  }

  if (historyQuery.isLoading) {
    return <Skeleton className="h-32 rounded-lg" />;
  }

  if (historyQuery.isError) {
    return (
      <div className="rounded-lg border border-destructive bg-destructive/10 p-4 text-sm text-destructive">
        Failed to load download history.
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="rounded-lg border p-8 text-center text-muted-foreground">
        No download history yet.
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
                  "cursor-pointer select-none px-4",
                  col.align === "right" && "text-right",
                )}
                onClick={() => handleSort(col.key)}
              >
                {col.label}
                {sortIndicator(col.key)}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((item, idx) => (
            <TableRow key={`${String(item.symbol)}-${String(item.timestamp)}-${idx}`}>
              <TableCell className="px-4 font-mono font-medium">
                {String(item.symbol ?? "--")}
              </TableCell>
              <TableCell className="px-4">{String(item.interval ?? "--")}</TableCell>
              <TableCell className="px-4">{formatDate(item.start_date)}</TableCell>
              <TableCell className="px-4">{formatDate(item.end_date)}</TableCell>
              <TableCell className="px-4 text-right font-mono">
                {typeof item.rows === "number"
                  ? item.rows.toLocaleString()
                  : String(item.rows ?? "--")}
              </TableCell>
              <TableCell className="px-4 text-muted-foreground">
                {formatTimestamp(item.timestamp)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
