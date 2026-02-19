import { useMemo, useState } from "react";
import { useDownloadHistoryApiDataHistoryGet } from "@/api/generated/data/data";

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
    return (
      <div
        className="h-32 animate-pulse rounded-lg"
        style={{ backgroundColor: "hsl(var(--muted))" }}
      />
    );
  }

  if (historyQuery.isError) {
    return (
      <div
        className="rounded-lg border p-4 text-sm"
        style={{
          borderColor: "hsl(0 84% 60%)",
          backgroundColor: "hsl(0 84% 60% / 0.1)",
          color: "hsl(0 84% 60%)",
        }}
      >
        Failed to load download history.
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div
        className="rounded-lg border p-8 text-center"
        style={{
          borderColor: "hsl(var(--border))",
          color: "hsl(var(--muted-foreground))",
        }}
      >
        No download history yet.
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
          </tr>
        </thead>
        <tbody>
          {sorted.map((item, idx) => (
            <tr
              key={`${String(item.symbol)}-${String(item.timestamp)}-${idx}`}
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
                {String(item.symbol ?? "--")}
              </td>
              <td className="px-4 py-2" style={{ color: "hsl(var(--foreground))" }}>
                {String(item.interval ?? "--")}
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
                {typeof item.rows === "number"
                  ? item.rows.toLocaleString()
                  : String(item.rows ?? "--")}
              </td>
              <td className="px-4 py-2" style={{ color: "hsl(var(--muted-foreground))" }}>
                {formatTimestamp(item.timestamp)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
