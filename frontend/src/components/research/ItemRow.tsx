import { Link } from "@tanstack/react-router";
import { memo } from "react";
import type { ResearchItemResponse } from "@/api/generated/models";
import { ConfidenceBar } from "@/components/research/ConfidenceBar";
import { StatusBadge } from "@/components/research/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { TableCell, TableRow } from "@/components/ui/table";

function formatRelative(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const diff = Date.now() - t;
  const min = Math.floor(diff / 60_000);
  if (min < 1) return "just now";
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const days = Math.floor(hr / 24);
  return `${days}d ago`;
}

function getConfidence(extras: { [k: string]: unknown } | null | undefined): number | null {
  if (!extras) return null;
  const c = extras.latest_confidence;
  return typeof c === "number" ? c : null;
}

function getNumComments(extras: { [k: string]: unknown } | null | undefined): number | null {
  if (!extras) return null;
  const n = extras.num_comments;
  return typeof n === "number" ? n : null;
}

interface Props {
  item: ResearchItemResponse;
}

function ItemRowImpl({ item }: Props) {
  const confidence = getConfidence(item.extras);
  const numComments = getNumComments(item.extras);
  const showConfidence =
    item.extraction_status === "extracted" || item.extraction_status === "parsed";

  return (
    <TableRow className="cursor-pointer hover:bg-white/[0.02]">
      <TableCell className="text-xs text-muted-foreground tabular-nums">
        {formatRelative(item.posted_at ?? item.fetched_at)}
      </TableCell>
      <TableCell className="max-w-md">
        <Link
          to="/research/$itemId"
          params={{ itemId: String(item.id) }}
          className="block truncate text-sm text-foreground hover:text-primary hover:underline"
          title={item.title ?? undefined}
        >
          {item.title || "(untitled)"}
        </Link>
      </TableCell>
      <TableCell>
        <Badge variant="outline" className="font-mono text-[10px] text-muted-foreground">
          {item.source}
        </Badge>
      </TableCell>
      <TableCell className="text-right font-mono text-xs tabular-nums text-muted-foreground">
        {item.score ?? "—"}
      </TableCell>
      <TableCell className="text-right font-mono text-xs tabular-nums text-muted-foreground">
        {numComments ?? "—"}
      </TableCell>
      <TableCell>
        <StatusBadge status={item.extraction_status} />
      </TableCell>
      <TableCell>
        {showConfidence ? (
          <ConfidenceBar value={confidence} />
        ) : (
          <span className="text-xs text-muted-foreground">—</span>
        )}
      </TableCell>
    </TableRow>
  );
}

export const ItemRow = memo(ItemRowImpl);
