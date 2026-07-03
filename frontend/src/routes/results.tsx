import { useNavigate } from "@tanstack/react-router";
import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  type SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { ArrowUpDown } from "lucide-react";
import { useMemo, useState } from "react";
import type { RunSummaryItem } from "@/api/generated/models";
import { useListRunsSummaryApiResultsRunsSummaryGet } from "@/api/generated/results/results";
import { useListStrategiesApiStrategiesGet } from "@/api/generated/strategies/strategies";
import { EmptyState } from "@/components/ui";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
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

const ALL = "__all__";
const SHOW_ALL_MODES = "__show_all__";

function formatPct(v: number | null | undefined): string {
  if (v == null) return "-";
  return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;
}

function formatNum(v: number | null | undefined, decimals = 2): string {
  if (v == null) return "-";
  return v.toFixed(decimals);
}

function formatDate(d: string | null | undefined): string {
  if (!d) return "-";
  return new Date(d).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function statusVariant(status: string): "default" | "destructive" | "secondary" | "outline" {
  switch (status) {
    case "completed":
      return "default";
    case "failed":
      return "destructive";
    case "running":
      return "secondary";
    default:
      return "outline";
  }
}

function modeVariant(mode: string): "default" | "secondary" | "outline" {
  return mode === "validation" ? "default" : "outline";
}

function SortHeader({
  column,
  children,
  align,
}: {
  column: {
    toggleSorting: (desc?: boolean) => void;
    getIsSorted: () => false | "asc" | "desc";
  };
  children: React.ReactNode;
  align?: "right";
}) {
  const sorted = column.getIsSorted();
  return (
    <button
      type="button"
      className={cn(
        "flex w-full items-center gap-1 hover:text-foreground",
        align === "right" && "justify-end",
        sorted && "text-foreground",
      )}
      onClick={() => column.toggleSorting(sorted === "asc")}
    >
      {children}
      <ArrowUpDown className={cn("h-3 w-3", sorted && "text-primary")} />
    </button>
  );
}

/** Column meta: alignment class applied to both header and cells */
interface ColumnMetaWithClass {
  className?: string;
}

function columnClass(meta: unknown): string | undefined {
  return (meta as ColumnMetaWithClass | undefined)?.className;
}

const columns: ColumnDef<RunSummaryItem>[] = [
  {
    accessorKey: "strategy_name",
    header: ({ column }) => <SortHeader column={column}>Strategy</SortHeader>,
    cell: ({ row }) =>
      row.original.strategy_name ? (
        <span className="font-medium">{row.original.strategy_name}</span>
      ) : (
        <span className="text-muted-foreground/60 italic">Unnamed run</span>
      ),
  },
  {
    accessorKey: "run_mode",
    header: "Mode",
    cell: ({ row }) => (
      <Badge variant={modeVariant(row.original.run_mode)} className="capitalize">
        {row.original.run_mode}
      </Badge>
    ),
  },
  {
    accessorKey: "symbols",
    header: "Symbols",
    cell: ({ row }) => (
      <span className="text-xs text-muted-foreground">{row.original.symbols.join(", ")}</span>
    ),
    enableSorting: false,
  },
  {
    accessorKey: "total_return",
    header: ({ column }) => (
      <SortHeader column={column} align="right">
        Return
      </SortHeader>
    ),
    meta: { className: "text-right" },
    cell: ({ row }) => {
      const v = row.original.total_return;
      return (
        <span
          className={cn(
            "font-mono font-medium",
            v != null && v > 0
              ? "text-profit"
              : v != null && v < 0
                ? "text-loss"
                : "text-muted-foreground",
          )}
        >
          {formatPct(v)}
        </span>
      );
    },
  },
  {
    accessorKey: "sharpe_ratio",
    header: ({ column }) => (
      <SortHeader column={column} align="right">
        Sharpe
      </SortHeader>
    ),
    meta: { className: "text-right" },
    cell: ({ row }) => <span className="font-mono">{formatNum(row.original.sharpe_ratio)}</span>,
  },
  {
    accessorKey: "max_drawdown",
    header: ({ column }) => (
      <SortHeader column={column} align="right">
        Max DD
      </SortHeader>
    ),
    meta: { className: "text-right" },
    cell: ({ row }) => {
      const v = row.original.max_drawdown;
      return (
        <span
          className={cn("font-mono", v != null && v > 0 ? "text-loss" : "text-muted-foreground")}
        >
          {v != null ? `${(v * 100).toFixed(2)}%` : "-"}
        </span>
      );
    },
  },
  {
    accessorKey: "total_trades",
    header: ({ column }) => (
      <SortHeader column={column} align="right">
        Trades
      </SortHeader>
    ),
    meta: { className: "text-right" },
    cell: ({ row }) => <span className="font-mono">{row.original.total_trades ?? "-"}</span>,
  },
  {
    id: "win_loss",
    header: () => <div className="text-right">W/L</div>,
    meta: { className: "text-right" },
    cell: ({ row }) => {
      const w = row.original.winning_trades;
      const l = row.original.losing_trades;
      if (w == null && l == null) return <span className="text-muted-foreground">-</span>;
      return (
        <span className="font-mono text-xs">
          <span className="text-profit">{w ?? 0}</span>
          <span className="text-muted-foreground">/</span>
          <span className="text-loss">{l ?? 0}</span>
        </span>
      );
    },
    enableSorting: false,
  },
  {
    accessorKey: "win_rate",
    header: ({ column }) => (
      <SortHeader column={column} align="right">
        Win%
      </SortHeader>
    ),
    meta: { className: "text-right" },
    cell: ({ row }) => {
      const v = row.original.win_rate;
      return <span className="font-mono">{v != null ? `${(v * 100).toFixed(1)}%` : "-"}</span>;
    },
  },
  {
    accessorKey: "profit_factor",
    header: ({ column }) => (
      <SortHeader column={column} align="right">
        PF
      </SortHeader>
    ),
    meta: { className: "text-right" },
    cell: ({ row }) => <span className="font-mono">{formatNum(row.original.profit_factor)}</span>,
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: ({ row }) => (
      <Badge variant={statusVariant(row.original.status)} className="capitalize">
        {row.original.status}
      </Badge>
    ),
  },
  {
    accessorKey: "created_at",
    header: ({ column }) => <SortHeader column={column}>Date</SortHeader>,
    cell: ({ row }) => (
      <span className="text-xs text-muted-foreground">{formatDate(row.original.created_at)}</span>
    ),
  },
];

function TableSkeleton() {
  return (
    <div className="space-y-2">
      <Skeleton className="h-10 w-full" />
      {Array.from({ length: 6 }).map((_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: skeleton rows
        <Skeleton key={i} className="h-12 w-full" />
      ))}
    </div>
  );
}

export function ResultsPage() {
  const navigate = useNavigate();
  const [modeFilter, setModeFilter] = useState<string>(ALL);
  const [strategyFilter, setStrategyFilter] = useState<string>(ALL);
  const [statusFilter, setStatusFilter] = useState<string>(ALL);
  const [sorting, setSorting] = useState<SortingState>([{ id: "created_at", desc: true }]);

  const apiParams = useMemo(() => {
    const p: Record<string, string | number> = {};
    if (modeFilter !== ALL && modeFilter !== SHOW_ALL_MODES) p.run_mode = modeFilter;
    if (statusFilter !== ALL) p.status = statusFilter;
    if (strategyFilter !== ALL) p.strategy_id = Number(strategyFilter);
    return Object.keys(p).length > 0 ? p : undefined;
  }, [modeFilter, statusFilter, strategyFilter]);

  const query = useListRunsSummaryApiResultsRunsSummaryGet(apiParams);
  const resp = query.data;
  const runs = resp && resp.status === 200 ? resp.data.runs : [];

  // Default (ALL): exclude screening; SHOW_ALL_MODES: show everything
  const filteredRuns = useMemo(() => {
    if (modeFilter === ALL) return runs.filter((r) => r.run_mode !== "screening");
    return runs;
  }, [runs, modeFilter]);

  const strategiesQuery = useListStrategiesApiStrategiesGet();
  const strategiesResp = strategiesQuery.data;
  const strategies =
    strategiesResp && strategiesResp.status === 200 ? strategiesResp.data.strategies : [];

  const table = useReactTable({
    data: filteredRuns,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="flex flex-col gap-6">
      <Card className="flex-row flex-wrap items-center gap-3 px-4 py-3">
        <Select value={modeFilter} onValueChange={setModeFilter}>
          <SelectTrigger size="sm" className="w-[150px]">
            <SelectValue placeholder="Mode" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>Validation only</SelectItem>
            <SelectItem value="validation">Validation</SelectItem>
            <SelectItem value="screening">Screening</SelectItem>
            <SelectItem value={SHOW_ALL_MODES}>All modes</SelectItem>
          </SelectContent>
        </Select>

        <Select value={strategyFilter} onValueChange={setStrategyFilter}>
          <SelectTrigger size="sm" className="w-[180px]">
            <SelectValue placeholder="All strategies" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>All strategies</SelectItem>
            {strategies.map((s) => (
              <SelectItem key={s.id} value={String(s.id)}>
                {s.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger size="sm" className="w-[140px]">
            <SelectValue placeholder="All statuses" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>All statuses</SelectItem>
            <SelectItem value="completed">Completed</SelectItem>
            <SelectItem value="running">Running</SelectItem>
            <SelectItem value="failed">Failed</SelectItem>
            <SelectItem value="pending">Pending</SelectItem>
          </SelectContent>
        </Select>
      </Card>

      {query.isLoading ? (
        <TableSkeleton />
      ) : filteredRuns.length === 0 ? (
        <EmptyState
          title="No results yet"
          description="Run a backtest from the Backtest page to see results here."
        />
      ) : (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground tabular-nums">
            {filteredRuns.length} run{filteredRuns.length !== 1 ? "s" : ""}
            {modeFilter === ALL && " — screening runs hidden; pick “All modes” to include them"}
          </p>
          <div className="overflow-auto rounded-md border">
            <Table>
              <TableHeader className="sticky top-0 z-10 bg-card">
                {table.getHeaderGroups().map((hg) => (
                  <TableRow key={hg.id}>
                    {hg.headers.map((header) => (
                      <TableHead
                        key={header.id}
                        className={cn(
                          "whitespace-nowrap",
                          columnClass(header.column.columnDef.meta),
                        )}
                      >
                        {header.isPlaceholder
                          ? null
                          : flexRender(header.column.columnDef.header, header.getContext())}
                      </TableHead>
                    ))}
                  </TableRow>
                ))}
              </TableHeader>
              <TableBody>
                {table.getRowModel().rows.map((row) => (
                  <TableRow
                    key={row.id}
                    className="cursor-pointer"
                    onClick={() =>
                      navigate({
                        to: "/results/$runId",
                        params: { runId: String(row.original.run_id) },
                      })
                    }
                  >
                    {row.getVisibleCells().map((cell) => (
                      <TableCell
                        key={cell.id}
                        className={cn("whitespace-nowrap", columnClass(cell.column.columnDef.meta))}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      )}
    </div>
  );
}
