import {
  type ColumnDef,
  createExpandedRowModel,
  createSortedRowModel,
  type ExpandedState,
  flexRender,
  type OnChangeFn,
  rowExpandingFeature,
  rowSortingFeature,
  type SortFn,
  type SortingState,
  tableFeatures,
  useTable,
} from "@tanstack/react-table";
import { ArrowUpDown, ChevronRight, Square } from "lucide-react";
import { Fragment, useCallback, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  getListDiscoveryJobsApiDiscoveryJobsGetQueryKey,
  useKillDiscoveryJobApiDiscoveryJobsRunIdDelete,
  useListDiscoveryJobsApiDiscoveryJobsGet,
} from "@/api/generated/discovery/discovery";
import type { DiscoveryJobResponse } from "@/api/generated/models";
import { queryClient } from "@/api/query-client";
import { EmptyState } from "@/components/ui";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/StatusBadge";
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
import { parseServerTimestamp } from "@/lib/time";
import { cn } from "@/lib/utils";
import { DiscoveryResults } from "./DiscoveryResults";

const ALL = "__all__";

function formatDuration(
  start: string | null | undefined,
  end: string | null | undefined,
  status: string,
): string {
  const s = parseServerTimestamp(start);
  if (s === undefined) return "-";
  const e = status === "running" ? Date.now() : (parseServerTimestamp(end) ?? Date.now());
  const secs = Math.max(0, Math.round((e - s) / 1000));
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.round(secs / 60)}m`;
  const h = Math.floor(secs / 3600);
  const m = Math.round((secs % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function formatStartedAt(start: string | null | undefined): string {
  const ms = parseServerTimestamp(start);
  if (ms === undefined) return "-";
  return new Date(ms).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function bestFitnessFromProgress(job: DiscoveryJobResponse): string {
  const p = job.progress as Record<string, unknown> | null;
  if (!p) return "-";
  const f = p.best_fitness ?? p.fitness;
  return f != null ? Number(f).toFixed(4) : "-";
}

function SortHeader({
  column,
  children,
}: {
  column: {
    toggleSorting: (desc?: boolean) => void;
    getIsSorted: () => false | "asc" | "desc";
  };
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      className="flex items-center gap-1 hover:text-foreground"
      onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
    >
      {children}
      <ArrowUpDown className="h-3 w-3" />
    </button>
  );
}

interface TableMeta {
  onKill: (id: number) => void;
  isKilling: boolean;
}

const features = tableFeatures({
  rowSortingFeature,
  rowExpandingFeature,
  sortedRowModel: createSortedRowModel(),
  expandedRowModel: createExpandedRowModel(),
  tableMeta: {} as TableMeta,
});

/** Chronological sort where a missing started_at (queued/running job) counts as newest. */
const startedAtSort: SortFn<typeof features, DiscoveryJobResponse> = (rowA, rowB, columnId) => {
  const a = rowA.getValue<string | null>(columnId);
  const b = rowB.getValue<string | null>(columnId);
  const at = parseServerTimestamp(a) ?? Number.POSITIVE_INFINITY;
  const bt = parseServerTimestamp(b) ?? Number.POSITIVE_INFINITY;
  return at === bt ? 0 : at < bt ? -1 : 1;
};

const columns: ColumnDef<typeof features, DiscoveryJobResponse>[] = [
  {
    id: "expander",
    header: "",
    cell: ({ row }) => {
      if (!row.getCanExpand()) return null;
      return (
        <ChevronRight
          className={cn(
            "h-4 w-4 text-muted-foreground transition-transform",
            row.getIsExpanded() && "rotate-90",
          )}
        />
      );
    },
    enableSorting: false,
  },
  {
    accessorKey: "run_id",
    header: ({ column }) => <SortHeader column={column}>Run</SortHeader>,
    cell: ({ row }) => (
      <div className="flex flex-col">
        <span className="font-mono text-xs">#{row.original.run_id}</span>
        {row.original.seed != null && (
          // biome-ignore lint/a11y: this span only stops click propagation to the row; it is not an interactive control
          <span
            className="select-all font-mono text-[10px] text-muted-foreground"
            onClick={(e) => e.stopPropagation()}
          >
            seed {row.original.seed}
          </span>
        )}
      </div>
    ),
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: ({ row }) => <StatusBadge status={row.original.status} />,
  },
  {
    accessorKey: "symbols",
    header: "Symbols",
    cell: ({ row }) => (
      <span className="text-xs text-muted-foreground">
        {row.original.symbols?.join(", ") ?? "-"}
      </span>
    ),
    enableSorting: false,
  },
  {
    accessorKey: "timeframe",
    header: "TF",
    cell: ({ row }) => <span className="font-mono text-xs">{row.original.timeframe ?? "-"}</span>,
  },
  {
    accessorKey: "generations",
    header: ({ column }) => <SortHeader column={column}>Gens</SortHeader>,
    cell: ({ row }) => {
      const p = row.original.progress as Record<string, unknown> | null;
      const current = p ? Number(p.generation ?? p.current_generation ?? 0) : 0;
      const max = row.original.generations ?? 0;
      if (!max) return <span className="font-mono text-xs">-</span>;
      const isRunning = row.original.status.toLowerCase() === "running";
      return (
        <span className="font-mono text-xs">{isRunning ? `${current}/${max}` : String(max)}</span>
      );
    },
  },
  {
    accessorKey: "population",
    header: "Pop",
    cell: ({ row }) => <span className="font-mono text-xs">{row.original.population ?? "-"}</span>,
  },
  {
    id: "best_fitness",
    header: ({ column }) => <SortHeader column={column}>Best Fitness</SortHeader>,
    accessorFn: (row) => {
      const p = row.progress as Record<string, unknown> | null;
      if (!p) return null;
      const f = p.best_fitness ?? p.fitness;
      return f != null ? Number(f) : null;
    },
    cell: ({ row }) => (
      <span className="font-mono text-xs font-medium">{bestFitnessFromProgress(row.original)}</span>
    ),
  },
  {
    accessorKey: "strategies_found",
    header: ({ column }) => <SortHeader column={column}>Strategies</SortHeader>,
    cell: ({ row }) => (
      <span className="font-mono text-xs">{row.original.strategies_found ?? "-"}</span>
    ),
  },
  {
    id: "duration",
    header: "Duration",
    cell: ({ row }) => (
      <span className="text-xs text-muted-foreground">
        {formatDuration(row.original.started_at, row.original.completed_at, row.original.status)}
      </span>
    ),
    enableSorting: false,
  },
  {
    accessorKey: "started_at",
    sortFn: startedAtSort,
    header: ({ column }) => <SortHeader column={column}>Started</SortHeader>,
    cell: ({ row }) => (
      <span className="text-xs text-muted-foreground">
        {formatStartedAt(row.original.started_at)}
      </span>
    ),
  },
  {
    id: "actions",
    header: "",
    cell: ({ row, table }) => {
      const isRunning = row.original.status.toLowerCase() === "running";
      if (!isRunning) return null;
      const meta = table.options.meta;
      return (
        <Button
          type="button"
          variant="destructive"
          size="xs"
          disabled={meta?.isKilling}
          onClick={(e) => {
            e.stopPropagation();
            meta?.onKill(row.original.run_id);
          }}
        >
          <Square className="mr-1 h-3 w-3" /> Kill
        </Button>
      );
    },
    enableSorting: false,
  },
];

function TableSkeleton() {
  return (
    <div className="space-y-2">
      <Skeleton className="h-10 w-full" />
      {Array.from({ length: 5 }).map((_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: skeleton rows
        <Skeleton key={i} className="h-12 w-full" />
      ))}
    </div>
  );
}

export function DiscoveryJobsTable({
  jobs,
  sorting,
  onSortingChange,
  expanded,
  onExpandedChange,
  onKill,
  isKilling,
}: {
  jobs: DiscoveryJobResponse[];
  sorting: SortingState;
  onSortingChange: OnChangeFn<SortingState>;
  expanded: ExpandedState;
  onExpandedChange: OnChangeFn<ExpandedState>;
  onKill: (id: number) => void;
  isKilling: boolean;
}) {
  const table = useTable({
    features,
    data: jobs,
    columns,
    state: { sorting, expanded },
    onSortingChange,
    onExpandedChange,
    getRowCanExpand: (row) => {
      const s = row.original.status.toLowerCase();
      return s === "completed" || s === "failed";
    },
    getRowId: (row) => String(row.run_id),
    meta: { onKill, isKilling } satisfies TableMeta,
  });

  return (
    <div className="overflow-auto rounded-md border">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((hg) => (
            <TableRow key={hg.id}>
              {hg.headers.map((header) => (
                <TableHead key={header.id}>
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
            <Fragment key={row.id}>
              <TableRow
                className={cn(
                  row.getCanExpand() && "cursor-pointer hover:bg-muted/50",
                  row.getIsExpanded() && "bg-muted/30",
                )}
                onClick={() => row.getCanExpand() && row.toggleExpanded()}
              >
                {row.getAllCells().map((cell) => (
                  <TableCell key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
              {row.getIsExpanded() && (
                <TableRow>
                  <TableCell colSpan={columns.length} className="p-0">
                    <div className="border-t border-border bg-muted/10 p-4">
                      {row.original.status.toLowerCase() === "failed" ? (
                        <div className="space-y-2">
                          <h4 className="text-sm font-semibold text-red-400">Error Details</h4>
                          <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-md border border-red-500/20 bg-red-950/20 p-3 font-mono text-xs text-red-300">
                            {row.original.error_message ||
                              "No error details available. Check server logs for run #" +
                                row.original.run_id +
                                "."}
                          </pre>
                        </div>
                      ) : (
                        <DiscoveryResults runId={row.original.run_id} />
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              )}
            </Fragment>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function DiscoveryResultsPage() {
  const [statusFilter, setStatusFilter] = useState<string>(ALL);
  const [sorting, setSorting] = useState<SortingState>([{ id: "started_at", desc: true }]);
  const [expanded, setExpanded] = useState<ExpandedState>({});

  const { data: jobsResp, isLoading } = useListDiscoveryJobsApiDiscoveryJobsGet({
    query: {
      refetchInterval: (query) => {
        const data = query.state.data;
        const jobs = data && data.status === 200 ? data.data : [];
        return jobs.some((j: DiscoveryJobResponse) => j.status.toLowerCase() === "running")
          ? 5000
          : false;
      },
    },
  });

  const jobs: DiscoveryJobResponse[] = useMemo(() => {
    if (!jobsResp || jobsResp.status !== 200) return [];
    return jobsResp.data;
  }, [jobsResp]);

  const filteredJobs = useMemo(() => {
    if (statusFilter === ALL) return jobs;
    return jobs.filter((j) => j.status.toLowerCase() === statusFilter);
  }, [jobs, statusFilter]);

  const killMutation = useKillDiscoveryJobApiDiscoveryJobsRunIdDelete({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getListDiscoveryJobsApiDiscoveryJobsGetQueryKey(),
        });
        toast.success("Discovery job killed");
      },
      onError: (err: unknown) => {
        toast.error("Failed to kill job", {
          description: err instanceof Error ? err.message : "Unknown error",
        });
      },
    },
  });

  const handleKill = useCallback(
    (runId: number) => {
      killMutation.mutate({ runId });
    },
    [killMutation],
  );

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <h1 className="text-lg font-semibold text-foreground">Discovery Results</h1>

      <Card className="flex-row flex-wrap items-center gap-3 px-4 py-3">
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger size="sm" className="w-[150px]">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>All statuses</SelectItem>
            <SelectItem value="running">Running</SelectItem>
            <SelectItem value="completed">Completed</SelectItem>
            <SelectItem value="failed">Failed</SelectItem>
            <SelectItem value="cancelled">Cancelled</SelectItem>
          </SelectContent>
        </Select>
      </Card>

      {isLoading ? (
        <TableSkeleton />
      ) : filteredJobs.length === 0 ? (
        <EmptyState
          title="No discovery runs"
          description="Launch a discovery from the Discovery page to see results here."
        />
      ) : (
        <DiscoveryJobsTable
          jobs={filteredJobs}
          sorting={sorting}
          onSortingChange={setSorting}
          expanded={expanded}
          onExpandedChange={setExpanded}
          onKill={handleKill}
          isKilling={killMutation.isPending}
        />
      )}
    </div>
  );
}
