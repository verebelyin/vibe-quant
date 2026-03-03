# Discovery Results Sub-page Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move discovery results from the main discovery page into a dedicated `/discovery/results` sub-route with a sortable/filterable TanStack Table, live status updates, kill buttons, and expandable strategy detail rows.

**Architecture:** New route at `/discovery/results` with a TanStack React Table listing all discovery runs. Sidebar gets a tree-style sub-item using shadcn's built-in `SidebarMenuSub`. The main discovery page keeps config + a compact running-jobs indicator. Backend needs a richer endpoint returning symbols/timeframe/dates per discovery run.

**Tech Stack:** TanStack React Table, TanStack Router, shadcn sidebar sub-menu components, existing discovery API hooks (orval-generated).

---

### Task 1: Enrich backend discovery jobs endpoint

The current `DiscoveryJobResponse` only has `run_id`, `status`, `started_at`, `progress`. The results table needs symbols, timeframe, generations, population, best fitness, duration.

**Files:**
- Modify: `vibe_quant/api/schemas/discovery.py:24-31`
- Modify: `vibe_quant/api/routers/discovery.py:82-89`

**Step 1: Add fields to DiscoveryJobResponse schema**

```python
class DiscoveryJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: int
    status: str
    started_at: str | None
    completed_at: str | None = None
    progress: dict[str, object] | None = None
    symbols: list[str] | None = None
    timeframe: str | None = None
    generations: int | None = None
    population: int | None = None
    strategies_found: int | None = None
```

**Step 2: Enrich `_job_info_to_discovery_response` to pull run data from DB**

```python
def _job_info_to_discovery_response(
    info: JobInfo, state: StateManager | None = None
) -> DiscoveryJobResponse:
    progress = _read_progress_file(info.run_id)
    resp = DiscoveryJobResponse(
        run_id=info.run_id,
        status=info.status.value,
        started_at=info.started_at.isoformat() if info.started_at else None,
        progress=progress,
    )
    if state is not None:
        run = state.get_backtest_run(info.run_id)
        if run:
            resp.symbols = run.get("symbols")
            resp.timeframe = run.get("timeframe")
            params = run.get("parameters", {})
            resp.generations = params.get("generations")
            resp.population = params.get("population")
            resp.completed_at = run.get("completed_at")
            # Count strategies from results
            strategies = _load_discovery_strategies(state, info.run_id)
            resp.strategies_found = len(strategies) if strategies else None
    return resp
```

**Step 3: Update `list_discovery_jobs` endpoint to pass state**

Change signature to also accept `state: StateMgr` and pass it through:

```python
@router.get("/jobs", response_model=list[DiscoveryJobResponse])
async def list_discovery_jobs(jobs: JobMgr, state: StateMgr) -> list[DiscoveryJobResponse]:
    all_jobs = jobs.list_all_jobs(job_type="discovery")
    return [_job_info_to_discovery_response(j, state) for j in all_jobs]
```

Also update `launch_discovery` to pass state in its final return.

**Step 4: Regenerate OpenAPI + frontend types**

Run:
```bash
cd /Users/verebelyin/projects/vibe-quant && .venv/bin/python -c "from vibe_quant.api.app import create_app; import json; app = create_app(); from fastapi.openapi.utils import get_openapi; spec = get_openapi(title=app.title, version=app.version, routes=app.routes); open('frontend/openapi.json','w').write(json.dumps(spec, indent=2))"
cd frontend && pnpm orval
```

**Step 5: Commit**

```bash
git add vibe_quant/api/schemas/discovery.py vibe_quant/api/routers/discovery.py frontend/openapi.json frontend/src/api/generated/
git commit -m "feat: enrich discovery jobs endpoint with symbols/timeframe/params"
```

---

### Task 2: Add sidebar tree-style sub-navigation for Discovery

**Files:**
- Modify: `frontend/src/components/layout/Sidebar.tsx`

**Step 1: Update imports to include sub-menu components**

Add `SidebarMenuSub`, `SidebarMenuSubButton`, `SidebarMenuSubItem` to the imports from `@/components/ui/sidebar`.

**Step 2: Restructure nav data model to support children**

```typescript
interface NavItem {
  label: string;
  path: string;
  icon: React.ReactNode;
  children?: { label: string; path: string }[];
}
```

**Step 3: Add children to Discovery nav item**

```typescript
{
  label: "Discovery",
  path: "/discovery",
  icon: (/* existing search icon SVG */),
  children: [
    { label: "Results", path: "/discovery/results" },
  ],
},
```

**Step 4: Render sub-items using SidebarMenuSub**

In the map over `group.items`, after the `SidebarMenuButton`, conditionally render:

```tsx
{item.children && (
  <SidebarMenuSub>
    {item.children.map((child) => {
      const isChildActive =
        location.pathname === child.path;
      return (
        <SidebarMenuSubItem key={child.path}>
          <SidebarMenuSubButton
            asChild
            isActive={isChildActive}
          >
            <Link to={child.path}>
              <span>{child.label}</span>
            </Link>
          </SidebarMenuSubButton>
        </SidebarMenuSubItem>
      );
    })}
  </SidebarMenuSub>
)}
```

Also update the `isActive` check for parent items to use `startsWith` so Discovery stays highlighted when on `/discovery/results`:

```typescript
const isActive = item.children
  ? location.pathname.startsWith(item.path)
  : location.pathname === item.path;
```

**Step 5: Commit**

```bash
git add frontend/src/components/layout/Sidebar.tsx
git commit -m "feat: add tree-style Discovery > Results sub-nav in sidebar"
```

---

### Task 3: Create Discovery Results route and page component

**Files:**
- Create: `frontend/src/routes/discovery-results.tsx`
- Modify: `frontend/src/app.tsx:28-30,81-91,190-203`

**Step 1: Create the route page file**

Create `frontend/src/routes/discovery-results.tsx` with the `DiscoveryResultsPage` component (main table implementation in Task 4, this is just the route shell):

```tsx
export { DiscoveryResultsPage } from "@/components/discovery/DiscoveryResultsPage";
```

**Step 2: Register the route in app.tsx**

Add lazy import:
```typescript
const DiscoveryResultsPage = lazy(() =>
  import("./routes/discovery-results").then((m) => ({ default: m.DiscoveryResultsPage })),
);
```

Add route definition:
```typescript
const discoveryResultsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/discovery/results",
  component: function DiscoveryResultsRouteComponent() {
    return (
      <SuspensePage>
        <DiscoveryResultsPage />
      </SuspensePage>
    );
  },
});
```

Add to route tree (BEFORE discoveryRoute so it matches first):
```typescript
const routeTree = rootRoute.addChildren([
  indexRoute,
  strategiesRoute,
  strategyEditRoute,
  discoveryResultsRoute,  // must be before discoveryRoute
  discoveryRoute,
  backtestRoute,
  ...
]);
```

**Step 3: Commit**

```bash
git add frontend/src/routes/discovery-results.tsx frontend/src/app.tsx
git commit -m "feat: register /discovery/results route"
```

---

### Task 4: Build the Discovery Results table page component

**Files:**
- Create: `frontend/src/components/discovery/DiscoveryResultsPage.tsx`

**Step 1: Create the main page component**

Build a TanStack React Table following the exact pattern from `results.tsx`:

```tsx
import { useCallback, useMemo, useState } from "react";
import {
  type ColumnDef,
  type SortingState,
  type ExpandedState,
  flexRender,
  getCoreRowModel,
  getExpandedRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { ArrowUpDown, Loader2, Square } from "lucide-react";
import { toast } from "sonner";
import {
  getListDiscoveryJobsApiDiscoveryJobsGetQueryKey,
  useKillDiscoveryJobApiDiscoveryJobsRunIdDelete,
  useListDiscoveryJobsApiDiscoveryJobsGet,
} from "@/api/generated/discovery/discovery";
import type { DiscoveryJobResponse } from "@/api/generated/models";
import { queryClient } from "@/api/query-client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui";
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
import { DiscoveryResults } from "./DiscoveryResults";
```

Key features:
- **Columns:** Run ID, Status (badge w/ pulse for running), Symbols, Timeframe, Generations, Population, Best Fitness (from progress), Strategies Found, Duration, Started At, Actions (Kill button)
- **Filters:** Status dropdown (All / Running / Completed / Failed / Killed)
- **Sorting:** Default by started_at desc, all numeric columns sortable
- **Expandable rows:** Click to expand, show `DiscoveryResults` inline for that run_id
- **Live updates:** `refetchInterval: 5000` when any job is running
- **Kill button:** Inline on running rows, uses existing kill mutation

Status badge styles (reuse from DiscoveryJobList):
```typescript
const STATUS_STYLES: Record<string, string> = {
  running: "bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300",
  completed: "bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300",
  failed: "bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300",
  cancelled: "bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-300",
};
```

Duration formatter:
```typescript
function formatDuration(start: string | null, end: string | null, status: string): string {
  if (!start) return "-";
  const s = new Date(start).getTime();
  const e = status === "running" ? Date.now() : end ? new Date(end).getTime() : s;
  const secs = Math.round((e - s) / 1000);
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.round(secs / 60)}m`;
  const h = Math.floor(secs / 3600);
  const m = Math.round((secs % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}
```

Progress-derived fields helper:
```typescript
function bestFitnessFromProgress(job: DiscoveryJobResponse): string {
  const p = job.progress as Record<string, unknown> | null;
  if (!p) return "-";
  const f = p.best_fitness ?? p.fitness;
  return f != null ? Number(f).toFixed(4) : "-";
}
```

SortHeader component — reuse the exact same pattern from results.tsx:
```tsx
function SortHeader({ column, children }: { column: { toggleSorting: ... }; children: React.ReactNode }) {
  return (
    <button type="button" className="flex items-center gap-1 hover:text-foreground"
      onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}>
      {children}
      <ArrowUpDown className="h-3 w-3" />
    </button>
  );
}
```

Column definitions:
```typescript
const columns: ColumnDef<DiscoveryJobResponse>[] = [
  {
    accessorKey: "run_id",
    header: ({ column }) => <SortHeader column={column}>Run</SortHeader>,
    cell: ({ row }) => <span className="font-mono text-xs">#{row.original.run_id}</span>,
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: ({ row }) => {
      const s = row.original.status.toLowerCase();
      return (
        <Badge variant="outline"
          className={cn("border-transparent text-[10px] capitalize",
            STATUS_STYLES[s] ?? "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
            s === "running" && "animate-pulse"
          )}>
          {row.original.status}
        </Badge>
      );
    },
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
    cell: ({ row }) => (
      <span className="font-mono text-xs">{row.original.timeframe ?? "-"}</span>
    ),
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
        <span className="font-mono text-xs">
          {isRunning ? `${current}/${max}` : String(max)}
        </span>
      );
    },
  },
  {
    accessorKey: "population",
    header: "Pop",
    cell: ({ row }) => (
      <span className="font-mono text-xs">{row.original.population ?? "-"}</span>
    ),
  },
  {
    id: "best_fitness",
    header: ({ column }) => <SortHeader column={column}>Best Fitness</SortHeader>,
    accessorFn: (row) => {
      const p = row.progress as Record<string, unknown> | null;
      if (!p) return null;
      return Number(p.best_fitness ?? p.fitness ?? null);
    },
    cell: ({ row }) => (
      <span className="font-mono text-xs font-medium">
        {bestFitnessFromProgress(row.original)}
      </span>
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
    header: ({ column }) => <SortHeader column={column}>Started</SortHeader>,
    cell: ({ row }) => (
      <span className="text-xs text-muted-foreground">
        {row.original.started_at
          ? new Date(row.original.started_at).toLocaleDateString("en-US", {
              month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
            })
          : "-"}
      </span>
    ),
  },
  {
    id: "actions",
    header: "",
    cell: ({ row, table }) => {
      const isRunning = row.original.status.toLowerCase() === "running";
      if (!isRunning) return null;
      const meta = table.options.meta as { onKill: (id: number) => void; isKilling: boolean } | undefined;
      return (
        <Button type="button" variant="destructive" size="xs"
          disabled={meta?.isKilling}
          onClick={(e) => { e.stopPropagation(); meta?.onKill(row.original.run_id); }}>
          <Square className="mr-1 h-3 w-3" /> Kill
        </Button>
      );
    },
    enableSorting: false,
  },
];
```

Main component:
```tsx
export function DiscoveryResultsPage() {
  const ALL = "__all__";
  const [statusFilter, setStatusFilter] = useState<string>(ALL);
  const [sorting, setSorting] = useState<SortingState>([{ id: "started_at", desc: true }]);
  const [expanded, setExpanded] = useState<ExpandedState>({});

  const { data: jobsResp, isLoading } = useListDiscoveryJobsApiDiscoveryJobsGet({
    query: {
      refetchInterval: (query) => {
        const jobs = query.state.data?.status === 200 ? query.state.data.data : [];
        return jobs.some((j) => j.status.toLowerCase() === "running") ? 5000 : false;
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

  const handleKill = useCallback((runId: number) => {
    killMutation.mutate({ runId });
  }, [killMutation]);

  const table = useReactTable({
    data: filteredJobs,
    columns,
    state: { sorting, expanded },
    onSortingChange: setSorting,
    onExpandedChange: setExpanded,
    getRowCanExpand: (row) => row.original.status.toLowerCase() === "completed",
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
    meta: { onKill: handleKill, isKilling: killMutation.isPending },
  });

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold text-foreground">Discovery Results</h1>
      </div>

      {/* Filters */}
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

      {/* Table */}
      {isLoading ? (
        <TableSkeleton />
      ) : filteredJobs.length === 0 ? (
        <EmptyState
          title="No discovery runs"
          description="Launch a discovery from the Discovery page to see results here."
        />
      ) : (
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
                <>
                  <TableRow
                    key={row.id}
                    className={cn(
                      row.getCanExpand() && "cursor-pointer",
                      row.getIsExpanded() && "bg-muted/30",
                    )}
                    onClick={() => row.getCanExpand() && row.toggleExpanded()}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <TableCell key={cell.id}>
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    ))}
                  </TableRow>
                  {row.getIsExpanded() && (
                    <TableRow key={`${row.id}-expanded`}>
                      <TableCell colSpan={columns.length} className="p-0">
                        <div className="border-t border-border bg-muted/10 p-4">
                          <DiscoveryResults runId={row.original.run_id} />
                        </div>
                      </TableCell>
                    </TableRow>
                  )}
                </>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}

function TableSkeleton() {
  return (
    <div className="space-y-2">
      <Skeleton className="h-10 w-full" />
      {Array.from({ length: 5 }).map((_, i) => (
        <Skeleton key={i} className="h-12 w-full" />
      ))}
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/discovery/DiscoveryResultsPage.tsx
git commit -m "feat: discovery results table page with sort/filter/expand/kill"
```

---

### Task 5: Simplify the main Discovery page

Remove the full job list and results from the main discovery page. Keep only config + a compact "running jobs" indicator.

**Files:**
- Modify: `frontend/src/routes/discovery.tsx`

**Step 1: Replace current layout**

```tsx
import { useMemo } from "react";
import { Link } from "@tanstack/react-router";
import { useListDiscoveryJobsApiDiscoveryJobsGet } from "@/api/generated/discovery/discovery";
import type { DiscoveryJobResponse } from "@/api/generated/models";
import { DiscoveryConfig } from "@/components/discovery/DiscoveryConfig";
import { Badge } from "@/components/ui/badge";

export function DiscoveryPage() {
  const { data: jobsResp } = useListDiscoveryJobsApiDiscoveryJobsGet({
    query: { refetchInterval: 10_000 },
  });

  const runningJobs: DiscoveryJobResponse[] = useMemo(() => {
    if (!jobsResp || jobsResp.status !== 200) return [];
    return jobsResp.data.filter((j) => j.status.toLowerCase() === "running");
  }, [jobsResp]);

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      {/* Running jobs indicator */}
      {runningJobs.length > 0 && (
        <div className="rounded-lg border border-blue-500/30 bg-blue-950/20 p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 animate-pulse rounded-full bg-blue-400" />
              <span className="text-sm font-medium text-foreground">
                {runningJobs.length} discovery {runningJobs.length === 1 ? "run" : "runs"} active
              </span>
            </div>
            <Link
              to="/discovery/results"
              className="text-xs text-blue-400 hover:text-blue-300 hover:underline"
            >
              View results →
            </Link>
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {runningJobs.map((job) => {
              const p = job.progress as Record<string, unknown> | null;
              const gen = p ? Number(p.generation ?? 0) : 0;
              const maxGen = p ? Number(p.max_generations ?? 0) : 0;
              return (
                <Badge key={job.run_id} variant="outline" className="font-mono text-[10px]">
                  #{job.run_id} — Gen {gen}/{maxGen || "?"}
                </Badge>
              );
            })}
          </div>
        </div>
      )}

      <DiscoveryConfig />
    </div>
  );
}
```

**Step 2: Commit**

```bash
git add frontend/src/routes/discovery.tsx
git commit -m "feat: simplify discovery page to config + running indicator"
```

---

### Task 6: Build check + test

**Step 1: Run TypeScript check**

```bash
cd frontend && pnpm tsc --noEmit
```

Fix any type errors.

**Step 2: Run frontend build**

```bash
cd frontend && pnpm build
```

Fix any build errors.

**Step 3: Run backend tests**

```bash
cd /Users/verebelyin/projects/vibe-quant && .venv/bin/pytest tests/unit/api/ -x -q
```

**Step 4: Commit any fixes**

```bash
git add -u && git commit -m "fix: type errors and build fixes"
```

---

### Task 7: Final cleanup + push

**Step 1: Remove unused imports from discovery.tsx**

The old discovery page imported `DiscoveryJobList`, `DiscoveryProgress`, `DiscoveryResults`, and `useState`. Remove those.

**Step 2: Verify no dead code**

Check that `DiscoveryJobList` is still used by `DiscoveryResultsPage` (it's not — results page has its own table). `DiscoveryJobList` component file can stay for now (not deleted, just unused by the page — future cleanup).

**Step 3: Final commit + push**

```bash
git add -u
git commit -m "chore: cleanup unused imports"
git push
```
