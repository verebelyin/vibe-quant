import { useMemo, useState } from "react";
import { toast } from "sonner";
import {
  useExportDiscoveredStrategyApiDiscoveryResultsRunIdExportStrategyIndexPost,
  useGetDiscoveryResultsApiDiscoveryResultsRunIdGet,
  useGetLatestResultsApiDiscoveryResultsLatestGet,
} from "@/api/generated/discovery/discovery";
import type { DiscoveryResultResponseStrategiesItem } from "@/api/generated/models";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

/** Typed accessor for untyped strategy dict. */
function num(s: DiscoveryResultResponseStrategiesItem, key: string): number | null {
  const v = s[key];
  if (v == null) return null;
  return Number(v);
}

function str(s: DiscoveryResultResponseStrategiesItem, key: string): string {
  const v = s[key];
  if (v == null) return "--";
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.join(", ");
  return String(v);
}

function fmtNum(v: number | null, decimals = 4): string {
  if (v == null) return "--";
  return v.toFixed(decimals);
}

function fmtPct(v: number | null, decimals = 1): string {
  if (v == null) return "--";
  return `${v.toFixed(decimals)}%`;
}

interface SummaryStats {
  totalStrategies: number;
  bestFitness: number | null;
  avgFitness: number | null;
  generationsCompleted: number | null;
}

function computeSummary(strategies: DiscoveryResultResponseStrategiesItem[]): SummaryStats {
  if (strategies.length === 0) {
    return { totalStrategies: 0, bestFitness: null, avgFitness: null, generationsCompleted: null };
  }

  const fitnesses = strategies
    .map((s) => num(s, "fitness") ?? num(s, "adjusted_score") ?? num(s, "score"))
    .filter((f): f is number => f != null);

  const bestFitness = fitnesses.length > 0 ? Math.max(...fitnesses) : null;
  const avgFitness =
    fitnesses.length > 0 ? fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length : null;

  // Try to find generation info from first strategy
  const gen = num(strategies[0], "generations_completed") ?? num(strategies[0], "generation");

  return {
    totalStrategies: strategies.length,
    bestFitness,
    avgFitness,
    generationsCompleted: gen,
  };
}

const RANK_STYLES: Record<number, string> = {
  1: "bg-amber-50 dark:bg-amber-950/30 font-semibold",
  2: "bg-slate-50 dark:bg-slate-900/30 font-semibold",
  3: "bg-orange-50 dark:bg-orange-950/20 font-medium",
};

interface DiscoveryResultsProps {
  /** Specific run ID to fetch results for. If null, fetches latest. */
  runId: number | null;
}

export function DiscoveryResults({ runId }: DiscoveryResultsProps) {
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const [exportedIndices, setExportedIndices] = useState<Set<number>>(new Set());

  // Fetch results for specific run
  const runResults = useGetDiscoveryResultsApiDiscoveryResultsRunIdGet(runId ?? 0, {
    query: { enabled: runId != null },
  });

  // Fallback: fetch latest
  const latestResults = useGetLatestResultsApiDiscoveryResultsLatestGet({
    query: { enabled: runId == null },
  });

  const exportMutation = useExportDiscoveredStrategyApiDiscoveryResultsRunIdExportStrategyIndexPost(
    {
      mutation: {
        onSuccess: (_data, variables) => {
          toast.success("Strategy exported to library");
          setExportedIndices((prev) => new Set(prev).add(variables.strategyIndex));
        },
        onError: (err: unknown) => {
          const message = err instanceof Error ? err.message : "Export failed";
          toast.error("Export failed", { description: message });
        },
      },
    },
  );

  const activeResp = runId != null ? runResults : latestResults;
  const isLoading = activeResp.isLoading;

  const strategies: DiscoveryResultResponseStrategiesItem[] = useMemo(() => {
    if (!activeResp.data) return [];
    if (activeResp.data.status === 200) return activeResp.data.data.strategies;
    return [];
  }, [activeResp.data]);

  // Sort by fitness descending
  const sorted = useMemo(() => {
    return [...strategies].sort((a, b) => {
      const fa = num(a, "fitness") ?? num(a, "adjusted_score") ?? num(a, "score") ?? -Infinity;
      const fb = num(b, "fitness") ?? num(b, "adjusted_score") ?? num(b, "score") ?? -Infinity;
      return fb - fa;
    });
  }, [strategies]);

  const summary = useMemo(() => computeSummary(sorted), [sorted]);

  function handleExport(strategyIndex: number) {
    if (runId == null) return;
    exportMutation.mutate({ runId, strategyIndex });
  }

  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <p className="text-xs text-muted-foreground">Loading results...</p>
      </div>
    );
  }

  if (sorted.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
          Discovery Results
        </h3>
        <p className="mt-2 text-xs text-muted-foreground">
          {runId != null
            ? "No strategies found for this run."
            : "No discovery results available yet. Run a discovery job first."}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
        Discovery Results
        {runId != null && (
          <Badge variant="outline" className="ml-2 font-mono text-[10px]">
            Run #{runId}
          </Badge>
        )}
      </h3>

      {/* Summary stats */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <SummaryStat label="Strategies Evaluated" value={String(summary.totalStrategies)} />
        <SummaryStat label="Best Fitness" value={fmtNum(summary.bestFitness)} />
        <SummaryStat label="Avg Fitness" value={fmtNum(summary.avgFitness)} />
        <SummaryStat
          label="Generations"
          value={summary.generationsCompleted != null ? String(summary.generationsCompleted) : "--"}
        />
      </div>

      {/* Strategies table */}
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-12 text-xs">Rank</TableHead>
              <TableHead className="text-xs">Fitness</TableHead>
              <TableHead className="text-xs">Sharpe</TableHead>
              <TableHead className="text-xs">Return</TableHead>
              <TableHead className="text-xs">Max DD</TableHead>
              <TableHead className="text-xs">Trades</TableHead>
              <TableHead className="text-xs">PF</TableHead>
              <TableHead className="text-xs">Indicators</TableHead>
              <TableHead className="w-20 text-xs">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sorted.map((s, idx) => {
              const rank = idx + 1;
              const isExpanded = expandedRow === idx;
              const isExported = exportedIndices.has(idx);
              const fitness = num(s, "fitness") ?? num(s, "adjusted_score") ?? num(s, "score");
              const sharpe = num(s, "sharpe") ?? num(s, "sharpe_ratio");
              const returnPct = num(s, "return_pct") ?? num(s, "total_return") ?? num(s, "return");
              const maxDD = num(s, "max_drawdown") ?? num(s, "max_dd");
              const winRate = num(s, "win_rate");
              const trades = num(s, "trades") ?? num(s, "total_trades");
              const pf = num(s, "pf") ?? num(s, "profit_factor");
              // Extract indicator info from DSL config
              const dsl = s.dsl as Record<string, unknown> | undefined;
              const dslIndicators = dsl?.indicators;
              const indicators = dslIndicators
                ? Object.keys(dslIndicators as Record<string, unknown>).map(k => k.split("_")[0].toUpperCase()).filter((v, i, a) => a.indexOf(v) === i).join(", ")
                : str(s, "indicators");
              const conditions =
                num(s, "conditions") ?? num(s, "genes") ?? num(s, "total_conditions");

              // Build a stable key from strategy identity fields
              const stableKey = `${str(s, "name") ?? ""}-${fitness ?? ""}-${sharpe ?? ""}-${idx}`;

              return (
                <TableRow
                  key={stableKey}
                  className={cn(
                    "cursor-pointer transition-colors hover:bg-muted/50",
                    RANK_STYLES[rank],
                    isExpanded && "bg-muted/30",
                  )}
                  onClick={() => setExpandedRow(isExpanded ? null : idx)}
                >
                  <TableCell className="font-mono text-xs text-foreground">
                    {rank <= 3 ? (
                      <Badge
                        variant="outline"
                        className={cn(
                          "border-transparent text-[10px]",
                          rank === 1 &&
                            "bg-amber-200 text-amber-900 dark:bg-amber-800 dark:text-amber-100",
                          rank === 2 &&
                            "bg-slate-200 text-slate-900 dark:bg-slate-700 dark:text-slate-100",
                          rank === 3 &&
                            "bg-orange-200 text-orange-900 dark:bg-orange-800 dark:text-orange-100",
                        )}
                      >
                        #{rank}
                      </Badge>
                    ) : (
                      <span>#{rank}</span>
                    )}
                  </TableCell>
                  <TableCell className="font-mono text-xs text-foreground">
                    {fmtNum(fitness)}
                  </TableCell>
                  <TableCell className="font-mono text-xs text-foreground">
                    {fmtNum(sharpe, 2)}
                  </TableCell>
                  <TableCell
                    className={cn(
                      "font-mono text-xs",
                      returnPct != null && returnPct >= 0 && "text-green-600",
                      returnPct != null && returnPct < 0 && "text-red-600",
                    )}
                  >
                    {fmtPct(returnPct)}
                  </TableCell>
                  <TableCell
                    className={cn(
                      "font-mono text-xs",
                      maxDD != null && maxDD > 20 && "text-red-600",
                      maxDD != null && maxDD <= 20 && "text-foreground",
                    )}
                  >
                    {fmtPct(maxDD)}
                  </TableCell>
                  <TableCell className="font-mono text-xs text-foreground">
                    {trades != null ? String(trades) : "--"}
                  </TableCell>
                  <TableCell className="font-mono text-xs text-foreground">
                    {fmtNum(pf, 2)}
                  </TableCell>
                  <TableCell className="max-w-40 truncate text-xs text-foreground" title={indicators}>
                    {indicators}
                  </TableCell>
                  <TableCell className="text-xs" onClick={(e) => e.stopPropagation()}>
                    {runId != null && (
                      <Button
                        type="button"
                        variant={isExported ? "outline" : "secondary"}
                        size="xs"
                        disabled={isExported || exportMutation.isPending}
                        onClick={() => handleExport(idx)}
                      >
                        {isExported ? "Exported" : "Export"}
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>

      {/* Expanded row detail */}
      {expandedRow != null && sorted[expandedRow] && (
        <StrategyDetail strategy={sorted[expandedRow]} rank={expandedRow + 1} />
      )}
    </div>
  );
}

function SummaryStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p className="font-mono text-sm font-medium text-foreground">{value}</p>
    </div>
  );
}

function StrategyDetail({
  strategy,
  rank,
}: {
  strategy: DiscoveryResultResponseStrategiesItem;
  rank: number;
}) {
  // Build a readable summary of the strategy config
  const entries = Object.entries(strategy).filter(
    ([key]) =>
      ![
        "fitness",
        "adjusted_score",
        "score",
        "sharpe",
        "sharpe_ratio",
        "return_pct",
        "total_return",
        "return",
        "max_drawdown",
        "max_dd",
        "win_rate",
        "indicators",
        "conditions",
        "genes",
        "total_conditions",
        "rank",
        "generations_completed",
        "generation",
      ].includes(key),
  );

  return (
    <div className="rounded-md border border-border bg-background p-4">
      <h4 className="mb-2 text-xs font-semibold text-muted-foreground">
        Strategy #{rank} -- Full Config
      </h4>
      {entries.length === 0 ? (
        <p className="text-xs italic text-muted-foreground">No additional config data.</p>
      ) : (
        <div className="max-h-64 overflow-y-auto">
          <pre className="whitespace-pre-wrap font-mono text-xs text-foreground">
            {JSON.stringify(Object.fromEntries(entries), null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
