import { Search, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { StrategyResponse } from "@/api/generated/models";
import { useListStrategiesApiStrategiesGet } from "@/api/generated/strategies/strategies";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { StrategyCard } from "@/components/ui/StrategyCard";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const STRATEGY_TYPES = [
  { value: "momentum", label: "Momentum" },
  { value: "mean_reversion", label: "Mean Reversion" },
  { value: "breakout", label: "Breakout" },
  { value: "trend_following", label: "Trend Following" },
  { value: "arbitrage", label: "Arbitrage" },
  { value: "volatility", label: "Volatility" },
] as const;

const SORT_OPTIONS = [
  { value: "updated_at", label: "Last Updated" },
  { value: "created_at", label: "Created" },
  { value: "name", label: "Name" },
  { value: "version", label: "Version" },
] as const;

type SortKey = (typeof SORT_OPTIONS)[number]["value"];

interface StrategyListProps {
  onSelect: (strategy: StrategyResponse) => void;
  onDelete: (strategy: StrategyResponse) => void;
}

function StrategyCardWithDelete({
  strategy,
  onSelect,
  onDelete,
}: {
  strategy: StrategyResponse;
  onSelect: () => void;
  onDelete: () => void;
}) {
  // Extract detail fields from opaque dsl_config
  const dsl = (strategy.dsl_config ?? {}) as Record<string, unknown>;
  const timeframe = typeof dsl.timeframe === "string" ? dsl.timeframe : undefined;
  const symbols = Array.isArray(dsl.symbols) ? (dsl.symbols as string[]) : [];
  const indicators = Array.isArray(dsl.indicators) ? dsl.indicators : [];

  return (
    <div className="group relative">
      <StrategyCard
        name={strategy.name}
        description={strategy.description ?? undefined}
        strategyType={strategy.strategy_type ?? undefined}
        version={strategy.version}
        isActive={strategy.is_active}
        timeframe={timeframe}
        symbols={symbols}
        indicatorCount={indicators.length}
        updatedAt={strategy.updated_at}
        onClick={onSelect}
      />
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        className="absolute bottom-2.5 right-2.5 hidden group-hover:flex items-center justify-center w-6 h-6 rounded-md text-white/30 hover:text-destructive hover:bg-destructive/15 transition-colors duration-150"
        aria-label="Delete strategy"
      >
        <Trash2 className="w-3 h-3" />
      </button>
    </div>
  );
}

export function StrategyList({ onSelect, onDelete }: StrategyListProps) {
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<SortKey>("updated_at");
  const timerRef = useRef<ReturnType<typeof setTimeout>>(null);

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setDebouncedSearch(search), 300);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [search]);

  const query = useListStrategiesApiStrategiesGet();
  const data = query.data?.data;

  const filtered = useMemo(() => {
    if (!data?.strategies) return [];
    let items = [...data.strategies];

    if (debouncedSearch) {
      const q = debouncedSearch.toLowerCase();
      items = items.filter(
        (s) => s.name.toLowerCase().includes(q) || s.description?.toLowerCase().includes(q),
      );
    }

    if (typeFilter && typeFilter !== "all") {
      items = items.filter((s) => s.strategy_type === typeFilter);
    }

    items.sort((a, b) => {
      switch (sortBy) {
        case "name":
          return a.name.localeCompare(b.name);
        case "created_at":
          return b.created_at.localeCompare(a.created_at);
        case "updated_at":
          return b.updated_at.localeCompare(a.updated_at);
        case "version":
          return b.version - a.version;
        default:
          return 0;
      }
    });

    return items;
  }, [data?.strategies, debouncedSearch, typeFilter, sortBy]);

  if (query.isLoading) {
    return (
      <div className="space-y-5">
        <div className="flex gap-3">
          <div className="h-9 flex-1 max-w-xs animate-pulse rounded-lg bg-muted" />
          <div className="h-9 w-36 animate-pulse rounded-lg bg-muted" />
          <div className="h-9 w-36 animate-pulse rounded-lg bg-muted" />
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {(["a", "b", "c", "d", "e", "f"] as const).map((id) => (
            <div key={id} className="h-[172px] animate-pulse rounded-lg bg-muted" />
          ))}
        </div>
      </div>
    );
  }

  if (query.isError) {
    return (
      <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-6 text-destructive">
        <p className="font-medium text-sm">Failed to load strategies</p>
        <p className="mt-1 text-xs opacity-70">
          {query.error instanceof Error ? query.error.message : "Unknown error"}
        </p>
      </div>
    );
  }

  const totalCount = data?.strategies?.length ?? 0;
  const showingCount = filtered.length;

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Search */}
        <div className="relative min-w-[180px] max-w-[280px] flex-1">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3 w-3 text-white/25 pointer-events-none" />
          <input
            type="text"
            placeholder="Search…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className={cn(
              "w-full pl-7 pr-3 h-8 text-xs rounded-lg",
              "bg-white/[0.04] border border-white/[0.07] text-foreground placeholder:text-white/25",
              "focus:outline-none focus:ring-1 focus:ring-primary/40 focus:border-primary/25",
              "transition-colors duration-150",
            )}
          />
        </div>

        {/* Type filter */}
        <Select value={typeFilter} onValueChange={setTypeFilter}>
          <SelectTrigger className="w-[136px] h-8 text-xs bg-white/[0.04] border-white/[0.07]">
            <SelectValue placeholder="All types" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All types</SelectItem>
            {STRATEGY_TYPES.map((t) => (
              <SelectItem key={t.value} value={t.value}>
                {t.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Sort */}
        <Select value={sortBy} onValueChange={(v) => setSortBy(v as SortKey)}>
          <SelectTrigger className="w-[136px] h-8 text-xs bg-white/[0.04] border-white/[0.07]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SORT_OPTIONS.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Count */}
        {totalCount > 0 && (
          <span className="ml-auto font-mono text-[10px] text-white/25 tabular-nums">
            {showingCount === totalCount ? `${totalCount}` : `${showingCount}/${totalCount}`}
            {" "}strat{totalCount !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Grid */}
      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-lg border border-white/[0.06] bg-muted/10 py-20 text-center">
          <div className="mb-3 text-3xl opacity-30">◈</div>
          <p className="text-sm font-medium text-foreground/70">No strategies found</p>
          <p className="mt-1 text-xs text-muted-foreground/50">
            {debouncedSearch || (typeFilter && typeFilter !== "all")
              ? "Try adjusting your search or filter."
              : "Create your first strategy to get started."}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filtered.map((strategy) => (
            <StrategyCardWithDelete
              key={strategy.id}
              strategy={strategy}
              onSelect={() => onSelect(strategy)}
              onDelete={() => onDelete(strategy)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
