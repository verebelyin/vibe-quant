import { useEffect, useMemo, useRef, useState } from "react";
import type { StrategyResponse } from "@/api/generated/models";
import { useListStrategiesApiStrategiesGet } from "@/api/generated/strategies/strategies";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StrategyCard } from "@/components/ui/StrategyCard";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const STRATEGY_TYPES = [
  "momentum",
  "mean_reversion",
  "breakout",
  "trend_following",
  "arbitrage",
  "volatility",
] as const;

const SORT_OPTIONS = [
  { value: "name", label: "Name" },
  { value: "created_at", label: "Created" },
  { value: "updated_at", label: "Updated" },
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
  return (
    <div className="group relative">
      <StrategyCard
        name={strategy.name}
        description={strategy.description ?? undefined}
        strategyType={strategy.strategy_type ?? undefined}
        version={strategy.version}
        onClick={onSelect}
      />
      <Button
        variant="ghost"
        size="xs"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        className="absolute right-2 top-2 hidden text-destructive opacity-70 hover:opacity-100 group-hover:block"
      >
        Delete
      </Button>
    </div>
  );
}

export function StrategyList({ onSelect, onDelete }: StrategyListProps) {
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<SortKey>("updated_at");
  const timerRef = useRef<ReturnType<typeof setTimeout>>(null);

  // Debounce search input
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setDebouncedSearch(search), 300);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [search]);

  const query = useListStrategiesApiStrategiesGet();
  const data = query.data?.data;

  // Client-side filter + sort (API params only support active_only)
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
      <div className="space-y-4">
        <div className="flex gap-3">
          {(["a", "b", "c"] as const).map((id) => (
            <div key={id} className="h-10 w-48 animate-pulse rounded-lg bg-muted" />
          ))}
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {(["a", "b", "c", "d", "e", "f"] as const).map((id) => (
            <div key={id} className="h-32 animate-pulse rounded-lg bg-muted" />
          ))}
        </div>
      </div>
    );
  }

  if (query.isError) {
    return (
      <div className="rounded-lg border border-destructive bg-destructive/10 p-6 text-destructive">
        <p className="font-medium">Failed to load strategies</p>
        <p className="mt-1 text-sm opacity-80">
          {query.error instanceof Error ? query.error.message : "Unknown error"}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <Input
          type="text"
          placeholder="Search strategies..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="min-w-[220px] max-w-xs"
        />
        <Select value={typeFilter} onValueChange={setTypeFilter}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="All types" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All types</SelectItem>
            {STRATEGY_TYPES.map((t) => (
              <SelectItem key={t} value={t}>
                {t.replace(/_/g, " ")}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={sortBy} onValueChange={(v) => setSortBy(v as SortKey)}>
          <SelectTrigger className="w-[160px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SORT_OPTIONS.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                Sort: {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Grid */}
      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-lg border border-border py-16 text-muted-foreground">
          <p className="text-lg font-medium">No strategies found</p>
          <p className="mt-1 text-sm">
            {debouncedSearch || (typeFilter && typeFilter !== "all")
              ? "Try adjusting your search or filters."
              : "Create your first strategy to get started."}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
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
