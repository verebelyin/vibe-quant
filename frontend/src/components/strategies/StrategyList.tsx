import { useEffect, useMemo, useRef, useState } from "react";
import type { StrategyResponse } from "@/api/generated/models";
import { useListStrategiesApiStrategiesGet } from "@/api/generated/strategies/strategies";
import { StrategyCard } from "@/components/ui/StrategyCard";

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
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        className="absolute right-2 top-2 hidden cursor-pointer rounded p-1 text-xs opacity-70 transition-opacity hover:opacity-100 group-hover:block"
        style={{ color: "hsl(0 84% 60%)" }}
        title="Delete strategy"
      >
        Delete
      </button>
    </div>
  );
}

export function StrategyList({ onSelect, onDelete }: StrategyListProps) {
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("");
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

    if (typeFilter) {
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
            <div
              key={id}
              className="h-10 w-48 animate-pulse rounded-lg"
              style={{ backgroundColor: "hsl(var(--muted))" }}
            />
          ))}
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {(["a", "b", "c", "d", "e", "f"] as const).map((id) => (
            <div
              key={id}
              className="h-32 animate-pulse rounded-lg"
              style={{ backgroundColor: "hsl(var(--muted))" }}
            />
          ))}
        </div>
      </div>
    );
  }

  if (query.isError) {
    return (
      <div
        className="rounded-lg border p-6"
        style={{
          borderColor: "hsl(0 84% 60%)",
          backgroundColor: "hsl(0 84% 60% / 0.1)",
          color: "hsl(0 84% 60%)",
        }}
      >
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
        <input
          type="text"
          placeholder="Search strategies..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="h-10 rounded-lg border px-3 text-sm outline-none focus:ring-2"
          style={{
            backgroundColor: "hsl(var(--card))",
            borderColor: "hsl(var(--border))",
            color: "hsl(var(--foreground))",
            minWidth: "220px",
          }}
        />
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="h-10 rounded-lg border px-3 text-sm outline-none"
          style={{
            backgroundColor: "hsl(var(--card))",
            borderColor: "hsl(var(--border))",
            color: "hsl(var(--foreground))",
          }}
        >
          <option value="">All types</option>
          {STRATEGY_TYPES.map((t) => (
            <option key={t} value={t}>
              {t.replace(/_/g, " ")}
            </option>
          ))}
        </select>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as SortKey)}
          className="h-10 rounded-lg border px-3 text-sm outline-none"
          style={{
            backgroundColor: "hsl(var(--card))",
            borderColor: "hsl(var(--border))",
            color: "hsl(var(--foreground))",
          }}
        >
          {SORT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              Sort: {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Grid */}
      {filtered.length === 0 ? (
        <div
          className="flex flex-col items-center justify-center rounded-lg border py-16"
          style={{
            borderColor: "hsl(var(--border))",
            color: "hsl(var(--muted-foreground))",
          }}
        >
          <p className="text-lg font-medium">No strategies found</p>
          <p className="mt-1 text-sm">
            {debouncedSearch || typeFilter
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
