import { Search, Trash2 } from "lucide-react";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { StrategyResponse } from "@/api/generated/models";
import { useListStrategiesApiStrategiesGet } from "@/api/generated/strategies/strategies";
import { cn } from "@/lib/utils";
import { StrategyCard, TYPE_META } from "@/components/ui/StrategyCard";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const SORT_OPTIONS = [
  { value: "updated_at", label: "Last Updated" },
  { value: "created_at", label: "Created"      },
  { value: "name",       label: "Name"         },
  { value: "version",    label: "Version"      },
] as const;

type SortKey = (typeof SORT_OPTIONS)[number]["value"];

/** Derive a short display label from a strategy_type value */
function typeShortLabel(type: string): string {
  const meta = TYPE_META[type.toLowerCase()];
  if (meta) return meta.label.length <= 5 ? meta.label : meta.label.slice(0, 4).toUpperCase();
  return type.replace(/_/g, " ").toUpperCase().slice(0, 5);
}

interface StrategyListProps {
  onSelect: (s: StrategyResponse) => void;
  onDelete: (s: StrategyResponse) => void;
}

const StrategyCardWithDelete = memo(function StrategyCardWithDelete({
  strategy,
  index,
  onSelect,
  onDelete,
}: {
  strategy: StrategyResponse;
  index: number;
  onSelect: () => void;
  onDelete: () => void;
}) {
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
        index={index}
        onClick={onSelect}
      />
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        className="absolute bottom-2.5 right-2.5 hidden group-hover:flex cursor-pointer items-center justify-center w-6 h-6 rounded-sm text-white/20 hover:text-red-400 hover:bg-red-500/10 transition-colors duration-100"
        aria-label="Delete strategy"
      >
        <Trash2 className="w-3 h-3" />
      </button>
    </div>
  );
});

function MemoizedCardWrapper({
  strategy,
  index,
  onSelect,
  onDelete,
}: {
  strategy: StrategyResponse;
  index: number;
  onSelect: (s: StrategyResponse) => void;
  onDelete: (s: StrategyResponse) => void;
}) {
  const handleSelect = useCallback(() => onSelect(strategy), [onSelect, strategy]);
  const handleDelete = useCallback(() => onDelete(strategy), [onDelete, strategy]);
  return (
    <StrategyCardWithDelete
      strategy={strategy}
      index={index}
      onSelect={handleSelect}
      onDelete={handleDelete}
    />
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
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [search]);

  const query = useListStrategiesApiStrategiesGet();
  const data = query.data?.data;

  // Derive unique types actually present in the data
  const presentTypes = useMemo(() => {
    if (!data?.strategies) return [];
    const seen = new Set<string>();
    for (const s of data.strategies) {
      if (s.strategy_type) seen.add(s.strategy_type.toLowerCase());
    }
    return Array.from(seen).sort();
  }, [data?.strategies]);

  const filtered = useMemo(() => {
    if (!data?.strategies) return [];
    let items = [...data.strategies];
    if (debouncedSearch) {
      const q = debouncedSearch.toLowerCase();
      items = items.filter(s => s.name.toLowerCase().includes(q) || s.description?.toLowerCase().includes(q));
    }
    if (typeFilter !== "all") {
      items = items.filter(s => (s.strategy_type?.toLowerCase() ?? "") === typeFilter);
    }
    items.sort((a, b) => {
      switch (sortBy) {
        case "name":       return a.name.localeCompare(b.name);
        case "created_at": return b.created_at.localeCompare(a.created_at);
        case "updated_at": return b.updated_at.localeCompare(a.updated_at);
        case "version":    return b.version - a.version;
        default:           return 0;
      }
    });
    return items;
  }, [data?.strategies, debouncedSearch, typeFilter, sortBy]);

  if (query.isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex gap-2">
          {(["a","b","c","d"] as const).map(id => (
            <div key={id} className="h-8 w-16 animate-pulse rounded-sm bg-white/[0.05]" />
          ))}
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {(["a","b","c","d","e","f","g","h"] as const).map(id => (
            <div key={id} className="h-[172px] animate-pulse rounded-sm bg-white/[0.04]" />
          ))}
        </div>
      </div>
    );
  }

  if (query.isError) {
    return (
      <div className="rounded-sm border border-red-500/20 bg-red-500/5 p-5">
        <p className="font-mono text-xs text-red-400/80">ERR — Failed to load strategies</p>
        <p className="font-mono text-[10px] text-red-400/40 mt-1">
          {query.error instanceof Error ? query.error.message : "Unknown error"}
        </p>
      </div>
    );
  }

  const total = data?.strategies?.length ?? 0;
  const shown = filtered.length;

  return (
    <div className="space-y-4">

      {/* ── Control bar ─────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3">

        {/* Search */}
        <div className="relative min-w-[160px] max-w-[240px] flex-1">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3 w-3 text-white/20 pointer-events-none" />
          <input
            type="text"
            placeholder="Search…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className={cn(
              "w-full pl-7 pr-3 h-8 rounded-sm font-mono text-[11px] cursor-text",
              "bg-white/[0.04] border border-white/[0.07] text-white/70 placeholder:text-white/20",
              "focus:outline-none focus:border-white/20 transition-colors duration-150",
            )}
          />
        </div>

        {/* ── Type tabs — built from actual data ────────────── */}
        <div className="flex items-center gap-[3px]">
          <button
            type="button"
            onClick={() => setTypeFilter("all")}
            className={cn(
              "cursor-pointer h-8 px-3 rounded-sm font-mono text-[10px] font-bold tracking-[0.15em] uppercase transition-all duration-100",
              typeFilter === "all"
                ? "bg-white/[0.09] text-white/80"
                : "text-white/30 hover:text-white/55 hover:bg-white/[0.04]",
            )}
          >
            ALL
          </button>
          {presentTypes.map(type => {
            const meta = TYPE_META[type];
            const color = meta?.color ?? "#22d3ee";
            const active = typeFilter === type;
            return (
              <button
                key={type}
                type="button"
                onClick={() => setTypeFilter(type)}
                className="cursor-pointer h-8 px-2.5 rounded-sm font-mono text-[10px] font-bold tracking-[0.13em] uppercase transition-all duration-100"
                style={
                  active
                    ? { color, background: `${color}18`, border: `1px solid ${color}35` }
                    : { color: "rgba(255,255,255,0.28)", border: "1px solid transparent" }
                }
              >
                {typeShortLabel(type)}
              </button>
            );
          })}
        </div>

        {/* Sort + count */}
        <div className="flex items-center gap-2 ml-auto">
          {total > 0 && (
            <span className="font-mono text-[10px] tabular-nums text-white/22">
              {shown < total ? `${shown}/${total}` : total} strat{total !== 1 ? "s" : ""}
            </span>
          )}
          <Select value={sortBy} onValueChange={v => setSortBy(v as SortKey)}>
            <SelectTrigger className="w-[130px] h-8 rounded-sm font-mono text-[10px] bg-white/[0.03] border-white/[0.07] text-white/50 cursor-pointer">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SORT_OPTIONS.map(opt => (
                <SelectItem key={opt.value} value={opt.value} className="font-mono text-xs cursor-pointer">
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* ── Grid ─────────────────────────────────────────────── */}
      {filtered.length === 0 ? (
        <div
          className="flex flex-col items-center justify-center py-20 border border-white/[0.05] rounded-sm"
          style={{
            backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.01) 2px, rgba(255,255,255,0.01) 3px)",
          }}
        >
          <span className="font-mono text-[28px] text-white/[0.07] mb-3">◈</span>
          <p className="font-mono text-[11px] text-white/30 tracking-widest uppercase">No strategies found</p>
          <p className="font-mono text-[10px] text-white/15 mt-1.5">
            {debouncedSearch || typeFilter !== "all"
              ? "adjust search or filter"
              : "create your first strategy"}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filtered.map((strategy, i) => (
            <MemoizedCardWrapper
              key={strategy.id}
              strategy={strategy}
              index={i}
              onSelect={onSelect}
              onDelete={onDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
}
