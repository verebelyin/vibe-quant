import { useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { usePromoteProposedIndicatorApiResearchIndicatorsNamePromotePost } from "@/api/generated/research/research";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { LoadingSpinner } from "@/components/ui";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { type DslConfig, type DslIndicator, TIMEFRAMES } from "./types";
import { useIndicatorCatalog } from "@/hooks/useIndicatorCatalog";

interface IndicatorsTabProps {
  config: DslConfig;
  onConfigChange: (config: DslConfig) => void;
}

interface CatalogDisplayEntry {
  type: string;
  name: string;
  description: string;
  category: string;
  defaultParams: Record<string, number>;
  isProposed: boolean;
  sourceFile: string | null;
}

const CATEGORY_COLORS: Record<string, string> = {
  trend: "bg-blue-500/10 text-blue-600 border-blue-200",
  momentum: "bg-orange-500/10 text-orange-600 border-orange-200",
  volatility: "bg-purple-500/10 text-purple-600 border-purple-200",
  volume: "bg-green-500/10 text-green-600 border-green-200",
  moving_average: "bg-blue-500/10 text-blue-600 border-blue-200",
};

function categoryClass(category: string): string {
  return CATEGORY_COLORS[category.toLowerCase()] ?? "bg-gray-500/10 text-gray-600 border-gray-200";
}

function IndicatorCatalog({
  catalog,
  categories,
  onAdd,
  onPromote,
  promotingName,
  promotedNames,
}: {
  catalog: CatalogDisplayEntry[];
  categories: string[];
  onAdd: (type: string, defaults: Record<string, number>) => void;
  onPromote: (type: string) => void;
  promotingName: string | null;
  promotedNames: ReadonlySet<string>;
}) {
  const [filterCategory, setFilterCategory] = useState<string>("all");

  const filtered =
    filterCategory === "all" ? catalog : catalog.filter((i) => i.category === filterCategory);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">Filter:</span>
        <Badge
          variant={filterCategory === "all" ? "default" : "outline"}
          className="cursor-pointer text-xs hover:bg-accent"
          onClick={() => setFilterCategory("all")}
        >
          All
        </Badge>
        {categories.map((cat) => (
          <Badge
            key={cat}
            variant={filterCategory === cat ? "default" : "outline"}
            className="cursor-pointer text-xs capitalize hover:bg-accent"
            onClick={() => setFilterCategory(cat)}
          >
            {cat}
          </Badge>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {filtered.map((entry) => {
          const justPromoted = promotedNames.has(entry.type);
          return (
            <div
              key={entry.type}
              className={`overflow-hidden rounded-lg border ${categoryClass(entry.category)}`}
            >
              {/* Add-to-strategy affordance. Kept a separate button from the
                  promote control below so we never nest interactive elements. */}
              <button
                type="button"
                onClick={() => onAdd(entry.type, entry.defaultParams)}
                className="block w-full p-3 text-left transition-colors hover:bg-accent"
              >
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-semibold">{entry.type}</span>
                  {entry.isProposed && (
                    <Badge
                      variant="outline"
                      className="text-[9px] border-amber-500/40 text-amber-300"
                    >
                      proposed
                    </Badge>
                  )}
                  {!entry.isProposed && justPromoted && (
                    <Badge
                      variant="outline"
                      className="text-[9px] border-emerald-500/40 text-emerald-300"
                    >
                      promoted
                    </Badge>
                  )}
                </div>
                <div className="mt-1 text-[11px] font-medium">{entry.name}</div>
                <div className="mt-0.5 text-[10px] opacity-70">{entry.description}</div>
              </button>
              {entry.isProposed && (
                <div className="flex items-center justify-between border-t border-amber-500/20 bg-amber-500/5 px-3 py-1.5">
                  <span className="text-[10px] text-amber-300/80">LLM-scaffolded</span>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    className="h-6 text-[10px]"
                    disabled={promotingName !== null}
                    onClick={() => onPromote(entry.type)}
                    title="Drop the proposed_ prefix, strip the AUTO-GENERATED header, and commit the plugin"
                  >
                    {promotingName === entry.type ? "Promoting…" : "Promote"}
                  </Button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function IndicatorParamFields({
  indicator,
  onChange,
}: {
  indicator: DslIndicator;
  onChange: (params: Record<string, number>) => void;
}) {
  const entries = Object.entries(indicator.params);
  if (entries.length === 0) {
    return <p className="text-xs text-muted-foreground">No parameters</p>;
  }

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {entries.map(([key, val]) => (
        <div key={key} className="space-y-1">
          <Label className="text-xs">{key.replace(/_/g, " ")}</Label>
          <Input
            type="number"
            value={val}
            onChange={(e) => onChange({ ...indicator.params, [key]: Number(e.target.value) })}
            className="h-8 text-xs"
          />
        </div>
      ))}
    </div>
  );
}

export function IndicatorsTab({ config, onConfigChange }: IndicatorsTabProps) {
  const [showCatalog, setShowCatalog] = useState(config.indicators.length === 0);
  const catalogQuery = useIndicatorCatalog();
  const queryClient = useQueryClient();
  const promoteMut = usePromoteProposedIndicatorApiResearchIndicatorsNamePromotePost();

  // Names promoted in this session — drives the transient "promoted" badge.
  // The catalog itself refreshes via invalidateQueries, dropping is_proposed.
  const [promotedNames, setPromotedNames] = useState<ReadonlySet<string>>(() => new Set());

  const handlePromote = (type: string) => {
    promoteMut.mutate(
      { name: type },
      {
        onSuccess: (resp) => {
          // Errors now return real 4xx/5xx → customInstance throws → onError.
          // Only the 200 "ok" PromoteIndicatorResponse reaches here.
          if (resp.status !== 200) return;
          const body = resp.data;
          const sha = body.commit_sha ? ` (${body.commit_sha.slice(0, 7)})` : "";
          toast.success(`Promoted ${body.name}${sha}`);
          if (body.bd_remember_ok === false) {
            toast.warning("Promoted, but provenance note (bd remember) failed");
          }
          setPromotedNames((prev) => new Set(prev).add(type));
          queryClient.invalidateQueries({ queryKey: ["indicators", "catalog"] });
        },
        onError: (error) => {
          // The thrown Error embeds the HTTP status ("API error: 409 …"),
          // matching how research.queue.tsx handles cancel conflicts.
          const msg = error instanceof Error ? error.message : "";
          if (msg.includes("409")) {
            toast.error(`Cannot promote ${type}: ${type.toLowerCase()}.py already exists`);
          } else if (msg.includes("400")) {
            toast.error(`Cannot promote ${type}: not a valid indicator name`);
          } else if (msg.includes("404")) {
            toast.error(`Cannot promote ${type}: no scaffolded proposed_${type.toLowerCase()}.py found`);
          } else {
            toast.error(`Promote ${type} failed`);
          }
        },
      },
    );
  };

  const promotingName = promoteMut.isPending ? (promoteMut.variables?.name ?? null) : null;

  const { catalog, categories } = useMemo<{
    catalog: CatalogDisplayEntry[];
    categories: string[];
  }>(() => {
    if (catalogQuery.data?.status !== 200) return { catalog: [], categories: [] };
    const apiEntries = catalogQuery.data.data.indicators;
    const items: CatalogDisplayEntry[] = apiEntries.map((api) => ({
      type: api.type_name,
      name: api.display_name || api.type_name,
      description: api.description || "",
      category: api.category || "other",
      defaultParams: api.default_params,
      isProposed: api.is_proposed,
      sourceFile: api.source_file,
    }));
    const cats = catalogQuery.data.data.categories ?? [];
    return { catalog: items, categories: cats };
  }, [catalogQuery.data]);

  const indicatorTypes = useMemo(() => catalog.map((c) => c.type), [catalog]);

  const updateIndicators = (indicators: DslIndicator[]) => {
    onConfigChange({ ...config, indicators });
  };

  const addIndicator = (type: string, defaults: Record<string, number>) => {
    updateIndicators([...config.indicators, { type, params: defaults }]);
    setShowCatalog(false);
  };

  const removeIndicator = (idx: number) => {
    updateIndicators(config.indicators.filter((_, i) => i !== idx));
  };

  const updateType = (idx: number, type: string) => {
    const entry = catalog.find((c) => c.type === type);
    const defaults = entry?.defaultParams ?? {};
    const updated = [...config.indicators];
    updated[idx] = { type, params: defaults };
    updateIndicators(updated);
  };

  const updateParams = (idx: number, params: Record<string, number>) => {
    const updated = [...config.indicators];
    const existing = updated[idx];
    if (existing) updated[idx] = { ...existing, params };
    updateIndicators(updated);
  };

  const updateTimeframeOverride = (idx: number, value: string) => {
    const updated = [...config.indicators];
    const existing = updated[idx];
    if (existing) {
      updated[idx] = {
        ...existing,
        timeframe_override: value === "default" ? undefined : value,
      };
    }
    updateIndicators(updated);
  };

  if (catalogQuery.isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <LoadingSpinner size="sm" />
        <span className="ml-2 text-xs text-muted-foreground">Loading indicator catalog…</span>
      </div>
    );
  }

  if (catalogQuery.isError) {
    return (
      <div className="rounded-lg border border-destructive bg-destructive/10 p-4">
        <p className="text-sm text-destructive">Failed to load indicator catalog.</p>
        <Button
          variant="outline"
          size="sm"
          className="mt-2"
          onClick={() => catalogQuery.refetch()}
        >
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          {config.indicators.length} indicator{config.indicators.length !== 1 ? "s" : ""} configured
        </p>
        <Button variant="outline" size="sm" onClick={() => setShowCatalog(!showCatalog)}>
          {showCatalog ? "Hide Catalog" : "+ Add Indicator"}
        </Button>
      </div>

      {showCatalog && (
        <IndicatorCatalog
          catalog={catalog}
          categories={categories}
          onAdd={addIndicator}
          onPromote={handlePromote}
          promotingName={promotingName}
          promotedNames={promotedNames}
        />
      )}

      {config.indicators.length === 0 && !showCatalog && (
        <div className="rounded-lg border border-dashed py-8 text-center text-sm text-muted-foreground">
          No indicators yet. Add one to get started.
        </div>
      )}

      {config.indicators.map((ind, idx) => {
        const catalogEntry = catalog.find((c) => c.type === ind.type);
        return (
          <Card key={`ind-${idx.toString()}`}>
            <CardContent className="space-y-3 p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Select value={ind.type} onValueChange={(v) => updateType(idx, v)}>
                    <SelectTrigger className="w-[180px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {indicatorTypes.map((t) => (
                        <SelectItem key={t} value={t}>
                          {t}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {catalogEntry && (
                    <Badge
                      variant="outline"
                      className={`text-[10px] capitalize ${categoryClass(catalogEntry.category)}`}
                    >
                      {catalogEntry.category}
                    </Badge>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Select
                    value={ind.timeframe_override ?? "default"}
                    onValueChange={(v) => updateTimeframeOverride(idx, v)}
                  >
                    <SelectTrigger className="h-8 w-[140px] text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">Strategy default</SelectItem>
                      {TIMEFRAMES.map((tf) => (
                        <SelectItem key={tf} value={tf}>
                          {tf}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-destructive"
                    onClick={() => removeIndicator(idx)}
                  >
                    Remove
                  </Button>
                </div>
              </div>
              <IndicatorParamFields
                indicator={ind}
                onChange={(params) => updateParams(idx, params)}
              />
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
