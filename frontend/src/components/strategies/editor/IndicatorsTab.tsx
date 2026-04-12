import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  type DslConfig,
  type DslIndicator,
  getDefaultParams,
  INDICATOR_CATALOG,
  type IndicatorCategory,
  TIMEFRAMES,
} from "./types";
import { useIndicatorCatalog } from "@/hooks/useIndicatorCatalog";

interface IndicatorsTabProps {
  config: DslConfig;
  onConfigChange: (config: DslConfig) => void;
}

type ExtendedCategory = IndicatorCategory | "Custom";
const CATEGORIES: ExtendedCategory[] = ["Trend", "Momentum", "Volatility", "Volume", "Custom"];
const KNOWN_CATEGORIES = new Set<string>(CATEGORIES.filter((c) => c !== "Custom"));

const CATEGORY_COLORS: Record<ExtendedCategory, string> = {
  Trend: "bg-blue-500/10 text-blue-600 border-blue-200",
  Momentum: "bg-orange-500/10 text-orange-600 border-orange-200",
  Volatility: "bg-purple-500/10 text-purple-600 border-purple-200",
  Volume: "bg-green-500/10 text-green-600 border-green-200",
  Custom: "bg-gray-500/10 text-gray-600 border-gray-200",
};

interface CatalogDisplayEntry {
  type: string;
  name: string;
  emoji: string;
  description: string;
  category: ExtendedCategory;
  defaultParams: Record<string, number>;
}

/** Merge hardcoded catalog with API entries, adding plugin indicators. */
function useMergedCatalog(): CatalogDisplayEntry[] {
  const catalogQuery = useIndicatorCatalog();

  return useMemo(() => {
    const hardcodedByType = new Map(INDICATOR_CATALOG.map((e) => [e.type, e]));
    const merged: CatalogDisplayEntry[] = INDICATOR_CATALOG.map((e) => ({
      type: e.type,
      name: e.name,
      emoji: e.emoji,
      description: e.description,
      category: e.category,
      defaultParams: getDefaultParams(e.type),
    }));

    const apiEntries =
      catalogQuery.data?.status === 200 ? catalogQuery.data.data.indicators : [];

    for (const api of apiEntries) {
      if (!hardcodedByType.has(api.type_name)) {
        merged.push({
          type: api.type_name,
          name: api.display_name || api.type_name,
          emoji: "\u{1F9EA}",
          description: api.description || "Custom indicator plugin",
          category: (KNOWN_CATEGORIES.has(api.category)
            ? api.category
            : "Custom") as ExtendedCategory,
          defaultParams: api.default_params,
        });
      }
    }
    return merged;
  }, [catalogQuery.data]);
}

function IndicatorCatalog({
  catalog,
  onAdd,
}: {
  catalog: CatalogDisplayEntry[];
  onAdd: (type: string, defaults: Record<string, number>) => void;
}) {
  const [filterCategory, setFilterCategory] = useState<ExtendedCategory | "all">("all");

  const filtered =
    filterCategory === "all"
      ? catalog
      : catalog.filter((i) => i.category === filterCategory);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">Filter:</span>
        <div className="flex gap-1">
          <Badge
            variant={filterCategory === "all" ? "default" : "outline"}
            className="cursor-pointer text-xs hover:bg-accent"
            onClick={() => setFilterCategory("all")}
          >
            All
          </Badge>
          {CATEGORIES.map((cat) => (
            <Badge
              key={cat}
              variant={filterCategory === cat ? "default" : "outline"}
              className="cursor-pointer text-xs hover:bg-accent"
              onClick={() => setFilterCategory(cat)}
            >
              {cat}
            </Badge>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {filtered.map((entry) => (
          <button
            key={entry.type}
            type="button"
            onClick={() => onAdd(entry.type, entry.defaultParams)}
            className={`rounded-lg border p-3 text-left transition-colors hover:bg-accent ${CATEGORY_COLORS[entry.category]}`}
          >
            <div className="text-lg">{entry.emoji}</div>
            <div className="mt-1 text-xs font-medium">{entry.name}</div>
            <div className="mt-0.5 text-[10px] opacity-70">{entry.description}</div>
          </button>
        ))}
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
  const catalog = useMergedCatalog();

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
    const defaults = entry?.defaultParams ?? getDefaultParams(type);
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

      {showCatalog && <IndicatorCatalog catalog={catalog} onAdd={addIndicator} />}

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
                  {catalogEntry && <span className="text-lg">{catalogEntry.emoji}</span>}
                  <Select value={ind.type} onValueChange={(v) => updateType(idx, v)}>
                    <SelectTrigger className="w-[140px]">
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
                      className={`text-[10px] ${CATEGORY_COLORS[catalogEntry.category]}`}
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
