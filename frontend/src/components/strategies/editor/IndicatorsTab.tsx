import { useState } from "react";
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
  INDICATOR_TYPES,
  type IndicatorCategory,
  TIMEFRAMES,
} from "./types";

interface IndicatorsTabProps {
  config: DslConfig;
  onConfigChange: (config: DslConfig) => void;
}

const CATEGORIES: IndicatorCategory[] = ["Trend", "Momentum", "Volatility", "Volume"];

const CATEGORY_COLORS: Record<IndicatorCategory, string> = {
  Trend: "bg-blue-500/10 text-blue-600 border-blue-200",
  Momentum: "bg-orange-500/10 text-orange-600 border-orange-200",
  Volatility: "bg-purple-500/10 text-purple-600 border-purple-200",
  Volume: "bg-green-500/10 text-green-600 border-green-200",
};

function IndicatorCatalog({ onAdd }: { onAdd: (type: string) => void }) {
  const [filterCategory, setFilterCategory] = useState<IndicatorCategory | "all">("all");

  const filtered =
    filterCategory === "all"
      ? INDICATOR_CATALOG
      : INDICATOR_CATALOG.filter((i) => i.category === filterCategory);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">Filter:</span>
        <div className="flex gap-1">
          <Badge
            variant={filterCategory === "all" ? "default" : "outline"}
            className="cursor-pointer text-xs"
            onClick={() => setFilterCategory("all")}
          >
            All
          </Badge>
          {CATEGORIES.map((cat) => (
            <Badge
              key={cat}
              variant={filterCategory === cat ? "default" : "outline"}
              className="cursor-pointer text-xs"
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
            onClick={() => onAdd(entry.type)}
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

  const updateIndicators = (indicators: DslIndicator[]) => {
    onConfigChange({ ...config, indicators });
  };

  const addIndicator = (type: string) => {
    updateIndicators([...config.indicators, { type, params: getDefaultParams(type) }]);
    setShowCatalog(false);
  };

  const removeIndicator = (idx: number) => {
    updateIndicators(config.indicators.filter((_, i) => i !== idx));
  };

  const updateType = (idx: number, type: string) => {
    const updated = [...config.indicators];
    updated[idx] = { type, params: getDefaultParams(type) };
    updateIndicators(updated);
  };

  const updateParams = (idx: number, params: Record<string, number>) => {
    const updated = [...config.indicators];
    updated[idx] = { ...updated[idx], params };
    updateIndicators(updated);
  };

  const updateTimeframeOverride = (idx: number, value: string) => {
    const updated = [...config.indicators];
    updated[idx] = {
      ...updated[idx],
      timeframe_override: value === "default" ? undefined : value,
    };
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

      {showCatalog && <IndicatorCatalog onAdd={addIndicator} />}

      {config.indicators.length === 0 && !showCatalog && (
        <div className="rounded-lg border border-dashed py-8 text-center text-sm text-muted-foreground">
          No indicators yet. Add one to get started.
        </div>
      )}

      {config.indicators.map((ind, idx) => {
        const catalogEntry = INDICATOR_CATALOG.find((c) => c.type === ind.type);
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
                      {INDICATOR_TYPES.map((t) => (
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
