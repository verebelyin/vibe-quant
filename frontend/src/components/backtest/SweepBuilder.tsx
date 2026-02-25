import { useMemo, useState } from "react";
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

type SweepValueType = "fixed" | "range" | "list";

export interface SweepParam {
  /** e.g. "SMA_0.period" */
  key: string;
  label: string;
  type: SweepValueType;
  fixedValue: number;
  rangeMin: number;
  rangeMax: number;
  rangeStep: number;
  listValues: string;
}

export interface SweepConfig {
  params: SweepParam[];
}

/** Compute how many values a single param generates */
function paramValueCount(p: SweepParam): number {
  switch (p.type) {
    case "fixed":
      return 1;
    case "range": {
      if (p.rangeStep <= 0 || p.rangeMax < p.rangeMin) return 0;
      return Math.floor((p.rangeMax - p.rangeMin) / p.rangeStep) + 1;
    }
    case "list": {
      const vals = p.listValues
        .split(",")
        .map((v) => v.trim())
        .filter((v) => v !== "");
      return vals.length || 1;
    }
    default:
      return 1;
  }
}

/** Expand a sweep param to its concrete values */
export function expandParam(p: SweepParam): number[] {
  switch (p.type) {
    case "fixed":
      return [p.fixedValue];
    case "range": {
      if (p.rangeStep <= 0 || p.rangeMax < p.rangeMin) return [];
      const vals: number[] = [];
      for (let v = p.rangeMin; v <= p.rangeMax; v += p.rangeStep) {
        vals.push(v);
      }
      return vals;
    }
    case "list":
      return p.listValues
        .split(",")
        .map((v) => v.trim())
        .filter((v) => v !== "")
        .map(Number)
        .filter((n) => !Number.isNaN(n));
    default:
      return [];
  }
}

/** Convert sweep config to a serializable payload for the API */
export function sweepToPayload(
  sweep: SweepConfig,
): Record<string, { type: string; values?: number[]; min?: number; max?: number; step?: number }> {
  const result: Record<
    string,
    { type: string; values?: number[]; min?: number; max?: number; step?: number }
  > = {};
  for (const p of sweep.params) {
    switch (p.type) {
      case "fixed":
        result[p.key] = { type: "fixed", values: [p.fixedValue] };
        break;
      case "range":
        result[p.key] = { type: "range", min: p.rangeMin, max: p.rangeMax, step: p.rangeStep };
        break;
      case "list":
        result[p.key] = { type: "list", values: expandParam(p) };
        break;
    }
  }
  return result;
}

interface IndicatorParam {
  indicatorType: string;
  indicatorIndex: number;
  paramName: string;
  currentValue: number;
}

interface SweepBuilderProps {
  /** Strategy indicators to extract sweepable params from */
  indicators: Array<{ type: string; params: Record<string, number> }>;
  /** Callback when sweep config changes */
  onChange: (config: SweepConfig) => void;
  /** Current sweep config */
  value: SweepConfig;
}

const PRESETS = {
  quick: { label: "Quick Scan", description: "3 values per range param" },
  fine: { label: "Fine Grid", description: "10+ values per range param" },
  custom: { label: "Custom", description: "Manual configuration" },
} as const;

type PresetKey = keyof typeof PRESETS;

function buildParamsFromIndicators(
  indicators: Array<{ type: string; params: Record<string, number> }>,
): IndicatorParam[] {
  const result: IndicatorParam[] = [];
  for (let i = 0; i < indicators.length; i++) {
    const ind = indicators[i]!;
    for (const [paramName, value] of Object.entries(ind.params)) {
      result.push({
        indicatorType: ind.type,
        indicatorIndex: i,
        paramName,
        currentValue: value,
      });
    }
  }
  return result;
}

function applyPreset(params: IndicatorParam[], preset: "quick" | "fine"): SweepParam[] {
  return params.map((p) => {
    const key = `${p.indicatorType}_${p.indicatorIndex}.${p.paramName}`;
    const label = `${p.indicatorType}[${p.indicatorIndex}].${p.paramName}`;
    const base = p.currentValue;

    if (preset === "quick") {
      // 3 values: 0.5x, 1x, 1.5x of current
      const step = Math.max(1, Math.round(base * 0.5));
      return {
        key,
        label,
        type: "range" as const,
        fixedValue: base,
        rangeMin: Math.max(1, base - step),
        rangeMax: base + step,
        rangeStep: step,
        listValues: "",
      };
    }
    // fine: 10 steps from 0.25x to 2x
    const min = Math.max(1, Math.round(base * 0.25));
    const max = Math.round(base * 2);
    const step = Math.max(1, Math.round((max - min) / 10));
    return {
      key,
      label,
      type: "range" as const,
      fixedValue: base,
      rangeMin: min,
      rangeMax: max,
      rangeStep: step,
      listValues: "",
    };
  });
}

export function SweepBuilder({ indicators, onChange, value }: SweepBuilderProps) {
  const [preset, setPreset] = useState<PresetKey>("custom");

  const indicatorParams = useMemo(() => buildParamsFromIndicators(indicators), [indicators]);

  const totalCombos = useMemo(() => {
    if (value.params.length === 0) return 0;
    return value.params.reduce((acc, p) => acc * paramValueCount(p), 1);
  }, [value.params]);

  const handleInitialize = () => {
    const params = indicatorParams.map((p) => ({
      key: `${p.indicatorType}_${p.indicatorIndex}.${p.paramName}`,
      label: `${p.indicatorType}[${p.indicatorIndex}].${p.paramName}`,
      type: "fixed" as const,
      fixedValue: p.currentValue,
      rangeMin: 1,
      rangeMax: p.currentValue * 2,
      rangeStep: 1,
      listValues: "",
    }));
    onChange({ params });
  };

  const handleApplyPreset = (p: PresetKey) => {
    setPreset(p);
    if (p === "custom") return;
    const params = applyPreset(indicatorParams, p);
    onChange({ params });
  };

  const updateParam = (idx: number, updates: Partial<SweepParam>) => {
    const next = [...value.params];
    const existing = next[idx]!;
    next[idx] = { ...existing, ...updates } as SweepParam;
    setPreset("custom");
    onChange({ params: next });
  };

  if (indicatorParams.length === 0) {
    return (
      <div className="rounded-lg border border-dashed py-6 text-center text-sm text-muted-foreground">
        No sweepable parameters. Add indicators to the strategy first.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Parameter Sweep</h3>
          {totalCombos > 0 && (
            <p className="text-xs text-muted-foreground">
              {totalCombos.toLocaleString()} total combination{totalCombos !== 1 ? "s" : ""}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {(Object.keys(PRESETS) as PresetKey[]).map((key) => (
            <Button
              key={key}
              variant={preset === key ? "default" : "outline"}
              size="sm"
              onClick={() => handleApplyPreset(key)}
            >
              {PRESETS[key].label}
            </Button>
          ))}
        </div>
      </div>

      {/* Initialize if empty */}
      {value.params.length === 0 ? (
        <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed py-6">
          <p className="text-sm text-muted-foreground">
            {indicatorParams.length} parameter{indicatorParams.length !== 1 ? "s" : ""} available
            for sweep
          </p>
          <Button variant="outline" size="sm" onClick={handleInitialize}>
            Initialize Sweep Parameters
          </Button>
        </div>
      ) : (
        <div className="space-y-3">
          {value.params.map((param, idx) => (
            <Card key={param.key}>
              <CardContent className="space-y-3 p-4">
                <div className="flex items-center justify-between">
                  <Label className="font-mono text-sm">{param.label}</Label>
                  <Select
                    value={param.type}
                    onValueChange={(v) => updateParam(idx, { type: v as SweepValueType })}
                  >
                    <SelectTrigger className="w-[120px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fixed">Fixed</SelectItem>
                      <SelectItem value="range">Range</SelectItem>
                      <SelectItem value="list">List</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {param.type === "fixed" && (
                  <div className="space-y-1">
                    <Label className="text-xs">Value</Label>
                    <Input
                      type="number"
                      value={param.fixedValue}
                      onChange={(e) => updateParam(idx, { fixedValue: Number(e.target.value) })}
                      className="h-8 w-32 text-xs"
                    />
                  </div>
                )}

                {param.type === "range" && (
                  <div className="grid grid-cols-3 gap-3">
                    <div className="space-y-1">
                      <Label className="text-xs">Min</Label>
                      <Input
                        type="number"
                        value={param.rangeMin}
                        onChange={(e) => updateParam(idx, { rangeMin: Number(e.target.value) })}
                        className="h-8 text-xs"
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">Max</Label>
                      <Input
                        type="number"
                        value={param.rangeMax}
                        onChange={(e) => updateParam(idx, { rangeMax: Number(e.target.value) })}
                        className="h-8 text-xs"
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">Step</Label>
                      <Input
                        type="number"
                        value={param.rangeStep}
                        onChange={(e) => updateParam(idx, { rangeStep: Number(e.target.value) })}
                        className="h-8 text-xs"
                      />
                    </div>
                    <p className="col-span-3 text-xs text-muted-foreground">
                      {paramValueCount(param)} value{paramValueCount(param) !== 1 ? "s" : ""}
                    </p>
                  </div>
                )}

                {param.type === "list" && (
                  <div className="space-y-1">
                    <Label className="text-xs">Values (comma-separated)</Label>
                    <Input
                      type="text"
                      value={param.listValues}
                      onChange={(e) => updateParam(idx, { listValues: e.target.value })}
                      placeholder="e.g. 5, 10, 20, 50"
                      className="h-8 text-xs"
                    />
                    <p className="text-xs text-muted-foreground">
                      {paramValueCount(param)} value{paramValueCount(param) !== 1 ? "s" : ""}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Combo warning */}
      {totalCombos > 1000 && (
        <div className="rounded-md border border-yellow-600 bg-yellow-600/10 px-3 py-2">
          <p className="text-sm text-yellow-600">
            {totalCombos.toLocaleString()} combinations may take a long time to run. Consider
            reducing ranges or using Quick Scan preset.
          </p>
        </div>
      )}
    </div>
  );
}
