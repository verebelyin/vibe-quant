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
import { type DslConfig, type DslIndicator, getDefaultParams, INDICATOR_TYPES } from "./types";

interface IndicatorsTabProps {
  config: DslConfig;
  onConfigChange: (config: DslConfig) => void;
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
  const updateIndicators = (indicators: DslIndicator[]) => {
    onConfigChange({ ...config, indicators });
  };

  const addIndicator = () => {
    const type = "SMA";
    updateIndicators([...config.indicators, { type, params: getDefaultParams(type) }]);
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

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          {config.indicators.length} indicator{config.indicators.length !== 1 ? "s" : ""} configured
        </p>
        <Button variant="outline" size="sm" onClick={addIndicator}>
          + Add Indicator
        </Button>
      </div>

      {config.indicators.length === 0 && (
        <div className="rounded-lg border border-dashed py-8 text-center text-sm text-muted-foreground">
          No indicators yet. Add one to get started.
        </div>
      )}

      {config.indicators.map((ind, idx) => (
        <Card key={`ind-${idx.toString()}`}>
          <CardContent className="space-y-3 p-4">
            <div className="flex items-center justify-between">
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
              <Button
                variant="ghost"
                size="sm"
                className="text-destructive"
                onClick={() => removeIndicator(idx)}
              >
                Remove
              </Button>
            </div>
            <IndicatorParamFields
              indicator={ind}
              onChange={(params) => updateParams(idx, params)}
            />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
