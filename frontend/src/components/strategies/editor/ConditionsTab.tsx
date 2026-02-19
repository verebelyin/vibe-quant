import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { type DslCondition, type DslConfig, OPERATORS } from "./types";

interface ConditionsTabProps {
  config: DslConfig;
  onConfigChange: (config: DslConfig) => void;
}

function ConditionRow({
  condition,
  onChange,
  onRemove,
  showLogic,
}: {
  condition: DslCondition;
  onChange: (c: DslCondition) => void;
  onRemove: () => void;
  showLogic: boolean;
}) {
  return (
    <div className="space-y-2">
      {showLogic && (
        <Select
          value={condition.logic ?? "and"}
          onValueChange={(v) => onChange({ ...condition, logic: v as "and" | "or" })}
        >
          <SelectTrigger className="w-[80px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="and">AND</SelectItem>
            <SelectItem value="or">OR</SelectItem>
          </SelectContent>
        </Select>
      )}
      <div className="flex items-end gap-2">
        <div className="flex-1 space-y-1">
          <Label className="text-xs">Left</Label>
          <Input
            value={condition.left}
            onChange={(e) => onChange({ ...condition, left: e.target.value })}
            placeholder="e.g. SMA_20"
            className="h-8 text-xs"
          />
        </div>
        <div className="w-[140px] space-y-1">
          <Label className="text-xs">Operator</Label>
          <Select
            value={condition.operator}
            onValueChange={(v) => onChange({ ...condition, operator: v })}
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {OPERATORS.map((op) => (
                <SelectItem key={op.value} value={op.value}>
                  {op.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex-1 space-y-1">
          <Label className="text-xs">Right</Label>
          <Input
            value={condition.right}
            onChange={(e) => onChange({ ...condition, right: e.target.value })}
            placeholder="e.g. price or 70"
            className="h-8 text-xs"
          />
        </div>
        <Button variant="ghost" size="sm" className="h-8 text-destructive" onClick={onRemove}>
          x
        </Button>
      </div>
    </div>
  );
}

function ConditionSection({
  title,
  conditions,
  onChange,
}: {
  title: string;
  conditions: DslCondition[];
  onChange: (conditions: DslCondition[]) => void;
}) {
  const add = () => {
    onChange([
      ...conditions,
      { left: "", operator: ">", right: "", logic: conditions.length > 0 ? "and" : undefined },
    ]);
  };

  const update = (idx: number, c: DslCondition) => {
    const updated = [...conditions];
    updated[idx] = c;
    onChange(updated);
  };

  const remove = (idx: number) => {
    const updated = conditions.filter((_, i) => i !== idx);
    // Remove logic from first item
    if (updated.length > 0 && updated[0].logic) {
      updated[0] = { ...updated[0], logic: undefined };
    }
    onChange(updated);
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">{title}</CardTitle>
          <Button variant="outline" size="sm" onClick={add}>
            + Add
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {conditions.length === 0 && (
          <p className="py-4 text-center text-xs text-muted-foreground">No conditions defined.</p>
        )}
        {conditions.map((cond, idx) => (
          <ConditionRow
            key={`cond-${idx.toString()}`}
            condition={cond}
            onChange={(c) => update(idx, c)}
            onRemove={() => remove(idx)}
            showLogic={idx > 0}
          />
        ))}
      </CardContent>
    </Card>
  );
}

export function ConditionsTab({ config, onConfigChange }: ConditionsTabProps) {
  const updateConditions = (patch: Partial<DslConfig["conditions"]>) => {
    onConfigChange({
      ...config,
      conditions: { ...config.conditions, ...patch },
    });
  };

  return (
    <div className="space-y-4">
      <ConditionSection
        title="Entry Conditions"
        conditions={config.conditions.entry}
        onChange={(entry) => updateConditions({ entry })}
      />
      <ConditionSection
        title="Exit Conditions"
        conditions={config.conditions.exit}
        onChange={(exit) => updateConditions({ exit })}
      />
    </div>
  );
}
