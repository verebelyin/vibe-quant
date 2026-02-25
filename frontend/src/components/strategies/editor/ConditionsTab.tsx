import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { buildOperandOptions, type DslCondition, type DslConfig, OPERATORS } from "./types";

interface ConditionsTabProps {
  config: DslConfig;
  onConfigChange: (config: DslConfig) => void;
}

function OperandSelect({
  value,
  operandOptions,
  onChange,
  placeholder,
}: {
  value: string;
  operandOptions: string[];
  onChange: (v: string) => void;
  placeholder: string;
}) {
  const isCustom = value !== "" && !operandOptions.includes(value);

  return (
    <Select
      value={isCustom ? "__custom__" : value}
      onValueChange={(v) => {
        if (v !== "__custom__") onChange(v);
      }}
    >
      <SelectTrigger className="h-8 text-xs">
        <SelectValue placeholder={placeholder}>{isCustom ? value : undefined}</SelectValue>
      </SelectTrigger>
      <SelectContent>
        {operandOptions.map((opt) => (
          <SelectItem key={opt} value={opt}>
            {opt}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function ConditionRow({
  condition,
  onChange,
  onRemove,
  operandOptions,
}: {
  condition: DslCondition;
  onChange: (c: DslCondition) => void;
  onRemove: () => void;
  operandOptions: string[];
}) {
  return (
    <div className="flex items-end gap-2">
      <div className="flex-1 space-y-1">
        <Label className="text-xs">Left</Label>
        <OperandSelect
          value={condition.left}
          operandOptions={operandOptions}
          onChange={(v) => onChange({ ...condition, left: v })}
          placeholder="e.g. SMA(20)"
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
        <OperandSelect
          value={condition.right}
          operandOptions={operandOptions}
          onChange={(v) => onChange({ ...condition, right: v })}
          placeholder="e.g. price or 70"
        />
      </div>
      <Button variant="ghost" size="sm" className="h-8 text-destructive" onClick={onRemove}>
        x
      </Button>
    </div>
  );
}

function LogicToggle({
  value,
  onChange,
}: {
  value: "and" | "or";
  onChange: (v: "and" | "or") => void;
}) {
  return (
    <div className="flex items-center gap-2 pl-2">
      <div className="h-4 w-px bg-border" />
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={value}
        onValueChange={(v) => {
          if (v === "and" || v === "or") onChange(v);
        }}
      >
        <ToggleGroupItem value="and" className="h-6 px-2 text-xs">
          AND
        </ToggleGroupItem>
        <ToggleGroupItem value="or" className="h-6 px-2 text-xs">
          OR
        </ToggleGroupItem>
      </ToggleGroup>
      <div className="h-4 w-px bg-border" />
    </div>
  );
}

function ConditionSection({
  title,
  conditions,
  operandOptions,
  onChange,
}: {
  title: string;
  conditions: DslCondition[];
  operandOptions: string[];
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
    if (updated.length > 0 && updated[0]?.logic) {
      updated[0] = { ...updated[0], logic: undefined };
    }
    onChange(updated);
  };

  const updateLogic = (idx: number, logic: "and" | "or") => {
    const updated = [...conditions];
    const existing = updated[idx];
    if (existing) updated[idx] = { ...existing, logic };
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
      <CardContent className="space-y-1">
        {conditions.length === 0 && (
          <p className="py-4 text-center text-xs text-muted-foreground">No conditions defined.</p>
        )}
        {conditions.map((cond, idx) => (
          <div key={`cond-${idx.toString()}`}>
            {idx > 0 && (
              <LogicToggle value={cond.logic ?? "and"} onChange={(v) => updateLogic(idx, v)} />
            )}
            <div className={idx > 0 ? "ml-4 border-l-2 border-muted pl-3" : ""}>
              <ConditionRow
                condition={cond}
                onChange={(c) => update(idx, c)}
                onRemove={() => remove(idx)}
                operandOptions={operandOptions}
              />
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function ConditionsTab({ config, onConfigChange }: ConditionsTabProps) {
  const operandOptions = buildOperandOptions(config.indicators);
  const [separateLongShort, setSeparateLongShort] = useState(
    !!(
      config.conditions.long_entry?.length ||
      config.conditions.long_exit?.length ||
      config.conditions.short_entry?.length ||
      config.conditions.short_exit?.length
    ),
  );

  const updateConditions = (patch: Partial<DslConfig["conditions"]>) => {
    onConfigChange({
      ...config,
      conditions: { ...config.conditions, ...patch },
    });
  };

  return (
    <div className="space-y-4">
      {config.indicators.length === 0 && (
        <div className="rounded-md border border-dashed bg-muted/50 p-3 text-center text-xs text-muted-foreground">
          Add indicators first to populate operand dropdowns.
        </div>
      )}

      <Label className="flex items-center gap-2 text-sm">
        <Checkbox
          checked={separateLongShort}
          onCheckedChange={(v) => setSeparateLongShort(v === true)}
        />
        Separate long/short conditions
      </Label>

      {separateLongShort ? (
        <>
          <ConditionSection
            title="Long Entry"
            conditions={config.conditions.long_entry ?? []}
            operandOptions={operandOptions}
            onChange={(long_entry) => updateConditions({ long_entry })}
          />
          <ConditionSection
            title="Long Exit"
            conditions={config.conditions.long_exit ?? []}
            operandOptions={operandOptions}
            onChange={(long_exit) => updateConditions({ long_exit })}
          />
          <ConditionSection
            title="Short Entry"
            conditions={config.conditions.short_entry ?? []}
            operandOptions={operandOptions}
            onChange={(short_entry) => updateConditions({ short_entry })}
          />
          <ConditionSection
            title="Short Exit"
            conditions={config.conditions.short_exit ?? []}
            operandOptions={operandOptions}
            onChange={(short_exit) => updateConditions({ short_exit })}
          />
        </>
      ) : (
        <>
          <ConditionSection
            title="Entry Conditions"
            conditions={config.conditions.entry}
            operandOptions={operandOptions}
            onChange={(entry) => updateConditions({ entry })}
          />
          <ConditionSection
            title="Exit Conditions"
            conditions={config.conditions.exit}
            operandOptions={operandOptions}
            onChange={(exit) => updateConditions({ exit })}
          />
        </>
      )}
    </div>
  );
}
