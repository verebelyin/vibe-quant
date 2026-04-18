import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useIndicatorCatalog } from "@/hooks/useIndicatorCatalog";
import {
  buildOperandOptions,
  type DslCondition,
  type DslConfig,
  type DslIndicator,
  OPERATORS,
} from "./types";

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

function AddMaConditionDialog({
  existingIndicators,
  onApply,
}: {
  existingIndicators: DslIndicator[];
  onApply: (indicator: DslIndicator | null, condition: DslCondition) => void;
}) {
  const [open, setOpen] = useState(false);
  const [maKind, setMaKind] = useState("EMA");
  const [period, setPeriod] = useState(20);
  const [operator, setOperator] = useState("crosses_above");
  const catalogQuery = useIndicatorCatalog();

  const maKinds = useMemo(() => {
    if (catalogQuery.data?.status !== 200) return ["SMA", "EMA"];
    return catalogQuery.data.data.indicators
      .filter((i) => (i.category || "").toLowerCase().includes("moving_average"))
      .map((i) => i.type_name);
  }, [catalogQuery.data]);

  const handleApply = () => {
    const existing = existingIndicators.find(
      (i) => i.type === maKind && i.params.period === period,
    );
    const indicator: DslIndicator | null = existing
      ? null
      : { type: maKind, params: { period } };
    const maName = `${maKind}(${period})`;
    onApply(indicator, {
      left: "close",
      operator,
      right: maName,
    });
    setOpen(false);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          + Price vs MA
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Add Price-vs-MA Condition</DialogTitle>
        </DialogHeader>
        <div className="space-y-3">
          <div className="space-y-1">
            <Label className="text-xs">MA Kind</Label>
            <Select value={maKind} onValueChange={setMaKind}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {maKinds.length === 0 ? (
                  <SelectItem value="EMA">EMA (catalog unavailable)</SelectItem>
                ) : (
                  maKinds.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Period</Label>
            <Input
              type="number"
              min={2}
              max={500}
              value={period}
              onChange={(e) => setPeriod(Number(e.target.value))}
              className="h-8 text-xs"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Comparator</Label>
            <Select value={operator} onValueChange={setOperator}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {OPERATORS.filter((o) =>
                  ["crosses_above", "crosses_below", ">", "<"].includes(o.value),
                ).map((o) => (
                  <SelectItem key={o.value} value={o.value}>
                    close {o.label} MA
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleApply}>Add Condition</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ConditionSection({
  title,
  conditions,
  operandOptions,
  existingIndicators,
  onChange,
  onAddIndicator,
}: {
  title: string;
  conditions: DslCondition[];
  operandOptions: string[];
  existingIndicators: DslIndicator[];
  onChange: (conditions: DslCondition[]) => void;
  onAddIndicator: (ind: DslIndicator) => void;
}) {
  const add = () => {
    onChange([
      ...conditions,
      { left: "", operator: ">", right: "", logic: conditions.length > 0 ? "and" : undefined },
    ]);
  };

  const addMa = (indicator: DslIndicator | null, condition: DslCondition) => {
    if (indicator) onAddIndicator(indicator);
    onChange([
      ...conditions,
      { ...condition, logic: conditions.length > 0 ? "and" : undefined },
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
          <div className="flex gap-2">
            <AddMaConditionDialog
              existingIndicators={existingIndicators}
              onApply={addMa}
            />
            <Button variant="outline" size="sm" onClick={add}>
              + Add
            </Button>
          </div>
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

  const addIndicator = (indicator: DslIndicator) => {
    onConfigChange({
      ...config,
      indicators: [...config.indicators, indicator],
    });
  };

  const commonProps = {
    existingIndicators: config.indicators,
    onAddIndicator: addIndicator,
    operandOptions,
  };

  return (
    <div className="space-y-4">
      {config.indicators.length === 0 && (
        <div className="rounded-md border border-dashed bg-muted/50 p-3 text-center text-xs text-muted-foreground">
          Add indicators first to populate operand dropdowns. Or use <strong>+ Price vs MA</strong>
          below to auto-create one.
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
            {...commonProps}
            onChange={(long_entry) => updateConditions({ long_entry })}
          />
          <ConditionSection
            title="Long Exit"
            conditions={config.conditions.long_exit ?? []}
            {...commonProps}
            onChange={(long_exit) => updateConditions({ long_exit })}
          />
          <ConditionSection
            title="Short Entry"
            conditions={config.conditions.short_entry ?? []}
            {...commonProps}
            onChange={(short_entry) => updateConditions({ short_entry })}
          />
          <ConditionSection
            title="Short Exit"
            conditions={config.conditions.short_exit ?? []}
            {...commonProps}
            onChange={(short_exit) => updateConditions({ short_exit })}
          />
        </>
      ) : (
        <>
          <ConditionSection
            title="Entry Conditions"
            conditions={config.conditions.entry}
            {...commonProps}
            onChange={(entry) => updateConditions({ entry })}
          />
          <ConditionSection
            title="Exit Conditions"
            conditions={config.conditions.exit}
            {...commonProps}
            onChange={(exit) => updateConditions({ exit })}
          />
        </>
      )}
    </div>
  );
}
