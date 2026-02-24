import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";
import {
  getListStrategiesApiStrategiesGetQueryKey,
  useCreateStrategyApiStrategiesPost,
} from "@/api/generated/strategies/strategies";
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
import { cn } from "@/lib/utils";
import { ConditionsTab } from "./editor/ConditionsTab";
import { IndicatorsTab } from "./editor/IndicatorsTab";
import { RiskTab } from "./editor/RiskTab";
import { type DslConfig, emptyDslConfig, STRATEGY_TYPES, TIMEFRAMES } from "./editor/types";

const STEPS = [
  { label: "Name & Type", description: "Basic info" },
  { label: "Markets", description: "Symbols & timeframe" },
  { label: "Indicators", description: "Technical indicators" },
  { label: "Entry/Exit", description: "Trading conditions" },
  { label: "Risk", description: "Risk management" },
  { label: "Review", description: "Summary & create" },
] as const;

const STRATEGY_TYPE_DESCRIPTIONS: Record<string, string> = {
  momentum: "Trade in the direction of recent price movement",
  mean_reversion: "Bet on prices returning to average levels",
  breakout: "Enter when price breaks key support/resistance",
  trend_following: "Follow established market trends",
  arbitrage: "Exploit price differences across markets",
  volatility: "Trade based on volatility expansion/contraction",
};

interface StrategyWizardProps {
  onCancel: () => void;
}

function StepIndicator({ currentStep }: { currentStep: number }) {
  return (
    <div className="mb-8 flex items-center justify-center">
      {STEPS.map((step, idx) => (
        <div key={step.label} className="flex items-center">
          <div className="flex flex-col items-center">
            <div
              className={cn(
                "flex h-8 w-8 items-center justify-center rounded-full border-2 text-xs font-bold transition-colors",
                idx < currentStep && "border-primary bg-primary text-primary-foreground",
                idx === currentStep && "border-primary bg-primary/10 text-primary",
                idx > currentStep && "border-muted text-muted-foreground",
              )}
            >
              {idx < currentStep ? "\u2713" : idx + 1}
            </div>
            <span
              className={cn(
                "mt-1 text-[10px]",
                idx === currentStep ? "font-medium text-foreground" : "text-muted-foreground",
              )}
            >
              {step.label}
            </span>
          </div>
          {idx < STEPS.length - 1 && (
            <div
              className={cn("mx-2 mb-4 h-0.5 w-8", idx < currentStep ? "bg-primary" : "bg-muted")}
            />
          )}
        </div>
      ))}
    </div>
  );
}

function StepNameType({
  name,
  description,
  strategyType,
  onNameChange,
  onDescriptionChange,
  onStrategyTypeChange,
}: {
  name: string;
  description: string;
  strategyType: string;
  onNameChange: (v: string) => void;
  onDescriptionChange: (v: string) => void;
  onStrategyTypeChange: (v: string) => void;
}) {
  return (
    <div className="space-y-5">
      <div className="space-y-1.5">
        <Label htmlFor="wiz-name">Strategy Name</Label>
        <Input
          id="wiz-name"
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          placeholder="My Strategy"
        />
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="wiz-desc">Description</Label>
        <textarea
          id="wiz-desc"
          value={description}
          onChange={(e) => onDescriptionChange(e.target.value)}
          placeholder="Describe what this strategy does..."
          rows={3}
          className="border-input bg-background placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 w-full rounded-md border px-3 py-2 text-sm focus-visible:ring-[3px] focus-visible:outline-1"
        />
      </div>
      <div className="space-y-1.5">
        <Label>Strategy Type</Label>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {STRATEGY_TYPES.map((t) => (
            <Card
              key={t}
              className={cn(
                "cursor-pointer gap-0 py-0 transition-colors hover:bg-muted/50",
                strategyType === t ? "border-primary bg-primary/10" : "",
              )}
              onClick={() => onStrategyTypeChange(t)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onStrategyTypeChange(t);
                }
              }}
            >
              <CardContent className="p-3">
                <div className="text-sm font-medium">{t.replace(/_/g, " ")}</div>
                <div className="mt-0.5 text-[10px] text-muted-foreground">
                  {STRATEGY_TYPE_DESCRIPTIONS[t] ?? ""}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}

function StepMarkets({
  config,
  onConfigChange,
}: {
  config: DslConfig;
  onConfigChange: (c: DslConfig) => void;
}) {
  const updateGeneral = (patch: Partial<DslConfig["general"]>) => {
    onConfigChange({ ...config, general: { ...config.general, ...patch } });
  };

  const handleSymbolsKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      const input = e.currentTarget;
      const val = input.value.trim().toUpperCase();
      if (val && !config.general.symbols.includes(val)) {
        updateGeneral({ symbols: [...config.general.symbols, val] });
      }
      input.value = "";
    }
  };

  const removeSymbol = (sym: string) => {
    updateGeneral({ symbols: config.general.symbols.filter((s) => s !== sym) });
  };

  return (
    <div className="space-y-5">
      <div className="space-y-1.5">
        <Label>Timeframe</Label>
        <Select
          value={config.general.timeframe}
          onValueChange={(v) => updateGeneral({ timeframe: v })}
        >
          <SelectTrigger className="w-[200px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {TIMEFRAMES.map((t) => (
              <SelectItem key={t} value={t}>
                {t}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-1.5">
        <Label>Symbols</Label>
        <Input
          placeholder="Type symbol and press Enter (e.g. BTCUSDT)"
          onKeyDown={handleSymbolsKeyDown}
        />
        {config.general.symbols.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {config.general.symbols.map((sym) => (
              <Badge
                key={sym}
                variant="secondary"
                className="cursor-pointer hover:bg-destructive/20 hover:text-destructive"
                onClick={() => removeSymbol(sym)}
              >
                {sym} x
              </Badge>
            ))}
          </div>
        )}
        {config.general.symbols.length === 0 && (
          <p className="text-xs text-muted-foreground">Add at least one symbol to continue.</p>
        )}
      </div>
    </div>
  );
}

function StepReview({
  name,
  description,
  config,
}: {
  name: string;
  description: string;
  config: DslConfig;
}) {
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium">Review your strategy before creating:</h3>
      <div className="space-y-3 rounded-lg border p-4">
        <div>
          <span className="text-xs text-muted-foreground">Name:</span>
          <p className="text-sm font-medium">{name || "Untitled"}</p>
        </div>
        {description && (
          <div>
            <span className="text-xs text-muted-foreground">Description:</span>
            <p className="text-sm">{description}</p>
          </div>
        )}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <span className="text-xs text-muted-foreground">Type:</span>
            <p className="text-sm">{config.general.strategy_type.replace(/_/g, " ")}</p>
          </div>
          <div>
            <span className="text-xs text-muted-foreground">Timeframe:</span>
            <p className="text-sm">{config.general.timeframe}</p>
          </div>
        </div>
        <div>
          <span className="text-xs text-muted-foreground">Symbols:</span>
          <div className="mt-1 flex flex-wrap gap-1">
            {config.general.symbols.map((s) => (
              <Badge key={s} variant="secondary">
                {s}
              </Badge>
            ))}
            {config.general.symbols.length === 0 && (
              <span className="text-xs text-muted-foreground">None</span>
            )}
          </div>
        </div>
        <div>
          <span className="text-xs text-muted-foreground">Indicators:</span>
          <p className="text-sm">
            {config.indicators.length > 0
              ? config.indicators.map((i) => i.type).join(", ")
              : "None"}
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <span className="text-xs text-muted-foreground">Entry conditions:</span>
            <p className="text-sm">{config.conditions.entry.length}</p>
          </div>
          <div>
            <span className="text-xs text-muted-foreground">Exit conditions:</span>
            <p className="text-sm">{config.conditions.exit.length}</p>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <div>
            <span className="text-xs text-muted-foreground">Stop loss:</span>
            <p className="text-sm">
              {config.risk.stop_loss.type === "fixed_pct"
                ? `${config.risk.stop_loss.value}%`
                : `ATR x${config.risk.stop_loss.value}`}
            </p>
          </div>
          <div>
            <span className="text-xs text-muted-foreground">Take profit:</span>
            <p className="text-sm">
              {config.risk.take_profit.type === "fixed_pct"
                ? `${config.risk.take_profit.value}%`
                : `${config.risk.take_profit.value}R`}
            </p>
          </div>
          <div>
            <span className="text-xs text-muted-foreground">Position sizing:</span>
            <p className="text-sm">
              {config.risk.position_sizing.type}
              {config.risk.position_sizing.value != null
                ? ` (${config.risk.position_sizing.value})`
                : ""}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export function StrategyWizard({ onCancel }: StrategyWizardProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const createMutation = useCreateStrategyApiStrategiesPost();

  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [config, setConfig] = useState<DslConfig>(emptyDslConfig());

  const updateStrategyType = (type: string) => {
    setConfig({ ...config, general: { ...config.general, strategy_type: type } });
  };

  const canNext = (): boolean => {
    switch (step) {
      case 0:
        return name.trim().length > 0;
      case 1:
        return config.general.symbols.length > 0;
      case 2:
        return config.indicators.length > 0;
      case 3:
        return true; // conditions optional
      case 4:
        return true; // risk has defaults
      case 5:
        return true;
      default:
        return false;
    }
  };

  const handleCreate = () => {
    createMutation.mutate(
      {
        data: {
          name,
          description: description || undefined,
          strategy_type: config.general.strategy_type,
          dsl_config: config as unknown as Record<string, unknown>,
        },
      },
      {
        onSuccess: (res) => {
          queryClient.invalidateQueries({
            queryKey: getListStrategiesApiStrategiesGetQueryKey(),
          });
          toast.success("Strategy created");
          navigate({
            to: "/strategies/$strategyId",
            params: { strategyId: String(res.data.id) },
          });
        },
        onError: () => toast.error("Failed to create strategy"),
      },
    );
  };

  return (
    <div className="mx-auto max-w-3xl p-6">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-xl font-bold">New Strategy Wizard</h1>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>

      <StepIndicator currentStep={step} />

      <div className="min-h-[300px]">
        {step === 0 && (
          <StepNameType
            name={name}
            description={description}
            strategyType={config.general.strategy_type}
            onNameChange={setName}
            onDescriptionChange={setDescription}
            onStrategyTypeChange={updateStrategyType}
          />
        )}
        {step === 1 && <StepMarkets config={config} onConfigChange={setConfig} />}
        {step === 2 && <IndicatorsTab config={config} onConfigChange={setConfig} />}
        {step === 3 && <ConditionsTab config={config} onConfigChange={setConfig} />}
        {step === 4 && <RiskTab config={config} onConfigChange={setConfig} />}
        {step === 5 && <StepReview name={name} description={description} config={config} />}
      </div>

      <div className="mt-8 flex items-center justify-between border-t pt-4">
        <Button variant="outline" onClick={() => setStep((s) => s - 1)} disabled={step === 0}>
          Back
        </Button>
        <span className="text-xs text-muted-foreground">
          Step {step + 1} of {STEPS.length}
        </span>
        {step < STEPS.length - 1 ? (
          <Button onClick={() => setStep((s) => s + 1)} disabled={!canNext()}>
            Next
          </Button>
        ) : (
          <Button onClick={handleCreate} disabled={createMutation.isPending}>
            {createMutation.isPending ? "Creating..." : "Create Strategy"}
          </Button>
        )}
      </div>
    </div>
  );
}
