import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { DslConfig } from "./types";

interface RiskTabProps {
  config: DslConfig;
  onConfigChange: (config: DslConfig) => void;
}

const STOP_LOSS_TYPES = [
  { value: "fixed_pct", label: "Fixed %" },
  { value: "atr_based", label: "ATR-based" },
] as const;

const TAKE_PROFIT_TYPES = [
  { value: "fixed_pct", label: "Fixed %" },
  { value: "rr_ratio", label: "R:R Ratio" },
] as const;

const POSITION_SIZING_TYPES = [
  { value: "fixed", label: "Fixed Size" },
  { value: "percent_equity", label: "% Equity" },
  { value: "kelly", label: "Kelly Criterion" },
] as const;

interface RiskPreset {
  label: string;
  description: string;
  sl: number;
  tp: number;
}

const RISK_PRESETS: RiskPreset[] = [
  { label: "Conservative", description: "1% SL / 2% TP", sl: 1, tp: 2 },
  { label: "Moderate", description: "2% SL / 4% TP", sl: 2, tp: 4 },
  { label: "Aggressive", description: "3% SL / 6% TP", sl: 3, tp: 6 },
];

function RiskRewardDisplay({
  slValue,
  tpValue,
  slType,
  tpType,
}: {
  slValue: number;
  tpValue: number;
  slType: string;
  tpType: string;
}) {
  // Only show R:R for comparable types (both fixed_pct)
  if (slType !== "fixed_pct" || tpType !== "fixed_pct" || slValue === 0) {
    return null;
  }

  const ratio = tpValue / slValue;
  const riskPct = Math.max(0, Math.min(100, (slValue / (slValue + tpValue)) * 100));
  const rewardPct = 100 - riskPct;

  let ratingColor = "text-red-500";
  if (ratio >= 2) ratingColor = "text-green-500";
  else if (ratio >= 1.5) ratingColor = "text-yellow-500";

  return (
    <div className="space-y-2 rounded-lg border bg-muted/30 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">Risk:Reward Ratio</span>
        <Badge variant="outline" className={ratingColor}>
          1:{ratio.toFixed(1)}
        </Badge>
      </div>
      <div className="flex h-3 overflow-hidden rounded-full">
        <div
          className="bg-red-400 transition-all"
          style={{ width: `${riskPct}%` }}
          title={`Risk: ${slValue}%`}
        />
        <div
          className="bg-green-400 transition-all"
          style={{ width: `${rewardPct}%` }}
          title={`Reward: ${tpValue}%`}
        />
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>Risk: {slValue}%</span>
        <span>Reward: {tpValue}%</span>
      </div>
    </div>
  );
}

export function RiskTab({ config, onConfigChange }: RiskTabProps) {
  const updateRisk = (patch: Partial<DslConfig["risk"]>) => {
    onConfigChange({ ...config, risk: { ...config.risk, ...patch } });
  };

  const applyPreset = (preset: RiskPreset) => {
    updateRisk({
      stop_loss: { type: "fixed_pct", value: preset.sl },
      take_profit: { type: "fixed_pct", value: preset.tp },
    });
  };

  const isPresetActive = (preset: RiskPreset) =>
    config.risk.stop_loss.type === "fixed_pct" &&
    config.risk.stop_loss.value === preset.sl &&
    config.risk.take_profit.type === "fixed_pct" &&
    config.risk.take_profit.value === preset.tp;

  return (
    <div className="space-y-6">
      {/* Presets */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium">Quick Presets</h3>
        <div className="flex gap-2">
          {RISK_PRESETS.map((preset) => (
            <Button
              key={preset.label}
              variant={isPresetActive(preset) ? "default" : "outline"}
              size="sm"
              onClick={() => applyPreset(preset)}
              className="flex-1"
            >
              <div className="text-center">
                <div className="text-xs font-medium">{preset.label}</div>
                <div className="text-[10px] opacity-70">{preset.description}</div>
              </div>
            </Button>
          ))}
        </div>
      </div>

      {/* R:R Visualization */}
      <RiskRewardDisplay
        slValue={config.risk.stop_loss.value}
        tpValue={config.risk.take_profit.value}
        slType={config.risk.stop_loss.type}
        tpType={config.risk.take_profit.type}
      />

      {/* Stop Loss */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Stop Loss</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <Label className="text-xs">Type</Label>
            <Select
              value={config.risk.stop_loss.type}
              onValueChange={(v) =>
                updateRisk({ stop_loss: { ...config.risk.stop_loss, type: v } })
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {STOP_LOSS_TYPES.map((t) => (
                  <SelectItem key={t.value} value={t.value}>
                    {t.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">
              Value {config.risk.stop_loss.type === "fixed_pct" ? "(%)" : "(ATR multiplier)"}
            </Label>
            <Input
              type="number"
              step="0.1"
              value={config.risk.stop_loss.value}
              onChange={(e) =>
                updateRisk({
                  stop_loss: { ...config.risk.stop_loss, value: Number(e.target.value) },
                })
              }
            />
          </div>
        </div>
        {config.risk.stop_loss.type === "atr_based" && (
          <p className="text-[10px] text-muted-foreground">
            Stop loss = ATR(14) x {config.risk.stop_loss.value}. Ensure ATR indicator is configured.
          </p>
        )}
      </div>

      {/* Take Profit */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Take Profit</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <Label className="text-xs">Type</Label>
            <Select
              value={config.risk.take_profit.type}
              onValueChange={(v) =>
                updateRisk({ take_profit: { ...config.risk.take_profit, type: v } })
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TAKE_PROFIT_TYPES.map((t) => (
                  <SelectItem key={t.value} value={t.value}>
                    {t.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">
              Value {config.risk.take_profit.type === "fixed_pct" ? "(%)" : "(R:R)"}
            </Label>
            <Input
              type="number"
              step="0.1"
              value={config.risk.take_profit.value}
              onChange={(e) =>
                updateRisk({
                  take_profit: { ...config.risk.take_profit, value: Number(e.target.value) },
                })
              }
            />
          </div>
        </div>
      </div>

      {/* Trailing Stop */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Trailing Stop</h3>
        <div className="flex items-center gap-3">
          <Checkbox
            id="trailing-stop-enabled"
            checked={config.risk.trailing_stop_pct != null}
            onCheckedChange={(v) =>
              updateRisk({ trailing_stop_pct: v === true ? 1.5 : undefined })
            }
          />
          <Label htmlFor="trailing-stop-enabled" className="text-sm font-normal">
            Enable trailing stop
          </Label>
        </div>
        {config.risk.trailing_stop_pct != null && (
          <div className="space-y-1.5">
            <Label className="text-xs">Trail Distance (%)</Label>
            <Input
              type="number"
              step="0.1"
              min="0.1"
              value={config.risk.trailing_stop_pct}
              onChange={(e) => updateRisk({ trailing_stop_pct: Number(e.target.value) })}
            />
            <p className="text-[10px] text-muted-foreground">
              Stop moves with price by {config.risk.trailing_stop_pct}%.
            </p>
          </div>
        )}
      </div>

      {/* Position Sizing */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Position Sizing</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <Label className="text-xs">Method</Label>
            <Select
              value={config.risk.position_sizing.type}
              onValueChange={(v) =>
                updateRisk({ position_sizing: { ...config.risk.position_sizing, type: v } })
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {POSITION_SIZING_TYPES.map((t) => (
                  <SelectItem key={t.value} value={t.value}>
                    {t.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {config.risk.position_sizing.type !== "kelly" && (
            <div className="space-y-1.5">
              <Label className="text-xs">
                {config.risk.position_sizing.type === "fixed" ? "Size" : "% of Equity"}
              </Label>
              <Input
                type="number"
                step="0.1"
                value={config.risk.position_sizing.value ?? 0}
                onChange={(e) =>
                  updateRisk({
                    position_sizing: {
                      ...config.risk.position_sizing,
                      value: Number(e.target.value),
                    },
                  })
                }
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
