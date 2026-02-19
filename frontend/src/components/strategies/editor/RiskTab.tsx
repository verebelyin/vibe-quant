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

export function RiskTab({ config, onConfigChange }: RiskTabProps) {
  const updateRisk = (patch: Partial<DslConfig["risk"]>) => {
    onConfigChange({ ...config, risk: { ...config.risk, ...patch } });
  };

  return (
    <div className="space-y-6">
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
              Value {config.risk.stop_loss.type === "fixed_pct" ? "(%)" : "(ATR mult)"}
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
