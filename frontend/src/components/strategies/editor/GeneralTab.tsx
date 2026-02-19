import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { type DslConfig, STRATEGY_TYPES, TIMEFRAMES } from "./types";

interface GeneralTabProps {
  name: string;
  description: string;
  config: DslConfig;
  onNameChange: (name: string) => void;
  onDescriptionChange: (desc: string) => void;
  onConfigChange: (config: DslConfig) => void;
}

export function GeneralTab({
  name,
  description,
  config,
  onNameChange,
  onDescriptionChange,
  onConfigChange,
}: GeneralTabProps) {
  const updateGeneral = (patch: Partial<DslConfig["general"]>) => {
    onConfigChange({
      ...config,
      general: { ...config.general, ...patch },
    });
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
    updateGeneral({
      symbols: config.general.symbols.filter((s) => s !== sym),
    });
  };

  return (
    <div className="space-y-5">
      <div className="space-y-1.5">
        <Label htmlFor="strategy-name">Name</Label>
        <Input
          id="strategy-name"
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          placeholder="My Strategy"
        />
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="strategy-desc">Description</Label>
        <textarea
          id="strategy-desc"
          value={description}
          onChange={(e) => onDescriptionChange(e.target.value)}
          placeholder="Describe what this strategy does..."
          rows={3}
          className="border-input bg-background placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 w-full rounded-md border px-3 py-2 text-sm focus-visible:ring-[3px] focus-visible:outline-1"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <Label>Strategy Type</Label>
          <Select
            value={config.general.strategy_type}
            onValueChange={(v) => updateGeneral({ strategy_type: v })}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {STRATEGY_TYPES.map((t) => (
                <SelectItem key={t} value={t}>
                  {t.replace(/_/g, " ")}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label>Timeframe</Label>
          <Select
            value={config.general.timeframe}
            onValueChange={(v) => updateGeneral({ timeframe: v })}
          >
            <SelectTrigger>
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
                className="cursor-pointer"
                onClick={() => removeSymbol(sym)}
              >
                {sym} x
              </Badge>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
