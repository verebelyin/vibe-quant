import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DslConfig } from "./types";

interface ValidationPanelProps {
  config: DslConfig;
}

interface CheckItem {
  label: string;
  passed: boolean;
}

interface Warning {
  message: string;
}

function computeChecks(config: DslConfig): CheckItem[] {
  return [
    { label: "Has entry conditions", passed: config.conditions.entry.length > 0 },
    { label: "Has exit conditions", passed: config.conditions.exit.length > 0 },
    { label: "Has stop loss", passed: config.risk.stop_loss.value > 0 },
    { label: "Has at least 2 indicators", passed: config.indicators.length >= 2 },
    { label: "Has symbols configured", passed: config.general.symbols.length > 0 },
  ];
}

function computeWarnings(config: DslConfig): Warning[] {
  const warnings: Warning[] = [];
  if (config.conditions.exit.length === 0) {
    warnings.push({ message: "No exit conditions defined" });
  }
  if (config.indicators.length === 0) {
    warnings.push({ message: "No indicators configured" });
  } else if (config.indicators.length === 1) {
    warnings.push({ message: "Only 1 indicator -- consider adding more" });
  }
  if (config.risk.stop_loss.value === 0) {
    warnings.push({ message: "No stop loss configured" });
  }
  if (config.conditions.entry.length === 0) {
    warnings.push({ message: "No entry conditions defined" });
  }
  if (config.general.symbols.length === 0) {
    warnings.push({ message: "No symbols configured" });
  }
  if (
    config.risk.stop_loss.type === "atr_based" &&
    !config.indicators.some((i) => i.type === "ATR")
  ) {
    warnings.push({ message: "ATR-based stop loss requires ATR indicator" });
  }
  return warnings;
}

function computeComplexity(config: DslConfig): number {
  return config.indicators.length + config.conditions.entry.length + config.conditions.exit.length;
}

export function ValidationPanel({ config }: ValidationPanelProps) {
  const checks = computeChecks(config);
  const warnings = computeWarnings(config);
  const complexity = computeComplexity(config);
  const passedCount = checks.filter((c) => c.passed).length;

  let complexityLabel = "Low";
  let complexityColor = "bg-green-500/10 text-green-600";
  if (complexity >= 8) {
    complexityLabel = "High";
    complexityColor = "bg-red-500/10 text-red-600";
  } else if (complexity >= 4) {
    complexityLabel = "Medium";
    complexityColor = "bg-yellow-500/10 text-yellow-600";
  }

  return (
    <div className="grid gap-4 md:grid-cols-3">
      {/* Complexity */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Complexity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold">{complexity}</span>
            <Badge className={complexityColor}>{complexityLabel}</Badge>
          </div>
          <p className="mt-1 text-[10px] text-muted-foreground">
            {config.indicators.length} indicators +{" "}
            {config.conditions.entry.length + config.conditions.exit.length} conditions
          </p>
        </CardContent>
      </Card>

      {/* Readiness Checklist */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">
            Readiness ({passedCount}/{checks.length})
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {checks.map((check) => (
            <div key={check.label} className="flex items-center gap-2 text-xs">
              <span className={check.passed ? "text-green-500" : "text-red-500"}>
                {check.passed ? "\u2713" : "\u2717"}
              </span>
              <span className={check.passed ? "text-foreground" : "text-muted-foreground"}>
                {check.label}
              </span>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Warnings */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Warnings ({warnings.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {warnings.length === 0 ? (
            <p className="text-xs text-green-500">No warnings</p>
          ) : (
            <ul className="space-y-1">
              {warnings.map((w) => (
                <li key={w.message} className="text-xs text-yellow-600">
                  {w.message}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
