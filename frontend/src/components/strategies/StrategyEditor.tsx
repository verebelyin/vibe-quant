import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import type { StrategyResponse, ValidationResult } from "@/api/generated/models";
import {
  getGetStrategyApiStrategiesStrategyIdGetQueryKey,
  getListStrategiesApiStrategiesGetQueryKey,
  useUpdateStrategyApiStrategiesStrategyIdPut,
  useValidateStrategyApiStrategiesStrategyIdValidatePost,
} from "@/api/generated/strategies/strategies";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { ConditionsTab } from "./editor/ConditionsTab";
import { GeneralTab } from "./editor/GeneralTab";
import { IndicatorsTab } from "./editor/IndicatorsTab";
import { RiskTab } from "./editor/RiskTab";
import { TimeTab } from "./editor/TimeTab";
import { type DslConfig, emptyDslConfig, parseDslConfig } from "./editor/types";
import { ValidationPanel } from "./editor/ValidationPanel";
import { YamlEditor } from "./editor/YamlEditor";

type EditorMode = "visual" | "yaml";

interface StrategyEditorProps {
  strategy: StrategyResponse;
}

export function StrategyEditor({ strategy }: StrategyEditorProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const initialConfig = useMemo(
    () => parseDslConfig(strategy.dsl_config as Record<string, unknown>),
    [strategy.dsl_config],
  );

  const [name, setName] = useState(strategy.name);
  const [description, setDescription] = useState(strategy.description ?? "");
  const [config, setConfig] = useState<DslConfig>(
    Object.keys(strategy.dsl_config).length > 0 ? initialConfig : emptyDslConfig(),
  );
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [showValidationPanel, setShowValidationPanel] = useState(false);
  const [editorMode, setEditorMode] = useState<EditorMode>("visual");

  const updateMutation = useUpdateStrategyApiStrategiesStrategyIdPut();
  const validateMutation = useValidateStrategyApiStrategiesStrategyIdValidatePost();

  const handleSave = () => {
    updateMutation.mutate(
      {
        strategyId: strategy.id,
        data: {
          description: description || null,
          dsl_config: config as unknown as Record<string, unknown>,
        },
      },
      {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getGetStrategyApiStrategiesStrategyIdGetQueryKey(strategy.id),
          });
          queryClient.invalidateQueries({
            queryKey: getListStrategiesApiStrategiesGetQueryKey(),
          });
          toast.success("Strategy saved");
        },
        onError: () => toast.error("Failed to save strategy"),
      },
    );
  };

  const handleValidate = () => {
    // Save first, then validate
    updateMutation.mutate(
      {
        strategyId: strategy.id,
        data: {
          description: description || null,
          dsl_config: config as unknown as Record<string, unknown>,
        },
      },
      {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getGetStrategyApiStrategiesStrategyIdGetQueryKey(strategy.id),
          });
          validateMutation.mutate(
            { strategyId: strategy.id },
            {
              onSuccess: (res) => {
                const result = res.data;
                setValidation(result);
                setShowValidationPanel(true);
                if (result.valid) {
                  toast.success("Strategy is valid");
                } else {
                  toast.error(`Validation failed: ${result.errors.length} error(s)`);
                }
              },
              onError: () => toast.error("Validation request failed"),
            },
          );
        },
        onError: () => toast.error("Failed to save before validation"),
      },
    );
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => navigate({ to: "/strategies" })}>
            &larr; Back
          </Button>
          <h1 className="text-xl font-bold">{name || "Untitled Strategy"}</h1>
          <Badge variant="secondary">v{strategy.version}</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={handleValidate}
            disabled={updateMutation.isPending || validateMutation.isPending}
          >
            {validateMutation.isPending ? "Validating..." : "Validate"}
          </Button>
          <Button onClick={handleSave} disabled={updateMutation.isPending}>
            {updateMutation.isPending ? "Saving..." : "Save"}
          </Button>
        </div>
      </div>

      {/* Validation results */}
      {validation && !validation.valid && (
        <div className="mb-4 rounded-lg border border-destructive bg-destructive/10 p-3">
          <p className="text-sm font-medium text-destructive">Validation Errors</p>
          <ul className="mt-1 space-y-0.5">
            {validation.errors.map((err) => (
              <li key={err} className="text-xs text-destructive">
                {err}
              </li>
            ))}
          </ul>
        </div>
      )}

      {validation?.valid && (
        <div className="mb-4 rounded-lg border border-green-600 bg-green-600/10 p-3">
          <p className="text-sm font-medium text-green-600">Strategy is valid</p>
        </div>
      )}

      {/* Editor mode toggle */}
      <div className="mb-4 inline-flex rounded-md border border-border">
        {(["visual", "yaml"] as const).map((m) => (
          <Button
            key={m}
            type="button"
            variant={editorMode === m ? "default" : "ghost"}
            size="sm"
            className={cn(
              "capitalize first:rounded-r-none last:rounded-l-none",
              editorMode !== m && "text-foreground",
            )}
            onClick={() => setEditorMode(m)}
          >
            {m === "yaml" ? "YAML" : "Visual"}
          </Button>
        ))}
      </div>

      {editorMode === "yaml" ? (
        <YamlEditor config={config} onConfigChange={setConfig} />
      ) : (
        <Tabs defaultValue="general">
          <TabsList>
            <TabsTrigger value="general">General</TabsTrigger>
            <TabsTrigger value="indicators">Indicators</TabsTrigger>
            <TabsTrigger value="conditions">Conditions</TabsTrigger>
            <TabsTrigger value="risk">Risk</TabsTrigger>
            <TabsTrigger value="time">Time</TabsTrigger>
          </TabsList>

          <TabsContent value="general" className="mt-4">
            <GeneralTab
              name={name}
              description={description}
              config={config}
              onNameChange={setName}
              onDescriptionChange={setDescription}
              onConfigChange={setConfig}
            />
          </TabsContent>

          <TabsContent value="indicators" className="mt-4">
            <IndicatorsTab config={config} onConfigChange={setConfig} />
          </TabsContent>

          <TabsContent value="conditions" className="mt-4">
            <ConditionsTab config={config} onConfigChange={setConfig} />
          </TabsContent>

          <TabsContent value="risk" className="mt-4">
            <RiskTab config={config} onConfigChange={setConfig} />
          </TabsContent>

          <TabsContent value="time" className="mt-4">
            <TimeTab config={config} onConfigChange={setConfig} />
          </TabsContent>
        </Tabs>
      )}

      {/* Validation summary panel */}
      {showValidationPanel && (
        <div className="mt-6">
          <ValidationPanel config={config} />
        </div>
      )}
    </div>
  );
}
