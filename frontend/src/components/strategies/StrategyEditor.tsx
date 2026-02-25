import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import type { StrategyResponse, ValidationResult } from "@/api/generated/models";
import {
  getGetStrategyApiStrategiesStrategyIdGetQueryKey,
  getListStrategiesApiStrategiesGetQueryKey,
  useListTemplatesApiStrategiesTemplatesGet,
  useUpdateStrategyApiStrategiesStrategyIdPut,
  useValidateStrategyApiStrategiesStrategyIdValidatePost,
} from "@/api/generated/strategies/strategies";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
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

type EditorMode = "visual" | "yaml" | "split";

interface StrategyEditorProps {
  strategy: StrategyResponse;
}

interface TemplateItem {
  name?: string;
  description?: string;
  strategy_type?: string;
  dsl_config?: Record<string, unknown>;
}

function VisualEditorContent({
  name,
  description,
  config,
  onNameChange,
  onDescriptionChange,
  onConfigChange,
}: {
  name: string;
  description: string;
  config: DslConfig;
  onNameChange: (name: string) => void;
  onDescriptionChange: (desc: string) => void;
  onConfigChange: (config: DslConfig) => void;
}) {
  return (
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
          onNameChange={onNameChange}
          onDescriptionChange={onDescriptionChange}
          onConfigChange={onConfigChange}
        />
      </TabsContent>

      <TabsContent value="indicators" className="mt-4">
        <IndicatorsTab config={config} onConfigChange={onConfigChange} />
      </TabsContent>

      <TabsContent value="conditions" className="mt-4">
        <ConditionsTab config={config} onConfigChange={onConfigChange} />
      </TabsContent>

      <TabsContent value="risk" className="mt-4">
        <RiskTab config={config} onConfigChange={onConfigChange} />
      </TabsContent>

      <TabsContent value="time" className="mt-4">
        <TimeTab config={config} onConfigChange={onConfigChange} />
      </TabsContent>
    </Tabs>
  );
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
  const [confirmTemplateReplace, setConfirmTemplateReplace] = useState<TemplateItem | null>(null);

  const updateMutation = useUpdateStrategyApiStrategiesStrategyIdPut();
  const validateMutation = useValidateStrategyApiStrategiesStrategyIdValidatePost();
  const templatesQuery = useListTemplatesApiStrategiesTemplatesGet();
  const templates = (templatesQuery.data?.data ?? []) as TemplateItem[];

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
                const result = res.data as { valid: boolean; errors: string[]; warnings?: string[] };
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

  const handleLoadTemplate = (tmpl: TemplateItem) => {
    setConfirmTemplateReplace(tmpl);
  };

  const confirmLoadTemplate = () => {
    if (!confirmTemplateReplace?.dsl_config) return;
    const parsed = parseDslConfig(confirmTemplateReplace.dsl_config);
    setConfig(parsed);
    setConfirmTemplateReplace(null);
    toast.success(`Template "${confirmTemplateReplace.name}" loaded`);
  };

  const handleExportTemplate = () => {
    // Copy current config as JSON to clipboard
    const json = JSON.stringify(config, null, 2);
    navigator.clipboard.writeText(json).then(
      () => toast.success("Config copied to clipboard as JSON template"),
      () => toast.error("Failed to copy to clipboard"),
    );
  };

  const editorModes: EditorMode[] = ["visual", "yaml", "split"];
  const modeLabels: Record<EditorMode, string> = {
    visual: "Visual",
    yaml: "YAML",
    split: "Split",
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
          {/* Template dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                Load Template
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {templatesQuery.isLoading && <DropdownMenuItem disabled>Loading...</DropdownMenuItem>}
              {templates.length === 0 && !templatesQuery.isLoading && (
                <DropdownMenuItem disabled>No templates available</DropdownMenuItem>
              )}
              {templates.map((tmpl, idx) => (
                <DropdownMenuItem key={tmpl.name ?? idx} onClick={() => handleLoadTemplate(tmpl)}>
                  <div>
                    <div className="text-sm font-medium">{tmpl.name ?? `Template ${idx + 1}`}</div>
                    {tmpl.description && (
                      <div className="text-xs text-muted-foreground">{tmpl.description}</div>
                    )}
                  </div>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button variant="outline" size="sm" onClick={handleExportTemplate}>
            Export as Template
          </Button>
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
        {editorModes.map((m) => (
          <Button
            key={m}
            type="button"
            variant={editorMode === m ? "default" : "ghost"}
            size="sm"
            className={cn(
              "first:rounded-r-none last:rounded-l-none [&:not(:first-child):not(:last-child)]:rounded-none",
              editorMode !== m && "text-foreground",
            )}
            onClick={() => setEditorMode(m)}
          >
            {modeLabels[m]}
          </Button>
        ))}
      </div>

      {/* Editor content */}
      {editorMode === "yaml" && <YamlEditor config={config} onConfigChange={setConfig} />}

      {editorMode === "visual" && (
        <VisualEditorContent
          name={name}
          description={description}
          config={config}
          onNameChange={setName}
          onDescriptionChange={setDescription}
          onConfigChange={setConfig}
        />
      )}

      {editorMode === "split" && (
        <ResizablePanelGroup orientation="horizontal" className="min-h-[500px] rounded-lg border">
          <ResizablePanel defaultSize={50} minSize={30}>
            <div className="h-full overflow-auto p-4">
              <VisualEditorContent
                name={name}
                description={description}
                config={config}
                onNameChange={setName}
                onDescriptionChange={setDescription}
                onConfigChange={setConfig}
              />
            </div>
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel defaultSize={50} minSize={30}>
            <div className="h-full overflow-auto p-4">
              <YamlEditor config={config} onConfigChange={setConfig} />
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      )}

      {/* Validation summary panel */}
      {showValidationPanel && (
        <div className="mt-6">
          <ValidationPanel config={config} />
        </div>
      )}

      {/* Confirm template replace dialog */}
      <Dialog
        open={!!confirmTemplateReplace}
        onOpenChange={(isOpen) => !isOpen && setConfirmTemplateReplace(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Load Template</DialogTitle>
            <DialogDescription>
              This will replace your current DSL configuration with the template &quot;
              {confirmTemplateReplace?.name}&quot;. Unsaved changes will be lost.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmTemplateReplace(null)}>
              Cancel
            </Button>
            <Button onClick={confirmLoadTemplate}>Replace Config</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
