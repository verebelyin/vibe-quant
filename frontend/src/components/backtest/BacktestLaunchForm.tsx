import { Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import {
  useLaunchScreeningApiBacktestScreeningPost,
  useLaunchValidationApiBacktestValidationPost,
  useValidateCoverageApiBacktestValidateCoveragePost,
} from "@/api/generated/backtest/backtest";
import { useListSymbolsApiDataSymbolsGet } from "@/api/generated/data/data";
import type { CoverageCheckResponseCoverage } from "@/api/generated/models";
import {
  useListLatencyPresetsApiSettingsLatencyPresetsGet,
  useListRiskConfigsApiSettingsRiskGet,
  useListSizingConfigsApiSettingsSizingGet,
} from "@/api/generated/settings/settings";
import { useListStrategiesApiStrategiesGet } from "@/api/generated/strategies/strategies";
import { parseDslConfig } from "@/components/strategies/editor/types";
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
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { PreflightStatus } from "./PreflightStatus";
import { SweepBuilder, type SweepConfig, sweepToPayload } from "./SweepBuilder";

type BacktestMode = "screening" | "validation";

export function BacktestLaunchForm() {
  // Form state
  const [strategyId, setStrategyId] = useState<string>("");
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [mode, setMode] = useState<BacktestMode>("screening");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [initialBalance, setInitialBalance] = useState(10000);
  const [leverage, setLeverage] = useState(10);
  const [timeframe, setTimeframe] = useState("1m");
  // Validation-only fields
  const [latencyPreset, setLatencyPreset] = useState("");
  const [sizingConfigId, setSizingConfigId] = useState<string>("");
  const [riskConfigId, setRiskConfigId] = useState<string>("");
  // Sweep state
  const [sweepEnabled, setSweepEnabled] = useState(false);
  const [sweepConfig, setSweepConfig] = useState<SweepConfig>({ params: [] });
  // Overfitting filter state
  const [dsrEnabled, setDsrEnabled] = useState(false);
  const [wfaEnabled, setWfaEnabled] = useState(false);
  const [wfaSplits, setWfaSplits] = useState(5);
  const [purgedKfoldEnabled, setPurgedKfoldEnabled] = useState(false);
  const [purgeEmbargoPct, setPurgeEmbargoPct] = useState(1);

  // Preflight / launch result state
  const [coverageResult, setCoverageResult] = useState<CoverageCheckResponseCoverage | null>(null);
  const [launchResult, setLaunchResult] = useState<{
    id: number;
    runMode: string;
  } | null>(null);

  // Queries
  const strategiesQuery = useListStrategiesApiStrategiesGet();
  const symbolsQuery = useListSymbolsApiDataSymbolsGet();
  const latencyQuery = useListLatencyPresetsApiSettingsLatencyPresetsGet();
  const sizingQuery = useListSizingConfigsApiSettingsSizingGet();
  const riskQuery = useListRiskConfigsApiSettingsRiskGet();

  // Mutations
  const coverageMutation = useValidateCoverageApiBacktestValidateCoveragePost();
  const screeningMutation = useLaunchScreeningApiBacktestScreeningPost();
  const validationMutation = useLaunchValidationApiBacktestValidationPost();

  const strategies =
    strategiesQuery.data?.status === 200 ? strategiesQuery.data.data.strategies : [];
  const symbols = symbolsQuery.data?.status === 200 ? symbolsQuery.data.data : [];
  const latencyPresets = latencyQuery.data?.status === 200 ? latencyQuery.data.data : [];
  const sizingConfigs = sizingQuery.data?.status === 200 ? sizingQuery.data.data : [];
  const riskConfigs = riskQuery.data?.status === 200 ? riskQuery.data.data : [];

  // Extract indicators from selected strategy for sweep builder
  const selectedStrategy = strategies.find((s) => String(s.id) === strategyId);
  const strategyIndicators = useMemo(() => {
    if (!selectedStrategy) return [];
    const dsl = parseDslConfig(selectedStrategy.dsl_config as Record<string, unknown>);
    return dsl.indicators;
  }, [selectedStrategy]);

  const isLaunching = screeningMutation.isPending || validationMutation.isPending;

  const canSubmit =
    strategyId !== "" &&
    selectedSymbols.length > 0 &&
    startDate !== "" &&
    endDate !== "" &&
    !isLaunching;

  function handleSymbolToggle(symbol: string) {
    setSelectedSymbols((prev) =>
      prev.includes(symbol) ? prev.filter((s) => s !== symbol) : [...prev, symbol],
    );
  }

  function applyDatePreset(months: number) {
    const end = new Date();
    const start = new Date();
    start.setMonth(start.getMonth() - months);
    setEndDate(end.toISOString().slice(0, 10));
    setStartDate(start.toISOString().slice(0, 10));
  }

  function handleSelectAllSymbols() {
    if (selectedSymbols.length === symbols.length) {
      setSelectedSymbols([]);
    } else {
      setSelectedSymbols([...symbols]);
    }
  }

  function handlePreflight() {
    if (selectedSymbols.length === 0 || !startDate || !endDate) return;
    setCoverageResult(null);
    coverageMutation.mutate(
      {
        data: {
          symbols: selectedSymbols,
          timeframe,
          start_date: startDate,
          end_date: endDate,
        },
      },
      {
        onSuccess: (resp) => {
          if (resp.status === 200) {
            setCoverageResult(resp.data.coverage);
          }
        },
      },
    );
  }

  function handleLaunch() {
    if (strategyId === "" || selectedSymbols.length === 0) return;
    setLaunchResult(null);

    const overfittingFilters: Record<string, boolean> = {};
    if (dsrEnabled) overfittingFilters.deflated_sharpe_ratio = true;
    if (wfaEnabled) overfittingFilters.walk_forward_analysis = true;
    if (purgedKfoldEnabled) overfittingFilters.purged_kfold_cv = true;

    const payload = {
      strategy_id: Number(strategyId),
      symbols: selectedSymbols,
      timeframe,
      start_date: startDate,
      end_date: endDate,
      parameters: {
        initial_balance: initialBalance,
        leverage,
        ...(sweepEnabled &&
          sweepConfig.params.length > 0 && {
            sweep: sweepToPayload(sweepConfig),
          }),
        ...(wfaEnabled && { wfa_splits: wfaSplits }),
        ...(purgedKfoldEnabled && { purge_embargo_pct: purgeEmbargoPct }),
      },
      ...(Object.keys(overfittingFilters).length > 0 && {
        overfitting_filters: overfittingFilters,
      }),
      ...(mode === "validation" && {
        latency_preset: latencyPreset || null,
        sizing_config_id: sizingConfigId === "" ? null : Number(sizingConfigId),
        risk_config_id: riskConfigId === "" ? null : Number(riskConfigId),
      }),
    };

    const mutation = mode === "screening" ? screeningMutation : validationMutation;

    mutation.mutate(
      { data: payload },
      {
        onSuccess: (resp) => {
          if (resp.status === 201) {
            setLaunchResult({
              id: resp.data.id,
              runMode: resp.data.run_mode,
            });
            toast.success("Backtest launched successfully", {
              description: `Run ID: ${resp.data.id} | Mode: ${resp.data.run_mode}`,
            });
          }
        },
        onError: (err: unknown) => {
          const message = err instanceof Error ? err.message : "Launch failed";
          toast.error("Launch failed", { description: message });
        },
      },
    );
  }

  const sectionClass = "space-y-4";

  return (
    <div className="mx-auto max-w-5xl space-y-8 p-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Backtest Launch</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Configure and launch screening or validation backtests.
        </p>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        {/* Left column */}
        <div className={sectionClass}>
          {/* Strategy */}
          <div className="space-y-2">
            <Label htmlFor="strategy-select">Strategy</Label>
            <Select value={strategyId} onValueChange={setStrategyId}>
              <SelectTrigger id="strategy-select" className="w-full">
                <SelectValue placeholder="Select a strategy..." />
              </SelectTrigger>
              <SelectContent>
                {strategies.map((s) => (
                  <SelectItem key={s.id} value={String(s.id)}>
                    {s.name} (v{s.version})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {strategiesQuery.isLoading && (
              <p className="text-xs text-muted-foreground">Loading strategies...</p>
            )}
          </div>

          {/* Mode toggle */}
          <div className="space-y-2">
            <Label>Mode</Label>
            <div className="inline-flex rounded-md border border-border">
              {(["screening", "validation"] as const).map((m) => (
                <Button
                  key={m}
                  type="button"
                  variant={mode === m ? "default" : "ghost"}
                  size="sm"
                  className={cn(
                    "capitalize first:rounded-r-none last:rounded-l-none",
                    mode !== m && "text-foreground",
                  )}
                  onClick={() => setMode(m)}
                >
                  {m}
                </Button>
              ))}
            </div>
          </div>

          {/* Parameter sweep toggle */}
          {strategyId !== "" && strategyIndicators.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Parameter Sweep</Label>
                <Button
                  type="button"
                  variant={sweepEnabled ? "default" : "outline"}
                  size="sm"
                  onClick={() => {
                    setSweepEnabled((v) => !v);
                    if (!sweepEnabled) setSweepConfig({ params: [] });
                  }}
                >
                  {sweepEnabled ? "Enabled" : "Disabled"}
                </Button>
              </div>
              {sweepEnabled && (
                <SweepBuilder
                  indicators={strategyIndicators}
                  value={sweepConfig}
                  onChange={setSweepConfig}
                />
              )}
            </div>
          )}

          {/* Date range */}
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="start-date">Start Date</Label>
                <Input
                  id="start-date"
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="end-date">End Date</Label>
                <Input
                  id="end-date"
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
            </div>
            <div className="flex items-center gap-1">
              <span className="mr-1 text-xs text-muted-foreground">Presets:</span>
              {(
                [
                  { label: "1M", months: 1 },
                  { label: "3M", months: 3 },
                  { label: "6M", months: 6 },
                  { label: "1Y", months: 12 },
                  { label: "2Y", months: 24 },
                ] as const
              ).map((p) => (
                <Button
                  key={p.label}
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-6 px-2 text-xs"
                  onClick={() => applyDatePreset(p.months)}
                >
                  {p.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Timeframe */}
          <div className="space-y-2">
            <Label htmlFor="timeframe-select">Timeframe</Label>
            <Select value={timeframe} onValueChange={setTimeframe}>
              <SelectTrigger id="timeframe-select" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1m">1 minute</SelectItem>
                <SelectItem value="5m">5 minutes</SelectItem>
                <SelectItem value="15m">15 minutes</SelectItem>
                <SelectItem value="1h">1 hour</SelectItem>
                <SelectItem value="4h">4 hours</SelectItem>
                <SelectItem value="1d">1 day</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Initial balance & leverage */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="initial-balance">Initial Balance (USD)</Label>
              <Input
                id="initial-balance"
                type="number"
                min={100}
                step={100}
                value={initialBalance}
                onChange={(e) => setInitialBalance(Number(e.target.value))}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="leverage">Leverage (1-125)</Label>
              <Input
                id="leverage"
                type="number"
                min={1}
                max={125}
                value={leverage}
                onChange={(e) => setLeverage(Number(e.target.value))}
              />
            </div>
          </div>

          {/* Overfitting Filters */}
          <div className="space-y-3 rounded-lg border border-border bg-card p-4">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
              Overfitting Filters
            </h3>

            <div className="flex items-center justify-between">
              <Label htmlFor="dsr-toggle" className="cursor-pointer">
                Deflated Sharpe Ratio (DSR)
              </Label>
              <Switch id="dsr-toggle" checked={dsrEnabled} onCheckedChange={setDsrEnabled} />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="wfa-toggle" className="cursor-pointer">
                  Walk-Forward Analysis (WFA)
                </Label>
                <Switch id="wfa-toggle" checked={wfaEnabled} onCheckedChange={setWfaEnabled} />
              </div>
              {wfaEnabled && (
                <div className="ml-4 space-y-1">
                  <Label htmlFor="wfa-splits" className="text-xs">
                    Splits
                  </Label>
                  <Input
                    id="wfa-splits"
                    type="number"
                    min={2}
                    max={20}
                    value={wfaSplits}
                    onChange={(e) => setWfaSplits(Number(e.target.value))}
                    className="h-8 w-24"
                  />
                </div>
              )}
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="kfold-toggle" className="cursor-pointer">
                  Purged K-Fold CV
                </Label>
                <Switch
                  id="kfold-toggle"
                  checked={purgedKfoldEnabled}
                  onCheckedChange={setPurgedKfoldEnabled}
                />
              </div>
              {purgedKfoldEnabled && (
                <div className="ml-4 space-y-1">
                  <Label htmlFor="purge-embargo" className="text-xs">
                    Purge embargo %
                  </Label>
                  <Input
                    id="purge-embargo"
                    type="number"
                    min={0}
                    max={50}
                    step={0.5}
                    value={purgeEmbargoPct}
                    onChange={(e) => setPurgeEmbargoPct(Number(e.target.value))}
                    className="h-8 w-24"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Validation-only fields */}
          {mode === "validation" && (
            <div className="space-y-4 rounded-lg border border-border bg-card p-4">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
                Validation Settings
              </h3>

              {/* Latency preset */}
              <div className="space-y-2">
                <Label htmlFor="latency-preset">Latency Preset</Label>
                <Select value={latencyPreset} onValueChange={setLatencyPreset}>
                  <SelectTrigger id="latency-preset" className="w-full">
                    <SelectValue placeholder="None (no latency simulation)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">None (no latency simulation)</SelectItem>
                    {latencyPresets.map((p) => (
                      <SelectItem key={p.name} value={p.name}>
                        {p.name} - {p.description} ({p.base_latency_ms}ms)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Sizing config */}
              <div className="space-y-2">
                <Label htmlFor="sizing-config">Sizing Config</Label>
                <Select value={sizingConfigId} onValueChange={setSizingConfigId}>
                  <SelectTrigger id="sizing-config" className="w-full">
                    <SelectValue placeholder="Default sizing" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__default__">Default sizing</SelectItem>
                    {sizingConfigs.map((c) => (
                      <SelectItem key={c.id} value={String(c.id)}>
                        {c.name} ({c.method})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Risk config */}
              <div className="space-y-2">
                <Label htmlFor="risk-config">Risk Config</Label>
                <Select value={riskConfigId} onValueChange={setRiskConfigId}>
                  <SelectTrigger id="risk-config" className="w-full">
                    <SelectValue placeholder="Default risk" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__default__">Default risk</SelectItem>
                    {riskConfigs.map((c) => (
                      <SelectItem key={c.id} value={String(c.id)}>
                        {c.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}
        </div>

        {/* Right column */}
        <div className={sectionClass}>
          {/* Symbol multi-select */}
          <div>
            <div className="mb-2 flex items-center justify-between">
              <Label>Symbols ({selectedSymbols.length} selected)</Label>
              <Button type="button" variant="link" size="xs" onClick={handleSelectAllSymbols}>
                {selectedSymbols.length === symbols.length ? "Deselect all" : "Select all"}
              </Button>
            </div>
            <div className="max-h-64 overflow-y-auto rounded-md border border-border bg-input p-2 dark:bg-input/30">
              {symbolsQuery.isLoading && (
                <p className="p-2 text-sm text-muted-foreground">Loading symbols...</p>
              )}
              {symbols.length === 0 && !symbolsQuery.isLoading && (
                <p className="p-2 text-sm italic text-muted-foreground">
                  No symbols available. Ingest data first.
                </p>
              )}
              {symbols.map((sym) => (
                <div
                  key={sym}
                  className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm text-foreground transition-colors hover:opacity-80"
                >
                  <Checkbox
                    id={`sym-${sym}`}
                    checked={selectedSymbols.includes(sym)}
                    onCheckedChange={() => handleSymbolToggle(sym)}
                  />
                  <Label htmlFor={`sym-${sym}`} className="cursor-pointer font-mono">
                    {sym}
                  </Label>
                </div>
              ))}
            </div>
          </div>

          {/* Preflight check */}
          <div>
            <Button
              type="button"
              variant="outline"
              className="w-full"
              disabled={
                selectedSymbols.length === 0 || !startDate || !endDate || coverageMutation.isPending
              }
              onClick={handlePreflight}
            >
              {coverageMutation.isPending ? "Checking coverage..." : "Run Preflight Check"}
            </Button>
          </div>

          {/* Preflight results */}
          {coverageResult && (
            <PreflightStatus
              coverage={coverageResult}
              requestedStart={startDate}
              requestedEnd={endDate}
            />
          )}

          {/* Launch button */}
          <div>
            <Button
              type="button"
              className="w-full py-3 font-semibold"
              disabled={!canSubmit}
              onClick={handleLaunch}
            >
              {isLaunching
                ? "Launching..."
                : `Launch ${mode === "screening" ? "Screening" : "Validation"}`}
            </Button>
          </div>

          {/* Launch success */}
          {launchResult && (
            <div className="rounded-md border border-green-600 bg-green-600/10 p-4">
              <p className="text-sm font-medium text-green-600">Backtest launched successfully!</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Run ID: {launchResult.id} | Mode: {launchResult.runMode}
              </p>
              <Link
                to={`/results?run=${launchResult.id}`}
                className="mt-2 inline-block text-sm font-medium text-accent-foreground underline"
              >
                View Results
              </Link>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
