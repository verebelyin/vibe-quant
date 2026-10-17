import { Link } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  useLaunchScreeningApiBacktestScreeningPost,
  useLaunchValidationApiBacktestValidationPost,
  useValidateCoverageApiBacktestValidateCoveragePost,
} from "@/api/generated/backtest/backtest";
import { useListSymbolsApiDataSymbolsGet } from "@/api/generated/data/data";
import type {
  BacktestLaunchRequestEntryOrder,
  CoverageCheckResponseCoverage,
} from "@/api/generated/models";
import { useListLatencyPresetsApiSettingsLatencyPresetsGet } from "@/api/generated/settings/settings";
import { useListStrategiesApiStrategiesGet } from "@/api/generated/strategies/strategies";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { DatasetRangeIndicator } from "@/components/ui/DatasetRangeIndicator";
import { DatePicker } from "@/components/ui/date-picker";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useDatasetDateRange } from "@/hooks/useDatasetDateRange";
import { useIndicatorCatalog } from "@/hooks/useIndicatorCatalog";
import { LIQUIDATION_MODE_OPTIONS, type LiquidationMode } from "@/lib/liquidation";
import {
  DEFAULT_MAX_POSITION_PCT,
  DEFAULT_RISK_PCT,
  MAX_MAX_POSITION_PCT,
  MAX_RISK_PCT,
  sizingProblems,
} from "@/lib/sizing";
import { cn } from "@/lib/utils";
import { buildBacktestLaunchBody } from "./backtestLaunchBody";
import { parseMaxMarginPct } from "./marginLimit";
import { PreflightStatus } from "./PreflightStatus";
import {
  SweepBuilder,
  type SweepCatalog,
  type SweepConfig,
  strategyTranslationProblem,
  sweepableParamsFor,
  sweepProblem,
  sweepToPayload,
} from "./SweepBuilder";

type BacktestMode = "screening" | "validation";

/** Maker knobs are integer ticks/bars >= 1 (the API rejects 0 and non-integers). */
function positiveInt(raw: string): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return 1;
  return Math.max(1, Math.trunc(n));
}

export function BacktestLaunchForm() {
  // Form state
  const [strategyId, setStrategyId] = useState<string>("");
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [mode, setMode] = useState<BacktestMode>("screening");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [initialBalance, setInitialBalance] = useState(10000);
  const [leverage, setLeverage] = useState(10);
  // Sizing (vibe-quant-2gyk1.2): "" = an empty field, which is a problem.
  const [maxPos, setMaxPos] = useState<number | "">(DEFAULT_MAX_POSITION_PCT);
  const [riskPct, setRiskPct] = useState<number | "">(DEFAULT_RISK_PCT);
  // Margin limit (T6, vibe-quant-t8nrv): a percent of equity, "" = no limit at all.
  const [maxMarginPct, setMaxMarginPct] = useState<string>("");
  // Liquidation (vibe-quant-2gyk1.1.10, E1): off sends no liquidation keys.
  const [liquidationMode, setLiquidationMode] = useState<LiquidationMode>("off");
  const [timeframe, setTimeframe] = useState("1m");
  // Validation-only fields
  const [latencyPreset, setLatencyPreset] = useState("");
  const [bnbDiscount, setBnbDiscount] = useState(false);
  // Validation-only: post-only maker entries (D3-6), with their knobs.
  const [entryOrder, setEntryOrder] = useState<BacktestLaunchRequestEntryOrder>("market");
  const [makerOffsetTicks, setMakerOffsetTicks] = useState(1);
  const [makerEntryTimeoutBars, setMakerEntryTimeoutBars] = useState(3);
  // Sweep state
  const [sweepEnabled, setSweepEnabled] = useState(false);
  const [sweepConfig, setSweepConfig] = useState<SweepConfig>({ params: [] });

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
  const datasetRange = useDatasetDateRange();
  const catalogQuery = useIndicatorCatalog();

  // Reset sweep config when strategy changes to avoid stale indicator indices
  useEffect(() => {
    setSweepEnabled(false);
    setSweepConfig({ params: [] });
  }, [strategyId]);

  // Auto-populate dates from dataset coverage
  useEffect(() => {
    if (!startDate && datasetRange.minStart) setStartDate(datasetRange.minStart);
    if (!endDate && datasetRange.maxEnd) setEndDate(datasetRange.maxEnd);
  }, [datasetRange.minStart, datasetRange.maxEnd]); // eslint-disable-line react-hooks/exhaustive-deps

  // Mutations
  const coverageMutation = useValidateCoverageApiBacktestValidateCoveragePost();
  const screeningMutation = useLaunchScreeningApiBacktestScreeningPost();
  const validationMutation = useLaunchValidationApiBacktestValidationPost();

  const strategies =
    strategiesQuery.data?.status === 200 ? strategiesQuery.data.data.strategies : [];
  const symbols = symbolsQuery.data?.status === 200 ? symbolsQuery.data.data : [];
  const latencyPresets = latencyQuery.data?.status === 200 ? latencyQuery.data.data : [];

  // Sweepable params of the selected strategy, keyed as the backend's compiled config fields
  const selectedStrategy = strategies.find((s) => String(s.id) === strategyId);
  // Int-ness of a param comes from the catalog: no sweep is built or launched without it
  const catalogReady = catalogQuery.data?.status === 200;
  const sweepableParams = useMemo(() => {
    if (!selectedStrategy) return [];
    const catalog: SweepCatalog | undefined =
      catalogQuery.data?.status === 200
        ? Object.fromEntries(catalogQuery.data.data.indicators.map((i) => [i.type_name, i]))
        : undefined;
    return sweepableParamsFor(selectedStrategy.dsl_config as Record<string, unknown>, catalog);
  }, [selectedStrategy, catalogQuery.data]);

  const strategyProblem = useMemo(
    () =>
      selectedStrategy
        ? strategyTranslationProblem(selectedStrategy.dsl_config as Record<string, unknown>)
        : null,
    [selectedStrategy],
  );

  // The margin limit (T6) is optional — an empty field means OFF — but a value outside
  // (0, 100] is refused here, before the launch.
  const marginLimit = parseMaxMarginPct(maxMarginPct);
  const sizingIssues = [
    ...sizingProblems(maxPos, riskPct, leverage),
    ...(marginLimit.kind === "invalid" ? [marginLimit.message] : []),
  ];

  const isLaunching = screeningMutation.isPending || validationMutation.isPending;

  const canSubmit =
    strategyId !== "" &&
    selectedSymbols.length > 0 &&
    startDate !== "" &&
    endDate !== "" &&
    !strategyProblem &&
    sizingIssues.length === 0 &&
    !(sweepEnabled && (!catalogReady || sweepProblem(sweepConfig, sweepableParams))) &&
    !isLaunching;

  function handleSymbolToggle(symbol: string) {
    setSelectedSymbols((prev) =>
      prev.includes(symbol) ? prev.filter((s) => s !== symbol) : [...prev, symbol],
    );
  }

  function applyDatePreset(months: number) {
    const end = datasetRange.maxEnd ? new Date(datasetRange.maxEnd + "T00:00:00") : new Date();
    const start = new Date(end);
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

    // Overfitting toggles used to be sent here but no backtest runner ever
    // applied them (the API now rejects them); they run in discovery / the
    // overfitting CLI. Sizing is validated by sizingProblems (canSubmit).
    if (maxPos === "" || riskPct === "" || sizingIssues.length > 0) return;
    const payload = buildBacktestLaunchBody({
      strategyId: Number(strategyId),
      symbols: selectedSymbols,
      timeframe,
      startDate,
      endDate,
      initialBalance,
      leverage,
      maxPositionPct: maxPos,
      riskPct,
      maxMarginPct,
      liquidationMode,
      sweep: sweepEnabled && sweepConfig.params.length > 0 ? sweepToPayload(sweepConfig) : null,
      mode,
      latencyPreset,
      bnbDiscount,
      entryOrder,
      makerOffsetTicks,
      makerEntryTimeoutBars,
    });

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
    <div className="mx-auto max-w-5xl space-y-8">
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

          {strategyProblem && (
            <div className="rounded-md border border-destructive bg-destructive/10 px-3 py-2">
              <p className="text-sm text-destructive">
                {strategyProblem}. Fix the strategy before launching: the backtest would fail.
              </p>
            </div>
          )}

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
          {strategyId !== "" && sweepableParams.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Parameter Sweep</Label>
                <Button
                  type="button"
                  variant={sweepEnabled ? "default" : "outline"}
                  size="sm"
                  disabled={!catalogReady}
                  onClick={() => {
                    setSweepEnabled((v) => !v);
                    if (!sweepEnabled) setSweepConfig({ params: [] });
                  }}
                >
                  {sweepEnabled ? "Enabled" : "Disabled"}
                </Button>
              </div>
              {!catalogReady && (
                <p className="text-xs text-muted-foreground">
                  {catalogQuery.isError
                    ? "Indicator catalog failed to load: sweeps are unavailable."
                    : "Loading indicator catalog..."}
                </p>
              )}
              {sweepEnabled && (
                <SweepBuilder
                  params={sweepableParams}
                  value={sweepConfig}
                  onChange={setSweepConfig}
                />
              )}
            </div>
          )}

          {/* Date range */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Label>Date Range</Label>
              <DatasetRangeIndicator
                items={datasetRange.items}
                minStart={datasetRange.minStart}
                maxEnd={datasetRange.maxEnd}
                isLoading={datasetRange.isLoading}
                onApply={(start, end) => {
                  setStartDate(start);
                  setEndDate(end);
                }}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="start-date" className="text-xs text-muted-foreground">
                  Start
                </Label>
                <DatePicker
                  id="start-date"
                  value={startDate}
                  onChange={setStartDate}
                  placeholder="Start date"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="end-date" className="text-xs text-muted-foreground">
                  End
                </Label>
                <DatePicker
                  id="end-date"
                  value={endDate}
                  onChange={setEndDate}
                  placeholder="End date"
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
                <SelectItem value="1s">1 second (sub-bar data required)</SelectItem>
                <SelectItem value="5s">5 seconds (sub-bar data required)</SelectItem>
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
                step={1}
                value={leverage}
                onChange={(e) => setLeverage(Number(e.target.value))}
              />
              <p className="text-xs text-muted-foreground">
                Venue leverage; changes results only with Liquidation on.
              </p>
            </div>
          </div>

          {/* Sizing (vibe-quant-2gyk1.2) */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="max-exposure">Max exposure (× equity)</Label>
              <Input
                id="max-exposure"
                type="number"
                min={0}
                max={MAX_MAX_POSITION_PCT}
                step="any"
                value={maxPos}
                onChange={(e) => setMaxPos(e.target.value === "" ? "" : Number(e.target.value))}
              />
              <p className="text-xs text-muted-foreground">
                Largest position per symbol, as a multiple of your account (0.5 = half the account).
                Must not exceed leverage.
                {selectedSymbols.length > 1 &&
                  ` Applies per symbol: ${selectedSymbols.length} symbols can hold up to ${selectedSymbols.length} × this in total.`}
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="risk-per-trade">Risk per trade (%)</Label>
              <Input
                id="risk-per-trade"
                type="number"
                min={0}
                max={MAX_RISK_PCT}
                step="any"
                value={riskPct}
                onChange={(e) => setRiskPct(e.target.value === "" ? "" : Number(e.target.value))}
              />
              <p className="text-xs text-muted-foreground">
                How much of the account one stopped-out trade may lose (2 = 2%).
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="max-margin-pct">Max margin in use (% of equity)</Label>
              <Input
                id="max-margin-pct"
                type="text"
                inputMode="decimal"
                placeholder="off"
                value={maxMarginPct}
                onChange={(e) => setMaxMarginPct(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Refuse an entry that would push the margin in use over this share of the account (50
                = half). Empty means no limit.
              </p>
            </div>
            {sizingIssues.length > 0 && (
              <ul className="col-span-2 space-y-1 text-xs text-destructive" role="alert">
                {sizingIssues.map((m) => (
                  <li key={m}>{m}</li>
                ))}
              </ul>
            )}
          </div>

          {/* Liquidation (vibe-quant-2gyk1.1.10, E1) */}
          <div className="space-y-2">
            <Label htmlFor="liquidation-mode">Liquidation</Label>
            <Select
              value={liquidationMode}
              onValueChange={(v) => setLiquidationMode(v as LiquidationMode)}
            >
              <SelectTrigger id="liquidation-mode" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {LIQUIDATION_MODE_OPTIONS.map((o) => (
                  <SelectItem key={o.value} value={o.value}>
                    {o.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {liquidationMode !== "off" && (
              <p className="text-xs text-muted-foreground">
                Positions are liquidated at the Binance maintenance margin for the chosen leverage.
              </p>
            )}
          </div>

          {/* Validation-only fields */}
          {mode === "validation" && (
            <div className="space-y-4 rounded-lg border border-border bg-card p-4">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
                Validation Settings
              </h3>

              {/* Latency preset */}
              <div className="space-y-2">
                <Label htmlFor="latency-preset">Latency Preset (tick data only)</Label>
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
                <p className="text-xs text-muted-foreground">
                  Validation releases orders on each symbol's next 1m bar; latency presets apply to
                  tick data only.
                </p>
              </div>

              {/* Entry orders: market fills or post-only maker limits (D3-6) */}
              <div className="space-y-2">
                <Label htmlFor="entry-order">Entry Orders</Label>
                <Select
                  value={entryOrder}
                  onValueChange={(v) => setEntryOrder(v as BacktestLaunchRequestEntryOrder)}
                >
                  <SelectTrigger id="entry-order" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="market">Market (fill at the release bar)</SelectItem>
                    <SelectItem value="post_only">Post-only maker (limit)</SelectItem>
                  </SelectContent>
                </Select>
                {entryOrder === "post_only" && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label
                          htmlFor="maker-offset-ticks"
                          className="text-xs text-muted-foreground"
                        >
                          Maker Offset (ticks)
                        </Label>
                        <Input
                          id="maker-offset-ticks"
                          type="number"
                          min={1}
                          step={1}
                          value={makerOffsetTicks}
                          onChange={(e) => setMakerOffsetTicks(positiveInt(e.target.value))}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label
                          htmlFor="maker-timeout-bars"
                          className="text-xs text-muted-foreground"
                        >
                          Entry Timeout (bars)
                        </Label>
                        <Input
                          id="maker-timeout-bars"
                          type="number"
                          min={1}
                          step={1}
                          value={makerEntryTimeoutBars}
                          onChange={(e) => setMakerEntryTimeoutBars(positiveInt(e.target.value))}
                        />
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Post-only entries are modelled optimistically (full size when price trades
                      through the limit). Each post-only validation run also backtests a market twin
                      for comparison.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* BNB fee discount (screening + validation; vibe-quant-t1ee0) */}
          <div className="flex items-center gap-2">
            <Checkbox
              id="bnb-discount"
              checked={bnbDiscount}
              onCheckedChange={(v) => setBnbDiscount(v === true)}
            />
            <Label htmlFor="bnb-discount" className="cursor-pointer">
              Pay fees in BNB (−10%)
            </Label>
          </div>
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
                to={`/results/$runId` as const}
                params={{ runId: String(launchResult.id) }}
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
