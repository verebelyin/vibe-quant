import { Link } from "@tanstack/react-router";
import { useState } from "react";
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
import { PreflightStatus } from "./PreflightStatus";

type BacktestMode = "screening" | "validation";

export function BacktestLaunchForm() {
  // Form state
  const [strategyId, setStrategyId] = useState<number | "">("");
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [mode, setMode] = useState<BacktestMode>("screening");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [initialBalance, setInitialBalance] = useState(10000);
  const [leverage, setLeverage] = useState(10);
  const [timeframe, setTimeframe] = useState("1m");
  // Validation-only fields
  const [latencyPreset, setLatencyPreset] = useState("");
  const [sizingConfigId, setSizingConfigId] = useState<number | "">("");
  const [riskConfigId, setRiskConfigId] = useState<number | "">("");

  // Preflight / launch result state
  const [coverageResult, setCoverageResult] = useState<CoverageCheckResponseCoverage | null>(null);
  const [launchResult, setLaunchResult] = useState<{
    id: number;
    runMode: string;
  } | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);

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
    setLaunchError(null);
    setLaunchResult(null);

    const payload = {
      strategy_id: strategyId as number,
      symbols: selectedSymbols,
      timeframe,
      start_date: startDate,
      end_date: endDate,
      parameters: {
        initial_balance: initialBalance,
        leverage,
      },
      ...(mode === "validation" && {
        latency_preset: latencyPreset || null,
        sizing_config_id: sizingConfigId === "" ? null : sizingConfigId,
        risk_config_id: riskConfigId === "" ? null : riskConfigId,
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
          }
        },
        onError: (err: unknown) => {
          const message = err instanceof Error ? err.message : "Launch failed";
          setLaunchError(message);
        },
      },
    );
  }

  // --- Styles ---
  const labelClass = "block text-sm font-medium mb-1";
  const inputClass = "w-full rounded-md border px-3 py-2 text-sm focus:outline-none focus:ring-2";
  const sectionClass = "space-y-4";

  return (
    <div className="mx-auto max-w-5xl space-y-8 p-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: "hsl(var(--foreground))" }}>
          Backtest Launch
        </h1>
        <p className="mt-1 text-sm" style={{ color: "hsl(var(--muted-foreground))" }}>
          Configure and launch screening or validation backtests.
        </p>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        {/* Left column */}
        <div className={sectionClass}>
          {/* Strategy */}
          <div>
            <label className={labelClass} style={{ color: "hsl(var(--foreground))" }}>
              Strategy
            </label>
            <select
              className={inputClass}
              style={{
                backgroundColor: "hsl(var(--input))",
                borderColor: "hsl(var(--border))",
                color: "hsl(var(--foreground))",
              }}
              value={strategyId}
              onChange={(e) => setStrategyId(e.target.value ? Number(e.target.value) : "")}
            >
              <option value="">Select a strategy...</option>
              {strategies.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} (v{s.version})
                </option>
              ))}
            </select>
            {strategiesQuery.isLoading && (
              <p className="mt-1 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
                Loading strategies...
              </p>
            )}
          </div>

          {/* Mode toggle */}
          <div>
            <label className={labelClass} style={{ color: "hsl(var(--foreground))" }}>
              Mode
            </label>
            <div
              className="inline-flex rounded-md border"
              style={{ borderColor: "hsl(var(--border))" }}
            >
              {(["screening", "validation"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  className="px-4 py-2 text-sm font-medium capitalize transition-colors first:rounded-l-md last:rounded-r-md"
                  style={{
                    backgroundColor: mode === m ? "hsl(var(--accent))" : "hsl(var(--input))",
                    color: mode === m ? "hsl(var(--accent-foreground))" : "hsl(var(--foreground))",
                  }}
                  onClick={() => setMode(m)}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          {/* Date range */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className={labelClass} style={{ color: "hsl(var(--foreground))" }}>
                Start Date
              </label>
              <input
                type="date"
                className={inputClass}
                style={{
                  backgroundColor: "hsl(var(--input))",
                  borderColor: "hsl(var(--border))",
                  color: "hsl(var(--foreground))",
                }}
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>
            <div>
              <label className={labelClass} style={{ color: "hsl(var(--foreground))" }}>
                End Date
              </label>
              <input
                type="date"
                className={inputClass}
                style={{
                  backgroundColor: "hsl(var(--input))",
                  borderColor: "hsl(var(--border))",
                  color: "hsl(var(--foreground))",
                }}
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>
          </div>

          {/* Timeframe */}
          <div>
            <label className={labelClass} style={{ color: "hsl(var(--foreground))" }}>
              Timeframe
            </label>
            <select
              className={inputClass}
              style={{
                backgroundColor: "hsl(var(--input))",
                borderColor: "hsl(var(--border))",
                color: "hsl(var(--foreground))",
              }}
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
            >
              <option value="1m">1 minute</option>
              <option value="5m">5 minutes</option>
              <option value="15m">15 minutes</option>
              <option value="1h">1 hour</option>
              <option value="4h">4 hours</option>
              <option value="1d">1 day</option>
            </select>
          </div>

          {/* Initial balance & leverage */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className={labelClass} style={{ color: "hsl(var(--foreground))" }}>
                Initial Balance (USD)
              </label>
              <input
                type="number"
                min={100}
                step={100}
                className={inputClass}
                style={{
                  backgroundColor: "hsl(var(--input))",
                  borderColor: "hsl(var(--border))",
                  color: "hsl(var(--foreground))",
                }}
                value={initialBalance}
                onChange={(e) => setInitialBalance(Number(e.target.value))}
              />
            </div>
            <div>
              <label className={labelClass} style={{ color: "hsl(var(--foreground))" }}>
                Leverage (1-125)
              </label>
              <input
                type="number"
                min={1}
                max={125}
                className={inputClass}
                style={{
                  backgroundColor: "hsl(var(--input))",
                  borderColor: "hsl(var(--border))",
                  color: "hsl(var(--foreground))",
                }}
                value={leverage}
                onChange={(e) => setLeverage(Number(e.target.value))}
              />
            </div>
          </div>

          {/* Validation-only fields */}
          {mode === "validation" && (
            <div
              className="space-y-4 rounded-lg border p-4"
              style={{
                borderColor: "hsl(var(--border))",
                backgroundColor: "hsl(var(--card))",
              }}
            >
              <h3
                className="text-sm font-semibold uppercase tracking-wider"
                style={{ color: "hsl(var(--foreground))" }}
              >
                Validation Settings
              </h3>

              {/* Latency preset */}
              <div>
                <label className={labelClass} style={{ color: "hsl(var(--foreground))" }}>
                  Latency Preset
                </label>
                <select
                  className={inputClass}
                  style={{
                    backgroundColor: "hsl(var(--input))",
                    borderColor: "hsl(var(--border))",
                    color: "hsl(var(--foreground))",
                  }}
                  value={latencyPreset}
                  onChange={(e) => setLatencyPreset(e.target.value)}
                >
                  <option value="">None (no latency simulation)</option>
                  {latencyPresets.map((p) => (
                    <option key={p.name} value={p.name}>
                      {p.name} - {p.description} ({p.base_latency_ms}ms)
                    </option>
                  ))}
                </select>
              </div>

              {/* Sizing config */}
              <div>
                <label className={labelClass} style={{ color: "hsl(var(--foreground))" }}>
                  Sizing Config
                </label>
                <select
                  className={inputClass}
                  style={{
                    backgroundColor: "hsl(var(--input))",
                    borderColor: "hsl(var(--border))",
                    color: "hsl(var(--foreground))",
                  }}
                  value={sizingConfigId}
                  onChange={(e) => setSizingConfigId(e.target.value ? Number(e.target.value) : "")}
                >
                  <option value="">Default sizing</option>
                  {sizingConfigs.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name} ({c.method})
                    </option>
                  ))}
                </select>
              </div>

              {/* Risk config */}
              <div>
                <label className={labelClass} style={{ color: "hsl(var(--foreground))" }}>
                  Risk Config
                </label>
                <select
                  className={inputClass}
                  style={{
                    backgroundColor: "hsl(var(--input))",
                    borderColor: "hsl(var(--border))",
                    color: "hsl(var(--foreground))",
                  }}
                  value={riskConfigId}
                  onChange={(e) => setRiskConfigId(e.target.value ? Number(e.target.value) : "")}
                >
                  <option value="">Default risk</option>
                  {riskConfigs.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Right column */}
        <div className={sectionClass}>
          {/* Symbol multi-select */}
          <div>
            <div className="mb-1 flex items-center justify-between">
              <label className="text-sm font-medium" style={{ color: "hsl(var(--foreground))" }}>
                Symbols ({selectedSymbols.length} selected)
              </label>
              <button
                type="button"
                className="text-xs underline"
                style={{ color: "hsl(var(--accent))" }}
                onClick={handleSelectAllSymbols}
              >
                {selectedSymbols.length === symbols.length ? "Deselect all" : "Select all"}
              </button>
            </div>
            <div
              className="max-h-64 overflow-y-auto rounded-md border p-2"
              style={{
                borderColor: "hsl(var(--border))",
                backgroundColor: "hsl(var(--input))",
              }}
            >
              {symbolsQuery.isLoading && (
                <p className="p-2 text-sm" style={{ color: "hsl(var(--muted-foreground))" }}>
                  Loading symbols...
                </p>
              )}
              {symbols.length === 0 && !symbolsQuery.isLoading && (
                <p className="p-2 text-sm italic" style={{ color: "hsl(var(--muted-foreground))" }}>
                  No symbols available. Ingest data first.
                </p>
              )}
              {symbols.map((sym) => (
                <label
                  key={sym}
                  className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm transition-colors hover:opacity-80"
                  style={{ color: "hsl(var(--foreground))" }}
                >
                  <input
                    type="checkbox"
                    checked={selectedSymbols.includes(sym)}
                    onChange={() => handleSymbolToggle(sym)}
                    className="accent-[hsl(var(--accent))]"
                  />
                  <span className="font-mono">{sym}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Preflight check */}
          <div>
            <button
              type="button"
              disabled={
                selectedSymbols.length === 0 || !startDate || !endDate || coverageMutation.isPending
              }
              className="w-full rounded-md border px-4 py-2 text-sm font-medium transition-opacity disabled:cursor-not-allowed disabled:opacity-50"
              style={{
                borderColor: "hsl(var(--border))",
                backgroundColor: "hsl(var(--muted))",
                color: "hsl(var(--foreground))",
              }}
              onClick={handlePreflight}
            >
              {coverageMutation.isPending ? "Checking coverage..." : "Run Preflight Check"}
            </button>
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
            <button
              type="button"
              disabled={!canSubmit}
              className="w-full rounded-md px-4 py-3 text-sm font-semibold transition-opacity disabled:cursor-not-allowed disabled:opacity-50"
              style={{
                backgroundColor: "hsl(var(--accent))",
                color: "hsl(var(--accent-foreground))",
              }}
              onClick={handleLaunch}
            >
              {isLaunching
                ? "Launching..."
                : `Launch ${mode === "screening" ? "Screening" : "Validation"}`}
            </button>
          </div>

          {/* Launch error */}
          {launchError && (
            <div
              className="rounded-md border p-3 text-sm"
              style={{
                borderColor: "hsl(0 84% 60%)",
                backgroundColor: "hsl(0 84% 60% / 0.1)",
                color: "hsl(0 84% 60%)",
              }}
            >
              {launchError}
            </div>
          )}

          {/* Launch success */}
          {launchResult && (
            <div
              className="rounded-md border p-4"
              style={{
                borderColor: "hsl(142 76% 36%)",
                backgroundColor: "hsl(142 76% 36% / 0.1)",
              }}
            >
              <p className="text-sm font-medium" style={{ color: "hsl(142 76% 36%)" }}>
                Backtest launched successfully!
              </p>
              <p className="mt-1 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
                Run ID: {launchResult.id} | Mode: {launchResult.runMode}
              </p>
              <Link
                to={`/results?run=${launchResult.id}`}
                className="mt-2 inline-block text-sm font-medium underline"
                style={{ color: "hsl(var(--accent))" }}
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
