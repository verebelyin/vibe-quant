import { useEffect, useState } from "react";
import { toast } from "sonner";
import type { DiscoveryLaunchRequest } from "@/api/generated/models";
import {
  getListDiscoveryJobsApiDiscoveryJobsGetQueryKey,
  useGetIndicatorPoolApiDiscoveryIndicatorPoolGet,
  useLaunchDiscoveryApiDiscoveryLaunchPost,
} from "@/api/generated/discovery/discovery";
import { queryClient } from "@/api/query-client";
import { Badge } from "@/components/ui/badge";
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

export interface DiscoveryConvergenceConfig {
  convergenceWindow: number;
  convergenceThreshold: number;
}

interface DiscoveryConfigProps {
  onConvergenceChange?: (config: DiscoveryConvergenceConfig) => void;
}

export function DiscoveryConfig({ onConvergenceChange }: DiscoveryConfigProps) {
  // GA parameters (sensible defaults for quick exploration)
  const [population, setPopulation] = useState(20);
  const [generations, setGenerations] = useState(15);
  const [crossoverRate, setCrossoverRate] = useState(0.8);
  const [mutationRate, setMutationRate] = useState(0.1);
  const [eliteCount, setEliteCount] = useState(2);
  const [tournamentSize, setTournamentSize] = useState(3);
  const [convergenceWindow, setConvergenceWindow] = useState(5);
  const [convergenceThreshold, setConvergenceThreshold] = useState(0.001);

  // Target config
  const [symbols, setSymbols] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState("4h");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  // Indicator pool
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>([]);

  // Overfitting & robustness knobs (match DiscoveryLaunchRequest defaults)
  const [robustnessOpen, setRobustnessOpen] = useState(false);
  const [direction, setDirection] = useState<"long" | "short" | "both" | "random">("random");
  const [evalWindows, setEvalWindows] = useState(3);
  const [trainTestSplit, setTrainTestSplit] = useState(0);
  const [numSeeds, setNumSeeds] = useState(1);
  const [wfaOosStepDays, setWfaOosStepDays] = useState(0);
  const [wfaMinConsistency, setWfaMinConsistency] = useState(0.75);
  const [crossWindowMonths, setCrossWindowMonths] = useState("");
  const [crossWindowMinSharpe, setCrossWindowMinSharpe] = useState(0.5);

  const parsedCrossWindowMonths = crossWindowMonths
    .split(",")
    .map((s) => Number.parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n) && n > 0);

  const indicatorPoolQuery = useGetIndicatorPoolApiDiscoveryIndicatorPoolGet();
  const launchMutation = useLaunchDiscoveryApiDiscoveryLaunchPost();
  const datasetRange = useDatasetDateRange();

  // Auto-populate dates: default to last 3 months of available data
  useEffect(() => {
    if (!endDate && datasetRange.maxEnd) setEndDate(datasetRange.maxEnd);
    if (!startDate && datasetRange.maxEnd) {
      // 3 months before dataset end (not full range — avoids 4x runtime on 1m)
      const end = new Date(datasetRange.maxEnd);
      end.setMonth(end.getMonth() - 3);
      const threeMonthsBack = end.toISOString().slice(0, 10);
      // Use whichever is later: 3mo back or dataset start
      const bounded =
        datasetRange.minStart && threeMonthsBack < datasetRange.minStart
          ? datasetRange.minStart
          : threeMonthsBack;
      setStartDate(bounded);
    }
  }, [datasetRange.minStart, datasetRange.maxEnd]); // eslint-disable-line react-hooks/exhaustive-deps

  const indicators: Array<{ name: string; [key: string]: unknown }> =
    indicatorPoolQuery.data?.status === 200
      ? (indicatorPoolQuery.data.data as Array<{ name: string; [key: string]: unknown }>)
      : [];

  const indicatorNames = indicators.map((i) => String(i.name ?? i));

  function handleToggleIndicator(name: string) {
    setSelectedIndicators((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name],
    );
  }

  function handleSelectAll() {
    setSelectedIndicators(
      selectedIndicators.length === indicatorNames.length ? [] : [...indicatorNames],
    );
  }

  function handleLaunch() {
    const symbolList = symbols
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (symbolList.length === 0) {
      toast.error("Enter at least one symbol");
      return;
    }
    if (eliteCount >= population) {
      toast.error("Elite count must be less than population size");
      return;
    }

    launchMutation.mutate(
      {
        data: {
          population,
          generations,
          mutation_rate: mutationRate,
          crossover_rate: crossoverRate,
          elite_count: eliteCount,
          tournament_size: tournamentSize,
          convergence_generations: convergenceWindow,
          symbols: symbolList,
          timeframes: [timeframe],
          indicator_pool: selectedIndicators.length > 0 ? selectedIndicators : null,
          ...(startDate && { start_date: startDate }),
          ...(endDate && { end_date: endDate }),
          ...(direction !== "random" && { direction }),
          eval_windows: evalWindows,
          train_test_split: trainTestSplit,
          num_seeds: numSeeds,
          wfa_oos_step_days: wfaOosStepDays,
          wfa_min_consistency: wfaMinConsistency,
          ...(parsedCrossWindowMonths.length > 0 && {
            cross_window_months: parsedCrossWindowMonths,
            cross_window_min_sharpe: crossWindowMinSharpe,
          }),
        } as DiscoveryLaunchRequest,
      },
      {
        onSuccess: (resp) => {
          if (resp.status === 201) {
            toast.success("Discovery launched", {
              description: `Run ID: ${resp.data.run_id}`,
            });
            queryClient.invalidateQueries({
              queryKey: getListDiscoveryJobsApiDiscoveryJobsGetQueryKey(),
            });
          }
        },
        onError: (err: unknown) => {
          let message = "Launch failed";
          if (err instanceof Error) {
            message = err.message;
          }
          // Try to extract detail from Axios-style error response
          const axiosErr = err as { response?: { data?: { detail?: string }; status?: number } };
          if (axiosErr.response?.data?.detail) {
            message = axiosErr.response.data.detail;
          }
          toast.error("Discovery launch failed", {
            description: message,
            duration: 8000,
          });
        },
      },
    );
  }

  return (
    <div className="space-y-6">
      {/* GA Parameters */}
      <div className="space-y-4 rounded-lg border border-border bg-card p-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
          GA Parameters
        </h3>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="population">Population Size</Label>
            <Input
              id="population"
              type="number"
              min={10}
              value={population}
              onChange={(e) => setPopulation(Number(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="generations">Generations</Label>
            <Input
              id="generations"
              type="number"
              min={1}
              value={generations}
              onChange={(e) => setGenerations(Number(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="crossover-rate">Crossover Rate</Label>
            <Input
              id="crossover-rate"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={crossoverRate}
              onChange={(e) => setCrossoverRate(Number(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="mutation-rate">Mutation Rate</Label>
            <Input
              id="mutation-rate"
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={mutationRate}
              onChange={(e) => setMutationRate(Number(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="elite-count">Elite Count</Label>
            <Input
              id="elite-count"
              type="number"
              min={0}
              value={eliteCount}
              onChange={(e) => setEliteCount(Number(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="tournament-size">Tournament Size</Label>
            <Input
              id="tournament-size"
              type="number"
              min={2}
              value={tournamentSize}
              onChange={(e) => setTournamentSize(Number(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="convergence-window">Convergence Window</Label>
            <Input
              id="convergence-window"
              type="number"
              min={2}
              max={50}
              value={convergenceWindow}
              onChange={(e) => {
                const v = Number(e.target.value);
                setConvergenceWindow(v);
                onConvergenceChange?.({ convergenceWindow: v, convergenceThreshold });
              }}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="convergence-threshold">Convergence Threshold</Label>
            <Input
              id="convergence-threshold"
              type="number"
              min={0}
              max={1}
              step={0.0001}
              value={convergenceThreshold}
              onChange={(e) => {
                const v = Number(e.target.value);
                setConvergenceThreshold(v);
                onConvergenceChange?.({ convergenceWindow, convergenceThreshold: v });
              }}
            />
          </div>
        </div>
      </div>

      {/* Indicator Pool */}
      <div className="space-y-4 rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
            Indicator Pool
          </h3>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              {selectedIndicators.length}/{indicatorNames.length}
            </Badge>
            <Button
              type="button"
              variant="link"
              size="xs"
              onClick={handleSelectAll}
              disabled={indicatorNames.length === 0}
            >
              {selectedIndicators.length === indicatorNames.length ? "Deselect All" : "Select All"}
            </Button>
          </div>
        </div>

        {indicatorPoolQuery.isLoading && (
          <p className="text-xs text-muted-foreground">Loading indicators...</p>
        )}
        {indicatorNames.length === 0 && !indicatorPoolQuery.isLoading && (
          <p className="text-xs italic text-muted-foreground">No indicators available.</p>
        )}

        <div className="max-h-48 overflow-y-auto rounded-md border border-border bg-input p-2 dark:bg-input/30">
          {indicatorNames.map((name) => (
            <div
              key={name}
              className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm text-foreground transition-colors hover:opacity-80"
            >
              <Checkbox
                id={`ind-${name}`}
                checked={selectedIndicators.includes(name)}
                onCheckedChange={() => handleToggleIndicator(name)}
              />
              <Label htmlFor={`ind-${name}`} className="cursor-pointer font-mono text-xs">
                {name}
              </Label>
            </div>
          ))}
        </div>
      </div>

      {/* Target Config */}
      <div className="space-y-4 rounded-lg border border-border bg-card p-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
          Target Config
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="symbols">Symbols (comma-separated)</Label>
            <Input
              id="symbols"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              placeholder="BTCUSDT, ETHUSDT"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="disc-timeframe">Timeframe</Label>
            <Select value={timeframe} onValueChange={setTimeframe}>
              <SelectTrigger id="disc-timeframe" className="w-full">
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
        </div>
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
            <Label htmlFor="disc-start-date" className="text-xs text-muted-foreground">Start</Label>
            <DatePicker
              id="disc-start-date"
              value={startDate}
              onChange={setStartDate}
              placeholder="Start date"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="disc-end-date" className="text-xs text-muted-foreground">End</Label>
            <DatePicker
              id="disc-end-date"
              value={endDate}
              onChange={setEndDate}
              placeholder="End date"
            />
          </div>
        </div>
      </div>

      {/* Overfitting & Robustness */}
      <div className="rounded-lg border border-border bg-card">
        <button
          type="button"
          onClick={() => setRobustnessOpen((v) => !v)}
          className="flex w-full items-center justify-between p-4 text-left"
          aria-expanded={robustnessOpen}
        >
          <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
            Overfitting & Robustness
          </h3>
          <div className="flex items-center gap-2">
            {(direction !== "random" ||
              evalWindows !== 3 ||
              trainTestSplit !== 0 ||
              numSeeds !== 1 ||
              wfaOosStepDays !== 0 ||
              parsedCrossWindowMonths.length > 0) && (
              <Badge variant="outline" className="text-[10px]">
                modified
              </Badge>
            )}
            <span className="text-xs text-muted-foreground">
              {robustnessOpen ? "▾" : "▸"}
            </span>
          </div>
        </button>
        {robustnessOpen && (
          <div className="space-y-4 border-t border-border p-4">
            <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="direction">Direction</Label>
                <Select
                  value={direction}
                  onValueChange={(v) =>
                    setDirection(v as "long" | "short" | "both" | "random")
                  }
                >
                  <SelectTrigger id="direction" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="random">Random (GA picks)</SelectItem>
                    <SelectItem value="long">Long only</SelectItem>
                    <SelectItem value="short">Short only</SelectItem>
                    <SelectItem value="both">Both</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="eval-windows">Eval Windows</Label>
                <Input
                  id="eval-windows"
                  type="number"
                  min={1}
                  max={5}
                  value={evalWindows}
                  onChange={(e) => setEvalWindows(Number(e.target.value))}
                />
                <p className="text-[10px] text-muted-foreground">
                  Worst-case fitness across N sub-windows (PKFOLD-biased).
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="num-seeds">Num Seeds</Label>
                <Input
                  id="num-seeds"
                  type="number"
                  min={1}
                  max={7}
                  value={numSeeds}
                  onChange={(e) => setNumSeeds(Number(e.target.value))}
                />
                <p className="text-[10px] text-muted-foreground">
                  Multi-seed ensemble — median Sharpe across runs.
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="train-test-split">Train/Test Split</Label>
                <Input
                  id="train-test-split"
                  type="number"
                  min={0}
                  max={0.9}
                  step={0.05}
                  value={trainTestSplit}
                  onChange={(e) => setTrainTestSplit(Number(e.target.value))}
                />
                <p className="text-[10px] text-muted-foreground">
                  Fraction used for <strong>training</strong> (not holdout). 0 = disabled.
                </p>
              </div>
            </div>

            <div className="space-y-2 rounded-md border border-border bg-input/30 p-3">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Walk-Forward Analysis
              </h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="wfa-step">OOS Step (days)</Label>
                  <Select
                    value={String(wfaOosStepDays)}
                    onValueChange={(v) => setWfaOosStepDays(Number(v))}
                  >
                    <SelectTrigger id="wfa-step" className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">Off</SelectItem>
                      <SelectItem value="7">7 (weekly)</SelectItem>
                      <SelectItem value="14">14 (biweekly)</SelectItem>
                      <SelectItem value="30">30 (monthly)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="wfa-consistency">Min Consistency</Label>
                  <Input
                    id="wfa-consistency"
                    type="number"
                    min={0.5}
                    max={1}
                    step={0.05}
                    value={wfaMinConsistency}
                    onChange={(e) => setWfaMinConsistency(Number(e.target.value))}
                    disabled={wfaOosStepDays === 0}
                  />
                </div>
              </div>
            </div>

            <div className="space-y-2 rounded-md border border-border bg-input/30 p-3">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Cross-Window Validation
              </h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="cw-months">Month Offsets</Label>
                  <Input
                    id="cw-months"
                    type="text"
                    placeholder="e.g. 1,2,3"
                    value={crossWindowMonths}
                    onChange={(e) => setCrossWindowMonths(e.target.value)}
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Comma-separated shifted-window offsets. Empty = disabled.
                  </p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="cw-min-sharpe">Min Sharpe</Label>
                  <Input
                    id="cw-min-sharpe"
                    type="number"
                    min={0}
                    step={0.1}
                    value={crossWindowMinSharpe}
                    onChange={(e) => setCrossWindowMinSharpe(Number(e.target.value))}
                    disabled={parsedCrossWindowMonths.length === 0}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ETA Estimate + Runtime Warning */}
      {(() => {
        const crossWindowMultiplier = Math.max(1, parsedCrossWindowMonths.length);
        const totalEvals = population * generations * numSeeds * crossWindowMultiplier;
        const secPerEval = 15;
        const cores = 8;
        const parallelSec = Math.ceil(totalEvals / cores) * secPerEval;
        const fmt = (s: number) => {
          if (s < 60) return `${s}s`;
          if (s < 3600) return `${Math.round(s / 60)}m`;
          const h = Math.floor(s / 3600);
          const m = Math.round((s % 3600) / 60);
          return m > 0 ? `${h}h ${m}m` : `${h}h`;
        };
        const danger = totalEvals > 500;
        const warn = totalEvals > 200;
        const borderCls = danger
          ? "border-red-500/50 bg-red-500/10 text-red-200"
          : warn
            ? "border-amber-500/50 bg-amber-500/10 text-amber-200"
            : "border-border bg-card text-muted-foreground";
        return (
          <div className={`rounded-lg border p-3 text-xs ${borderCls}`}>
            <div className="flex items-center justify-between">
              <span>Total evaluations: <span className="font-mono font-semibold text-foreground">{totalEvals.toLocaleString()}</span></span>
              <span>~{fmt(secPerEval)}/eval</span>
            </div>
            <div className="mt-1 flex items-center justify-between">
              <span>Parallel ({cores} cores): <span className="font-mono font-semibold text-foreground">{fmt(parallelSec)}</span></span>
            </div>
            {danger && (
              <p className="mt-1.5 text-[10px] font-medium text-red-300">
                {totalEvals.toLocaleString()} evals = ~{fmt(parallelSec)} runtime. Recommended: pop=20 x gen=15 = 300 evals for quick exploration.
              </p>
            )}
            {warn && !danger && (
              <p className="mt-1 text-[10px]">Consider reducing population or generations for faster iteration.</p>
            )}
          </div>
        );
      })()}

      {/* Launch */}
      <Button
        type="button"
        className="w-full py-3 font-semibold"
        disabled={launchMutation.isPending}
        onClick={handleLaunch}
      >
        {launchMutation.isPending ? "Launching..." : "Launch Discovery"}
      </Button>
    </div>
  );
}
