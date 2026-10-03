import { useMemo, useState } from "react";
import { toast } from "sonner";
import { isApiError } from "@/api/client";
import type { PaperStartRequest } from "@/api/generated/models";
import {
  useCloseAllPositionsApiPaperCloseAllPositionsPost,
  useGetStatusApiPaperStatusGet,
  useHaltPaperApiPaperHaltPost,
  useResumePaperApiPaperResumePost,
  useStartPaperApiPaperStartPost,
  useStopPaperApiPaperStopPost,
} from "@/api/generated/paper/paper";
import { useListRunsApiResultsRunsGet } from "@/api/generated/results/results";
import { useListStrategiesApiStrategiesGet } from "@/api/generated/strategies/strategies";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

const STATE_BADGE: Record<string, string> = {
  running: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
  paused: "bg-sky-100 text-sky-700 dark:bg-sky-900 dark:text-sky-300",
  halted: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300",
  stopped: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  error: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  starting: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 animate-pulse",
};
const FALLBACK = "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300";
const ACTIVE_STATES = new Set(["running", "paused", "halted", "starting"]);
const NO_RUN = "none";
// NautilusTrader TraderId is NAME-TAG; anything else aborts the node.
const TRADER_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_]*(-[A-Za-z0-9_]+)+$/;

function newTraderId(): string {
  return `PAPER-${crypto.randomUUID().slice(0, 8).toUpperCase()}`;
}

/** Percent input ("" = not set) -> fraction for the API, or null. */
function pctToFraction(value: number | ""): number | null {
  return value === "" ? null : value / 100;
}

function numberOrBlank(raw: string): number | "" {
  return raw === "" ? "" : Number(raw);
}

/** FastAPI error detail (string, or {error, result}) -> readable text. */
function apiErrorDetail(err: unknown, fallback: string): string {
  if (!isApiError(err)) return err instanceof Error ? err.message : fallback;
  const body = err.body;
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (typeof detail === "object" && detail !== null && "error" in detail) {
      const inner = detail.error;
      if (typeof inner === "string") return inner;
    }
    return JSON.stringify(detail);
  }
  return err.message;
}

export function SessionControl() {
  const [traderId, setTraderId] = useState(newTraderId);
  const [strategyId, setStrategyId] = useState("");
  const [validationRunId, setValidationRunId] = useState(NO_RUN);
  const [symbolsText, setSymbolsText] = useState("BTCUSDT");
  const [testnet, setTestnet] = useState(true);
  const [confirmLive, setConfirmLive] = useState(false);
  // Sizing overrides: blank = keep what the validation run used.
  const [maxLeverage, setMaxLeverage] = useState<number | "">("");
  const [maxPositionPct, setMaxPositionPct] = useState<number | "">("");
  const [riskPerTrade, setRiskPerTrade] = useState<number | "">("");
  // Risk limits enforced by the node (percent in the UI, fractions in the API).
  const [maxDrawdownPct, setMaxDrawdownPct] = useState<number | "">(20);
  const [maxDailyLossPct, setMaxDailyLossPct] = useState<number | "">(5);
  const [maxConsecutiveLosses, setMaxConsecutiveLosses] = useState<number | "">(5);
  const [maxPositionCount, setMaxPositionCount] = useState<number | "">(3);
  const [stopConfirmOpen, setStopConfirmOpen] = useState(false);
  const [closeAllConfirmOpen, setCloseAllConfirmOpen] = useState(false);
  const [validatedOnly, setValidatedOnly] = useState(true);

  const strategiesQuery = useListStrategiesApiStrategiesGet();
  const statusQuery = useGetStatusApiPaperStatusGet({
    query: { refetchInterval: 5_000 },
  });
  const runsQuery = useListRunsApiResultsRunsGet();

  const startMutation = useStartPaperApiPaperStartPost();
  const haltMutation = useHaltPaperApiPaperHaltPost();
  const resumeMutation = useResumePaperApiPaperResumePost();
  const stopMutation = useStopPaperApiPaperStopPost();
  const closeAllMutation = useCloseAllPositionsApiPaperCloseAllPositionsPost();

  const allStrategies =
    strategiesQuery.data?.status === 200 ? strategiesQuery.data.data.strategies : [];
  const allRuns = runsQuery.data?.status === 200 ? runsQuery.data.data.runs : [];

  const validationRuns = useMemo(
    () =>
      allRuns
        .filter((run) => run.run_mode === "validation" && run.status === "completed")
        .sort((a, b) => b.id - a.id),
    [allRuns],
  );
  const validatedStrategyIds = useMemo(
    () => new Set(validationRuns.map((run) => run.strategy_id)),
    [validationRuns],
  );
  const strategies = validatedOnly
    ? allStrategies.filter((s) => validatedStrategyIds.has(s.id))
    : allStrategies;
  const runsForStrategy = validationRuns.filter((run) => run.strategy_id === Number(strategyId));
  const selectedRun = runsForStrategy.find((run) => String(run.id) === validationRunId);

  const status = statusQuery.data?.status === 200 ? statusQuery.data.data : null;
  const currentState = status?.state?.toLowerCase() ?? "unknown";
  const isActive = ACTIVE_STATES.has(currentState);
  const traderIdValid = TRADER_ID_RE.test(traderId);

  function selectStrategy(value: string) {
    setStrategyId(value);
    const latest = validationRuns.find((run) => run.strategy_id === Number(value));
    setValidationRunId(latest ? String(latest.id) : NO_RUN);
  }

  function handleStart() {
    if (!strategyId) {
      toast.error("Select a strategy");
      return;
    }
    if (!traderIdValid) {
      toast.error("Invalid trader ID", { description: "Use NAME-TAG, e.g. PAPER-001" });
      return;
    }
    if (!testnet && !confirmLive) {
      toast.error("Live trading needs explicit confirmation");
      return;
    }
    const symbols = symbolsText
      .split(",")
      .map((s) => s.trim().toUpperCase())
      .filter((s) => s.length > 0);
    if (!selectedRun && symbols.length === 0) {
      toast.error("Enter at least one symbol");
      return;
    }
    const payload: PaperStartRequest = {
      strategy_id: Number(strategyId),
      validation_run_id: selectedRun ? selectedRun.id : null,
      symbols: selectedRun ? null : symbols,
      testnet,
      confirm_live: !testnet && confirmLive,
      trader_id: traderId,
      max_leverage: maxLeverage === "" ? null : maxLeverage,
      max_position_pct: pctToFraction(maxPositionPct),
      risk_per_trade: pctToFraction(riskPerTrade),
      max_drawdown_pct: pctToFraction(maxDrawdownPct),
      max_daily_loss_pct: pctToFraction(maxDailyLossPct),
      max_consecutive_losses: maxConsecutiveLosses === "" ? null : maxConsecutiveLosses,
      max_position_count: maxPositionCount === "" ? null : maxPositionCount,
    };
    startMutation.mutate(
      { data: payload },
      {
        onSuccess: (resp) => {
          if (resp.status === 201) {
            toast.success(`Paper trading starting (${resp.data.trader_id ?? traderId})`, {
              description: testnet ? "Binance testnet" : "LIVE trading",
            });
            setTraderId(newTraderId());
          }
        },
        onError: (err: unknown) => {
          toast.error("Failed to start paper trading", {
            description: apiErrorDetail(err, "Start failed"),
          });
        },
      },
    );
  }

  function handleHalt(mode: "halt" | "pause") {
    haltMutation.mutate(
      { params: { mode } },
      {
        onSuccess: () =>
          toast.success(
            mode === "halt"
              ? "Halted: positions flattened, strategies stopped"
              : "Paused: no new entries, SL/TP still active",
          ),
        onError: (err: unknown) =>
          toast.error(mode === "halt" ? "Halt failed" : "Pause failed", {
            description: apiErrorDetail(err, "Command failed"),
          }),
      },
    );
  }

  function handleResume() {
    resumeMutation.mutate(undefined, {
      onSuccess: () => toast.success("Paper trading resumed"),
      onError: (err: unknown) =>
        toast.error("Resume refused", { description: apiErrorDetail(err, "Resume failed") }),
    });
  }

  function handleStop() {
    setStopConfirmOpen(false);
    stopMutation.mutate(undefined, {
      onSuccess: () => toast.success("Paper trading stopped"),
      onError: (err: unknown) =>
        toast.error("Stop failed", { description: apiErrorDetail(err, "Stop failed") }),
    });
  }

  function handleCloseAll() {
    setCloseAllConfirmOpen(false);
    closeAllMutation.mutate(undefined, {
      onSuccess: (resp) => {
        const targeted = resp.status === 200 ? resp.data.targeted_positions : undefined;
        const count = Array.isArray(targeted) ? targeted.length : 0;
        toast.success(`All positions closed (${count})`);
      },
      onError: (err: unknown) =>
        toast.error("Close-all incomplete", {
          description: apiErrorDetail(err, "Close all failed"),
          duration: 15_000,
        }),
    });
  }

  return (
    <div className="space-y-6">
      {status && isActive && (
        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-foreground">
            Session Status
          </h3>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <div>
              <p className="text-xs text-muted-foreground">State</p>
              <Badge
                variant="outline"
                className={cn("mt-1 border-transparent", STATE_BADGE[currentState] ?? FALLBACK)}
              >
                {status.state}
              </Badge>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Venue</p>
              <Badge
                variant="outline"
                className={cn(
                  "mt-1 border-transparent",
                  status.testnet === false
                    ? "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300"
                    : FALLBACK,
                )}
              >
                {status.testnet === false ? "LIVE" : "TESTNET"}
              </Badge>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Session</p>
              <p className="mt-1 font-mono text-xs text-foreground">
                {status.trader_id ?? "--"} · run {status.run_id ?? "--"}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Trades today</p>
              <p className="mt-1 font-mono text-sm text-foreground">{status.trades_count}</p>
            </div>
          </div>
          {(status.halt_reason || status.message) && (
            <p className="mt-3 text-xs text-muted-foreground">
              {status.halt_reason && (
                <span className="font-semibold text-yellow-600">{status.halt_reason}: </span>
              )}
              {status.message}
            </p>
          )}

          <div className="mt-4 flex flex-wrap items-center gap-2">
            {currentState === "running" && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={haltMutation.isPending}
                onClick={() => handleHalt("pause")}
              >
                Pause entries
              </Button>
            )}
            {(currentState === "running" || currentState === "paused") && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={haltMutation.isPending}
                onClick={() => handleHalt("halt")}
              >
                {haltMutation.isPending ? "Working..." : "Halt & flatten"}
              </Button>
            )}
            {(currentState === "paused" || currentState === "halted") && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={resumeMutation.isPending}
                onClick={handleResume}
              >
                {resumeMutation.isPending ? "Resuming..." : "Resume"}
              </Button>
            )}
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="border-warning text-warning hover:bg-warning/10"
              disabled={closeAllMutation.isPending}
              onClick={() => setCloseAllConfirmOpen(true)}
            >
              {closeAllMutation.isPending ? "Closing..." : "Close All Positions"}
            </Button>
            <Button
              type="button"
              variant="destructive"
              size="sm"
              disabled={stopMutation.isPending}
              onClick={() => setStopConfirmOpen(true)}
            >
              {stopMutation.isPending ? "Stopping..." : "Stop"}
            </Button>
          </div>
        </div>
      )}

      {!isActive && (
        <div className="space-y-4 rounded-lg border border-border bg-card p-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
            Start Paper Trading
          </h3>

          <div className="space-y-2">
            <Label htmlFor="trader-id">Trader ID</Label>
            <div className="flex gap-2">
              <Input
                id="trader-id"
                value={traderId}
                onChange={(e) => setTraderId(e.target.value)}
                placeholder="PAPER-XXXXXXXX"
                className={cn("font-mono", !traderIdValid && "border-destructive")}
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setTraderId(newTraderId())}
              >
                Regenerate
              </Button>
            </div>
            {!traderIdValid && (
              <p className="text-xs text-destructive">Use NAME-TAG with a hyphen, e.g. PAPER-001</p>
            )}
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="paper-strategy">Strategy</Label>
              <Label className="flex items-center gap-2 text-xs font-normal">
                <Checkbox
                  checked={validatedOnly}
                  onCheckedChange={(v) => {
                    setValidatedOnly(v === true);
                    setStrategyId("");
                    setValidationRunId(NO_RUN);
                  }}
                />
                Validated only
              </Label>
            </div>
            <Select value={strategyId} onValueChange={selectStrategy}>
              <SelectTrigger id="paper-strategy" className="w-full">
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
            {validatedOnly && strategies.length === 0 && !strategiesQuery.isLoading && (
              <p className="text-xs text-muted-foreground">
                No validated strategies found. Run a validation backtest first.
              </p>
            )}
          </div>

          {strategyId && (
            <div className="space-y-2">
              <Label htmlFor="validation-run">Validation run (exact params + leverage)</Label>
              <Select value={validationRunId} onValueChange={setValidationRunId}>
                <SelectTrigger id="validation-run" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {runsForStrategy.map((run) => (
                    <SelectItem key={run.id} value={String(run.id)}>
                      Run {run.id} · {run.symbols.join(", ")} · {run.timeframe}
                    </SelectItem>
                  ))}
                  <SelectItem value={NO_RUN}>None (compiled DSL defaults)</SelectItem>
                </SelectContent>
              </Select>
              {selectedRun ? (
                <p className="text-xs text-muted-foreground">
                  Trades {selectedRun.symbols.join(", ")} with run {selectedRun.id}&apos;s
                  parameters and leverage; any override below shows up in the logged config diff.
                </p>
              ) : (
                <div className="space-y-1">
                  <Label htmlFor="paper-symbols">Symbols (comma-separated)</Label>
                  <Input
                    id="paper-symbols"
                    value={symbolsText}
                    onChange={(e) => setSymbolsText(e.target.value)}
                    placeholder="BTCUSDT, ETHUSDT"
                    className="font-mono"
                  />
                </div>
              )}
            </div>
          )}

          <div className="space-y-2">
            <Label className="flex items-center gap-2">
              <Checkbox
                checked={testnet}
                onCheckedChange={(v) => {
                  setTestnet(v === true);
                  setConfirmLive(false);
                }}
              />
              <span className="text-sm">Binance testnet (uses BINANCE_TESTNET_* keys)</span>
            </Label>
            {!testnet && (
              <div className="rounded-md border border-destructive bg-destructive/10 p-3">
                <Label className="flex items-center gap-2 text-sm font-normal text-destructive">
                  <Checkbox
                    checked={confirmLive}
                    onCheckedChange={(v) => setConfirmLive(v === true)}
                  />
                  I understand this trades REAL funds on Binance (BINANCE_API_* keys)
                </Label>
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              API keys are read from the backend environment only and never sent from the browser.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Sizing overrides (blank = as validated)
            </h4>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="max-leverage">Max Leverage (cap)</Label>
                <Input
                  id="max-leverage"
                  type="number"
                  min={1}
                  max={125}
                  placeholder="20"
                  value={maxLeverage}
                  onChange={(e) => setMaxLeverage(numberOrBlank(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="max-pos-pct">Max Position %</Label>
                <Input
                  id="max-pos-pct"
                  type="number"
                  min={1}
                  step={1}
                  placeholder="validated"
                  value={maxPositionPct}
                  onChange={(e) => setMaxPositionPct(numberOrBlank(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="risk-per-trade">Risk Per Trade %</Label>
                <Input
                  id="risk-per-trade"
                  type="number"
                  min={0.1}
                  max={50}
                  step={0.1}
                  placeholder="validated"
                  value={riskPerTrade}
                  onChange={(e) => setRiskPerTrade(numberOrBlank(e.target.value))}
                />
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Auto-Stop Limits
            </h4>
            <p className="text-xs text-muted-foreground">
              Drawdown / consecutive-loss breach: flatten and stop. Daily loss: flatten and block
              entries until 00:00 UTC.
            </p>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <div className="space-y-2">
                <Label htmlFor="max-drawdown-pct">Max Drawdown %</Label>
                <Input
                  id="max-drawdown-pct"
                  type="number"
                  min={1}
                  max={100}
                  step={0.5}
                  value={maxDrawdownPct}
                  onChange={(e) => setMaxDrawdownPct(numberOrBlank(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="max-daily-loss-pct">Max Daily Loss %</Label>
                <Input
                  id="max-daily-loss-pct"
                  type="number"
                  min={0.1}
                  max={100}
                  step={0.5}
                  value={maxDailyLossPct}
                  onChange={(e) => setMaxDailyLossPct(numberOrBlank(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="max-consecutive-losses">Max Consec. Losses</Label>
                <Input
                  id="max-consecutive-losses"
                  type="number"
                  min={1}
                  max={50}
                  value={maxConsecutiveLosses}
                  onChange={(e) => setMaxConsecutiveLosses(numberOrBlank(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="max-position-count">Max Positions</Label>
                <Input
                  id="max-position-count"
                  type="number"
                  min={1}
                  max={100}
                  value={maxPositionCount}
                  onChange={(e) => setMaxPositionCount(numberOrBlank(e.target.value))}
                />
              </div>
            </div>
          </div>

          <Button
            type="button"
            className="w-full py-3 font-semibold"
            disabled={
              !strategyId || startMutation.isPending || !traderIdValid || (!testnet && !confirmLive)
            }
            onClick={handleStart}
          >
            {startMutation.isPending
              ? "Starting..."
              : testnet
                ? "Start Paper Trading (testnet)"
                : "Start LIVE Trading"}
          </Button>
        </div>
      )}

      <Dialog open={stopConfirmOpen} onOpenChange={setStopConfirmOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Stop Paper Trading</DialogTitle>
            <DialogDescription>
              This stops the paper trading process. Use "Close All Positions" or "Halt & flatten"
              first if positions must not stay open.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setStopConfirmOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleStop}>
              Stop
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={closeAllConfirmOpen} onOpenChange={setCloseAllConfirmOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Close All Positions</DialogTitle>
            <DialogDescription>
              Closes every open position with reduce-only market orders and cancels open orders.
              Strategies keep running and may re-enter (use Pause first to prevent that).
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCloseAllConfirmOpen(false)}>
              Cancel
            </Button>
            <Button variant="default" onClick={handleCloseAll}>
              Close All
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
