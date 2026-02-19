import { useMemo, useState } from "react";
import { toast } from "sonner";
import {
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
  halted: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300",
  stopped: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  error: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  starting: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 animate-pulse",
};
const FALLBACK = "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300";

export function SessionControl() {
  // Start form state
  const [strategyId, setStrategyId] = useState("");
  const [testnet, setTestnet] = useState(true);
  const [sizingMethod, setSizingMethod] = useState("");
  const [maxLeverage, setMaxLeverage] = useState<number | "">(10);
  const [maxPositionPct, setMaxPositionPct] = useState<number | "">(25);
  const [riskPerTrade, setRiskPerTrade] = useState<number | "">(2);
  const [stopConfirmOpen, setStopConfirmOpen] = useState(false);
  // API credentials state
  const [showCredentials, setShowCredentials] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [apiSecret, setApiSecret] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [showApiSecret, setShowApiSecret] = useState(false);
  // Filter state
  const [validatedOnly, setValidatedOnly] = useState(false);

  // Queries
  const strategiesQuery = useListStrategiesApiStrategiesGet();
  const statusQuery = useGetStatusApiPaperStatusGet({
    query: { refetchInterval: 5_000 },
  });
  const runsQuery = useListRunsApiResultsRunsGet();

  // Mutations
  const startMutation = useStartPaperApiPaperStartPost();
  const haltMutation = useHaltPaperApiPaperHaltPost();
  const resumeMutation = useResumePaperApiPaperResumePost();
  const stopMutation = useStopPaperApiPaperStopPost();

  const allStrategies =
    strategiesQuery.data?.status === 200 ? strategiesQuery.data.data.strategies : [];
  const allRuns = runsQuery.data?.status === 200 ? runsQuery.data.data.runs : [];

  const validatedStrategyIds = useMemo(() => {
    const ids = new Set<number>();
    for (const run of allRuns) {
      if (run.run_mode === "validation" && run.status === "completed") {
        ids.add(run.strategy_id);
      }
    }
    return ids;
  }, [allRuns]);

  const strategies = validatedOnly
    ? allStrategies.filter((s) => validatedStrategyIds.has(s.id))
    : allStrategies;

  const status = statusQuery.data?.status === 200 ? statusQuery.data.data : null;

  const pnlMetrics = status?.pnl_metrics as Record<string, unknown> | null | undefined;
  const currentPnl = pnlMetrics?.total_pnl ?? pnlMetrics?.pnl;
  const currentState = status?.state?.toLowerCase() ?? "unknown";
  const isActive =
    currentState === "running" || currentState === "halted" || currentState === "starting";

  function handleStart() {
    if (!strategyId) {
      toast.error("Select a strategy");
      return;
    }
    startMutation.mutate(
      {
        data: {
          strategy_id: Number(strategyId),
          testnet,
          sizing_method: sizingMethod || null,
          max_leverage: maxLeverage === "" ? null : maxLeverage,
          max_position_pct: maxPositionPct === "" ? null : maxPositionPct,
          risk_per_trade: riskPerTrade === "" ? null : riskPerTrade,
          ...(apiKey && { api_key: apiKey }),
          ...(apiSecret && { api_secret: apiSecret }),
        } as Record<string, unknown>,
      },
      {
        onSuccess: (resp) => {
          if (resp.status === 201) {
            toast.success("Paper trading started", {
              description: `State: ${resp.data.state}`,
            });
          }
        },
        onError: (err: unknown) => {
          const message = err instanceof Error ? err.message : "Start failed";
          toast.error("Failed to start paper trading", { description: message });
        },
      },
    );
  }

  function handleHalt() {
    haltMutation.mutate(undefined, {
      onSuccess: () => toast.success("Paper trading halted"),
      onError: (err: unknown) => {
        const msg = err instanceof Error ? err.message : "Halt failed";
        toast.error("Halt failed", { description: msg });
      },
    });
  }

  function handleResume() {
    resumeMutation.mutate(undefined, {
      onSuccess: () => toast.success("Paper trading resumed"),
      onError: (err: unknown) => {
        const msg = err instanceof Error ? err.message : "Resume failed";
        toast.error("Resume failed", { description: msg });
      },
    });
  }

  function handleStop() {
    setStopConfirmOpen(false);
    stopMutation.mutate(undefined, {
      onSuccess: () => toast.success("Paper trading stopped"),
      onError: (err: unknown) => {
        const msg = err instanceof Error ? err.message : "Stop failed";
        toast.error("Stop failed", { description: msg });
      },
    });
  }

  return (
    <div className="space-y-6">
      {/* Status display */}
      {status && (
        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-foreground">
            Session Status
          </h3>
          <div className="grid grid-cols-3 gap-4">
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
              <p className="text-xs text-muted-foreground">Trades</p>
              <p className="mt-1 font-mono text-sm text-foreground">{status.trades_count}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">PnL</p>
              <p
                className={cn(
                  "mt-1 font-mono text-sm",
                  currentPnl != null && Number(currentPnl) >= 0 ? "text-green-600" : "text-red-600",
                )}
              >
                {currentPnl != null ? Number(currentPnl).toFixed(2) : "--"}
              </p>
            </div>
          </div>

          {/* Control buttons */}
          {isActive && (
            <div className="mt-4 flex items-center gap-2">
              {currentState === "running" && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={haltMutation.isPending}
                  onClick={handleHalt}
                >
                  {haltMutation.isPending ? "Halting..." : "Halt"}
                </Button>
              )}
              {currentState === "halted" && (
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
                variant="destructive"
                size="sm"
                disabled={stopMutation.isPending}
                onClick={() => setStopConfirmOpen(true)}
              >
                {stopMutation.isPending ? "Stopping..." : "Stop"}
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Start form (show when not active) */}
      {!isActive && (
        <div className="space-y-4 rounded-lg border border-border bg-card p-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
            Start Paper Trading
          </h3>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="paper-strategy">Strategy</Label>
              <Label className="flex items-center gap-2 text-xs font-normal">
                <Checkbox
                  checked={validatedOnly}
                  onCheckedChange={(v) => {
                    setValidatedOnly(v === true);
                    setStrategyId("");
                  }}
                />
                Validated only
              </Label>
            </div>
            <Select value={strategyId} onValueChange={setStrategyId}>
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
            {strategiesQuery.isLoading && (
              <p className="text-xs text-muted-foreground">Loading strategies...</p>
            )}
            {validatedOnly && strategies.length === 0 && !strategiesQuery.isLoading && (
              <p className="text-xs text-muted-foreground">
                No validated strategies found. Run a validation backtest first.
              </p>
            )}
          </div>

          <div className="flex items-center gap-4">
            <Label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={testnet}
                onChange={(e) => setTestnet(e.target.checked)}
                className="rounded border-border"
              />
              <span className="text-sm">Testnet</span>
            </Label>
          </div>

          <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="sizing-method">Sizing Method</Label>
              <Select value={sizingMethod} onValueChange={setSizingMethod}>
                <SelectTrigger id="sizing-method" className="w-full">
                  <SelectValue placeholder="Default" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fixed">Fixed</SelectItem>
                  <SelectItem value="kelly">Kelly</SelectItem>
                  <SelectItem value="risk_parity">Risk Parity</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="max-leverage">Max Leverage</Label>
              <Input
                id="max-leverage"
                type="number"
                min={1}
                max={125}
                value={maxLeverage}
                onChange={(e) =>
                  setMaxLeverage(e.target.value === "" ? "" : Number(e.target.value))
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="max-pos-pct">Max Position %</Label>
              <Input
                id="max-pos-pct"
                type="number"
                min={1}
                max={100}
                value={maxPositionPct}
                onChange={(e) =>
                  setMaxPositionPct(e.target.value === "" ? "" : Number(e.target.value))
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="risk-per-trade">Risk Per Trade %</Label>
              <Input
                id="risk-per-trade"
                type="number"
                min={0.1}
                max={100}
                step={0.1}
                value={riskPerTrade}
                onChange={(e) =>
                  setRiskPerTrade(e.target.value === "" ? "" : Number(e.target.value))
                }
              />
            </div>
          </div>

          {/* API Credentials */}
          <div className="space-y-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-auto p-0 text-xs text-muted-foreground hover:text-foreground"
              onClick={() => setShowCredentials((v) => !v)}
            >
              {showCredentials ? "Hide" : "Show"} API Credentials
            </Button>
            {showCredentials && (
              <div className="space-y-3 rounded-md border border-border p-3">
                <div className="space-y-2">
                  <Label htmlFor="api-key">API Key</Label>
                  <div className="flex gap-2">
                    <Input
                      id="api-key"
                      type={showApiKey ? "text" : "password"}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="Binance API key"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => setShowApiKey((v) => !v)}
                    >
                      {showApiKey ? "Hide" : "Show"}
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="api-secret">API Secret</Label>
                  <div className="flex gap-2">
                    <Input
                      id="api-secret"
                      type={showApiSecret ? "text" : "password"}
                      value={apiSecret}
                      onChange={(e) => setApiSecret(e.target.value)}
                      placeholder="Binance API secret"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => setShowApiSecret((v) => !v)}
                    >
                      {showApiSecret ? "Hide" : "Show"}
                    </Button>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">
                  Required for live testnet trading. Keys are sent to the backend and not stored in
                  the browser.
                </p>
              </div>
            )}
          </div>

          <Button
            type="button"
            className="w-full py-3 font-semibold"
            disabled={!strategyId || startMutation.isPending}
            onClick={handleStart}
          >
            {startMutation.isPending ? "Starting..." : "Start Paper Trading"}
          </Button>
        </div>
      )}

      {/* Stop confirmation dialog */}
      <Dialog open={stopConfirmOpen} onOpenChange={setStopConfirmOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Stop Paper Trading</DialogTitle>
            <DialogDescription>
              This will stop the paper trading session. All open positions will be closed.
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
    </div>
  );
}
