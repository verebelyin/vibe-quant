import { useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useState } from "react";
import { toast } from "sonner";
import { useKillJobApiBacktestJobsRunIdDelete } from "@/api/generated/backtest/backtest";
import type { BacktestRunResponse, BankrollReportResponse } from "@/api/generated/models";
import {
  getGetBankrollReportApiResultsRunsRunIdBankrollReportGetQueryKey,
  useGetBankrollReportApiResultsRunsRunIdBankrollReportGet,
  useGetRunMetaApiResultsRunsRunIdMetaGet,
  useGetRunSummaryApiResultsRunsRunIdGet,
  useLaunchBankrollReportApiResultsRunsRunIdBankrollReportPost,
} from "@/api/generated/results/results";
import { formatApiErrorDetail } from "@/components/discovery/apiErrorDetail";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

/**
 * Results-page sizing report (bead vibe-quant-2gyk1.3, design T5): Kelly and
 * vol-target numbers on the run's daily mark-to-market returns, a size-stress
 * table where every row is its own engine run, and a recommendation that is
 * always a size the report ran. Validation runs only.
 *
 * Every state decision is a pure exported helper (tested in node); the
 * component only wires them to the generated hooks.
 */

const POLL_MS = 5000;
const REVALIDATE = "re-validate first";
const DEFAULT_MULTIPLES = [0.25, 0.5, 0.75, 1, 2, 3, 5, 10];
/** The API's default margin grid (isolated, cross) x (3, 5, 10) x (0.5, 1.0): 12 cells. */
const DEFAULT_SWEEP_CELLS = 12;
export const SWEEP_CHECKBOX_LABEL = "Also sweep margin settings at the recommended size";
export const CROSS_NOTE = "Cross: leverage changes only how many positions fit.";

/** Design §2.7 / C9, verbatim: the 50% is a sizing budget, never a loss bound. */
export const RISK_HELD_TOOLTIP =
  "Risk is held at the 50% sizing-budget limit: the stop distance at the signal price " +
  "may use at most 50% of equity, before minimum quantity, fills and costs. " +
  "It does not cap the realized loss.";
export const RISK_HELD_LABEL = "risk held at limit";
export const REPLAY_CAPTION =
  "Replay on this report's captured market data; parent headlines matched; " +
  "the original market series was not verified.";
export const K_TOOLTIP =
  "Requested size step k: the run's validated max exposure and risk per trade are each " +
  "multiplied by k before sizing (risk held at the 50% limit). It is a sizing input; the " +
  "row's realized exposure, drawdown, fees and funding are measured from its own run.";

const REASON_TEXT: Record<string, string> = {
  insufficient_data: "Fewer than 60 daily returns",
  zero_variance: "Daily returns have zero variance",
  numeric: "Kelly numbers are not finite",
  non_positive_mean: "Mean daily return is not positive",
  no_eligible_row: "No tested size at or under the ceiling is eligible",
  daily_unavailable: "No daily mark-to-market series",
  intraday_wipeout: "Equity hit zero intraday",
  engine_force_stopped: "Engine stopped the run early",
  failed: "Run failed",
  not_runnable: "Not runnable at these settings",
  no_trades: "No trades",
  non_finite: "Non-finite result",
  non_positive_return: "Return is not positive",
  mtm_unavailable: "Mark-to-market path unavailable",
  dd_out_of_range: "Drawdown out of range",
  dd_over_limit: "Drawdown over your limit",
  liquidation_not_modeled: "Liquidation not modeled",
  liquidated: "Liquidated",
  bust: "Account ruined (cross bust)",
  path_divergence: "Liquidation overlay diverged from the engine path",
  move_unavailable: "Liquidating move unavailable",
  cross_path_dependent: "Not ranked: multi-coin cross",
  no_safe_setting: "No safe setting",
  no_recommended_size: "no recommended size",
  stale: `Parent changed since this report: ${REVALIDATE}`,
};

export function reasonText(reason: string | null | undefined): string {
  if (!reason) return "";
  return REASON_TEXT[reason] ?? reason;
}

// --- launch gate and ETA ----------------------------------------------------------

/** Python `json.dumps` writes NaN/Infinity as bare tokens; read them as null. */
const PY_NON_FINITE = /(?<=[:[,]\s*)(?:NaN|-?Infinity)(?=\s*[,}\]])/g;

function parseNotes(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    try {
      return JSON.parse(text.replace(PY_NON_FINITE, "null"));
    } catch {
      return undefined;
    }
  }
}

/** Python `round()`: an exact .5 tie goes to the even integer. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export type LaunchGate = "ok" | "not_validation" | "not_completed" | "inputs_unverified";

/** Whether the run may start a report: completed validation with recorded inputs. */
export function launchGate(
  run: Pick<BacktestRunResponse, "run_mode" | "status">,
  notes: string | null | undefined,
): LaunchGate {
  if (run.run_mode !== "validation") return "not_validation";
  if (run.status !== "completed") return "not_completed";
  const payload = parseNotes(notes ?? "");
  const identity =
    typeof payload === "object" && payload !== null && !Array.isArray(payload)
      ? (payload as Record<string, unknown>).validation_input
      : undefined;
  if (typeof identity !== "object" || identity === null || Array.isArray(identity)) {
    return "inputs_unverified";
  }
  return "ok";
}

/**
 * The API's upper estimate (`_estimated_seconds`): the parent's validation time
 * times the baseline plus every cell (1.0 always added) in waves of `workers`,
 * plus the margin-sweep cells (0 when the sweep is off) as their own waves.
 */
export function estimateSeconds(
  parentSeconds: number | null | undefined,
  multiples: number[],
  workers: number,
  sweepCells = 0,
): number | null {
  if (typeof parentSeconds !== "number" || !Number.isFinite(parentSeconds)) return null;
  const cells = new Set([...multiples, 1]).size;
  const waves = 1 + Math.ceil(cells / workers) + Math.ceil(sweepCells / workers);
  return roundHalfEven(parentSeconds * waves);
}

export function formatEta(seconds: number | null): string {
  if (seconds === null) return "unknown";
  if (seconds < 120) return `≈ ${seconds} s`;
  return `≈ ${Math.round(seconds / 60)} min`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "—";
  const days = Math.floor(seconds / 86_400);
  const hours = Math.floor((seconds % 86_400) / 3_600);
  const minutes = Math.floor((seconds % 3_600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

/** Workers default as the API does (`min(4, max(1, cpu // 2))`), sent explicitly. */
function defaultWorkers(): number {
  const cpu = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 1 : 1;
  return Math.min(4, Math.max(1, Math.floor(cpu / 2)));
}

/** Inline check of a launch field: a decimal comma or a non-number never reaches the API. */
export function numberFieldError(value: string): string | null {
  if (/^\s*-?\d+,\d+\s*$/.test(value))
    return `Use a dot for decimals (${value.trim().replace(",", ".")})`;
  return value.trim() !== "" && Number.isFinite(Number(value)) ? null : "Enter a number";
}

export function launchErrorText(err: unknown): string {
  const detail = formatApiErrorDetail(err);
  if (detail === "") return "Could not start the sizing report";
  if (detail.startsWith("report_running") || detail.startsWith("job_open")) {
    return `Another sizing report is running; one runs at a time. (${detail})`;
  }
  return detail;
}

// --- report states ----------------------------------------------------------------

interface QueryLike {
  isLoading: boolean;
  isError: boolean;
  error: unknown;
}

export type LoadView = { kind: "loading" } | { kind: "error"; text: string } | { kind: "ready" };

/**
 * The summary (launch gate, ETA) and report GETs: while either loads the panel
 * shows neither a gate warning nor the empty state; a failed GET shows its error.
 */
export function loadView(summary: QueryLike, report: QueryLike): LoadView {
  for (const [q, what] of [
    [summary, "run summary"],
    [report, "sizing report"],
  ] as const) {
    if (q.isError) {
      const detail = formatApiErrorDetail(q.error);
      return { kind: "error", text: `Could not load the ${what}${detail ? `: ${detail}` : ""}` };
    }
  }
  if (summary.isLoading || report.isLoading) return { kind: "loading" };
  return { kind: "ready" };
}

export type PanelState = "empty" | "running" | "completed" | "failed" | "stale";

export function panelState(resp: BankrollReportResponse | null | undefined): PanelState {
  if (resp == null) return "empty";
  if (resp.status === "running") return "running";
  if (resp.status === "failed") return "failed";
  if (resp.status === "stale_parent" || resp.freshness === "stale") return "stale";
  return "completed";
}

export interface HeadlineDiff {
  field: string;
  stored: string;
  replay: string;
}

/** `<field> stored=<a> replay=<b>` pairs from a `headline_mismatch` error message. */
export function parseHeadlineMismatch(message: string): HeadlineDiff[] {
  const out: HeadlineDiff[] = [];
  for (const [, field = "", stored = "", replay = ""] of message.matchAll(
    /(\w+) stored=([^;\s]+) replay=([^;\s:]+)/g,
  )) {
    out.push({ field, stored, replay });
  }
  return out;
}

export interface StaleView {
  title: string;
  reasons: string[];
  headline: HeadlineDiff[];
}

/** The "re-validate first" banner: a stale_parent report or a report whose parent moved. */
export function staleView(resp: BankrollReportResponse): StaleView | null {
  if (resp.status === "stale_parent") {
    const msg = resp.error_message ?? "";
    const body = msg.replace(/^stale_parent: /, "").replace(/: re-validate first$/, "");
    const headline = parseHeadlineMismatch(msg);
    const reason = headline.length > 0 ? body.split(":")[0] : body;
    const reproFailure = /^(headline_mismatch|inputs_mismatch)\b/.test(body);
    return {
      title: reproFailure
        ? `Parent results do not reproduce: ${REVALIDATE}`
        : `Parent result changed: ${REVALIDATE}`,
      reasons: reason ? [reason] : [],
      headline,
    };
  }
  if (resp.freshness === "stale") {
    return {
      title: `Parent changed since this report: ${REVALIDATE}`,
      reasons: resp.stale_reasons,
      headline: [],
    };
  }
  return null;
}

type Json = Record<string, unknown>;

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function obj(v: unknown): Json {
  return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Json) : {};
}

export interface RowView {
  k: number;
  role: string;
  status: string;
  effectiveCap: number | null;
  requestedRisk: number | null;
  effectiveRisk: number | null;
  leverage: number | null;
  riskHeld: boolean;
  trades: number | null;
  totalReturn: number | null;
  mtmDd: number | null;
  fees: number | null;
  funding: number | null;
  liquidations: number | null;
  firstLiquidationS: number | null;
  ruinS: number | null;
  eligible: boolean;
  reasonText: string;
  recommended: boolean;
}

export function rowViews(report: Json, recommendedK: number | null): RowView[] {
  const rows = Array.isArray(report.rows) ? report.rows : [];
  return rows.map((raw) => {
    const r = obj(raw);
    const k = num(r.k) ?? Number.NaN;
    const eligible = r.eligible === true;
    const why = eligible ? "" : reasonText(String(r.ineligible_reason ?? r.status ?? ""));
    return {
      k,
      role: String(r.role ?? "cell"),
      status: String(r.status ?? ""),
      effectiveCap: num(r.effective_cap),
      requestedRisk: num(r.requested_risk),
      effectiveRisk: num(r.effective_risk),
      leverage: num(r.leverage),
      riskHeld: r.risk_held_at_limit === true,
      trades: num(r.trades),
      totalReturn: num(r.total_return),
      mtmDd: num(r.mtm_dd),
      fees: num(r.total_fees),
      funding: num(r.total_funding),
      liquidations: num(r.liquidation_count),
      firstLiquidationS: num(r.time_to_first_liquidation_s),
      ruinS: num(r.time_to_ruin_s),
      eligible,
      reasonText: r.status === "ok" || eligible ? why : `${why}${r.error ? `: ${r.error}` : ""}`,
      recommended: recommendedK !== null && k === recommendedK,
    };
  });
}

export type RecommendationView =
  | { kind: "recommended"; k: number; ceiling: number | null; text: string }
  | { kind: "none"; reason: string; text: string; ceiling: number | null };

/** The recommendation line: a tested row only, masked when the report is stale. */
export function recommendationView(report: Json, freshness: string): RecommendationView {
  const rec = obj(report.recommendation);
  const ceiling = num(rec.ceiling);
  if (freshness === "stale") {
    return { kind: "none", reason: "stale", text: reasonText("stale"), ceiling };
  }
  const k = num(rec.k);
  const tested = rowViews(report, null).some((r) => r.k === k);
  if (k !== null && tested) {
    return { kind: "recommended", k, ceiling, text: typeof rec.text === "string" ? rec.text : "" };
  }
  const reason = typeof rec.reason === "string" ? rec.reason : "no_eligible_row";
  return { kind: "none", reason, text: reasonText(reason), ceiling };
}

// --- margin sweep -----------------------------------------------------------------

/** A cell's liquidating move (design R28): "32.80% short", never / at entry / unavailable. */
export function formatMove(raw: unknown): string {
  const m = obj(raw);
  const value = num(m.value);
  let text: string;
  if (m.status === "never") text = "not liquidatable in this scenario";
  else if (m.status === "at_entry") text = "0.00% (at entry)";
  else if (m.status === "unavailable")
    text = typeof m.reason === "string" ? `unavailable (${m.reason})` : "unavailable";
  else if (m.status === "ok" && value !== null)
    text = `${(value * 100).toFixed(2)}%${typeof m.side === "string" ? ` ${m.side}` : ""}`;
  else return "—";
  return m.path_dependent === true ? `${text} (simultaneous entry only)` : text;
}

export interface MarginRowView {
  key: string;
  mode: string;
  leverage: number | null;
  marginLimit: number | null;
  positions: string;
  move: string;
  totalReturn: number | null;
  mtmDd: number | null;
  liquidations: number | null;
  eligible: boolean;
  reasonText: string;
  safest: boolean;
}

export type SafestView = { kind: "safest"; text: string } | { kind: "none"; text: string };

export type SweepView =
  | { kind: "none" }
  | { kind: "message"; title: string; text: string }
  | {
      kind: "completed";
      title: string;
      rows: MarginRowView[];
      safest: SafestView;
      label: string;
      crossNote: boolean;
    };

function sameSetting(a: Json, b: Json): boolean {
  return a.mode === b.mode && a.leverage === b.leverage && a.max_margin_pct === b.max_margin_pct;
}

/**
 * The margin-settings card: none for a report without a sweep (legacy or not requested),
 * a skipped / failed line, or the cells with the safest setting (masked when stale).
 */
export function sweepView(report: Json, freshness: string): SweepView {
  if (report.margin_sweep == null) return { kind: "none" };
  const sweep = obj(report.margin_sweep);
  const k = num(sweep.k);
  const title = k === null ? "Margin settings" : `Margin settings at k = ${k}`;
  const reason = typeof sweep.reason === "string" ? sweep.reason : "unknown";
  if (sweep.status === "skipped") {
    return { kind: "message", title, text: `Sweep skipped: ${reasonText(reason)}` };
  }
  if (sweep.status !== "completed") {
    return { kind: "message", title, text: `Sweep failed: ${reason}` };
  }
  const stale = freshness === "stale";
  const best = stale || sweep.safest == null ? null : obj(sweep.safest);
  const cells = Array.isArray(sweep.cells) ? sweep.cells.map(obj) : [];
  const rows = cells.map((c, i): MarginRowView => {
    const eligible = c.eligible === true;
    const why = eligible ? "" : reasonText(String(c.ineligible_reason ?? c.status ?? ""));
    const positions = num(c.positions);
    const remainder = num(c.remainder) ?? 0;
    return {
      key: `${String(c.mode)}-${String(c.leverage)}-${String(c.max_margin_pct)}-${i}`,
      mode: String(c.mode ?? ""),
      leverage: num(c.leverage),
      marginLimit: num(c.max_margin_pct),
      positions: positions === null ? "—" : `${positions}${remainder > 0 ? " + 1 smaller" : ""}`,
      move: formatMove(c.move),
      totalReturn: num(c.total_return),
      mtmDd: num(c.mtm_dd),
      liquidations: num(c.liquidation_count),
      eligible,
      reasonText: c.status === "ok" || eligible ? why : `${why}${c.error ? `: ${c.error}` : ""}`,
      safest: best !== null && sameSetting(c, best),
    };
  });
  let safest: SafestView;
  if (stale) safest = { kind: "none", text: reasonText("stale") };
  else if (best === null) safest = { kind: "none", text: reasonText("no_safe_setting") };
  else {
    const limit = num(best.max_margin_pct);
    safest = {
      kind: "safest",
      text:
        `Safest setting: ${String(best.mode)}, leverage ${String(best.leverage)}, ` +
        `margin limit ${pct(limit, 0)} (liquidating move ${formatMove(best.move)})`,
    };
  }
  const scenario = obj(sweep.scenario);
  return {
    kind: "completed",
    title,
    rows,
    safest,
    label: typeof scenario.label === "string" ? scenario.label : "",
    crossNote: rows.some((r) => r.mode === "cross"),
  };
}

// --- rendering --------------------------------------------------------------------

function pct(v: number | null, digits = 1): string {
  return v === null ? "—" : `${(v * 100).toFixed(digits)}%`;
}

function fixed(v: number | null, digits = 2): string {
  return v === null ? "—" : v.toFixed(digits);
}

export function formatMultiple(v: number | null): string {
  return v === null ? "—" : `${fixed(v)}×`;
}

function Banner({ tone, children }: { tone: "warn" | "error"; children: ReactNode }) {
  const cls =
    tone === "error"
      ? "border-destructive/40 bg-destructive/10 text-destructive"
      : "border-yellow-500/40 bg-yellow-500/10 text-foreground";
  return <div className={`rounded-md border px-3 py-2 text-sm ${cls}`}>{children}</div>;
}

const GATE_TEXT: Record<Exclude<LaunchGate, "ok">, string> = {
  not_validation: "",
  not_completed: "The run must complete before a sizing report can start.",
  inputs_unverified: `This validation recorded no input identity: ${REVALIDATE}.`,
};

function LaunchForm({
  runId,
  parentSeconds,
  disabled,
  lastSweep,
}: {
  runId: number;
  parentSeconds: number | null;
  disabled: boolean;
  lastSweep: boolean;
}) {
  const queryClient = useQueryClient();
  // Off by default (the API's default); a re-launch keeps the last report's choice.
  const [sweep, setSweep] = useState(lastSweep);
  const [userCap, setUserCap] = useState("1");
  const [ddLimit, setDdLimit] = useState("0.3");
  const [targetVol, setTargetVol] = useState("0.2");
  const [workers, setWorkers] = useState(String(defaultWorkers()));
  const launch = useLaunchBankrollReportApiResultsRunsRunIdBankrollReportPost({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetBankrollReportApiResultsRunsRunIdBankrollReportGetQueryKey(runId),
        });
      },
      onError: (err) => {
        toast.error("Sizing report not started", { description: launchErrorText(err) });
      },
    },
  });
  const w = Number.parseInt(workers, 10);
  const eta = estimateSeconds(
    parentSeconds,
    DEFAULT_MULTIPLES,
    Number.isFinite(w) && w > 0 ? w : 1,
    sweep ? DEFAULT_SWEEP_CELLS : 0,
  );

  const invalid = [userCap, ddLimit, targetVol, workers].some((v) => numberFieldError(v) !== null);

  const field = (id: string, label: string, value: string, set: (v: string) => void) => {
    const error = numberFieldError(value);
    return (
      <div className="flex flex-col gap-1">
        <Label htmlFor={id} className="text-xs text-muted-foreground">
          {label}
        </Label>
        <Input
          id={id}
          className="h-8 w-24 tabular-nums"
          inputMode="decimal"
          value={value}
          aria-invalid={error !== null}
          onChange={(e) => set(e.target.value)}
        />
        {error && <span className="text-xs text-destructive">{error}</span>}
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-4">
        {field(`bk-cap-${runId}`, "Your size cap (k)", userCap, setUserCap)}
        {field(`bk-dd-${runId}`, "Drawdown limit", ddLimit, setDdLimit)}
        {field(`bk-vol-${runId}`, "Target volatility", targetVol, setTargetVol)}
        {field(`bk-workers-${runId}`, "Parallel engines (1–4)", workers, setWorkers)}
        <Button
          size="sm"
          disabled={disabled || invalid || launch.isPending}
          onClick={() =>
            launch.mutate({
              runId,
              data: {
                user_cap: Number(userCap),
                dd_limit: Number(ddLimit),
                target_vol: Number(targetVol),
                workers: Number(workers),
                margin_sweep: sweep,
              },
            })
          }
        >
          Start sizing report
        </Button>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox
          id={`bk-sweep-${runId}`}
          checked={sweep}
          onCheckedChange={(v) => setSweep(v === true)}
        />
        <Label htmlFor={`bk-sweep-${runId}`} className="cursor-pointer text-sm">
          {SWEEP_CHECKBOX_LABEL}
        </Label>
      </div>
      <p className="text-xs text-muted-foreground">
        Sizes k = {DEFAULT_MULTIPLES.join(", ")} at the run's own leverage and margin mode (cross if
        liquidation was off)
        {sweep &&
          `, then ${DEFAULT_SWEEP_CELLS} margin settings (isolated and cross, leverage 3, 5, 10, margin limit 50% and 100%) at the recommended size`}
        . Estimated time: <span className="font-medium text-foreground">{formatEta(eta)}</span>
        {eta === null && " (the run recorded no validation time)"}.
      </p>
    </div>
  );
}

function KellyCard({ kelly }: { kelly: Json }) {
  const status = typeof kelly.status === "string" ? kelly.status : null;
  const actionable = typeof kelly.actionable_reason === "string" ? kelly.actionable_reason : null;
  const items: [string, string][] = [
    ["Daily returns", String(num(kelly.n) ?? "—")],
    ["Full Kelly", formatMultiple(num(kelly.f_full))],
    ["Half Kelly", formatMultiple(num(kelly.f_half))],
    ["Vol target", formatMultiple(num(kelly.vol_target))],
    ["Annual vol", pct(num(kelly.sigma_ann))],
    ["Excess kurtosis", fixed(num(kelly.excess_kurtosis))],
  ];
  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-3 lg:grid-cols-6">
        {items.map(([label, value]) => (
          <div key={label}>
            <div className="text-xs text-muted-foreground">{label}</div>
            <div className="font-semibold tabular-nums">{value}</div>
          </div>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        Multiples of the validated sizing, from the baseline's daily mark-to-market returns. Vol
        target is shown only; it never drives the recommendation.
        {kelly.fat_tail === true && " Fat-tailed daily returns: Kelly overstates."}
      </p>
      {(status !== "ok" || actionable) && (
        <Banner tone="warn">
          Kelly not actionable: {reasonText(actionable ?? status ?? "daily_unavailable")}
        </Banner>
      )}
    </div>
  );
}

function StressTable({ rows }: { rows: RowView[] }) {
  return (
    // w-0 min-w-full: the 13-column table must not add its width to the page's
    // min-content width (it widened the sidebar inset), so it scrolls here instead.
    <div className="w-0 min-w-full overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>
              <Tooltip>
                <TooltipTrigger className="underline decoration-dotted">
                  k (size step)
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">{K_TOOLTIP}</TooltipContent>
              </Tooltip>
            </TableHead>
            <TableHead>Max exposure</TableHead>
            <TableHead>Risk per trade</TableHead>
            <TableHead>Leverage</TableHead>
            <TableHead className="text-right">Trades</TableHead>
            <TableHead className="text-right">Return</TableHead>
            <TableHead className="text-right">MTM drawdown</TableHead>
            <TableHead className="text-right">Fees</TableHead>
            <TableHead className="text-right">Funding</TableHead>
            <TableHead className="text-right">Liquidations</TableHead>
            <TableHead>First liquidation</TableHead>
            <TableHead>Time to ruin</TableHead>
            <TableHead>Verdict</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.k} className={r.recommended ? "bg-primary/10" : undefined}>
              <TableCell className="tabular-nums">
                {r.k}
                {r.role !== "cell" && (
                  <span className="ml-1 text-xs text-muted-foreground">({r.role})</span>
                )}
              </TableCell>
              <TableCell className="tabular-nums">{formatMultiple(r.effectiveCap)}</TableCell>
              <TableCell className="tabular-nums">
                {pct(r.effectiveRisk)}
                {r.riskHeld && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant="outline" className="ml-1 text-[10px]">
                        {RISK_HELD_LABEL} (asked {pct(r.requestedRisk)})
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">{RISK_HELD_TOOLTIP}</TooltipContent>
                  </Tooltip>
                )}
              </TableCell>
              <TableCell className="tabular-nums">{r.leverage ?? "—"}</TableCell>
              <TableCell className="text-right tabular-nums">{r.trades ?? "—"}</TableCell>
              <TableCell className="text-right tabular-nums">{pct(r.totalReturn)}</TableCell>
              <TableCell className="text-right tabular-nums">{pct(r.mtmDd)}</TableCell>
              <TableCell className="text-right tabular-nums">{fixed(r.fees)}</TableCell>
              <TableCell className="text-right tabular-nums">{fixed(r.funding)}</TableCell>
              <TableCell className="text-right tabular-nums">{r.liquidations ?? "—"}</TableCell>
              <TableCell className="tabular-nums">{formatDuration(r.firstLiquidationS)}</TableCell>
              <TableCell className="tabular-nums">{formatDuration(r.ruinS)}</TableCell>
              <TableCell className="text-xs">
                {r.recommended ? (
                  <Badge>recommended</Badge>
                ) : r.eligible ? (
                  <span className="text-muted-foreground">eligible</span>
                ) : (
                  <span className="text-muted-foreground">{r.reasonText}</span>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function MarginSweepCard({ view }: { view: Exclude<SweepView, { kind: "none" }> }) {
  return (
    <div className="flex flex-col gap-2 rounded-md border p-3">
      <div className="text-sm font-semibold">{view.title}</div>
      {view.kind === "message" ? (
        <p className="text-sm text-muted-foreground">{view.text}</p>
      ) : (
        <>
          <div className="text-sm">
            {view.safest.kind === "safest" ? (
              <span className="font-medium">{view.safest.text}.</span>
            ) : (
              <span>{view.safest.text}.</span>
            )}
          </div>
          <div className="w-0 min-w-full overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Mode</TableHead>
                  <TableHead>Leverage</TableHead>
                  <TableHead>Margin limit</TableHead>
                  <TableHead>Positions</TableHead>
                  <TableHead>Liquidating move (side)</TableHead>
                  <TableHead className="text-right">Return</TableHead>
                  <TableHead className="text-right">MTM DD</TableHead>
                  <TableHead className="text-right">Liquidations</TableHead>
                  <TableHead>Verdict</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {view.rows.map((r) => (
                  <TableRow key={r.key} className={r.safest ? "bg-primary/10" : undefined}>
                    <TableCell>{r.mode}</TableCell>
                    <TableCell className="tabular-nums">{r.leverage ?? "—"}</TableCell>
                    <TableCell className="tabular-nums">{pct(r.marginLimit, 0)}</TableCell>
                    <TableCell className="tabular-nums">{r.positions}</TableCell>
                    <TableCell className="tabular-nums">{r.move}</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(r.totalReturn)}</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(r.mtmDd)}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {r.liquidations ?? "—"}
                    </TableCell>
                    <TableCell className="text-xs">
                      {r.safest ? (
                        <Badge>safest</Badge>
                      ) : (
                        <span className="text-muted-foreground">
                          {r.eligible ? "eligible" : r.reasonText}
                        </span>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <div className="flex flex-col gap-1 text-xs text-muted-foreground">
            {view.label && <p>{view.label}</p>}
            {view.crossNote && <p>{CROSS_NOTE}</p>}
          </div>
        </>
      )}
    </div>
  );
}

function ReportBody({ resp }: { resp: BankrollReportResponse }) {
  const report = obj(resp.report);
  const rec = recommendationView(report, resp.freshness);
  const rows = rowViews(report, rec.kind === "recommended" ? rec.k : null);
  const sweep = sweepView(report, resp.freshness);
  const disclosures = obj(report.disclosures);
  const notes = [disclosures.slippage_timing, disclosures.k1_control].filter(
    (t): t is string => typeof t === "string",
  );
  return (
    <div className="flex flex-col gap-4">
      <KellyCard kelly={obj(report.kelly)} />
      <div className="text-sm">
        {rec.kind === "recommended" ? (
          <span>
            Recommended size: <span className="font-semibold">k = {rec.k}</span> (a tested row)
            {rec.text && ` — ${rec.text}`}. Ceiling min(half Kelly, your cap) = {fixed(rec.ceiling)}
            .
          </span>
        ) : (
          <span>
            No recommendation: {rec.text}
            {rec.ceiling !== null && ` (ceiling ${fixed(rec.ceiling)})`}.
          </span>
        )}
      </div>
      <StressTable rows={rows} />
      {sweep.kind !== "none" && <MarginSweepCard view={sweep} />}
      <div className="flex flex-col gap-1 text-xs text-muted-foreground">
        <p>{typeof report.label === "string" ? report.label : REPLAY_CAPTION}</p>
        {notes.map((t) => (
          <p key={t}>{t}</p>
        ))}
      </div>
    </div>
  );
}

function StaleBanner({ view }: { view: StaleView }) {
  return (
    <Banner tone="warn">
      <div className="font-medium">{view.title}</div>
      {view.reasons.length > 0 && <div className="text-xs">{view.reasons.join(", ")}</div>}
      {view.headline.length > 0 && (
        <table className="mt-1 text-xs tabular-nums">
          <thead>
            <tr className="text-muted-foreground">
              <th className="pr-4 text-left font-normal">Field</th>
              <th className="pr-4 text-left font-normal">Stored</th>
              <th className="text-left font-normal">Replay</th>
            </tr>
          </thead>
          <tbody>
            {view.headline.map((d) => (
              <tr key={d.field}>
                <td className="pr-4">{d.field}</td>
                <td className="pr-4">{d.stored}</td>
                <td>{d.replay}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Banner>
  );
}

export function BankrollPanel({ runId }: { runId: number }) {
  const metaQuery = useGetRunMetaApiResultsRunsRunIdMetaGet(runId);
  const run = metaQuery.data?.status === 200 ? metaQuery.data.data : undefined;
  const isValidation = run?.run_mode === "validation";
  const summaryQuery = useGetRunSummaryApiResultsRunsRunIdGet(runId, {
    query: { enabled: isValidation },
  });
  const summary = summaryQuery.data?.status === 200 ? summaryQuery.data.data : undefined;
  const reportQuery = useGetBankrollReportApiResultsRunsRunIdBankrollReportGet(runId, {
    query: {
      enabled: isValidation,
      refetchInterval: (query) => {
        const d = query.state.data;
        return d?.status === 200 && d.data?.status === "running" ? POLL_MS : false;
      },
    },
  });
  const queryClient = useQueryClient();
  const cancel = useKillJobApiBacktestJobsRunIdDelete({
    mutation: {
      onSuccess: () =>
        queryClient.invalidateQueries({
          queryKey: getGetBankrollReportApiResultsRunsRunIdBankrollReportGetQueryKey(runId),
        }),
      onError: (err) =>
        toast.error("Cancel failed", { description: formatApiErrorDetail(err) || undefined }),
    },
  });

  if (!run || !isValidation) return null;
  const load = loadView(summaryQuery, reportQuery);
  if (load.kind !== "ready") {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
            Sizing report
          </CardTitle>
        </CardHeader>
        <CardContent>
          {load.kind === "error" ? (
            <Banner tone="error">{load.text}</Banner>
          ) : (
            <p className="text-sm text-muted-foreground">Loading sizing report…</p>
          )}
        </CardContent>
      </Card>
    );
  }
  const resp = reportQuery.data?.status === 200 ? reportQuery.data.data : null;
  const state = panelState(resp);
  const gate = launchGate(run, summary?.notes);
  const stale = resp ? staleView(resp) : null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Sizing report
          {resp && <Badge variant="outline">{state}</Badge>}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {gate !== "ok" && gate !== "not_validation" && (
          <Banner tone="warn">{GATE_TEXT[gate]}</Banner>
        )}
        {state === "empty" && (
          <p className="text-sm text-muted-foreground">
            No sizing report yet. One re-runs this validation at several sizes and reads Kelly from
            its daily returns.
          </p>
        )}
        {state === "running" && resp && (
          <div className="flex items-center gap-3 text-sm">
            <span>
              Report #{resp.id} running since {resp.created_at} (checks every 5 s).
            </span>
            <Button
              size="sm"
              variant="outline"
              disabled={cancel.isPending}
              onClick={() => cancel.mutate({ runId: resp.job_id })}
            >
              Cancel
            </Button>
          </div>
        )}
        {state === "failed" && resp && (
          <Banner tone="error">
            Report #{resp.id} failed: {resp.error_message ?? "unknown"}
          </Banner>
        )}
        {stale && <StaleBanner view={stale} />}
        {resp?.report && <ReportBody resp={resp} />}
        {state !== "running" && (
          <LaunchForm
            runId={runId}
            parentSeconds={summary?.execution_time_seconds ?? null}
            disabled={gate !== "ok"}
            lastSweep={obj(resp?.request).margin_sweep === true}
          />
        )}
      </CardContent>
    </Card>
  );
}
