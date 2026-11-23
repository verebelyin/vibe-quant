import { Link } from "@tanstack/react-router";
import { fmtSharpe } from "@/lib/metrics";

export interface ScreenSummary {
  screen_sharpe?: number | null;
  screen_status?: string | null;
  screen_run_id?: number | null;
  screen_pf?: number | null;
  screen_max_dd?: number | null;
  screen_return?: number | null;
  screen_trades?: number | null;
  screen_error?: string | null;
  screen_completed_at?: string | null;
}

const MIN_TRADES = 50;
const SHARPE_WINNER_THRESHOLD = 1;

function fmt(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return String(value);
}

function fmtPct(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

/** Winner rule stays on the RAW stored value (>= 1 in √252 units). */
export function screenBadgeClasses(s: ScreenSummary): string {
  if (s.screen_status === "failed") {
    return "border-red-500/40 bg-red-500/10 text-red-300 hover:bg-red-500/15";
  }
  if (s.screen_status !== "done") {
    return "border-border/40 bg-muted/30 text-muted-foreground";
  }
  const trades = s.screen_trades ?? 0;
  const sharpe = s.screen_sharpe ?? null;
  if (trades < MIN_TRADES) {
    return "border-border/40 bg-muted/30 text-muted-foreground hover:bg-muted/40";
  }
  if (sharpe !== null && Number.isFinite(sharpe) && sharpe >= SHARPE_WINNER_THRESHOLD) {
    return "border-green-500/40 bg-green-500/10 text-green-300 hover:bg-green-500/15";
  }
  return "border-border/40 bg-muted/30 text-muted-foreground hover:bg-muted/40";
}

function buildTooltip(s: ScreenSummary): string {
  if (s.screen_status === "failed") {
    return `Status: failed\nError: ${s.screen_error ?? "—"}\nCompleted: ${s.screen_completed_at ?? "—"}`;
  }
  const lines = [
    `Status: ${s.screen_status ?? "—"}`,
    `Sharpe: ${fmtSharpe(s.screen_sharpe)} (ann √365)`,
    `Profit factor: ${fmt(s.screen_pf)}`,
    `Max drawdown: ${fmtPct(s.screen_max_dd)}`,
    `Total return: ${fmtPct(s.screen_return)}`,
    `Trades: ${fmtInt(s.screen_trades)}`,
    `Completed: ${s.screen_completed_at ?? "—"}`,
  ];
  return lines.join("\n");
}

export function screenBadgeText(s: ScreenSummary): string {
  return s.screen_status === "failed" ? "Sharpe: ✕" : `Sharpe: ${fmtSharpe(s.screen_sharpe)}`;
}

export function ScreenBadge({ summary }: { summary: ScreenSummary }) {
  const text = screenBadgeText(summary);
  const cls =
    "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 font-mono text-xs " +
    screenBadgeClasses(summary);
  const tooltip = buildTooltip(summary);

  if (summary.screen_run_id != null) {
    return (
      <Link
        to="/results/$runId"
        params={{ runId: String(summary.screen_run_id) }}
        title={tooltip}
        className={cls + " no-underline"}
      >
        {text}
      </Link>
    );
  }
  return (
    <span title={tooltip} className={cls}>
      {text}
    </span>
  );
}
