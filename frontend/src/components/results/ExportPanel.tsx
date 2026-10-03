import { FileJson, FileSpreadsheet, FileText } from "lucide-react";
import type { BacktestResultResponse } from "@/api/generated/models";
import { useGetRunSummaryApiResultsRunsRunIdGet } from "@/api/generated/results/results";
import { Button } from "@/components/ui/button";
import { formatSignedUsd, fractionToPercent, netPnlUsd } from "@/lib/metrics";

interface ExportPanelProps {
  runId: number;
}

function downloadBlob(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Backend sends fractions; every "(%)" column / "%" suffix is fraction * 100.
export function metricsToCSV(data: BacktestResultResponse): string {
  const fields: Array<[string, string | number | null]> = [
    ["Run ID", data.run_id],
    ["Total Return (%)", fractionToPercent(data.total_return)],
    ["Net PnL (USDT)", netPnlUsd(data.total_return, data.starting_balance)],
    ["CAGR (%)", fractionToPercent(data.cagr)],
    ["Sharpe Ratio", data.sharpe_ratio],
    ["Sortino Ratio", data.sortino_ratio],
    ["Calmar Ratio", data.calmar_ratio],
    ["Max Drawdown (%)", fractionToPercent(data.max_drawdown)],
    ["Max DD Duration (days)", data.max_drawdown_duration_days],
    ["Annual Volatility (%)", fractionToPercent(data.volatility_annual)],
    ["Total Trades", data.total_trades],
    ["Win Rate (%)", fractionToPercent(data.win_rate)],
    ["Profit Factor", data.profit_factor],
    ["Avg Win", data.avg_win],
    ["Avg Loss", data.avg_loss],
    ["Largest Win", data.largest_win],
    ["Largest Loss", data.largest_loss],
    ["Avg Trade Duration (hours)", data.avg_trade_duration_hours],
    ["Total Fees", data.total_fees],
    ["Total Funding", data.total_funding],
    ["Total Slippage", data.total_slippage],
    ["Deflated Sharpe", data.deflated_sharpe],
    ["Walk-Forward Efficiency", data.walk_forward_efficiency],
    ["Purged K-Fold Mean Sharpe", data.purged_kfold_mean_sharpe],
  ];

  const header = fields.map(([k]) => k).join(",");
  const values = fields.map(([, v]) => (v == null ? "" : String(v))).join(",");
  return `${header}\n${values}`;
}

export function metricsToReport(data: BacktestResultResponse): string {
  const fmt = (v: number | null | undefined, dec = 2, suffix = "") =>
    v == null ? "N/A" : `${v.toFixed(dec)}${suffix}`;
  const pct = (v: number | null | undefined, dec = 2) => fmt(fractionToPercent(v), dec, "%");

  const lines = [
    `=== Backtest Report: Run #${data.run_id} ===`,
    `Generated: ${new Date().toISOString()}`,
    "",
    "--- Performance ---",
    `Total Return:     ${pct(data.total_return)}`,
    `Net PnL:          ${formatSignedUsd(netPnlUsd(data.total_return, data.starting_balance))}`,
    `CAGR:             ${pct(data.cagr)}`,
    `Sharpe Ratio:     ${fmt(data.sharpe_ratio)}`,
    `Sortino Ratio:    ${fmt(data.sortino_ratio)}`,
    `Calmar Ratio:     ${fmt(data.calmar_ratio)}`,
    `Max Drawdown:     ${pct(data.max_drawdown)}`,
    `Annual Volatility: ${pct(data.volatility_annual)}`,
    "",
    "--- Trading ---",
    `Total Trades:     ${data.total_trades ?? "N/A"}`,
    `Win Rate:         ${pct(data.win_rate, 1)}`,
    `Profit Factor:    ${fmt(data.profit_factor)}`,
    `Winning Trades:   ${data.winning_trades ?? "N/A"}`,
    `Losing Trades:    ${data.losing_trades ?? "N/A"}`,
    "",
    "--- Costs ---",
    `Total Fees:       ${fmt(data.total_fees)}`,
    `Total Slippage:   ${fmt(data.total_slippage)}`,
    `Total Funding:    ${fmt(data.total_funding)}`,
    "",
    "--- Overfitting ---",
    `Deflated Sharpe:       ${fmt(data.deflated_sharpe)}`,
    `Walk-Forward Eff:      ${fmt(data.walk_forward_efficiency)}`,
    `Purged K-Fold Sharpe:  ${fmt(data.purged_kfold_mean_sharpe)}`,
  ];

  // user_notes only: `notes` is machine JSON (discovery payload, flags).
  if (data.user_notes) {
    lines.push("", "--- Notes ---", data.user_notes);
  }

  return lines.join("\n");
}

export function ExportPanel({ runId }: ExportPanelProps) {
  const query = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const data = query.data?.data as BacktestResultResponse | undefined;

  const exportCSV = () => {
    if (!data) return;
    downloadBlob(metricsToCSV(data), `run-${runId}-metrics.csv`, "text/csv");
  };

  const exportJSON = () => {
    if (!data) return;
    downloadBlob(JSON.stringify(data, null, 2), `run-${runId}-summary.json`, "application/json");
  };

  const exportReport = () => {
    if (!data) return;
    downloadBlob(metricsToReport(data), `run-${runId}-report.txt`, "text/plain");
  };

  const disabled = query.isLoading || !data;

  return (
    <div className="flex items-center gap-2">
      <Button variant="outline" size="sm" onClick={exportCSV} disabled={disabled}>
        <FileSpreadsheet className="size-3.5" />
        CSV
      </Button>
      <Button variant="outline" size="sm" onClick={exportJSON} disabled={disabled}>
        <FileJson className="size-3.5" />
        JSON
      </Button>
      <Button variant="outline" size="sm" onClick={exportReport} disabled={disabled}>
        <FileText className="size-3.5" />
        Report
      </Button>
    </div>
  );
}
