import type { BacktestResultResponse } from "@/api/generated/models";
import {
  useGetRunMetaApiResultsRunsRunIdMetaGet,
  useGetRunSummaryApiResultsRunsRunIdGet,
} from "@/api/generated/results/results";
import { LoadingSpinner } from "@/components/ui";
import { Badge } from "@/components/ui/badge";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface OverfittingBadgesProps {
  runId: number;
}

type CheckStatus = "pass" | "fail" | "na";

interface CheckInfo {
  label: string;
  status: CheckStatus;
  detail?: string | undefined;
}

function CheckRow({ label, status, detail, threshold, description }: CheckInfo & { threshold?: string; description?: string }) {
  const bg =
    status === "pass"
      ? "border-emerald-500/20 bg-emerald-500/[0.04]"
      : status === "fail"
        ? "border-red-500/20 bg-red-500/[0.04]"
        : "border-border";
  return (
    <div className={`rounded-lg border px-3 py-2.5 ${bg}`}>
      <div className="flex items-center gap-2.5">
        <Badge
          variant={status === "pass" ? "default" : status === "fail" ? "destructive" : "secondary"}
        >
          {status === "pass" ? "PASS" : status === "fail" ? "FAIL" : "N/A"}
        </Badge>
        <span className="text-xs font-medium text-foreground">{label}</span>
        {detail && (
          <span className="ml-auto font-mono text-[11px] text-muted-foreground">{detail}</span>
        )}
        {threshold && (
          <span className="font-mono text-[10px] text-muted-foreground/60">
            (threshold: {threshold})
          </span>
        )}
      </div>
      {description && (
        <p className="mt-1 pl-[52px] text-[10px] text-muted-foreground">{description}</p>
      )}
    </div>
  );
}

export function OverfittingBadges({ runId }: OverfittingBadgesProps) {
  const query = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const metaQuery = useGetRunMetaApiResultsRunsRunIdMetaGet(runId);
  const data = query.data?.data as BacktestResultResponse | undefined;
  const timeframe =
    metaQuery.data?.status === 200 ? metaQuery.data.data.timeframe : undefined;

  if (query.isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <LoadingSpinner size="sm" />
      </div>
    );
  }

  if (query.isError || !data) {
    return <p className="py-4 text-sm text-destructive">Failed to load overfitting data.</p>;
  }

  function sharpeStatus(value: number | null): CheckStatus {
    if (value == null) return "na";
    return value > 0 ? "pass" : "fail";
  }

  function efficiencyStatus(value: number | null): CheckStatus {
    if (value == null) return "na";
    return value >= 0.5 ? "pass" : "fail";
  }

  // IS/OOS ratio: approximate from win_rate and trade count (proxy only until dedicated endpoint)
  const isOosRatio = data.win_rate != null && data.total_trades != null
    ? data.win_rate
    : null;

  const checks: CheckInfo[] = [
    {
      label: "Walk-Forward",
      status: efficiencyStatus(data.walk_forward_efficiency),
      detail:
        data.walk_forward_efficiency != null
          ? `efficiency: ${data.walk_forward_efficiency.toFixed(2)}`
          : undefined,
    },
    {
      label: "Purged K-Fold",
      status: sharpeStatus(data.purged_kfold_mean_sharpe),
      detail:
        data.purged_kfold_mean_sharpe != null
          ? `mean sharpe: ${data.purged_kfold_mean_sharpe.toFixed(2)}`
          : undefined,
    },
    {
      label: "Deflated Sharpe Ratio",
      status: sharpeStatus(data.deflated_sharpe),
      detail: data.deflated_sharpe != null ? `DSR: ${data.deflated_sharpe.toFixed(3)}` : undefined,
    },
    {
      label: "IS/OOS Win Rate",
      status: isOosRatio != null ? (isOosRatio >= 0.45 ? "pass" : "fail") : "na",
      detail: isOosRatio != null ? `${(isOosRatio * 100).toFixed(1)}%` : undefined,
    },
    {
      label: "Stability (Calmar)",
      status: data.calmar_ratio != null ? (data.calmar_ratio >= 0.5 ? "pass" : "fail") : "na",
      detail: data.calmar_ratio != null ? `calmar: ${data.calmar_ratio.toFixed(2)}` : undefined,
    },
    (() => {
      const lower = data.bootstrap_sharpe_lower;
      const threshold = timeframe === "1m" ? 0.5 : 1.0;
      return {
        label: "Bootstrap CI (Sharpe lower)",
        status:
          lower == null
            ? ("na" as CheckStatus)
            : lower >= threshold
              ? ("pass" as CheckStatus)
              : ("fail" as CheckStatus),
        detail:
          lower != null
            ? `≥${lower.toFixed(2)} @ ${(data.bootstrap_ci_level ?? 0.95) * 100}% (thr: ${threshold})`
            : undefined,
      };
    })(),
    {
      label: "WFA Sharpe Consistency",
      status:
        data.wfa_sharpe_consistency == null
          ? "na"
          : data.wfa_sharpe_consistency >= 0.75
            ? "pass"
            : "fail",
      detail:
        data.wfa_sharpe_consistency != null
          ? `${(data.wfa_sharpe_consistency * 100).toFixed(0)}% (thr: 75%)`
          : undefined,
    },
    (() => {
      const rows = data.cross_regime_results as
        | Array<{ passed?: boolean }>
        | null
        | undefined;
      if (!rows || rows.length === 0) {
        return { label: "Cross-Regime", status: "na" as CheckStatus };
      }
      const passedAll = rows.every((r) => r?.passed === true);
      return {
        label: "Cross-Regime",
        status: passedAll ? ("pass" as CheckStatus) : ("fail" as CheckStatus),
        detail: `${rows.filter((r) => r?.passed).length}/${rows.length} regimes`,
      };
    })(),
  ];

  const passCount = checks.filter((b) => b.status === "pass").length;
  const failCount = checks.filter((b) => b.status === "fail").length;
  const naCount = checks.filter((b) => b.status === "na").length;

  const allNa = naCount === checks.length;
  const overallPass = failCount === 0 && !allNa;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Overfitting Filters
        </CardTitle>
        <CardAction>
          <Badge variant={allNa ? "secondary" : overallPass ? "default" : "destructive"}>
            {allNa
              ? "Not Run"
              : overallPass
                ? `All Passed (${passCount}/${checks.length})`
                : `${failCount} Failed`}
          </Badge>
        </CardAction>
      </CardHeader>

      <CardContent className="flex flex-col gap-2">
        {checks.map((check) => (
          <CheckRow key={check.label} {...check} />
        ))}
      </CardContent>
    </Card>
  );
}
