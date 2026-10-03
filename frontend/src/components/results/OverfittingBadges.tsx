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

export interface CheckInfo {
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

// Mirrors the backend gates (keep in sync):
// - DSR: deflated_sharpe is a z-score; significant when p = sf(z) < 0.05 (overfitting/dsr.py)
// - Purged k-fold: mean OOS Sharpe > 0.5 (and std < 1.0, not exposed) (overfitting/purged_kfold.py)
// - Walk-forward: discovery verdict wfa.passed (consistency >= 0.75); else WFA efficiency > 0.5
// - Bootstrap CI: lower >= floor; floor = the run's bootstrap_min_sharpe, else the
//   discovery timeframe default (discovery/__main__.py _BOOTSTRAP_FLOOR_BY_TIMEFRAME)
// - Cross-window: discovery verdict cross_window.passed
const DSR_Z_95 = 1.6448536269514722;
const PKFOLD_MIN_MEAN_SHARPE = 0.5;
const WFA_MIN_EFFICIENCY = 0.5;
const BOOTSTRAP_FLOOR_BY_TIMEFRAME: Record<string, number> = { "1m": 0.5, "4h": 0.0, "1d": 0.0 };
const BOOTSTRAP_FLOOR_DEFAULT = 1.0;

export function bootstrapFloor(
  data: BacktestResultResponse,
  timeframe: string | undefined,
): number {
  if (data.bootstrap_min_sharpe != null) return data.bootstrap_min_sharpe;
  const byTimeframe = timeframe ? BOOTSTRAP_FLOOR_BY_TIMEFRAME[timeframe] : undefined;
  return byTimeframe ?? BOOTSTRAP_FLOOR_DEFAULT;
}

function verdict(passed: boolean | null | undefined): CheckStatus {
  return passed == null ? "na" : passed ? "pass" : "fail";
}

export function overfittingChecks(
  data: BacktestResultResponse,
  timeframe: string | undefined,
): CheckInfo[] {
  const wfaStatus: CheckStatus =
    data.wfa_passed != null
      ? verdict(data.wfa_passed)
      : verdict(
          data.walk_forward_efficiency != null
            ? data.walk_forward_efficiency > WFA_MIN_EFFICIENCY
            : null,
        );
  const wfaDetail = [
    data.wfa_consistency != null ? `consistency ${(data.wfa_consistency * 100).toFixed(0)}%` : null,
    data.wfa_sharpe_consistency != null
      ? `sharpe>0 ${(data.wfa_sharpe_consistency * 100).toFixed(0)}%`
      : null,
    data.walk_forward_efficiency != null
      ? `efficiency ${data.walk_forward_efficiency.toFixed(2)}`
      : null,
  ]
    .filter(Boolean)
    .join(", ");
  const floor = bootstrapFloor(data, timeframe);

  const regimeRows = data.cross_regime_results as Array<{ passed?: boolean }> | null | undefined;
  const crossRegime: CheckInfo =
    !regimeRows || regimeRows.length === 0
      ? { label: "Cross-Regime", status: "na" }
      : {
          label: "Cross-Regime",
          status: verdict(regimeRows.every((r) => r?.passed === true)),
          detail: `${regimeRows.filter((r) => r?.passed).length}/${regimeRows.length} regimes`,
        };

  return [
    { label: "Walk-Forward", status: wfaStatus, detail: wfaDetail || undefined },
    {
      label: "Purged K-Fold",
      status: verdict(
        data.purged_kfold_mean_sharpe != null
          ? data.purged_kfold_mean_sharpe > PKFOLD_MIN_MEAN_SHARPE
          : null,
      ),
      detail:
        data.purged_kfold_mean_sharpe != null
          ? `mean OOS sharpe: ${data.purged_kfold_mean_sharpe.toFixed(2)} (thr: >0.5)`
          : undefined,
    },
    {
      label: "Deflated Sharpe Ratio",
      status: verdict(data.deflated_sharpe != null ? data.deflated_sharpe > DSR_Z_95 : null),
      detail:
        data.deflated_sharpe != null
          ? `z: ${data.deflated_sharpe.toFixed(2)} (thr: >1.645, p<0.05)`
          : undefined,
    },
    {
      label: "Bootstrap CI (Sharpe lower)",
      status: verdict(
        data.bootstrap_sharpe_lower != null ? data.bootstrap_sharpe_lower >= floor : null,
      ),
      detail:
        data.bootstrap_sharpe_lower != null
          ? `≥${data.bootstrap_sharpe_lower.toFixed(2)} @ ${(data.bootstrap_ci_level ?? 0.95) * 100}% (thr: ${floor})`
          : undefined,
    },
    { label: "Cross-Window", status: verdict(data.cross_window_passed) },
    crossRegime,
  ];
}

/** "Not Run" when no check produced a verdict; never "All Passed" on zero tests. */
export function overallVerdict(checks: CheckInfo[]): "not_run" | "pass" | "fail" {
  const ran = checks.filter((c) => c.status !== "na");
  if (ran.length === 0) return "not_run";
  return ran.some((c) => c.status === "fail") ? "fail" : "pass";
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

  const checks = overfittingChecks(data, timeframe);

  const passCount = checks.filter((b) => b.status === "pass").length;
  const failCount = checks.filter((b) => b.status === "fail").length;

  const overall = overallVerdict(checks);
  const allNa = overall === "not_run";
  const overallPass = overall === "pass";

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
                ? `All Passed (${passCount} run)`
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
