import { useGetRunSummaryApiResultsRunsRunIdGet } from "@/api/generated/results/results";
import { LoadingSpinner } from "@/components/ui";

interface OverfittingBadgesProps {
  runId: number;
}

type BadgeStatus = "pass" | "fail" | "na";

interface BadgeInfo {
  label: string;
  status: BadgeStatus;
  detail?: string;
}

const statusStyles: Record<BadgeStatus, { bg: string; text: string; border: string }> = {
  pass: {
    bg: "hsl(142, 70%, 45%, 0.15)",
    text: "hsl(142, 70%, 45%)",
    border: "hsl(142, 70%, 45%, 0.3)",
  },
  fail: {
    bg: "hsl(0, 70%, 55%, 0.15)",
    text: "hsl(0, 70%, 55%)",
    border: "hsl(0, 70%, 55%, 0.3)",
  },
  na: {
    bg: "hsl(var(--muted))",
    text: "hsl(var(--muted-foreground))",
    border: "hsl(var(--border))",
  },
};

function Badge({ label, status, detail }: BadgeInfo) {
  const style = statusStyles[status];
  const statusLabel = status === "pass" ? "PASS" : status === "fail" ? "FAIL" : "N/A";

  return (
    <div
      className="flex items-center gap-2 rounded-md border px-3 py-2"
      style={{
        backgroundColor: style.bg,
        borderColor: style.border,
      }}
    >
      <span className="text-xs font-bold uppercase" style={{ color: style.text }}>
        {statusLabel}
      </span>
      <span className="text-xs font-medium" style={{ color: "hsl(var(--foreground))" }}>
        {label}
      </span>
      {detail && (
        <span className="ml-auto text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
          {detail}
        </span>
      )}
    </div>
  );
}

export function OverfittingBadges({ runId }: OverfittingBadgesProps) {
  const query = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const data = query.data?.data;

  if (query.isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <LoadingSpinner size="sm" />
      </div>
    );
  }

  if (query.isError || !data) {
    return (
      <p className="py-4 text-sm" style={{ color: "hsl(var(--destructive))" }}>
        Failed to load overfitting data.
      </p>
    );
  }

  function sharpeStatus(value: number | null): BadgeStatus {
    if (value == null) return "na";
    return value > 0 ? "pass" : "fail";
  }

  function efficiencyStatus(value: number | null): BadgeStatus {
    if (value == null) return "na";
    return value >= 0.5 ? "pass" : "fail";
  }

  const badges: BadgeInfo[] = [
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
      label: "Deflated Sharpe",
      status: sharpeStatus(data.deflated_sharpe),
      detail: data.deflated_sharpe != null ? `DSR: ${data.deflated_sharpe.toFixed(2)}` : undefined,
    },
  ];

  const passCount = badges.filter((b) => b.status === "pass").length;
  const failCount = badges.filter((b) => b.status === "fail").length;
  const naCount = badges.filter((b) => b.status === "na").length;

  const allNa = naCount === badges.length;
  const overallPass = failCount === 0 && !allNa;

  return (
    <div
      className="rounded-lg border p-4"
      style={{
        backgroundColor: "hsl(var(--card))",
        borderColor: "hsl(var(--border))",
      }}
    >
      <div className="mb-3 flex items-center justify-between">
        <h3
          className="text-sm font-semibold uppercase tracking-wide"
          style={{ color: "hsl(var(--muted-foreground))" }}
        >
          Overfitting Filters
        </h3>
        <span
          className="rounded-full px-2 py-0.5 text-xs font-bold uppercase"
          style={{
            backgroundColor: allNa
              ? statusStyles.na.bg
              : overallPass
                ? statusStyles.pass.bg
                : statusStyles.fail.bg,
            color: allNa
              ? statusStyles.na.text
              : overallPass
                ? statusStyles.pass.text
                : statusStyles.fail.text,
          }}
        >
          {allNa
            ? "Not Run"
            : overallPass
              ? `All Passed (${passCount}/${badges.length})`
              : `${failCount} Failed`}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        {badges.map((badge) => (
          <Badge key={badge.label} {...badge} />
        ))}
      </div>
    </div>
  );
}
