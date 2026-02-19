import { useDataQualityApiDataQualitySymbolGet } from "@/api/generated/data/data";

interface DataQualityPanelProps {
  symbol: string;
}

interface GapItem {
  start: string;
  end: string;
  missing_bars?: number;
}

function qualityColor(score: number): string {
  if (score >= 95) return "hsl(142 71% 45%)";
  if (score >= 80) return "hsl(48 96% 53%)";
  return "hsl(0 84% 60%)";
}

function qualityLabel(score: number): string {
  if (score >= 95) return "Excellent";
  if (score >= 80) return "Fair";
  return "Poor";
}

export function DataQualityPanel({ symbol }: DataQualityPanelProps) {
  const query = useDataQualityApiDataQualitySymbolGet(symbol, {
    query: { enabled: !!symbol },
  });

  const data = query.data?.data;

  if (query.isLoading) {
    return (
      <div
        className="h-32 animate-pulse rounded-lg"
        style={{ backgroundColor: "hsl(var(--muted))" }}
      />
    );
  }

  if (query.isError) {
    return (
      <div
        className="rounded-lg border p-4"
        style={{
          borderColor: "hsl(0 84% 60%)",
          backgroundColor: "hsl(0 84% 60% / 0.1)",
          color: "hsl(0 84% 60%)",
        }}
      >
        Failed to load quality data
      </div>
    );
  }

  if (!data) return null;

  const score = data.quality_score * 100;
  const gaps = (data.gaps ?? []) as GapItem[];
  const gapCount = gaps.length;
  const missingBars = gaps.reduce((sum, g) => sum + (g.missing_bars ?? 0), 0);

  return (
    <div className="space-y-4">
      {/* Score + summary */}
      <div className="flex items-center gap-4">
        <div
          className="flex h-16 w-16 items-center justify-center rounded-full text-lg font-bold"
          style={{
            border: `3px solid ${qualityColor(score)}`,
            color: qualityColor(score),
          }}
        >
          {score.toFixed(0)}%
        </div>
        <div>
          <p className="text-sm font-semibold" style={{ color: qualityColor(score) }}>
            {qualityLabel(score)}
          </p>
          <p className="text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
            {gapCount} gap{gapCount !== 1 ? "s" : ""}, {missingBars} missing bar
            {missingBars !== 1 ? "s" : ""}
          </p>
        </div>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "Quality Score", value: `${score.toFixed(1)}%` },
          { label: "Gap Count", value: String(gapCount) },
          { label: "Missing Bars", value: String(missingBars) },
        ].map((m) => (
          <div
            key={m.label}
            className="rounded-lg border p-3"
            style={{
              backgroundColor: "hsl(var(--card))",
              borderColor: "hsl(var(--border))",
            }}
          >
            <p
              className="text-xs uppercase tracking-wider"
              style={{ color: "hsl(var(--muted-foreground))" }}
            >
              {m.label}
            </p>
            <p className="mt-0.5 text-lg font-bold" style={{ color: "hsl(var(--foreground))" }}>
              {m.value}
            </p>
          </div>
        ))}
      </div>

      {/* Gaps list */}
      {gaps.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold" style={{ color: "hsl(var(--foreground))" }}>
            Gaps
          </h4>
          <div
            className="max-h-48 overflow-y-auto rounded-lg border"
            style={{ borderColor: "hsl(var(--border))" }}
          >
            <table className="w-full text-xs">
              <thead>
                <tr style={{ backgroundColor: "hsl(var(--muted))" }}>
                  <th
                    className="px-3 py-1.5 text-left font-medium"
                    style={{ color: "hsl(var(--muted-foreground))" }}
                  >
                    Start
                  </th>
                  <th
                    className="px-3 py-1.5 text-left font-medium"
                    style={{ color: "hsl(var(--muted-foreground))" }}
                  >
                    End
                  </th>
                  <th
                    className="px-3 py-1.5 text-right font-medium"
                    style={{ color: "hsl(var(--muted-foreground))" }}
                  >
                    Missing
                  </th>
                </tr>
              </thead>
              <tbody>
                {gaps.map((gap, i) => (
                  <tr
                    key={`${gap.start}-${gap.end}`}
                    style={{
                      backgroundColor: i % 2 === 0 ? "hsl(var(--card))" : "hsl(var(--muted) / 0.3)",
                      borderTop: i > 0 ? "1px solid hsl(var(--border))" : undefined,
                    }}
                  >
                    <td
                      className="px-3 py-1.5 font-mono"
                      style={{ color: "hsl(var(--foreground))" }}
                    >
                      {gap.start}
                    </td>
                    <td
                      className="px-3 py-1.5 font-mono"
                      style={{ color: "hsl(var(--foreground))" }}
                    >
                      {gap.end}
                    </td>
                    <td
                      className="px-3 py-1.5 text-right font-mono"
                      style={{ color: "hsl(var(--foreground))" }}
                    >
                      {gap.missing_bars ?? "--"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
