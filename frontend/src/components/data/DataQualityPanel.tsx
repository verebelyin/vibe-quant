import type { DataQualityResponse } from "@/api/generated/models";
import { useDataQualityApiDataQualitySymbolGet } from "@/api/generated/data/data";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

interface DataQualityPanelProps {
  symbol: string;
}

interface GapItem {
  start: string;
  end: string;
  missing_bars?: number;
}

interface FillerRun {
  start: string;
  end: string;
  bars: number;
}

function qualityColorClass(score: number): string {
  if (score >= 95) return "text-green-500";
  if (score >= 80) return "text-yellow-500";
  return "text-destructive";
}

function qualityBorderClass(score: number): string {
  if (score >= 95) return "border-green-500";
  if (score >= 80) return "border-yellow-500";
  return "border-destructive";
}

function qualityLabel(score: number): string {
  if (score >= 95) return "Excellent";
  if (score >= 80) return "Fair";
  return "Poor";
}

function qualityBadgeVariant(score: number): "default" | "secondary" | "destructive" {
  if (score >= 95) return "default";
  if (score >= 80) return "secondary";
  return "destructive";
}

export function DataQualityPanel({ symbol }: DataQualityPanelProps) {
  const query = useDataQualityApiDataQualitySymbolGet(symbol, {
    query: { enabled: !!symbol },
  });

  const data = query.data?.data as DataQualityResponse | undefined;

  if (query.isLoading) {
    return <Skeleton className="h-32 rounded-lg" />;
  }

  if (query.isError) {
    return (
      <div className="rounded-lg border border-destructive bg-destructive/10 p-4 text-destructive">
        Failed to load quality data
      </div>
    );
  }

  if (!data) return null;
  if (data.error) {
    return (
      <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
        {data.error}
      </div>
    );
  }

  const score = (data.quality_score ?? 0) * 100;
  const gaps = (data.gaps ?? []) as unknown as GapItem[];
  // The list is capped server-side; the counts are not.
  const gapCount = data.gap_count ?? gaps.length;
  const missingBars = data.missing_bars ?? gaps.reduce((sum, g) => sum + (g.missing_bars ?? 0), 0);
  const fillerRuns = (data.zero_volume_runs ?? []) as unknown as FillerRun[];
  const fillerBars = data.zero_volume_bars ?? 0;
  const ohlcErrors = data.ohlc_errors ?? [];
  const ohlcErrorCount = data.ohlc_error_count ?? ohlcErrors.length;

  return (
    <div className="space-y-4">
      {/* Score + summary */}
      <div className="flex items-center gap-4">
        <div
          className={cn(
            "flex h-16 w-16 items-center justify-center rounded-full border-[3px] text-lg font-bold",
            qualityBorderClass(score),
            qualityColorClass(score),
          )}
        >
          {score.toFixed(0)}%
        </div>
        <div>
          <Badge variant={qualityBadgeVariant(score)}>{qualityLabel(score)}</Badge>
          <p className="mt-1 text-xs text-muted-foreground">
            {gapCount} gap{gapCount !== 1 ? "s" : ""}, {missingBars} missing bar
            {missingBars !== 1 ? "s" : ""}
          </p>
        </div>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-5 gap-3">
        {[
          { label: "Quality Score", value: `${score.toFixed(2)}%` },
          { label: "Gap Count", value: String(gapCount) },
          { label: "Missing Bars", value: String(missingBars) },
          { label: "Zero-Vol Filler", value: String(fillerBars) },
          { label: "OHLC Errors", value: String(ohlcErrorCount) },
        ].map((m) => (
          <Card key={m.label} className="gap-0 py-0">
            <CardContent className="p-3">
              <p className="text-xs uppercase tracking-wider text-muted-foreground">{m.label}</p>
              <p className="mt-0.5 text-lg font-bold text-foreground">{m.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Gaps list */}
      {gaps.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-foreground">Gaps</h4>
          <div className="max-h-48 overflow-y-auto rounded-lg border">
            <Table className="text-xs">
              <TableHeader>
                <TableRow className="bg-muted hover:bg-muted">
                  <TableHead className="px-3">Start</TableHead>
                  <TableHead className="px-3">End</TableHead>
                  <TableHead className="px-3 text-right">Missing</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {gaps.map((gap) => (
                  <TableRow key={`${gap.start}-${gap.end}`}>
                    <TableCell className="px-3 py-1.5 font-mono">{gap.start}</TableCell>
                    <TableCell className="px-3 py-1.5 font-mono">{gap.end}</TableCell>
                    <TableCell className="px-3 py-1.5 text-right font-mono">
                      {gap.missing_bars ?? "--"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      )}

      {/* Flat zero-volume filler candles (exchange outages filled by Binance Vision) */}
      {fillerRuns.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-foreground">
            Zero-volume filler runs
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              flat O=H=L=C bars with no volume — not real prices
            </span>
          </h4>
          <div className="max-h-48 overflow-y-auto rounded-lg border">
            <Table className="text-xs">
              <TableHeader>
                <TableRow className="bg-muted hover:bg-muted">
                  <TableHead className="px-3">Start</TableHead>
                  <TableHead className="px-3">End</TableHead>
                  <TableHead className="px-3 text-right">Bars</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {fillerRuns.map((run) => (
                  <TableRow key={`${run.start}-${run.end}`}>
                    <TableCell className="px-3 py-1.5 font-mono">{run.start}</TableCell>
                    <TableCell className="px-3 py-1.5 font-mono">{run.end}</TableCell>
                    <TableCell className="px-3 py-1.5 text-right font-mono">{run.bars}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      )}

      {/* OHLC errors */}
      {ohlcErrors.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-foreground">
            OHLC Errors
            {ohlcErrorCount > ohlcErrors.length && (
              <span className="ml-2 text-xs font-normal text-muted-foreground">
                (showing {ohlcErrors.length} of {ohlcErrorCount})
              </span>
            )}
          </h4>
          <div className="max-h-48 overflow-y-auto rounded-lg border border-destructive/30">
            <Table className="text-xs">
              <TableHeader>
                <TableRow className="bg-muted hover:bg-muted">
                  <TableHead className="px-3">Timestamp</TableHead>
                  <TableHead className="px-3">Error</TableHead>
                  <TableHead className="px-3">Values</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {ohlcErrors.map((err, i) => (
                  <TableRow key={i} className="text-destructive">
                    <TableCell className="px-3 py-1.5 font-mono">{err.timestamp}</TableCell>
                    <TableCell className="px-3 py-1.5">{err.error_type}</TableCell>
                    <TableCell className="px-3 py-1.5 font-mono text-xs">
                      {Object.entries(err.values)
                        .map(([k, v]) => `${k}=${v}`)
                        .join(", ")}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      )}
    </div>
  );
}
