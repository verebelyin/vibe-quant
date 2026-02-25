import {
  useDataCoverageApiDataCoverageGet,
  useDataStatusApiDataStatusGet,
  useListSymbolsApiDataSymbolsGet,
} from "@/api/generated/data/data";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { CoverageTable } from "./CoverageTable";

interface MetricCardProps {
  label: string;
  value: string;
  subtitle?: string | undefined;
}

function MetricCard({ label, value, subtitle }: MetricCardProps) {
  return (
    <Card className="gap-0 py-0">
      <CardContent className="p-5">
        <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </p>
        <p className="mt-1 text-2xl font-bold text-foreground">{value}</p>
        {subtitle && <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>}
      </CardContent>
    </Card>
  );
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const val = bytes / 1024 ** i;
  return `${val.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6 p-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {(["a", "b", "c", "d"] as const).map((id) => (
          <Skeleton key={id} className="h-24 rounded-lg" />
        ))}
      </div>
      <Skeleton className="h-64 rounded-lg" />
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="mx-6 rounded-lg border border-destructive bg-destructive/10 p-6 text-destructive">
      <p className="font-medium">Failed to load data status</p>
      <p className="mt-1 text-sm opacity-80">{message}</p>
    </div>
  );
}

export function DataStatusDashboard() {
  const statusQuery = useDataStatusApiDataStatusGet();
  const coverageQuery = useDataCoverageApiDataCoverageGet();
  const symbolsQuery = useListSymbolsApiDataSymbolsGet();

  const isLoading = statusQuery.isLoading || coverageQuery.isLoading || symbolsQuery.isLoading;
  const hasError = statusQuery.isError || coverageQuery.isError;

  if (isLoading) {
    return <LoadingSkeleton />;
  }

  if (hasError) {
    const errMsg = statusQuery.error instanceof Error ? statusQuery.error.message : "Unknown error";
    return (
      <div className="p-6">
        <h1 className="text-2xl font-bold text-foreground">Data Management</h1>
        <div className="mt-4">
          <ErrorState message={errMsg} />
        </div>
      </div>
    );
  }

  const status = statusQuery.data?.data;
  const coverage = coverageQuery.data?.data?.coverage ?? [];
  const symbols = symbolsQuery.data?.data ?? [];

  const totalBars = coverage.reduce((sum, c) => sum + c.bar_count, 0);
  const totalKlines = coverage.reduce((sum, c) => sum + c.kline_count, 0);
  const totalFunding = coverage.reduce((sum, c) => sum + c.funding_rate_count, 0);
  const totalRows = totalBars + totalKlines + totalFunding;

  const latestEnd = coverage.reduce((latest, c) => {
    if (!c.end_date) return latest;
    return c.end_date > latest ? c.end_date : latest;
  }, "");

  const lastUpdate = latestEnd
    ? new Date(latestEnd).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : "--";

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold text-foreground">Data Management</h1>

      {/* Status overview metrics */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Total Symbols"
          value={String(symbols.length)}
          subtitle={`${coverage.length} with coverage data`}
        />
        <MetricCard
          label="Total Data Points"
          value={totalRows.toLocaleString()}
          subtitle={`${totalKlines.toLocaleString()} klines, ${totalBars.toLocaleString()} bars`}
        />
        <MetricCard
          label="Database Size"
          value={status ? formatBytes(status.total_size_bytes) : "--"}
          subtitle={
            status
              ? `Archive: ${formatBytes(status.archive_size_bytes)}, Catalog: ${formatBytes(status.catalog_size_bytes)}`
              : undefined
          }
        />
        <MetricCard
          label="Latest Data"
          value={lastUpdate}
          subtitle={totalFunding > 0 ? `${totalFunding.toLocaleString()} funding rates` : undefined}
        />
      </div>

      {/* Coverage table */}
      <div>
        <h2 className="mb-3 text-lg font-semibold text-foreground">Symbol Coverage</h2>
        <CoverageTable coverage={coverage} />
      </div>
    </div>
  );
}
