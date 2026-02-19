import { useGetSystemInfoApiSettingsSystemInfoGet } from "@/api/generated/settings/settings";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const val = bytes / 1024 ** i;
  return `${val.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      <Badge variant="secondary" className="font-mono">
        {value}
      </Badge>
    </div>
  );
}

export function SystemTab() {
  const query = useGetSystemInfoApiSettingsSystemInfoGet();
  const info = query.data?.data;

  if (query.isLoading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (query.isError) {
    return (
      <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-destructive">
        <p className="font-medium">Failed to load system info</p>
      </div>
    );
  }

  if (!info) return null;

  const tableCounts = info.table_counts as Record<string, number>;

  return (
    <div className="space-y-6">
      {/* Runtime info */}
      <Card className="py-4">
        <CardContent>
          <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Runtime
          </p>
          <div className="divide-y divide-border">
            <InfoRow label="Python Version" value={info.python_version} />
            <InfoRow label="NautilusTrader Version" value={info.nt_version} />
          </div>
        </CardContent>
      </Card>

      {/* Storage */}
      <Card className="py-4">
        <CardContent>
          <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Storage
          </p>
          <div className="divide-y divide-border">
            <InfoRow label="Database Size" value={formatBytes(info.db_size_bytes)} />
            <InfoRow label="Catalog Size" value={formatBytes(info.catalog_size_bytes)} />
          </div>
        </CardContent>
      </Card>

      {/* Table counts */}
      {Object.keys(tableCounts).length > 0 && (
        <Card className="py-4">
          <CardContent>
            <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Table Row Counts
            </p>
            <div className="divide-y divide-border">
              {Object.entries(tableCounts)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([table, count]) => (
                  <InfoRow key={table} label={table} value={count.toLocaleString()} />
                ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
