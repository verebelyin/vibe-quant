import { useGetSystemInfoApiSettingsSystemInfoGet } from "@/api/generated/settings/settings";
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
      <span className="text-xs font-medium" style={{ color: "hsl(var(--muted-foreground))" }}>
        {label}
      </span>
      <span className="font-mono text-xs" style={{ color: "hsl(var(--foreground))" }}>
        {value}
      </span>
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
      <div
        className="rounded-lg border p-4"
        style={{
          borderColor: "hsl(0 84% 60%)",
          backgroundColor: "hsl(0 84% 60% / 0.1)",
          color: "hsl(0 84% 60%)",
        }}
      >
        <p className="font-medium">Failed to load system info</p>
      </div>
    );
  }

  if (!info) return null;

  const tableCounts = info.table_counts as Record<string, number>;

  return (
    <div className="space-y-6">
      {/* Runtime info */}
      <div
        className="rounded-lg border p-5"
        style={{
          backgroundColor: "hsl(var(--card))",
          borderColor: "hsl(var(--border))",
        }}
      >
        <p
          className="mb-3 text-xs font-semibold uppercase tracking-wider"
          style={{ color: "hsl(var(--muted-foreground))" }}
        >
          Runtime
        </p>
        <div className="divide-y" style={{ borderColor: "hsl(var(--border))" }}>
          <InfoRow label="Python Version" value={info.python_version} />
          <InfoRow label="NautilusTrader Version" value={info.nt_version} />
        </div>
      </div>

      {/* Storage */}
      <div
        className="rounded-lg border p-5"
        style={{
          backgroundColor: "hsl(var(--card))",
          borderColor: "hsl(var(--border))",
        }}
      >
        <p
          className="mb-3 text-xs font-semibold uppercase tracking-wider"
          style={{ color: "hsl(var(--muted-foreground))" }}
        >
          Storage
        </p>
        <div className="divide-y" style={{ borderColor: "hsl(var(--border))" }}>
          <InfoRow label="Database Size" value={formatBytes(info.db_size_bytes)} />
          <InfoRow label="Catalog Size" value={formatBytes(info.catalog_size_bytes)} />
        </div>
      </div>

      {/* Table counts */}
      {Object.keys(tableCounts).length > 0 && (
        <div
          className="rounded-lg border p-5"
          style={{
            backgroundColor: "hsl(var(--card))",
            borderColor: "hsl(var(--border))",
          }}
        >
          <p
            className="mb-3 text-xs font-semibold uppercase tracking-wider"
            style={{ color: "hsl(var(--muted-foreground))" }}
          >
            Table Row Counts
          </p>
          <div className="divide-y" style={{ borderColor: "hsl(var(--border))" }}>
            {Object.entries(tableCounts)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([table, count]) => (
                <InfoRow key={table} label={table} value={count.toLocaleString()} />
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
