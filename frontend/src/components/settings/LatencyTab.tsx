import { useListLatencyPresetsApiSettingsLatencyPresetsGet } from "@/api/generated/settings/settings";
import { EmptyState } from "@/components/ui/EmptyState";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

export function LatencyTab() {
  const query = useListLatencyPresetsApiSettingsLatencyPresetsGet();
  const presets = query.data?.data ?? [];

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
        <p className="font-medium">Failed to load latency presets</p>
      </div>
    );
  }

  if (presets.length === 0) {
    return (
      <EmptyState title="No latency presets" description="No presets configured on the backend." />
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm" style={{ color: "hsl(var(--muted-foreground))" }}>
        Latency presets are read-only and configured on the backend.
      </p>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {presets.map((preset) => (
          <div
            key={preset.name}
            className="rounded-lg border p-5"
            style={{
              backgroundColor: "hsl(var(--card))",
              borderColor: "hsl(var(--border))",
            }}
          >
            <p className="text-sm font-semibold" style={{ color: "hsl(var(--foreground))" }}>
              {preset.name}
            </p>
            <p className="mt-1 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
              {preset.description}
            </p>
            <div className="mt-3 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
                  Base Latency
                </span>
                <span
                  className="text-xs font-mono font-medium"
                  style={{ color: "hsl(var(--foreground))" }}
                >
                  {preset.base_latency_ms} ms
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
