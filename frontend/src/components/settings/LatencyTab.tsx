import { useListLatencyPresetsApiSettingsLatencyPresetsGet } from "@/api/generated/settings/settings";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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
      <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-destructive">
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
      <p className="text-sm text-muted-foreground">
        Latency presets are read-only and configured on the backend.
      </p>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {presets.map((preset) => (
          <Card key={preset.name} className="py-4">
            <CardHeader className="pb-0">
              <CardTitle className="text-sm">{preset.name}</CardTitle>
              <CardDescription className="text-xs">{preset.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Base Latency</span>
                <Badge variant="secondary" className="font-mono">
                  {preset.base_latency_ms} ms
                </Badge>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
