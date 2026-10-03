import { useGetDatabaseInfoApiSettingsDatabaseGet } from "@/api/generated/settings/settings";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

export function DatabaseTab() {
  const query = useGetDatabaseInfoApiSettingsDatabaseGet();
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
        <p className="font-medium">Failed to load database info</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Current DB info */}
      <Card className="py-4">
        <CardContent>
          <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Current Database
          </p>
          <div className="space-y-2">
            <div className="flex items-start justify-between gap-4">
              <span className="shrink-0 text-xs text-muted-foreground">Path</span>
              <span className="break-all text-right font-mono text-xs text-foreground">
                {info?.path ?? "N/A"}
              </span>
            </div>
          </div>

          {info?.tables && info.tables.length > 0 && (
            <div className="mt-4">
              <p className="mb-2 text-xs font-medium text-muted-foreground">
                Tables ({info.tables.length})
              </p>
              <div className="flex flex-wrap gap-1.5">
                {info.tables.map((t) => (
                  <Badge key={t} variant="secondary" className="font-mono text-[10px]">
                    {t}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* No runtime switch: it only re-pointed the API, while running jobs and
              their results kept using the old database. */}
          <p className="mt-4 text-xs text-muted-foreground">
            To use a different database, restart the backend with{" "}
            <code className="font-mono">VIBE_QUANT_DB=/path/to/file.db</code>. Every job started by
            the backend then uses that database.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
