import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import {
  getGetDatabaseInfoApiSettingsDatabaseGetQueryKey,
  useGetDatabaseInfoApiSettingsDatabaseGet,
  useSwitchDatabaseApiSettingsDatabasePut,
} from "@/api/generated/settings/settings";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";
import { Label } from "@/components/ui/label";

export function DatabaseTab() {
  const qc = useQueryClient();
  const query = useGetDatabaseInfoApiSettingsDatabaseGet();
  const switchMut = useSwitchDatabaseApiSettingsDatabasePut();

  const [newPath, setNewPath] = useState("");
  const [showSwitch, setShowSwitch] = useState(false);

  const info = query.data?.data;

  const invalidate = () =>
    qc.invalidateQueries({
      queryKey: getGetDatabaseInfoApiSettingsDatabaseGetQueryKey(),
    });

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
        </CardContent>
      </Card>

      {/* Switch DB */}
      {!showSwitch ? (
        <Button variant="outline" size="sm" onClick={() => setShowSwitch(true)}>
          Switch Database
        </Button>
      ) : (
        <Card className="py-4">
          <CardContent>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Switch Database
            </p>
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">New DB Path</Label>
              <Input
                type="text"
                value={newPath}
                onChange={(e) => setNewPath(e.target.value)}
                placeholder="/path/to/database.db"
              />
            </div>
            <div className="mt-3 flex gap-2">
              <Button
                size="sm"
                disabled={switchMut.isPending || !newPath.trim()}
                onClick={() => {
                  switchMut.mutate(
                    { data: { path: newPath.trim() } },
                    {
                      onSuccess: () => {
                        invalidate();
                        setShowSwitch(false);
                        setNewPath("");
                        toast.success("Database switched");
                      },
                    },
                  );
                }}
              >
                {switchMut.isPending ? "Switching..." : "Switch"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setShowSwitch(false);
                  setNewPath("");
                }}
              >
                Cancel
              </Button>
            </div>
            {switchMut.isError && (
              <p className="mt-2 text-xs text-destructive">
                Failed to switch database. Check the path is valid.
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
