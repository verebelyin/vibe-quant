import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getGetDatabaseInfoApiSettingsDatabaseGetQueryKey,
  useGetDatabaseInfoApiSettingsDatabaseGet,
  useSwitchDatabaseApiSettingsDatabasePut,
} from "@/api/generated/settings/settings";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

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
      <div
        className="rounded-lg border p-4"
        style={{
          borderColor: "hsl(0 84% 60%)",
          backgroundColor: "hsl(0 84% 60% / 0.1)",
          color: "hsl(0 84% 60%)",
        }}
      >
        <p className="font-medium">Failed to load database info</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Current DB info */}
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
          Current Database
        </p>
        <div className="space-y-2">
          <div className="flex items-start justify-between gap-4">
            <span className="shrink-0 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
              Path
            </span>
            <span
              className="break-all text-right font-mono text-xs"
              style={{ color: "hsl(var(--foreground))" }}
            >
              {info?.path ?? "N/A"}
            </span>
          </div>
        </div>

        {info?.tables && info.tables.length > 0 && (
          <div className="mt-4">
            <p
              className="mb-2 text-xs font-medium"
              style={{ color: "hsl(var(--muted-foreground))" }}
            >
              Tables ({info.tables.length})
            </p>
            <div className="flex flex-wrap gap-1.5">
              {info.tables.map((t) => (
                <span
                  key={t}
                  className="rounded px-1.5 py-0.5 font-mono text-[10px]"
                  style={{
                    backgroundColor: "hsl(var(--muted))",
                    color: "hsl(var(--muted-foreground))",
                  }}
                >
                  {t}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Switch DB */}
      {!showSwitch ? (
        <button
          type="button"
          onClick={() => setShowSwitch(true)}
          className="rounded-md border px-3 py-1.5 text-xs font-medium transition-colors hover:brightness-90"
          style={{
            borderColor: "hsl(var(--border))",
            color: "hsl(var(--foreground))",
          }}
        >
          Switch Database
        </button>
      ) : (
        <div
          className="rounded-lg border p-4"
          style={{
            backgroundColor: "hsl(var(--card))",
            borderColor: "hsl(var(--border))",
          }}
        >
          <p
            className="mb-2 text-xs font-semibold uppercase tracking-wider"
            style={{ color: "hsl(var(--muted-foreground))" }}
          >
            Switch Database
          </p>
          <label className="block">
            <span
              className="mb-1 block text-xs font-medium"
              style={{ color: "hsl(var(--muted-foreground))" }}
            >
              New DB Path
            </span>
            <input
              type="text"
              value={newPath}
              onChange={(e) => setNewPath(e.target.value)}
              placeholder="/path/to/database.db"
              className="w-full rounded-md border px-3 py-1.5 text-sm outline-none transition-colors focus:ring-1"
              style={{
                backgroundColor: "hsl(var(--background))",
                borderColor: "hsl(var(--border))",
                color: "hsl(var(--foreground))",
              }}
            />
          </label>
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              disabled={switchMut.isPending || !newPath.trim()}
              onClick={() => {
                switchMut.mutate(
                  { data: { path: newPath.trim() } },
                  {
                    onSuccess: () => {
                      invalidate();
                      setShowSwitch(false);
                      setNewPath("");
                    },
                  },
                );
              }}
              className="rounded-md px-3 py-1.5 text-xs font-medium transition-colors hover:brightness-90 disabled:opacity-50"
              style={{
                backgroundColor: "hsl(var(--primary))",
                color: "hsl(var(--primary-foreground))",
              }}
            >
              {switchMut.isPending ? "Switching..." : "Switch"}
            </button>
            <button
              type="button"
              onClick={() => {
                setShowSwitch(false);
                setNewPath("");
              }}
              className="rounded-md border px-3 py-1.5 text-xs font-medium transition-colors hover:brightness-90"
              style={{
                borderColor: "hsl(var(--border))",
                color: "hsl(var(--foreground))",
              }}
            >
              Cancel
            </button>
          </div>
          {switchMut.isError && (
            <p className="mt-2 text-xs" style={{ color: "hsl(0 84% 60%)" }}>
              Failed to switch database. Check the path is valid.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
