import { useState } from "react";
import {
  useIngestPreviewApiDataIngestPreviewPost,
  useListSymbolsApiDataSymbolsGet,
  useRebuildCatalogApiDataRebuildPost,
  useStartIngestApiDataIngestPost,
  useStartUpdateApiDataUpdatePost,
} from "@/api/generated/data/data";
import type { IngestPreviewResponse } from "@/api/generated/models";
import { DateRangePicker } from "@/components/ui/DateRangePicker";

const INTERVALS = ["1m", "5m", "15m", "1h", "4h"] as const;

interface IngestFormProps {
  onIngestStarted: (jobId: string) => void;
}

export function IngestForm({ onIngestStarted }: IngestFormProps) {
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [interval, setInterval] = useState<string>("1m");
  const [preview, setPreview] = useState<IngestPreviewResponse | null>(null);

  const symbolsQuery = useListSymbolsApiDataSymbolsGet();
  const symbols = symbolsQuery.data?.data ?? [];

  const previewMutation = useIngestPreviewApiDataIngestPreviewPost();
  const ingestMutation = useStartIngestApiDataIngestPost();
  const updateMutation = useStartUpdateApiDataUpdatePost();
  const rebuildMutation = useRebuildCatalogApiDataRebuildPost();

  const canPreview = selectedSymbols.length > 0 && startDate && endDate;

  function handleToggleSymbol(sym: string) {
    setSelectedSymbols((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym],
    );
    setPreview(null);
  }

  function handleSelectAll() {
    if (selectedSymbols.length === symbols.length) {
      setSelectedSymbols([]);
    } else {
      setSelectedSymbols([...symbols]);
    }
    setPreview(null);
  }

  function handlePreview() {
    if (!canPreview) return;
    previewMutation.mutate(
      { data: { symbols: selectedSymbols, start_date: startDate, end_date: endDate } },
      {
        onSuccess: (res) => {
          if (res.status === 200) {
            setPreview(res.data);
          }
        },
      },
    );
  }

  function handleStartDownload() {
    if (!canPreview) return;
    ingestMutation.mutate(
      { data: { symbols: selectedSymbols, start_date: startDate, end_date: endDate } },
      {
        onSuccess: (res) => {
          const data = res.data as Record<string, unknown>;
          const jobId = String(data?.job_id ?? data?.task_id ?? "");
          if (jobId) onIngestStarted(jobId);
        },
      },
    );
  }

  function handleUpdateAll() {
    updateMutation.mutate(undefined, {
      onSuccess: (res) => {
        const data = res.data as Record<string, unknown>;
        const jobId = String(data?.job_id ?? data?.task_id ?? "");
        if (jobId) onIngestStarted(jobId);
      },
    });
  }

  function handleRebuild() {
    rebuildMutation.mutate(undefined, {
      onSuccess: (res) => {
        const data = res.data as Record<string, unknown>;
        const jobId = String(data?.job_id ?? data?.task_id ?? "");
        if (jobId) onIngestStarted(jobId);
      },
    });
  }

  const inputStyle = {
    backgroundColor: "hsl(var(--input))",
    borderColor: "hsl(var(--border))",
    color: "hsl(var(--foreground))",
  };

  const btnPrimary = {
    backgroundColor: "hsl(var(--primary))",
    color: "hsl(var(--primary-foreground))",
  };

  const btnSecondary = {
    backgroundColor: "hsl(var(--secondary))",
    color: "hsl(var(--secondary-foreground))",
  };

  const isAnyMutating =
    previewMutation.isPending ||
    ingestMutation.isPending ||
    updateMutation.isPending ||
    rebuildMutation.isPending;

  return (
    <div
      className="rounded-lg border p-5"
      style={{
        backgroundColor: "hsl(var(--card))",
        borderColor: "hsl(var(--border))",
      }}
    >
      <h2 className="mb-4 text-lg font-semibold" style={{ color: "hsl(var(--foreground))" }}>
        Data Ingest
      </h2>

      <div className="space-y-4">
        {/* Symbol multi-select */}
        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <label
              className="text-xs font-medium uppercase tracking-wider"
              style={{ color: "hsl(var(--muted-foreground))" }}
            >
              Symbols
            </label>
            <button
              type="button"
              onClick={handleSelectAll}
              className="text-xs font-medium hover:underline"
              style={{ color: "hsl(var(--primary))" }}
            >
              {selectedSymbols.length === symbols.length ? "Deselect all" : "Select all"}
            </button>
          </div>
          <div
            className="flex max-h-32 flex-wrap gap-1.5 overflow-y-auto rounded-md border p-2"
            style={{ borderColor: "hsl(var(--border))", backgroundColor: "hsl(var(--input))" }}
          >
            {symbols.length === 0 && (
              <span className="text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
                No symbols available
              </span>
            )}
            {symbols.map((sym) => {
              const selected = selectedSymbols.includes(sym);
              return (
                <button
                  key={sym}
                  type="button"
                  onClick={() => handleToggleSymbol(sym)}
                  className="rounded-md px-2 py-0.5 font-mono text-xs font-medium transition-colors"
                  style={
                    selected
                      ? {
                          backgroundColor: "hsl(var(--primary))",
                          color: "hsl(var(--primary-foreground))",
                        }
                      : {
                          backgroundColor: "hsl(var(--accent))",
                          color: "hsl(var(--accent-foreground))",
                        }
                  }
                >
                  {sym}
                </button>
              );
            })}
          </div>
          {selectedSymbols.length > 0 && (
            <p className="mt-1 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
              {selectedSymbols.length} selected
            </p>
          )}
        </div>

        {/* Date range + interval */}
        <div className="flex flex-wrap items-end gap-4">
          <DateRangePicker
            startDate={startDate}
            endDate={endDate}
            onStartChange={(d) => {
              setStartDate(d);
              setPreview(null);
            }}
            onEndChange={(d) => {
              setEndDate(d);
              setPreview(null);
            }}
          />
          <div className="flex flex-col gap-1">
            <label
              htmlFor="ingest-interval"
              className="text-xs font-medium"
              style={{ color: "hsl(var(--muted-foreground))" }}
            >
              Interval
            </label>
            <select
              id="ingest-interval"
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
              className="rounded-md border px-2.5 py-1.5 text-sm outline-none focus:ring-2"
              style={
                { ...inputStyle, "--tw-ring-color": "hsl(var(--ring))" } as React.CSSProperties
              }
            >
              {INTERVALS.map((iv) => (
                <option key={iv} value={iv}>
                  {iv}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Preview result */}
        {preview && (
          <div
            className="rounded-md border p-3"
            style={{
              borderColor: "hsl(var(--border))",
              backgroundColor: "hsl(var(--muted) / 0.3)",
            }}
          >
            <p className="text-sm font-medium" style={{ color: "hsl(var(--foreground))" }}>
              Download Preview
            </p>
            <div className="mt-2 grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-xl font-bold" style={{ color: "hsl(var(--foreground))" }}>
                  {preview.total_months}
                </p>
                <p className="text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
                  Total months
                </p>
              </div>
              <div>
                <p className="text-xl font-bold" style={{ color: "hsl(142 71% 45%)" }}>
                  {preview.archived_months}
                </p>
                <p className="text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
                  Already archived
                </p>
              </div>
              <div>
                <p className="text-xl font-bold" style={{ color: "hsl(48 96% 53%)" }}>
                  {preview.new_months}
                </p>
                <p className="text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
                  New to download
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Error display */}
        {(previewMutation.isError || ingestMutation.isError) && (
          <div
            className="rounded-md border p-3 text-sm"
            style={{
              borderColor: "hsl(0 84% 60%)",
              backgroundColor: "hsl(0 84% 60% / 0.1)",
              color: "hsl(0 84% 60%)",
            }}
          >
            {previewMutation.error instanceof Error
              ? previewMutation.error.message
              : ingestMutation.error instanceof Error
                ? ingestMutation.error.message
                : "Request failed"}
          </div>
        )}

        {/* Action buttons */}
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={handlePreview}
            disabled={!canPreview || isAnyMutating}
            className="rounded-md px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
            style={btnSecondary}
          >
            {previewMutation.isPending ? "Loading..." : "Preview"}
          </button>
          <button
            type="button"
            onClick={handleStartDownload}
            disabled={!canPreview || isAnyMutating}
            className="rounded-md px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
            style={btnPrimary}
          >
            {ingestMutation.isPending ? "Starting..." : "Start Download"}
          </button>
          <div
            className="mx-2 self-stretch"
            style={{ borderLeft: "1px solid hsl(var(--border))" }}
          />
          <button
            type="button"
            onClick={handleUpdateAll}
            disabled={isAnyMutating}
            className="rounded-md px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
            style={btnSecondary}
          >
            {updateMutation.isPending ? "Starting..." : "Update All"}
          </button>
          <button
            type="button"
            onClick={handleRebuild}
            disabled={isAnyMutating}
            className="rounded-md px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
            style={btnSecondary}
          >
            {rebuildMutation.isPending ? "Rebuilding..." : "Rebuild Catalog"}
          </button>
        </div>

        {/* Success messages for update/rebuild */}
        {updateMutation.isSuccess && (
          <p className="text-sm" style={{ color: "hsl(142 71% 45%)" }}>
            Update started successfully.
          </p>
        )}
        {rebuildMutation.isSuccess && (
          <p className="text-sm" style={{ color: "hsl(142 71% 45%)" }}>
            Catalog rebuild started successfully.
          </p>
        )}
      </div>
    </div>
  );
}
