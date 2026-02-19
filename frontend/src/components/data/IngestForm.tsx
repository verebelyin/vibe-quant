import { useState } from "react";
import { toast } from "sonner";
import {
  useIngestPreviewApiDataIngestPreviewPost,
  useListSymbolsApiDataSymbolsGet,
  useRebuildCatalogApiDataRebuildPost,
  useStartIngestApiDataIngestPost,
  useStartUpdateApiDataUpdatePost,
} from "@/api/generated/data/data";
import type { IngestPreviewResponse } from "@/api/generated/models";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DateRangePicker } from "@/components/ui/DateRangePicker";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";

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
        onError: () => {
          toast.error("Failed to load preview");
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
          toast.success("Download started");
        },
        onError: () => {
          toast.error("Failed to start download");
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
        toast.success("Update started");
      },
      onError: () => {
        toast.error("Failed to start update");
      },
    });
  }

  function handleRebuild() {
    rebuildMutation.mutate(undefined, {
      onSuccess: (res) => {
        const data = res.data as Record<string, unknown>;
        const jobId = String(data?.job_id ?? data?.task_id ?? "");
        if (jobId) onIngestStarted(jobId);
        toast.success("Catalog rebuild started");
      },
      onError: () => {
        toast.error("Failed to start catalog rebuild");
      },
    });
  }

  const isAnyMutating =
    previewMutation.isPending ||
    ingestMutation.isPending ||
    updateMutation.isPending ||
    rebuildMutation.isPending;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Data Ingest</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Symbol multi-select */}
        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <Label className="text-xs uppercase tracking-wider">Symbols</Label>
            <Button
              type="button"
              variant="link"
              size="sm"
              className="h-auto p-0 text-xs"
              onClick={handleSelectAll}
            >
              {selectedSymbols.length === symbols.length ? "Deselect all" : "Select all"}
            </Button>
          </div>
          <div className="flex max-h-32 flex-wrap gap-1.5 overflow-y-auto rounded-md border border-input bg-transparent p-2 dark:bg-input/30">
            {symbols.length === 0 && (
              <span className="text-xs text-muted-foreground">No symbols available</span>
            )}
            {symbols.map((sym) => {
              const selected = selectedSymbols.includes(sym);
              return (
                <Button
                  key={sym}
                  type="button"
                  variant={selected ? "default" : "secondary"}
                  size="xs"
                  onClick={() => handleToggleSymbol(sym)}
                  className="font-mono"
                >
                  {sym}
                </Button>
              );
            })}
          </div>
          {selectedSymbols.length > 0 && (
            <p className="mt-1 text-xs text-muted-foreground">{selectedSymbols.length} selected</p>
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
            <Label className="text-xs text-muted-foreground">Interval</Label>
            <Select value={interval} onValueChange={setInterval}>
              <SelectTrigger className="w-24">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {INTERVALS.map((iv) => (
                  <SelectItem key={iv} value={iv}>
                    {iv}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Preview result */}
        {preview && (
          <div className="rounded-md border bg-muted/30 p-3">
            <p className="text-sm font-medium text-foreground">Download Preview</p>
            <div className="mt-2 grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-xl font-bold text-foreground">{preview.total_months}</p>
                <p className="text-xs text-muted-foreground">Total months</p>
              </div>
              <div>
                <p className="text-xl font-bold text-green-500">{preview.archived_months}</p>
                <p className="text-xs text-muted-foreground">Already archived</p>
              </div>
              <div>
                <p className="text-xl font-bold text-yellow-500">{preview.new_months}</p>
                <p className="text-xs text-muted-foreground">New to download</p>
              </div>
            </div>
          </div>
        )}

        {/* Error display */}
        {(previewMutation.isError || ingestMutation.isError) && (
          <div className="rounded-md border border-destructive bg-destructive/10 p-3 text-sm text-destructive">
            {previewMutation.error instanceof Error
              ? previewMutation.error.message
              : ingestMutation.error instanceof Error
                ? ingestMutation.error.message
                : "Request failed"}
          </div>
        )}

        {/* Action buttons */}
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="secondary"
            onClick={handlePreview}
            disabled={!canPreview || isAnyMutating}
          >
            {previewMutation.isPending ? "Loading..." : "Preview"}
          </Button>
          <Button onClick={handleStartDownload} disabled={!canPreview || isAnyMutating}>
            {ingestMutation.isPending ? "Starting..." : "Start Download"}
          </Button>
          <Separator orientation="vertical" className="mx-2 h-6" />
          <Button variant="secondary" onClick={handleUpdateAll} disabled={isAnyMutating}>
            {updateMutation.isPending ? "Starting..." : "Update All"}
          </Button>
          <Button variant="secondary" onClick={handleRebuild} disabled={isAnyMutating}>
            {rebuildMutation.isPending ? "Rebuilding..." : "Rebuild Catalog"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
