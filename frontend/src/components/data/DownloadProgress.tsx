import { useCallback, useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface ProgressEvent {
  progress?: number;
  message?: string;
  status?: string;
  error?: string;
}

interface DownloadProgressProps {
  jobId: string;
  onComplete: () => void;
  onCancel: () => void;
}

export function DownloadProgress({ jobId, onComplete, onCancel }: DownloadProgressProps) {
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<"connecting" | "running" | "complete" | "error">(
    "connecting",
  );
  const [errorMsg, setErrorMsg] = useState("");
  const esRef = useRef<EventSource | null>(null);
  const logsEndRef = useRef<HTMLDivElement | null>(null);

  const addLog = useCallback((msg: string) => {
    setLogs((prev) => [...prev.slice(-200), msg]);
  }, []);

  useEffect(() => {
    const es = new EventSource(`/api/data/ingest/${jobId}/progress`);
    esRef.current = es;

    es.onopen = () => {
      setStatus("running");
      addLog("Connected to progress stream.");
    };

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as ProgressEvent;

        if (data.progress !== undefined) {
          setProgress(Math.min(100, Math.max(0, data.progress)));
        }

        if (data.message) {
          addLog(data.message);
        }

        if (data.status === "complete" || data.status === "done") {
          setStatus("complete");
          setProgress(100);
          addLog("Download complete.");
          es.close();
        }

        if (data.status === "error" || data.error) {
          setStatus("error");
          setErrorMsg(data.error ?? "Unknown error");
          addLog(`Error: ${data.error ?? "Unknown"}`);
          es.close();
        }
      } catch {
        // Non-JSON message, treat as log line
        if (ev.data) addLog(ev.data);
      }
    };

    es.onerror = () => {
      // EventSource auto-reconnects on error, but if readyState is CLOSED it won't
      if (es.readyState === EventSource.CLOSED) {
        if (status !== "complete") {
          setStatus("error");
          setErrorMsg("Connection lost");
          addLog("Connection to progress stream lost.");
        }
      }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, addLog]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Notify parent on complete
  useEffect(() => {
    if (status === "complete") {
      const timer = setTimeout(onComplete, 1500);
      return () => clearTimeout(timer);
    }
  }, [status, onComplete]);

  function handleCancel() {
    esRef.current?.close();
    onCancel();
  }

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle>Download Progress</CardTitle>
        <div className="flex items-center gap-3">
          <Badge
            variant={
              status === "complete" ? "default" : status === "error" ? "destructive" : "secondary"
            }
          >
            {status === "connecting" && "Connecting..."}
            {status === "running" && `${progress.toFixed(0)}%`}
            {status === "complete" && "Complete"}
            {status === "error" && "Failed"}
          </Badge>
          {(status === "connecting" || status === "running") && (
            <Button variant="destructive" size="xs" onClick={handleCancel}>
              Cancel
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Progress bar */}
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-300",
              status === "error"
                ? "bg-destructive"
                : status === "complete"
                  ? "bg-green-500"
                  : "bg-primary",
            )}
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Error message */}
        {status === "error" && errorMsg && (
          <div className="rounded-md border border-destructive bg-destructive/10 p-2 text-sm text-destructive">
            {errorMsg}
          </div>
        )}

        {/* Log output */}
        <div className="max-h-48 overflow-y-auto rounded-md border bg-muted/30 p-3 font-mono text-xs text-muted-foreground">
          {logs.length === 0 && <span>Waiting for progress events...</span>}
          {logs.map((line, i) => (
            <div key={`${i}-${line.slice(0, 20)}`}>{line}</div>
          ))}
          <div ref={logsEndRef} />
        </div>
      </CardContent>
    </Card>
  );
}
