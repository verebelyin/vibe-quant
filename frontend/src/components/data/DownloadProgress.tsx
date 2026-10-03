import { useCallback, useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

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
  const statusRef = useRef(status);
  statusRef.current = status;

  const addLog = useCallback((msg: string) => {
    setLogs((prev) => [...prev.slice(-200), msg]);
    // Defer scroll to after render
    requestAnimationFrame(() => {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, []);

  useEffect(() => {
    // The backend streams NAMED SSE events (sse/progress.py): "log" (one log line),
    // "complete" (final job status: completed | failed | killed) and "error".
    // `onmessage` only receives unnamed events, so it never fired.
    const es = new EventSource(`/api/data/ingest/${jobId}/progress`);
    esRef.current = es;

    es.onopen = () => {
      setStatus("running");
    };

    es.addEventListener("log", (ev) => {
      const line = (ev as MessageEvent<string>).data;
      if (line) addLog(line);
    });

    es.addEventListener("complete", (ev) => {
      const final = (ev as MessageEvent<string>).data;
      es.close();
      if (final === "completed") {
        setStatus("complete");
        setProgress(100);
        addLog("Download complete.");
      } else {
        setStatus("error");
        setErrorMsg(`Job ${final}`);
        addLog(`Job ended: ${final}`);
      }
    });

    es.addEventListener("error", (ev) => {
      // Server-sent "error" events carry data; transport errors don't.
      const msg = (ev as MessageEvent<string>).data;
      if (msg) {
        es.close();
        setStatus("error");
        setErrorMsg(msg);
        addLog(`Error: ${msg}`);
      } else if (es.readyState === EventSource.CLOSED && statusRef.current !== "complete") {
        setStatus("error");
        setErrorMsg("Connection lost");
        addLog("Connection to progress stream lost.");
      }
    });

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [jobId, addLog]);

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
            {status === "running" && "Running..."}
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
                  : "animate-pulse bg-primary",
            )}
            // No percentage is streamed: full-width pulse while running.
            style={{ width: status === "running" ? "100%" : `${progress}%` }}
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
