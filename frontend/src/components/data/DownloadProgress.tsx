import { useCallback, useEffect, useRef, useState } from "react";

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

  const barColor =
    status === "error"
      ? "hsl(0 84% 60%)"
      : status === "complete"
        ? "hsl(142 71% 45%)"
        : "hsl(var(--primary))";

  return (
    <div
      className="rounded-lg border p-5"
      style={{
        backgroundColor: "hsl(var(--card))",
        borderColor: "hsl(var(--border))",
      }}
    >
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold" style={{ color: "hsl(var(--foreground))" }}>
          Download Progress
        </h2>
        <div className="flex items-center gap-3">
          <span
            className="text-xs font-medium uppercase tracking-wider"
            style={{
              color:
                status === "complete"
                  ? "hsl(142 71% 45%)"
                  : status === "error"
                    ? "hsl(0 84% 60%)"
                    : "hsl(var(--muted-foreground))",
            }}
          >
            {status === "connecting" && "Connecting..."}
            {status === "running" && `${progress.toFixed(0)}%`}
            {status === "complete" && "Complete"}
            {status === "error" && "Failed"}
          </span>
          {(status === "connecting" || status === "running") && (
            <button
              type="button"
              onClick={handleCancel}
              className="rounded-md px-3 py-1 text-xs font-medium transition-colors"
              style={{
                backgroundColor: "hsl(0 84% 60% / 0.1)",
                color: "hsl(0 84% 60%)",
              }}
            >
              Cancel
            </button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div
        className="h-2.5 w-full overflow-hidden rounded-full"
        style={{ backgroundColor: "hsl(var(--muted))" }}
      >
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{
            width: `${progress}%`,
            backgroundColor: barColor,
          }}
        />
      </div>

      {/* Error message */}
      {status === "error" && errorMsg && (
        <div
          className="mt-3 rounded-md border p-2 text-sm"
          style={{
            borderColor: "hsl(0 84% 60%)",
            backgroundColor: "hsl(0 84% 60% / 0.1)",
            color: "hsl(0 84% 60%)",
          }}
        >
          {errorMsg}
        </div>
      )}

      {/* Log output */}
      <div
        className="mt-3 max-h-48 overflow-y-auto rounded-md border p-3 font-mono text-xs"
        style={{
          borderColor: "hsl(var(--border))",
          backgroundColor: "hsl(var(--muted) / 0.3)",
          color: "hsl(var(--muted-foreground))",
        }}
      >
        {logs.length === 0 && <span>Waiting for progress events...</span>}
        {logs.map((line, i) => (
          <div key={`${i}-${line.slice(0, 20)}`}>{line}</div>
        ))}
        <div ref={logsEndRef} />
      </div>
    </div>
  );
}
