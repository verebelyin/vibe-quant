import { useState } from "react";
import type { ErrorComponentProps } from "@tanstack/react-router";

function getErrorReport(error: Error, url: string): string {
  const timestamp = new Date().toISOString();
  const lines = [
    `## Runtime Error`,
    `**URL:** ${url}`,
    `**Time:** ${timestamp}`,
    `**UA:** ${navigator.userAgent}`,
    "",
    `### Error`,
    `\`\`\``,
    error.toString(),
    `\`\`\``,
    "",
  ];

  if (error.stack) {
    lines.push(`### Stack Trace`, `\`\`\``, error.stack, `\`\`\``, "");
  }

  return lines.join("\n");
}

export function RouteErrorComponent({ error, reset }: ErrorComponentProps) {
  const [copied, setCopied] = useState(false);
  const err = error instanceof Error ? error : new Error(String(error));

  const handleCopy = () => {
    navigator.clipboard.writeText(getErrorReport(err, window.location.href)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div
      style={{
        minHeight: "60vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "2rem",
      }}
    >
      <div
        style={{
          maxWidth: "720px",
          width: "100%",
          background: "#141420",
          border: "1px solid #2a2a3e",
          borderRadius: "12px",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "20px 24px",
            borderBottom: "1px solid #2a2a3e",
            display: "flex",
            alignItems: "center",
            gap: "12px",
          }}
        >
          <div
            style={{
              width: "36px",
              height: "36px",
              borderRadius: "8px",
              background: "rgba(239, 68, 68, 0.15)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path
                d="M10 6v4m0 4h.01M19 10a9 9 0 11-18 0 9 9 0 0118 0z"
                stroke="#ef4444"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />
            </svg>
          </div>
          <div style={{ flex: 1 }}>
            <h1
              style={{
                margin: 0,
                fontSize: "16px",
                fontWeight: 600,
                color: "#f0f0f5",
                fontFamily: "'Geist', system-ui, sans-serif",
              }}
            >
              Runtime Error
            </h1>
            <p
              style={{
                margin: "2px 0 0",
                fontSize: "13px",
                color: "#888899",
                fontFamily: "'Geist', system-ui, sans-serif",
              }}
            >
              {window.location.pathname}
            </p>
          </div>
        </div>

        {/* Error message */}
        <div style={{ padding: "20px 24px" }}>
          <div
            style={{
              background: "rgba(239, 68, 68, 0.08)",
              border: "1px solid rgba(239, 68, 68, 0.2)",
              borderRadius: "8px",
              padding: "14px 16px",
              marginBottom: "16px",
            }}
          >
            <code
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: "13px",
                color: "#f87171",
                wordBreak: "break-word",
                lineHeight: 1.5,
              }}
            >
              {err.message}
            </code>
          </div>

          {/* Stack trace */}
          {err.stack && (
            <details open style={{ marginBottom: "16px" }}>
              <summary
                style={{
                  cursor: "pointer",
                  fontSize: "12px",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  color: "#888899",
                  marginBottom: "8px",
                  userSelect: "none",
                  fontFamily: "'Geist', system-ui, sans-serif",
                }}
              >
                Stack Trace
              </summary>
              <pre
                style={{
                  background: "#0d0d18",
                  border: "1px solid #2a2a3e",
                  borderRadius: "8px",
                  padding: "14px 16px",
                  margin: 0,
                  fontSize: "11.5px",
                  fontFamily: "'JetBrains Mono', monospace",
                  color: "#a0a0b0",
                  overflowX: "auto",
                  lineHeight: 1.6,
                  maxHeight: "280px",
                  overflowY: "auto",
                }}
              >
                {err.stack}
              </pre>
            </details>
          )}
        </div>

        {/* Actions */}
        <style>{`
          .rec-btn { transition: filter 150ms, opacity 150ms; }
          .rec-btn:hover { filter: brightness(1.3); }
          .rec-btn-primary:hover { opacity: 0.85; }
        `}</style>
        <div
          style={{
            padding: "16px 24px",
            borderTop: "1px solid #2a2a3e",
            display: "flex",
            gap: "10px",
            justifyContent: "flex-end",
            fontFamily: "'Geist', system-ui, sans-serif",
          }}
        >
          <button
            onClick={reset}
            type="button"
            className="rec-btn"
            style={{
              padding: "8px 16px",
              fontSize: "13px",
              fontWeight: 500,
              borderRadius: "6px",
              border: "1px solid #2a2a3e",
              background: "transparent",
              color: "#a0a0b0",
              cursor: "pointer",
            }}
          >
            Retry
          </button>
          <button
            onClick={handleCopy}
            type="button"
            className="rec-btn"
            style={{
              padding: "8px 16px",
              fontSize: "13px",
              fontWeight: 500,
              borderRadius: "6px",
              border: "1px solid #2a2a3e",
              background: copied ? "rgba(34, 197, 94, 0.15)" : "#1e1e30",
              color: copied ? "#4ade80" : "#e0e0e8",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              transition: "all 150ms",
            }}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              {copied ? (
                <path d="M20 6L9 17l-5-5" />
              ) : (
                <>
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                  <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
                </>
              )}
            </svg>
            {copied ? "Copied!" : "Copy Error Report"}
          </button>
          <button
            onClick={() => window.location.reload()}
            type="button"
            className="rec-btn rec-btn-primary"
            style={{
              padding: "8px 16px",
              fontSize: "13px",
              fontWeight: 500,
              borderRadius: "6px",
              border: "none",
              background: "oklch(0.72 0.19 230)",
              color: "#0a0a14",
              cursor: "pointer",
            }}
          >
            Reload Page
          </button>
        </div>
      </div>
    </div>
  );
}
