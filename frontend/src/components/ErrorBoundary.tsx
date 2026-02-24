import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  copied: boolean;
}

function getErrorReport(error: Error | null, errorInfo: ErrorInfo | null, url: string): string {
  const timestamp = new Date().toISOString();
  const lines = [
    `## Runtime Error`,
    `**URL:** ${url}`,
    `**Time:** ${timestamp}`,
    `**UA:** ${navigator.userAgent}`,
    "",
    `### Error`,
    `\`\`\``,
    error?.toString() ?? "Unknown error",
    `\`\`\``,
    "",
  ];

  if (error?.stack) {
    lines.push(`### Stack Trace`, `\`\`\``, error.stack, `\`\`\``, "");
  }

  if (errorInfo?.componentStack) {
    lines.push(`### Component Stack`, `\`\`\``, errorInfo.componentStack.trim(), `\`\`\``, "");
  }

  return lines.join("\n");
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null, copied: false };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ error, errorInfo });
  }

  handleCopy = () => {
    const report = getErrorReport(
      this.state.error,
      this.state.errorInfo,
      window.location.href,
    );
    navigator.clipboard.writeText(report).then(() => {
      this.setState({ copied: true });
      setTimeout(() => this.setState({ copied: false }), 2000);
    });
  };

  handleReload = () => {
    window.location.reload();
  };

  handleDismiss = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (!this.state.hasError) return this.props.children;

    const { error, errorInfo, copied } = this.state;

    return (
      <div
        style={{
          minHeight: "100vh",
          background: "#0a0a14",
          color: "#e0e0e8",
          fontFamily: "'Geist', 'Inter', system-ui, sans-serif",
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
          {/* Header bar */}
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
              <h1 style={{ margin: 0, fontSize: "16px", fontWeight: 600, color: "#f0f0f5" }}>
                Runtime Error
              </h1>
              <p style={{ margin: "2px 0 0", fontSize: "13px", color: "#888899" }}>
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
                {error?.message ?? "Unknown error"}
              </code>
            </div>

            {/* Stack trace */}
            {error?.stack && (
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
                    maxHeight: "240px",
                    overflowY: "auto",
                  }}
                >
                  {error.stack}
                </pre>
              </details>
            )}

            {/* Component stack */}
            {errorInfo?.componentStack && (
              <details style={{ marginBottom: "16px" }}>
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
                  }}
                >
                  Component Stack
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
                    maxHeight: "200px",
                    overflowY: "auto",
                  }}
                >
                  {errorInfo.componentStack.trim()}
                </pre>
              </details>
            )}
          </div>

          {/* Actions */}
          <div
            style={{
              padding: "16px 24px",
              borderTop: "1px solid #2a2a3e",
              display: "flex",
              gap: "10px",
              justifyContent: "flex-end",
            }}
          >
            <button
              onClick={this.handleDismiss}
              type="button"
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
              Dismiss
            </button>
            <button
              onClick={this.handleCopy}
              type="button"
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
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
              onClick={this.handleReload}
              type="button"
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
}
