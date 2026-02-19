import { useId, useState } from "react";
import { useListSymbolsApiDataSymbolsGet } from "@/api/generated/data/data";
import { DataBrowser } from "./DataBrowser";
import { DataQualityPanel } from "./DataQualityPanel";

const TABS = ["Browser", "Quality"] as const;
type Tab = (typeof TABS)[number];

export function DataBrowserTab() {
  const qualitySelectId = useId();
  const [activeTab, setActiveTab] = useState<Tab>("Browser");
  const [qualitySymbol, setQualitySymbol] = useState("");

  const symbolsQuery = useListSymbolsApiDataSymbolsGet();
  const symbols = symbolsQuery.data?.data ?? [];

  return (
    <div className="space-y-4">
      {/* Tab buttons */}
      <div
        className="flex gap-1 rounded-lg border p-1"
        style={{
          borderColor: "hsl(var(--border))",
          backgroundColor: "hsl(var(--muted) / 0.3)",
        }}
      >
        {TABS.map((tab) => (
          <button
            key={tab}
            type="button"
            onClick={() => setActiveTab(tab)}
            className="rounded-md px-4 py-1.5 text-sm font-medium transition-colors"
            style={{
              backgroundColor: activeTab === tab ? "hsl(var(--card))" : "transparent",
              color: activeTab === tab ? "hsl(var(--foreground))" : "hsl(var(--muted-foreground))",
              boxShadow: activeTab === tab ? "0 1px 2px hsl(var(--border))" : undefined,
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "Browser" && <DataBrowser />}

      {activeTab === "Quality" && (
        <div className="space-y-4">
          {/* Symbol selector for quality */}
          <div className="flex flex-col gap-1">
            <label
              htmlFor={qualitySelectId}
              className="text-xs font-medium uppercase tracking-wider"
              style={{ color: "hsl(var(--muted-foreground))" }}
            >
              Symbol
            </label>
            <select
              id={qualitySelectId}
              value={qualitySymbol}
              onChange={(e) => setQualitySymbol(e.target.value)}
              className="w-48 rounded-md border px-3 py-1.5 text-sm"
              style={{
                backgroundColor: "hsl(var(--card))",
                borderColor: "hsl(var(--border))",
                color: "hsl(var(--foreground))",
              }}
            >
              <option value="">Select symbol...</option>
              {symbols.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          {!qualitySymbol && (
            <div
              className="rounded-lg border p-8 text-center text-sm"
              style={{
                borderColor: "hsl(var(--border))",
                color: "hsl(var(--muted-foreground))",
              }}
            >
              Select a symbol to view quality metrics
            </div>
          )}

          {qualitySymbol && <DataQualityPanel symbol={qualitySymbol} />}
        </div>
      )}
    </div>
  );
}
