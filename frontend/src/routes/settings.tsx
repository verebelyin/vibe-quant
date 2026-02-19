import { useState } from "react";
import { DatabaseTab } from "@/components/settings/DatabaseTab";
import { LatencyTab } from "@/components/settings/LatencyTab";
import { RiskTab } from "@/components/settings/RiskTab";
import { SizingTab } from "@/components/settings/SizingTab";
import { SystemTab } from "@/components/settings/SystemTab";

const TABS = [
  { id: "sizing", label: "Sizing" },
  { id: "risk", label: "Risk" },
  { id: "latency", label: "Latency" },
  { id: "database", label: "Database" },
  { id: "system", label: "System" },
] as const;

type TabId = (typeof TABS)[number]["id"];

const TAB_CONTENT: Record<TabId, () => JSX.Element> = {
  sizing: SizingTab,
  risk: RiskTab,
  latency: LatencyTab,
  database: DatabaseTab,
  system: SystemTab,
};

export function SettingsPage() {
  const [activeTab, setActiveTab] = useState<TabId>("sizing");
  const ActiveComponent = TAB_CONTENT[activeTab];

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold" style={{ color: "hsl(var(--foreground))" }}>
        Settings
      </h1>

      {/* Tab bar */}
      <div className="mt-4 flex gap-1 border-b" style={{ borderColor: "hsl(var(--border))" }}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className="relative px-4 py-2 text-sm font-medium transition-colors"
            style={{
              color:
                activeTab === tab.id ? "hsl(var(--foreground))" : "hsl(var(--muted-foreground))",
            }}
          >
            {tab.label}
            {activeTab === tab.id && (
              <span
                className="absolute inset-x-0 bottom-0 h-0.5"
                style={{ backgroundColor: "hsl(var(--primary))" }}
              />
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="mt-6">
        <ActiveComponent />
      </div>
    </div>
  );
}
