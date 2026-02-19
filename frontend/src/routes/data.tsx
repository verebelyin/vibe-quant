import { useCallback, useState } from "react";
import { DataBrowserTab } from "@/components/data/DataBrowserTab";
import { DataStatusDashboard } from "@/components/data/DataStatusDashboard";
import { DownloadHistory } from "@/components/data/DownloadHistory";
import { DownloadProgress } from "@/components/data/DownloadProgress";
import { IngestForm } from "@/components/data/IngestForm";

export function DataPage() {
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const handleIngestStarted = useCallback((jobId: string) => {
    setActiveJobId(jobId);
  }, []);

  const handleComplete = useCallback(() => {
    setActiveJobId(null);
  }, []);

  const handleCancel = useCallback(() => {
    setActiveJobId(null);
  }, []);

  return (
    <div className="space-y-6">
      <DataStatusDashboard />
      <div className="px-6 pb-6">
        <div className="space-y-6">
          <IngestForm onIngestStarted={handleIngestStarted} />
          {activeJobId && (
            <DownloadProgress
              jobId={activeJobId}
              onComplete={handleComplete}
              onCancel={handleCancel}
            />
          )}
          <div>
            <h2 className="mb-3 text-lg font-semibold text-foreground">Data Browser & Quality</h2>
            <DataBrowserTab />
          </div>
          <div>
            <h2 className="mb-3 text-lg font-semibold text-foreground">Download History</h2>
            <DownloadHistory />
          </div>
        </div>
      </div>
    </div>
  );
}
