import { useQueryClient } from "@tanstack/react-query";
import { Loader2, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  getGetLatestScrapeApiResearchScrapeLatestGetQueryKey,
  getListItemsApiResearchItemsGetQueryKey,
  useGetScrapeApiResearchScrapeRunIdGet,
  useStartScrapeApiResearchScrapePost,
} from "@/api/generated/research/research";
import { Button } from "@/components/ui/button";

interface Props {
  source: string;
}

export function ScrapeButton({ source }: Props) {
  const queryClient = useQueryClient();
  const [activeRunId, setActiveRunId] = useState<number | null>(null);

  const startMutation = useStartScrapeApiResearchScrapePost({
    mutation: {
      onSuccess: (resp) => {
        if (resp.status === 201) {
          setActiveRunId(resp.data.id);
          toast.info(`Scrape started for '${source}'`);
        }
      },
      onError: (err) => {
        const detail =
          (err as { response?: { data?: { detail?: string } } }).response?.data?.detail ??
          "Failed to start scrape";
        toast.error(typeof detail === "string" ? detail : "Failed to start scrape");
      },
    },
  });

  const pollQuery = useGetScrapeApiResearchScrapeRunIdGet(activeRunId ?? 0, {
    query: {
      enabled: activeRunId != null,
      refetchInterval: (q) => {
        const data = q.state.data;
        if (!data || data.status !== 200) return false;
        return data.data.status === "running" ? 2000 : false;
      },
    },
  });

  useEffect(() => {
    const data = pollQuery.data;
    if (!data || data.status !== 200 || activeRunId == null) return;
    const run = data.data;
    if (run.status !== "running") {
      const final = run.status;
      if (final === "completed") {
        toast.success(
          `Scrape complete · ${run.items_new} new · ${run.items_extracted} parsed · ${run.items_failed} failed`,
        );
      } else if (final === "failed") {
        toast.error(`Scrape failed${run.error_message ? `: ${run.error_message}` : ""}`);
      } else if (final === "killed") {
        toast.warning("Scrape stopped");
      }
      setActiveRunId(null);
      queryClient.invalidateQueries({ queryKey: getListItemsApiResearchItemsGetQueryKey() });
      queryClient.invalidateQueries({
        queryKey: getGetLatestScrapeApiResearchScrapeLatestGetQueryKey({ source }),
      });
    }
  }, [pollQuery.data, activeRunId, queryClient, source]);

  const isRunning = activeRunId != null;
  const disabled = isRunning || startMutation.isPending;

  return (
    <Button
      size="sm"
      onClick={() => startMutation.mutate({ data: { source, limit: 50, extract: true } })}
      disabled={disabled}
      className="gap-1.5 h-8 text-xs"
    >
      {isRunning || startMutation.isPending ? (
        <Loader2 className="h-3 w-3 animate-spin" />
      ) : (
        <RefreshCw className="h-3 w-3" />
      )}
      {isRunning ? "Scraping…" : "Scrape now"}
    </Button>
  );
}
