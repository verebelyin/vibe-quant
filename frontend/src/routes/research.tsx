import { useNavigate, useSearch } from "@tanstack/react-router";
import { useEffect, useRef } from "react";
import { useGetSourcesApiResearchSourcesGet } from "@/api/generated/research/research";
import { ItemList } from "@/components/research/ItemList";
import { ScrapeButton } from "@/components/research/ScrapeButton";
import { ScrapeRunBanner } from "@/components/research/ScrapeRunBanner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  type ResearchSort,
  type ResearchStatusFilter,
  useResearchStore,
} from "@/stores/researchStore";

const VALID_SORTS: ResearchSort[] = [
  "newest_scraped",
  "newest_posted",
  "highest_score",
  "highest_confidence",
  "screen_sharpe",
];

const STATUS_OPTIONS: { value: ResearchStatusFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "pending", label: "Pending" },
  { value: "extracted", label: "Extracted" },
  { value: "failed", label: "Failed" },
  { value: "promoted", label: "Promoted" },
  { value: "rejected", label: "Rejected" },
  { value: "skipped", label: "Skipped" },
];

const SORT_OPTIONS: { value: ResearchSort; label: string }[] = [
  { value: "newest_scraped", label: "Newest scraped" },
  { value: "newest_posted", label: "Newest posted" },
  { value: "highest_score", label: "Highest score" },
  { value: "highest_confidence", label: "Highest confidence" },
  { value: "screen_sharpe", label: "Highest screen Sharpe" },
];

export function ResearchPage() {
  const source = useResearchStore((s) => s.source);
  const status = useResearchStore((s) => s.status);
  const sort = useResearchStore((s) => s.sort);
  const hideLowTrade = useResearchStore((s) => s.hideLowTrade);
  const setSource = useResearchStore((s) => s.setSource);
  const setStatus = useResearchStore((s) => s.setStatus);
  const setSort = useResearchStore((s) => s.setSort);
  const setHideLowTrade = useResearchStore((s) => s.setHideLowTrade);

  const navigate = useNavigate();
  const search = useSearch({ strict: false }) as {
    sort?: string;
    hide_low_trade?: boolean;
  };

  // On mount, hydrate store from URL (URL wins for shareable links).
  const hydratedRef = useRef(false);
  useEffect(() => {
    if (hydratedRef.current) return;
    hydratedRef.current = true;
    if (search.sort && (VALID_SORTS as string[]).includes(search.sort)) {
      setSort(search.sort as ResearchSort);
    }
    if (search.hide_low_trade === true) {
      setHideLowTrade(true);
    }
  }, [search.sort, search.hide_low_trade, setSort, setHideLowTrade]);

  // After hydration, mirror store back to URL.
  useEffect(() => {
    if (!hydratedRef.current) return;
    navigate({
      search: (prev: Record<string, unknown>) => ({
        ...prev,
        sort: sort === "newest_scraped" ? undefined : sort,
        hide_low_trade: hideLowTrade ? true : undefined,
      }),
      replace: true,
    });
  }, [sort, hideLowTrade, navigate]);

  const { data: sourcesData } = useGetSourcesApiResearchSourcesGet();
  const sources = sourcesData && sourcesData.status === 200 ? sourcesData.data.sources : ["reddit"];

  return (
    <div className="flex flex-col h-full gap-4">
      <div className="flex flex-wrap items-center gap-2 pb-2">
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted-foreground">Source</span>
          <Select value={source} onValueChange={setSource}>
            <SelectTrigger size="sm" className="h-8 text-xs w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {sources.map((s) => (
                <SelectItem key={s} value={s} className="text-xs">
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted-foreground">Status</span>
          <Select value={status} onValueChange={(v) => setStatus(v as ResearchStatusFilter)}>
            <SelectTrigger size="sm" className="h-8 text-xs w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {STATUS_OPTIONS.map((o) => (
                <SelectItem key={o.value} value={o.value} className="text-xs">
                  {o.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted-foreground">Sort</span>
          <Select value={sort} onValueChange={(v) => setSort(v as ResearchSort)}>
            <SelectTrigger size="sm" className="h-8 text-xs w-48">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SORT_OPTIONS.map((o) => (
                <SelectItem key={o.value} value={o.value} className="text-xs">
                  {o.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <label className="flex items-center gap-1.5 text-xs text-muted-foreground select-none cursor-pointer">
          <input
            type="checkbox"
            checked={hideLowTrade}
            onChange={(e) => setHideLowTrade(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-border/60 accent-primary cursor-pointer"
          />
          Hide &lt;50 trades
        </label>

        <div className="ml-auto">
          <ScrapeButton source={source} />
        </div>
      </div>

      <div className="rounded-md border border-border/40 bg-card/40 px-3 py-2">
        <ScrapeRunBanner source={source} />
      </div>

      <div className="flex-1 overflow-auto">
        <ItemList />
      </div>
    </div>
  );
}
