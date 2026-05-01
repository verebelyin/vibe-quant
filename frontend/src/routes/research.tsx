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
];

export function ResearchPage() {
  const source = useResearchStore((s) => s.source);
  const status = useResearchStore((s) => s.status);
  const sort = useResearchStore((s) => s.sort);
  const setSource = useResearchStore((s) => s.setSource);
  const setStatus = useResearchStore((s) => s.setStatus);
  const setSort = useResearchStore((s) => s.setSort);

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
            <SelectTrigger size="sm" className="h-8 text-xs w-44">
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
