import { ChevronLeft, ChevronRight } from "lucide-react";
import { useListItemsApiResearchItemsGet } from "@/api/generated/research/research";
import { ItemRow } from "@/components/research/ItemRow";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/EmptyState";
import { Table, TableBody, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { PAGE_SIZE, useResearchStore } from "@/stores/researchStore";

export function ItemList() {
  const source = useResearchStore((s) => s.source);
  const status = useResearchStore((s) => s.status);
  const sort = useResearchStore((s) => s.sort);
  const page = useResearchStore((s) => s.page);
  const hideLowTrade = useResearchStore((s) => s.hideLowTrade);
  const q = useResearchStore((s) => s.q);
  const setPage = useResearchStore((s) => s.setPage);
  const reset = useResearchStore((s) => s.reset);

  const debouncedQ = useDebouncedValue(q, 200);

  const params = {
    source,
    ...(status === "all" ? {} : { status }),
    sort,
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
    ...(hideLowTrade ? { hide_low_trade: true } : {}),
    ...(debouncedQ ? { q: debouncedQ } : {}),
  };

  const { data, isLoading, isError } = useListItemsApiResearchItemsGet(params);

  if (isLoading) {
    return <div className="py-16 text-center text-sm text-muted-foreground">Loading items…</div>;
  }

  if (isError || !data || data.status !== 200) {
    return <div className="py-16 text-center text-sm text-red-400">Failed to load items.</div>;
  }

  const list = data.data;
  const items = list.items;
  const total = list.total;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  if (items.length === 0) {
    const filteredOut = status !== "all" || !!debouncedQ || hideLowTrade;
    if (filteredOut) {
      return (
        <EmptyState
          title="No items match these filters"
          description={debouncedQ ? `No titles match "${debouncedQ}".` : "Try clearing the filters."}
          action={{ label: "Clear filters", onClick: reset }}
        />
      );
    }
    return (
      <EmptyState
        title="No items yet"
        description="Click 'Scrape now' above to fetch the latest posts."
      />
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-hidden rounded-lg border border-border/60">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="w-24">Posted</TableHead>
              <TableHead>Title</TableHead>
              <TableHead className="w-24">Source</TableHead>
              <TableHead className="w-16 text-right">Score</TableHead>
              <TableHead className="w-20 text-right">Comments</TableHead>
              <TableHead className="w-24">Status</TableHead>
              <TableHead className="w-32">Confidence</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((it) => (
              <ItemRow key={it.id} item={it} />
            ))}
          </TableBody>
        </Table>
      </div>

      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="tabular-nums">
          {page * PAGE_SIZE + 1}–{page * PAGE_SIZE + items.length} of {total}
        </span>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
            className="h-7 px-2 gap-1 text-xs"
          >
            <ChevronLeft className="h-3 w-3" />
            Prev
          </Button>
          <span className="tabular-nums">
            Page {page + 1} of {totalPages}
          </span>
          <Button
            size="sm"
            variant="outline"
            disabled={page >= totalPages - 1}
            onClick={() => setPage(page + 1)}
            className="h-7 px-2 gap-1 text-xs"
          >
            Next
            <ChevronRight className="h-3 w-3" />
          </Button>
        </div>
      </div>
    </div>
  );
}
