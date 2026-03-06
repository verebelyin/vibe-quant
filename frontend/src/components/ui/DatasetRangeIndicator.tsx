import { Database } from "lucide-react";
import type { DataCoverageItem } from "@/api/generated/models";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

interface DatasetRangeIndicatorProps {
  items: DataCoverageItem[];
  minStart: string;
  maxEnd: string;
  isLoading: boolean;
  /** Called when user clicks "Apply" to fill date pickers */
  onApply?: (start: string, end: string) => void;
}

export function DatasetRangeIndicator({
  items,
  minStart,
  maxEnd,
  isLoading,
  onApply,
}: DatasetRangeIndicatorProps) {
  if (isLoading || items.length === 0) return null;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="inline-flex items-center gap-1 rounded-md border border-border bg-muted/50 px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          title="Dataset date ranges"
        >
          <Database className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Dataset</span>
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-80 p-0" align="start">
        <div className="space-y-2 p-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-foreground">
              Dataset Coverage
            </h4>
            {onApply && minStart && maxEnd && (
              <button
                type="button"
                className="rounded px-2 py-0.5 text-[10px] font-medium text-accent-foreground underline-offset-2 hover:underline"
                onClick={() => onApply(minStart, maxEnd)}
              >
                Apply full range
              </button>
            )}
          </div>

          {/* Global range */}
          <div className="rounded-md bg-muted/50 px-2.5 py-1.5 text-xs">
            <span className="text-muted-foreground">Full range: </span>
            <span className="font-mono font-medium text-foreground">
              {minStart} &mdash; {maxEnd}
            </span>
          </div>

          {/* Per-symbol table */}
          <div className="max-h-48 overflow-y-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-left text-muted-foreground">
                  <th className="pb-1 pr-2 font-medium">Symbol</th>
                  <th className="pb-1 pr-2 font-medium">Start</th>
                  <th className="pb-1 pr-2 font-medium">End</th>
                  <th className="pb-1 text-right font-medium">Bars</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.symbol} className="border-b border-border/50 last:border-0">
                    <td className="py-1 pr-2 font-mono font-medium text-foreground">
                      {item.symbol}
                    </td>
                    <td className="py-1 pr-2 font-mono text-muted-foreground">
                      {item.start_date || "---"}
                    </td>
                    <td className="py-1 pr-2 font-mono text-muted-foreground">
                      {item.end_date || "---"}
                    </td>
                    <td className="py-1 text-right font-mono text-muted-foreground">
                      {item.kline_count.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
