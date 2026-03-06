import { useMemo } from "react";
import { useDataCoverageApiDataCoverageGet } from "@/api/generated/data/data";
import type { DataCoverageItem } from "@/api/generated/models";

export interface DatasetDateRange {
  /** Earliest start_date across all symbols (YYYY-MM-DD) */
  minStart: string;
  /** Latest end_date across all symbols (YYYY-MM-DD) */
  maxEnd: string;
  /** Per-symbol coverage items */
  items: DataCoverageItem[];
  /** Whether the query is still loading */
  isLoading: boolean;
}

/**
 * Fetches coverage data and computes the global min start / max end dates.
 * Returns empty strings when no data is available.
 */
export function useDatasetDateRange(): DatasetDateRange {
  const coverageQuery = useDataCoverageApiDataCoverageGet();
  const items: DataCoverageItem[] =
    coverageQuery.data?.status === 200 ? coverageQuery.data.data.coverage : [];

  const { minStart, maxEnd } = useMemo(() => {
    let min = "";
    let max = "";
    for (const item of items) {
      if (!item.start_date || !item.end_date) continue;
      if (!min || item.start_date < min) min = item.start_date;
      if (!max || item.end_date > max) max = item.end_date;
    }
    return { minStart: min, maxEnd: max };
  }, [items]);

  return {
    minStart,
    maxEnd,
    items,
    isLoading: coverageQuery.isLoading,
  };
}
