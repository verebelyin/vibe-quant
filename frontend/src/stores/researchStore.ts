import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ResearchStatusFilter =
  | "all"
  | "pending"
  | "extracted"
  | "parsed"
  | "failed"
  | "promoted"
  | "rejected"
  | "skipped";

export type ResearchSort =
  | "newest_scraped"
  | "newest_posted"
  | "highest_score"
  | "highest_confidence"
  | "credibility"
  | "screen_sharpe";

interface ResearchState {
  source: string;
  status: ResearchStatusFilter;
  sort: ResearchSort;
  page: number;
  hideLowTrade: boolean;
  q: string;
  setSource: (source: string) => void;
  setStatus: (status: ResearchStatusFilter) => void;
  setSort: (sort: ResearchSort) => void;
  setPage: (page: number) => void;
  setHideLowTrade: (hide: boolean) => void;
  setQ: (q: string) => void;
  reset: () => void;
}

const DEFAULTS = {
  source: "reddit",
  status: "all" as const,
  sort: "newest_scraped" as const,
  page: 0,
  hideLowTrade: false,
  q: "",
};

export const PAGE_SIZE = 50;

export const useResearchStore = create<ResearchState>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      setSource: (source) => set({ source, page: 0 }),
      setStatus: (status) => set({ status, page: 0 }),
      setSort: (sort) => set({ sort, page: 0 }),
      setPage: (page) => set({ page }),
      setHideLowTrade: (hide) => set({ hideLowTrade: hide, page: 0 }),
      setQ: (q) => set({ q, page: 0 }),
      reset: () => set({ ...DEFAULTS }),
    }),
    { name: "vq.research.filters" },
  ),
);
