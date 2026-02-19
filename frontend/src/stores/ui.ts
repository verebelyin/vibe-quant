import { create } from "zustand";
import { persist } from "zustand/middleware";

interface UIStore {
  sidebarCollapsed: boolean;
  editorMode: "visual" | "yaml";
  toggleSidebar: () => void;
  setEditorMode: (mode: "visual" | "yaml") => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      editorMode: "visual",
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setEditorMode: (mode) => set({ editorMode: mode }),
    }),
    { name: "vibe-quant-ui" },
  ),
);
