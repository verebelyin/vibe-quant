import { create } from "zustand";
import { persist } from "zustand/middleware";

type Theme = "dark" | "light" | "system";

interface UIStore {
  sidebarCollapsed: boolean;
  editorMode: "visual" | "yaml";
  theme: Theme;
  toggleSidebar: () => void;
  setEditorMode: (mode: "visual" | "yaml") => void;
  setTheme: (theme: Theme) => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      editorMode: "visual",
      theme: "dark",
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setEditorMode: (mode) => set({ editorMode: mode }),
      setTheme: (theme) => set({ theme }),
    }),
    { name: "vibe-quant-ui" },
  ),
);
