import { useEffect } from "react";
import { useUIStore } from "@/stores/ui";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";

export function PageLayout({ children }: { children: React.ReactNode }) {
  const theme = useUIStore((s) => s.theme);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />
        <main
          className="flex-1 overflow-y-auto"
          style={{ backgroundColor: "hsl(var(--background))" }}
        >
          {children}
        </main>
      </div>
    </div>
  );
}
