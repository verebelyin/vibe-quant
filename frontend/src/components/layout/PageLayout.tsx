import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";

export function PageLayout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <Sidebar />
      <SidebarInset>
        <Header />
        <main className="flex-1 overflow-y-auto bg-background p-4">{children}</main>
      </SidebarInset>
    </SidebarProvider>
  );
}
