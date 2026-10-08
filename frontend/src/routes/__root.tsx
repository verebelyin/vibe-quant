import { createRootRoute, Outlet } from "@tanstack/react-router";
import { PageLayout } from "@/components/layout/PageLayout";
import { Toaster } from "@/components/ui/sonner";

export const rootRoute = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <PageLayout>
      <Outlet />
      <Toaster richColors />
    </PageLayout>
  );
}
