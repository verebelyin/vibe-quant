import { createRootRoute, Outlet } from "@tanstack/react-router";
import { Toaster } from "sonner";
import { PageLayout } from "@/components/layout/PageLayout";

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
