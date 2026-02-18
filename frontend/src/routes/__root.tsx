import { createRootRoute, Outlet } from "@tanstack/react-router";
import { PageLayout } from "@/components/layout/PageLayout";

export const rootRoute = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <PageLayout>
      <Outlet />
    </PageLayout>
  );
}
