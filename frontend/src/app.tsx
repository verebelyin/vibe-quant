import { QueryClientProvider } from "@tanstack/react-query";
import { createRoute, createRouter, RouterProvider, redirect } from "@tanstack/react-router";
import { lazy, Suspense } from "react";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { RouteErrorComponent } from "@/components/RouteErrorComponent";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { queryClient } from "./api/query-client";
import { rootRoute } from "./routes/__root";

const DataPage = lazy(() => import("./routes/data").then((m) => ({ default: m.DataPage })));
const StrategiesPage = lazy(() =>
  import("./routes/strategies").then((m) => ({ default: m.StrategiesPage })),
);
const StrategyEditPage = lazy(() =>
  import("./routes/strategies.$strategyId").then((m) => ({ default: m.StrategyEditPage })),
);
const BacktestPage = lazy(() =>
  import("./routes/backtest").then((m) => ({ default: m.BacktestPage })),
);
const ResultsPage = lazy(() =>
  import("./routes/results").then((m) => ({ default: m.ResultsPage })),
);
const ResultsDetailPage = lazy(() =>
  import("./routes/results.$runId").then((m) => ({ default: m.ResultsDetailPage })),
);
const DiscoveryPage = lazy(() =>
  import("./routes/discovery").then((m) => ({ default: m.DiscoveryPage })),
);
const PaperTradingPage = lazy(() =>
  import("./routes/paper-trading").then((m) => ({ default: m.PaperTradingPage })),
);
const SettingsPage = lazy(() =>
  import("./routes/settings").then((m) => ({ default: m.SettingsPage })),
);
const GuidePage = lazy(() =>
  import("./routes/guide").then((m) => ({ default: m.GuidePage })),
);

function SuspensePage({ children }: { children: React.ReactNode }) {
  return <Suspense fallback={<LoadingSpinner />}>{children}</Suspense>;
}

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  beforeLoad: () => {
    throw redirect({ to: "/strategies" });
  },
});

const strategiesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/strategies",
  component: function StrategiesRouteComponent() {
    return (
      <SuspensePage>
        <StrategiesPage />
      </SuspensePage>
    );
  },
});

const strategyEditRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/strategies/$strategyId",
  component: function StrategyEditRouteComponent() {
    const { strategyId } = strategyEditRoute.useParams();
    return (
      <SuspensePage>
        <StrategyEditPage strategyId={Number(strategyId)} />
      </SuspensePage>
    );
  },
});

const discoveryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/discovery",
  component: function DiscoveryRouteComponent() {
    return (
      <SuspensePage>
        <DiscoveryPage />
      </SuspensePage>
    );
  },
});

const backtestRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/backtest",
  component: function BacktestRouteComponent() {
    return (
      <SuspensePage>
        <BacktestPage />
      </SuspensePage>
    );
  },
});

const resultsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/results",
  component: function ResultsRouteComponent() {
    return (
      <SuspensePage>
        <ResultsPage />
      </SuspensePage>
    );
  },
});

const resultsDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/results/$runId",
  component: function ResultsDetailRouteComponent() {
    const { runId } = resultsDetailRoute.useParams();
    return (
      <SuspensePage>
        <ResultsDetailPage runId={Number(runId)} />
      </SuspensePage>
    );
  },
});

const paperTradingRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/paper-trading",
  component: function PaperTradingRouteComponent() {
    return (
      <SuspensePage>
        <PaperTradingPage />
      </SuspensePage>
    );
  },
});

const dataRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/data",
  component: function DataRouteComponent() {
    return (
      <SuspensePage>
        <DataPage />
      </SuspensePage>
    );
  },
});

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: function SettingsRouteComponent() {
    return (
      <SuspensePage>
        <SettingsPage />
      </SuspensePage>
    );
  },
});

const guideRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/guide",
  component: function GuideRouteComponent() {
    return (
      <SuspensePage>
        <GuidePage />
      </SuspensePage>
    );
  },
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  strategiesRoute,
  strategyEditRoute,
  discoveryRoute,
  backtestRoute,
  resultsRoute,
  resultsDetailRoute,
  paperTradingRoute,
  dataRoute,
  settingsRoute,
  guideRoute,
]);

const router = createRouter({
  routeTree,
  defaultErrorComponent: RouteErrorComponent,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

export function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <RouterProvider router={router} />
          <Toaster />
        </TooltipProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
