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
const DiscoveryResultsPage = lazy(() =>
  import("./routes/discovery-results").then((m) => ({ default: m.DiscoveryResultsPage })),
);
const PaperTradingPage = lazy(() =>
  import("./routes/paper-trading").then((m) => ({ default: m.PaperTradingPage })),
);
const SettingsPage = lazy(() =>
  import("./routes/settings").then((m) => ({ default: m.SettingsPage })),
);
const GuidePage = lazy(() => import("./routes/guide").then((m) => ({ default: m.GuidePage })));
const BrowserPage = lazy(() =>
  import("./routes/browser").then((m) => ({ default: m.BrowserPage })),
);
const ResearchPage = lazy(() =>
  import("./routes/research").then((m) => ({ default: m.ResearchPage })),
);
const ResearchItemPage = lazy(() =>
  import("./routes/research.$itemId").then((m) => ({ default: m.ResearchItemPage })),
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

const discoveryResultsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/discovery/results",
  component: function DiscoveryResultsRouteComponent() {
    return (
      <SuspensePage>
        <DiscoveryResultsPage />
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

const browserRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/browser",
  component: function BrowserRouteComponent() {
    return (
      <SuspensePage>
        <BrowserPage />
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

type ResearchSearch = {
  sort?: string;
  hide_low_trade?: boolean;
};

const researchRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/research",
  validateSearch: (search: Record<string, unknown>): ResearchSearch => ({
    sort: typeof search.sort === "string" ? search.sort : undefined,
    hide_low_trade:
      search.hide_low_trade === true ||
      search.hide_low_trade === "1" ||
      search.hide_low_trade === "true"
        ? true
        : undefined,
  }),
  component: function ResearchRouteComponent() {
    return (
      <SuspensePage>
        <ResearchPage />
      </SuspensePage>
    );
  },
});

const researchItemRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/research/$itemId",
  component: function ResearchItemRouteComponent() {
    const { itemId } = researchItemRoute.useParams();
    return (
      <SuspensePage>
        <ResearchItemPage itemId={Number(itemId)} />
      </SuspensePage>
    );
  },
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  strategiesRoute,
  strategyEditRoute,
  discoveryResultsRoute,
  discoveryRoute,
  backtestRoute,
  resultsRoute,
  resultsDetailRoute,
  paperTradingRoute,
  browserRoute,
  dataRoute,
  settingsRoute,
  guideRoute,
  researchRoute,
  researchItemRoute,
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
