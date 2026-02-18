import { QueryClientProvider } from "@tanstack/react-query";
import { createRoute, createRouter, RouterProvider, redirect } from "@tanstack/react-router";
import { queryClient } from "./api/query-client";
import { rootRoute } from "./routes/__root";
import { BacktestPage } from "./routes/backtest";
import { DataPage } from "./routes/data";
import { DiscoveryPage } from "./routes/discovery";
import { PaperTradingPage } from "./routes/paper-trading";
import { ResultsPage } from "./routes/results";
import { SettingsPage } from "./routes/settings";
import { StrategiesPage } from "./routes/strategies";

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
  component: StrategiesPage,
});

const discoveryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/discovery",
  component: DiscoveryPage,
});

const backtestRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/backtest",
  component: BacktestPage,
});

const resultsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/results",
  component: ResultsPage,
});

const paperTradingRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/paper-trading",
  component: PaperTradingPage,
});

const dataRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/data",
  component: DataPage,
});

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: SettingsPage,
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  strategiesRoute,
  discoveryRoute,
  backtestRoute,
  resultsRoute,
  paperTradingRoute,
  dataRoute,
  settingsRoute,
]);

const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
