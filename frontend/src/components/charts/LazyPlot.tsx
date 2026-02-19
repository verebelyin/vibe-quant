import { lazy, Suspense } from "react";
import type { PlotParams } from "react-plotly.js";

const Plot = lazy(() => import("react-plotly.js"));

function PlotFallback() {
  return (
    <div className="flex items-center justify-center rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--muted))] p-8 text-sm text-[hsl(var(--muted-foreground))]">
      Loading chart...
    </div>
  );
}

export default function LazyPlot(props: PlotParams) {
  return (
    <Suspense fallback={<PlotFallback />}>
      <Plot {...props} />
    </Suspense>
  );
}
