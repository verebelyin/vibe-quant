import { useMemo } from "react";
import { useUIStore } from "../../stores/ui";
import LazyPlot from "./LazyPlot";

export interface ParetoPoint {
  sharpe_ratio: number;
  max_drawdown: number;
  total_return: number;
  is_pareto_optimal: boolean;
}

export interface ParetoSurface3DProps {
  data: ParetoPoint[];
  height?: number;
  className?: string;
}

export default function ParetoSurface3D({ data, height = 500, className }: ParetoSurface3DProps) {
  const theme = useUIStore((s) => s.theme);
  const isDark = theme === "dark";

  const bgColor = isDark ? "#1a1a2e" : "#ffffff";
  const fontColor = isDark ? "#a0a0b8" : "#4a4a68";
  const gridColor = isDark ? "#2a2a3e" : "#e8e8f0";

  const plotData = useMemo(() => {
    const pareto = data.filter((d) => d.is_pareto_optimal);
    const nonPareto = data.filter((d) => !d.is_pareto_optimal);

    const traces: Parameters<typeof LazyPlot>[0]["data"] = [];

    if (nonPareto.length > 0) {
      traces.push({
        type: "scatter3d" as const,
        mode: "markers" as const,
        name: "Non-optimal",
        x: nonPareto.map((d) => d.max_drawdown),
        y: nonPareto.map((d) => d.sharpe_ratio),
        z: nonPareto.map((d) => d.total_return),
        marker: {
          size: 3,
          color: isDark ? "#555570" : "#b0b0c0",
          opacity: 0.3,
        },
        hovertemplate: "Drawdown: %{x:.2%}<br>Sharpe: %{y:.2f}<br>Return: %{z:.2%}<extra></extra>",
      });
    }

    if (pareto.length > 0) {
      traces.push({
        type: "scatter3d" as const,
        mode: "markers" as const,
        name: "Pareto optimal",
        x: pareto.map((d) => d.max_drawdown),
        y: pareto.map((d) => d.sharpe_ratio),
        z: pareto.map((d) => d.total_return),
        marker: {
          size: 5,
          color: "#3b82f6",
          opacity: 0.9,
        },
        hovertemplate: "Drawdown: %{x:.2%}<br>Sharpe: %{y:.2f}<br>Return: %{z:.2%}<extra></extra>",
      });
    }

    return traces;
  }, [data, isDark]);

  const layout = useMemo(
    () => ({
      paper_bgcolor: bgColor,
      plot_bgcolor: bgColor,
      font: { color: fontColor },
      margin: { l: 0, r: 0, t: 32, b: 0 },
      scene: {
        xaxis: {
          title: { text: "Max Drawdown", font: { color: fontColor } },
          tickfont: { color: fontColor },
          gridcolor: gridColor,
          zerolinecolor: gridColor,
        },
        yaxis: {
          title: { text: "Sharpe Ratio", font: { color: fontColor } },
          tickfont: { color: fontColor },
          gridcolor: gridColor,
          zerolinecolor: gridColor,
        },
        zaxis: {
          title: { text: "Total Return", font: { color: fontColor } },
          tickfont: { color: fontColor },
          gridcolor: gridColor,
          zerolinecolor: gridColor,
        },
        bgcolor: bgColor,
        camera: {
          eye: { x: 1.6, y: -1.6, z: 0.8 },
          center: { x: 0, y: 0, z: -0.1 },
        },
      },
      legend: {
        font: { color: fontColor },
        bgcolor: "transparent",
      },
    }),
    [bgColor, fontColor, gridColor],
  );

  return (
    <div className={className}>
      <LazyPlot
        data={plotData}
        layout={layout}
        config={{ responsive: true, displayModeBar: true, displaylogo: false }}
        style={{ width: "100%", height }}
        useResizeHandler
      />
    </div>
  );
}
