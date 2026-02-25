import { useMemo } from "react";
import { useUIStore } from "../../stores/ui";
import LazyPlot from "./LazyPlot";

export interface HeatmapChartProps {
  data: { x: string[]; y: string[]; z: number[][] };
  title?: string;
  height?: number;
  className?: string;
}

export default function HeatmapChart({ data, title, height = 400, className }: HeatmapChartProps) {
  const theme = useUIStore((s) => s.theme);
  const isDark = theme === "dark";

  const bgColor = isDark ? "#1a1a2e" : "#ffffff";
  const fontColor = isDark ? "#a0a0b8" : "#4a4a68";

  const plotData = useMemo(
    () => [
      {
        type: "heatmap" as const,
        x: data.x,
        y: data.y,
        z: data.z,
        colorscale: "RdBu" as const,
        reversescale: true,
        colorbar: {
          tickfont: { color: fontColor },
        },
      },
    ],
    [data, fontColor],
  );

  const layout = useMemo(
    () => ({
      ...(title ? { title: { text: title, font: { color: fontColor, size: 14 } } } : {}),
      paper_bgcolor: bgColor,
      plot_bgcolor: bgColor,
      font: { color: fontColor },
      margin: { l: 80, r: 40, t: title ? 48 : 16, b: 80 },
      xaxis: {
        tickfont: { color: fontColor },
        gridcolor: isDark ? "#2a2a3e" : "#e8e8f0",
      },
      yaxis: {
        tickfont: { color: fontColor },
        gridcolor: isDark ? "#2a2a3e" : "#e8e8f0",
      },
    }),
    [title, bgColor, fontColor, isDark],
  );

  return (
    <div className={className}>
      <LazyPlot
        data={plotData}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: "100%", height }}
        useResizeHandler
      />
    </div>
  );
}
