import {
  CandlestickSeries,
  createChart,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type SeriesType,
} from "lightweight-charts";
import { useEffect, useRef } from "react";
import { useUIStore } from "../../stores/ui";

export type CandlestickData = {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
};

export type VolumeData = {
  time: string | number;
  value: number;
  color?: string;
};

interface CandlestickChartProps {
  data: CandlestickData[];
  volume?: VolumeData[];
  height?: number;
  className?: string;
}

const THEME_COLORS = {
  dark: {
    background: "#1a1a2e",
    text: "#a0a0b0",
    grid: "#2a2a3e",
  },
  light: {
    background: "#ffffff",
    text: "#333333",
    grid: "#e0e0e0",
  },
} as const;

const CANDLE_COLORS = {
  up: "#26a69a",
  down: "#ef5350",
} as const;

/** Format UTC unix seconds for crosshair label — shows date + time like TradingView */
function formatCrosshairTime(unixSec: number | string): string {
  if (typeof unixSec === "string") return unixSec;
  const d = new Date(unixSec * 1000);
  const year = d.getUTCFullYear();
  const mon = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  const hrs = String(d.getUTCHours()).padStart(2, "0");
  const min = String(d.getUTCMinutes()).padStart(2, "0");
  // Omit time portion if midnight (daily bars)
  if (hrs === "00" && min === "00") return `${year}-${mon}-${day}`;
  return `${year}-${mon}-${day} ${hrs}:${min}`;
}

/** Format tick marks on the time axis — context-aware like TradingView */
function formatTickMark(unixSec: number | string): string {
  if (typeof unixSec === "string") return unixSec;
  const d = new Date(unixSec * 1000);
  const hrs = String(d.getUTCHours()).padStart(2, "0");
  const min = String(d.getUTCMinutes()).padStart(2, "0");
  const day = d.getUTCDate();
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const mon = months[d.getUTCMonth()];
  // At midnight show "DD Mon", at month boundary show "Mon 'YY", otherwise show HH:MM
  if (hrs === "00" && min === "00") {
    return day === 1 ? `${mon} '${String(d.getUTCFullYear()).slice(2)}` : `${day} ${mon}`;
  }
  return `${hrs}:${min}`;
}

function getChartOptions(theme: "light" | "dark", height: number) {
  const colors = THEME_COLORS[theme];
  return {
    height,
    layout: {
      background: { color: colors.background },
      textColor: colors.text,
    },
    grid: {
      vertLines: { color: colors.grid },
      horzLines: { color: colors.grid },
    },
    crosshair: {
      mode: 0 as const,
    },
    localization: {
      timeFormatter: formatCrosshairTime,
    },
    timeScale: {
      borderColor: colors.grid,
      tickMarkFormatter: formatTickMark,
    },
    rightPriceScale: {
      borderColor: colors.grid,
    },
  };
}

export default function CandlestickChart({
  data,
  volume,
  height = 400,
  className,
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<SeriesType> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<SeriesType> | null>(null);
  const theme = useUIStore((s) => s.theme);

  // Create chart + series — mount-only; theme/height/volume handled in separate effects
  // biome-ignore lint/correctness/useExhaustiveDependencies: intentional mount-only effect
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, getChartOptions(theme === "system" ? "dark" : theme, height));
    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: CANDLE_COLORS.up,
      downColor: CANDLE_COLORS.down,
      borderVisible: false,
      wickUpColor: CANDLE_COLORS.up,
      wickDownColor: CANDLE_COLORS.down,
    });
    candleSeriesRef.current = candleSeries;

    if (volume) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });
      volumeSeriesRef.current = volumeSeries;

      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });
    }

    // Resize observer
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect;
        chart.applyOptions({ width });
      }
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
    };
  }, []);

  // Update theme
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.applyOptions(getChartOptions(theme === "system" ? "dark" : theme, height));
  }, [theme, height]);

  // Update candlestick data
  useEffect(() => {
    const series = candleSeriesRef.current;
    if (!series) return;
    series.setData(
      data.map((d) => ({
        time: d.time as never,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      })),
    );
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  // Update volume data
  useEffect(() => {
    const series = volumeSeriesRef.current;
    if (!series || !volume) return;
    series.setData(
      volume.map((v) => ({
        time: v.time as never,
        value: v.value,
        color: v.color ?? (v.value >= 0 ? "#26a69a80" : "#ef535080"),
      })),
    );
  }, [volume]);

  return <div ref={containerRef} className={className} />;
}
