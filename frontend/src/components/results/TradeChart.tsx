import { useEffect, useMemo, useRef, useState } from "react";
import {
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type SeriesType,
  type UTCTimestamp,
} from "lightweight-charts";
import {
  useGetTradesApiResultsRunsRunIdTradesGet,
  useListRunsApiResultsRunsGet,
} from "@/api/generated/results/results";
import { useBrowseDataApiDataBrowseSymbolGet } from "@/api/generated/data/data";
import { useUIStore } from "@/stores/ui";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

const INTERVALS = ["1m", "5m", "15m", "1h", "4h"] as const;

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

const CANDLE_COLORS = { up: "#26a69a", down: "#ef5350" } as const;

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
    crosshair: { mode: 0 as const },
    timeScale: { borderColor: colors.grid },
    rightPriceScale: { borderColor: colors.grid },
  };
}

interface TradeChartProps {
  runId: number;
}

export function TradeChart({ runId }: TradeChartProps) {
  const theme = useUIStore((s) => s.theme);
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<SeriesType> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<SeriesType> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<UTCTimestamp> | null>(null);

  // Get run metadata
  const runsQuery = useListRunsApiResultsRunsGet();
  const run = useMemo(() => {
    const resp = runsQuery.data;
    const runs = resp && resp.status === 200 ? resp.data.runs : [];
    return runs.find((r) => r.id === runId);
  }, [runsQuery.data, runId]);

  const symbols = run?.symbols ?? [];
  const [selectedSymbol, setSelectedSymbol] = useState<string>("");
  const [interval, setIntervalState] = useState<string>("");

  // Auto-select first symbol + default interval from run timeframe
  useEffect(() => {
    if (symbols.length > 0 && !symbols.includes(selectedSymbol)) {
      setSelectedSymbol(symbols[0]);
    }
  }, [symbols, selectedSymbol]);

  useEffect(() => {
    if (run?.timeframe && !interval) {
      setIntervalState(run.timeframe);
    }
  }, [run?.timeframe, interval]);

  // Fetch candle data
  const candleQuery = useBrowseDataApiDataBrowseSymbolGet(
    selectedSymbol,
    {
      interval: interval || run?.timeframe,
      start: run?.start_date,
      end: run?.end_date,
    },
    { query: { enabled: !!selectedSymbol && !!run && !!interval } },
  );

  // Fetch trades
  const tradesQuery = useGetTradesApiResultsRunsRunIdTradesGet(runId, undefined, {
    query: { enabled: !!run },
  });

  const candles = useMemo(() => {
    const resp = candleQuery.data;
    if (!resp || resp.status !== 200) return [];
    return resp.data.data.map((d: Record<string, unknown>) => ({
      time: Math.floor((d.open_time as number) / 1000) as UTCTimestamp,
      open: d.open as number,
      high: d.high as number,
      low: d.low as number,
      close: d.close as number,
      volume: d.volume as number,
    }));
  }, [candleQuery.data]);

  // Build a set of candle timestamps for snapping markers
  const candleTimeSet = useMemo(() => {
    const set = new Set<number>();
    for (const c of candles) set.add(c.time as number);
    return set;
  }, [candles]);

  // Sorted candle times for binary search snapping
  const sortedCandleTimes = useMemo(() => {
    return Array.from(candleTimeSet).sort((a, b) => a - b);
  }, [candleTimeSet]);

  const volumeData = useMemo(
    () =>
      candles.map((c) => ({
        time: c.time,
        value: c.volume,
        color: c.close >= c.open ? "#26a69a80" : "#ef535080",
      })),
    [candles],
  );

  const markers = useMemo(() => {
    const resp = tradesQuery.data;
    if (!resp || resp.status !== 200 || sortedCandleTimes.length === 0) return [];
    const trades = resp.data.filter((t) => t.symbol.startsWith(selectedSymbol));
    const result: SeriesMarker<UTCTimestamp>[] = [];

    // Snap a timestamp to the nearest candle time
    function snapToCandle(ts: number): UTCTimestamp {
      const times = sortedCandleTimes;
      let lo = 0;
      let hi = times.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (times[mid] < ts) lo = mid + 1;
        else hi = mid;
      }
      // lo is the first time >= ts; check if lo-1 is closer
      if (lo > 0 && ts - times[lo - 1] < times[lo] - ts) {
        return times[lo - 1] as UTCTimestamp;
      }
      return times[lo] as UTCTimestamp;
    }

    for (const trade of trades) {
      const entryTs = Math.floor(new Date(trade.entry_time).getTime() / 1000);
      const isLong = trade.direction?.toLowerCase() !== "short";

      result.push({
        time: snapToCandle(entryTs),
        position: "belowBar" as const,
        shape: "arrowUp" as const,
        color: "#26a69a",
        text: isLong ? "Long" : "Short",
        size: 5,
      });

      if (trade.exit_time && trade.exit_price != null) {
        const exitTs = Math.floor(new Date(trade.exit_time).getTime() / 1000);
        result.push({
          time: snapToCandle(exitTs),
          position: "aboveBar" as const,
          shape: "arrowDown" as const,
          color: "#ef5350",
          text: "Close",
          size: 5,
        });
      }
    }

    result.sort((a, b) => (a.time as number) - (b.time as number));
    return result;
  }, [tradesQuery.data, selectedSymbol, sortedCandleTimes]);

  // Create chart on mount
  // biome-ignore lint/correctness/useExhaustiveDependencies: mount-only
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, getChartOptions(theme, 450));
    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: CANDLE_COLORS.up,
      downColor: CANDLE_COLORS.down,
      borderVisible: false,
      wickUpColor: CANDLE_COLORS.up,
      wickDownColor: CANDLE_COLORS.down,
    });
    candleSeriesRef.current = candleSeries;

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    volumeSeriesRef.current = volumeSeries;
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    const seriesMarkers = createSeriesMarkers(candleSeries, [], {
      zOrder: "top",
    });
    markersRef.current = seriesMarkers;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width });
      }
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      markersRef.current = null;
    };
  }, []);

  // Update theme
  useEffect(() => {
    chartRef.current?.applyOptions(getChartOptions(theme, 450));
  }, [theme]);

  // Update candle + volume + markers together, reset scale on data change
  useEffect(() => {
    const candleSeries = candleSeriesRef.current;
    const volumeSeries = volumeSeriesRef.current;
    const chart = chartRef.current;
    if (!candleSeries || !volumeSeries || !chart) return;

    if (candles.length === 0) {
      candleSeries.setData([]);
      volumeSeries.setData([]);
      markersRef.current?.setMarkers([]);
      return;
    }

    candleSeries.setData(candles);
    volumeSeries.setData(volumeData);
    markersRef.current?.setMarkers(markers);

    // If we have markers, zoom to show the first trade with context; otherwise fit all
    if (markers.length > 0) {
      const firstMarkerTime = markers[0].time as number;
      const lastMarkerTime = markers[markers.length - 1].time as number;
      const span = Math.max(lastMarkerTime - firstMarkerTime, 3600 * 24); // min 1 day
      const padding = span * 0.2;
      chart.timeScale().setVisibleRange({
        from: (firstMarkerTime - padding) as UTCTimestamp,
        to: (lastMarkerTime + padding) as UTCTimestamp,
      });
    } else {
      chart.timeScale().fitContent();
    }
  }, [candles, volumeData, markers]);

  const isLoading = candleQuery.isLoading || tradesQuery.isLoading;

  if (!run) return null;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Trade Chart
        </h3>
        <div className="flex items-center gap-3">
          {/* Interval selector */}
          <div className="flex rounded-md border border-border">
            {INTERVALS.map((i) => (
              <button
                key={i}
                type="button"
                onClick={() => setIntervalState(i)}
                className={cn(
                  "px-2.5 py-1 text-xs font-medium transition-colors",
                  "first:rounded-l-md last:rounded-r-md",
                  interval === i
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                {i}
              </button>
            ))}
          </div>
          {/* Symbol selector */}
          {symbols.length > 1 && (
            <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Select symbol" />
              </SelectTrigger>
              <SelectContent>
                {symbols.map((s) => (
                  <SelectItem key={s} value={s}>
                    {s}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
      </div>
      {isLoading && (
        <div className="flex h-[450px] items-center justify-center text-muted-foreground">
          Loading chart data...
        </div>
      )}
      <div
        ref={containerRef}
        className={isLoading || candles.length === 0 ? "hidden" : ""}
      />
      {!isLoading && candles.length === 0 && selectedSymbol && (
        <div className="flex h-[450px] items-center justify-center text-muted-foreground">
          No candle data available for {selectedSymbol}
        </div>
      )}
    </div>
  );
}
