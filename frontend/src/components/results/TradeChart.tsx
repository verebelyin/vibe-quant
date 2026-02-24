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
  highlightedTradeId?: number | null;
}

export function TradeChart({ runId, highlightedTradeId }: TradeChartProps) {
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

  // Snap a timestamp to the nearest candle time
  const snapToCandle = useMemo(() => {
    return (ts: number): UTCTimestamp => {
      const times = sortedCandleTimes;
      if (times.length === 0) return ts as UTCTimestamp;
      let lo = 0;
      let hi = times.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (times[mid] < ts) lo = mid + 1;
        else hi = mid;
      }
      if (lo > 0 && ts - times[lo - 1] < times[lo] - ts) {
        return times[lo - 1] as UTCTimestamp;
      }
      return times[lo] as UTCTimestamp;
    };
  }, [sortedCandleTimes]);

  // Map trade id → snapped entry/exit times
  const tradeTimeMap = useMemo(() => {
    const resp = tradesQuery.data;
    if (!resp || resp.status !== 200 || sortedCandleTimes.length === 0) return new Map<number, { entry: UTCTimestamp; exit: UTCTimestamp | null }>();
    const map = new Map<number, { entry: UTCTimestamp; exit: UTCTimestamp | null }>();
    for (const trade of resp.data.filter((t) => t.symbol.startsWith(selectedSymbol))) {
      const entryTs = Math.floor(new Date(trade.entry_time).getTime() / 1000);
      const exitTs = trade.exit_time ? Math.floor(new Date(trade.exit_time).getTime() / 1000) : null;
      map.set(trade.id, {
        entry: snapToCandle(entryTs),
        exit: exitTs != null ? snapToCandle(exitTs) : null,
      });
    }
    return map;
  }, [tradesQuery.data, selectedSymbol, sortedCandleTimes, snapToCandle]);

  const markers = useMemo(() => {
    const resp = tradesQuery.data;
    if (!resp || resp.status !== 200 || sortedCandleTimes.length === 0) return [];
    const trades = resp.data.filter((t) => t.symbol.startsWith(selectedSymbol));
    const result: SeriesMarker<UTCTimestamp>[] = [];
    const hlId = highlightedTradeId;
    const hasFocus = hlId != null;

    for (const trade of trades) {
      const times = tradeTimeMap.get(trade.id);
      if (!times) continue;
      const isLong = trade.direction?.toLowerCase() !== "short";
      const isHighlighted = trade.id === hlId;
      const dimmed = hasFocus && !isHighlighted;

      result.push({
        time: times.entry,
        position: "belowBar" as const,
        shape: "arrowUp" as const,
        color: dimmed ? "#26a69a40" : isHighlighted ? "#00ffcc" : "#26a69a",
        text: isHighlighted ? (isLong ? "Long" : "Short") : dimmed ? "" : isLong ? "Long" : "Short",
        size: isHighlighted ? 3 : 2,
      });

      if (times.exit != null && trade.exit_price != null) {
        result.push({
          time: times.exit,
          position: "aboveBar" as const,
          shape: "arrowDown" as const,
          color: dimmed ? "#ef535040" : isHighlighted ? "#ff6b6b" : "#ef5350",
          text: isHighlighted ? "Close" : dimmed ? "" : "Close",
          size: isHighlighted ? 3 : 2,
        });
      }
    }

    result.sort((a, b) => (a.time as number) - (b.time as number));
    return result;
  }, [tradesQuery.data, selectedSymbol, sortedCandleTimes, highlightedTradeId, tradeTimeMap]);

  const hasCandles = candles.length > 0;

  // Create/recreate chart when candles become available (container must be visible)
  // biome-ignore lint/correctness/useExhaustiveDependencies: recreate on data availability
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !hasCandles) return;

    // Tear down previous chart if any
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      markersRef.current = null;
    }

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
  }, [hasCandles, theme]);

  // Set candle + volume data and initial range (only when data changes, NOT on highlight)
  useEffect(() => {
    const candleSeries = candleSeriesRef.current;
    const volumeSeries = volumeSeriesRef.current;
    const chart = chartRef.current;
    const container = containerRef.current;
    if (!candleSeries || !volumeSeries || !chart || !container) return;

    if (candles.length === 0) {
      candleSeries.setData([]);
      volumeSeries.setData([]);
      return;
    }

    // Force chart to match container width (container may have been hidden on mount)
    chart.applyOptions({ width: container.clientWidth });

    candleSeries.setData(candles);
    volumeSeries.setData(volumeData);

    chart.timeScale().fitContent();
  }, [candles, volumeData]);

  // Update markers separately (runs on highlight change without touching zoom)
  useEffect(() => {
    markersRef.current?.setMarkers(markers);
  }, [markers]);

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
        className={isLoading || candles.length === 0 ? "hidden" : "h-[450px]"}
      />
      {!isLoading && candles.length === 0 && selectedSymbol && (
        <div className="flex h-[450px] items-center justify-center text-muted-foreground">
          No candle data available for {selectedSymbol}
        </div>
      )}
    </div>
  );
}
