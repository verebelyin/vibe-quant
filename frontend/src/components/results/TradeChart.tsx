import { useEffect, useMemo, useRef, useState } from "react";
import {
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  HistogramSeries,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type SeriesType,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import {
  useGetTradesApiResultsRunsRunIdTradesGet,
  useListRunsApiResultsRunsGet,
} from "@/api/generated/results/results";
import {
  useBrowseDataApiDataBrowseSymbolGet,
  useComputeIndicatorsEndpointApiDataIndicatorsSymbolGet,
} from "@/api/generated/data/data";
import { useGetStrategyApiStrategiesStrategyIdGet } from "@/api/generated/strategies/strategies";
import type { IndicatorSeries } from "@/api/generated/models";
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

// Color palette per indicator type
const INDICATOR_COLORS: Record<string, string> = {
  SMA: "#ff9800",
  EMA: "#2196f3",
  WMA: "#9c27b0",
  DEMA: "#00bcd4",
  TEMA: "#e91e63",
  VWAP: "#ffeb3b",
  RSI: "#7c4dff",
  CCI: "#ff9800",
  WILLR: "#e91e63",
  ROC: "#00bcd4",
  ATR: "#ff9800",
  MFI: "#9c27b0",
  OBV: "#2196f3",
};

// Multi-output color mapping
const OUTPUT_COLORS: Record<string, Record<string, string>> = {
  BBANDS: { upper: "#7c4dff", middle: "#7c4dff80", lower: "#7c4dff" },
  KC: { upper: "#ff9800", middle: "#ff980080", lower: "#ff9800" },
  DONCHIAN: { upper: "#2196f3", middle: "#2196f380", lower: "#2196f3" },
  MACD: { macd: "#2196f3", signal: "#ff9800", histogram: "#26a69a" },
  STOCH: { k: "#2196f3", d: "#ff9800" },
  ICHIMOKU: { conversion: "#2196f3", base: "#ff9800", span_a: "#26a69a", span_b: "#ef5350" },
};

function getSeriesColor(series: IndicatorSeries): string {
  const typeColors = OUTPUT_COLORS[series.indicator_type];
  if (typeColors) {
    return typeColors[series.output_name] ?? "#a0a0b0";
  }
  return INDICATOR_COLORS[series.indicator_type] ?? "#a0a0b0";
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
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const indicatorSeriesRefs = useRef<{ key: string; series: ISeriesApi<SeriesType> }[]>([]);

  const [hiddenIndicators, setHiddenIndicators] = useState<Set<string>>(new Set());

  // Get run metadata
  const runsQuery = useListRunsApiResultsRunsGet();
  const run = useMemo(() => {
    const resp = runsQuery.data;
    const runs = resp && resp.status === 200 ? resp.data.runs : [];
    return runs.find((r) => r.id === runId);
  }, [runsQuery.data, runId]);

  const symbols = useMemo(() => run?.symbols ?? [], [run?.symbols]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("");
  const [interval, setIntervalState] = useState<string>("");

  // Auto-select first symbol + default interval from run timeframe
  useEffect(() => {
    if (symbols.length > 0 && !symbols.includes(selectedSymbol)) {
      setSelectedSymbol(symbols[0]!);
    }
  }, [symbols, selectedSymbol]);

  useEffect(() => {
    if (run?.timeframe && !interval) {
      setIntervalState(run.timeframe);
    }
  }, [run?.timeframe, interval]);

  // Fetch strategy to get indicator configs (only for non-discovery runs)
  const strategyId = run?.strategy_id;
  const strategyQuery = useGetStrategyApiStrategiesStrategyIdGet(strategyId ?? 0, {
    query: { enabled: !!strategyId },
  });

  const indicatorConfigs = useMemo(() => {
    const resp = strategyQuery.data;
    if (!resp || resp.status !== 200) return [];
    const dsl = resp.data.dsl_config as Record<string, unknown> | null;
    if (!dsl) return [];
    const indicators = dsl.indicators;
    if (!Array.isArray(indicators)) return [];
    return indicators as Record<string, unknown>[];
  }, [strategyQuery.data]);

  // Fetch candle data
  const effectiveInterval = interval || run?.timeframe || "";
  const browseParams = {
    interval: effectiveInterval,
    ...(run?.start_date ? { start: run.start_date } : {}),
    ...(run?.end_date ? { end: run.end_date } : {}),
  };
  const candleQuery = useBrowseDataApiDataBrowseSymbolGet(
    selectedSymbol,
    browseParams,
    { query: { enabled: !!selectedSymbol && !!run && !!interval } },
  );

  // Fetch trades
  const tradesQuery = useGetTradesApiResultsRunsRunIdTradesGet(runId, undefined, {
    query: { enabled: !!run },
  });

  // Fetch indicators
  const indicatorsJson = useMemo(() => {
    if (indicatorConfigs.length === 0) return "[]";
    return JSON.stringify(indicatorConfigs);
  }, [indicatorConfigs]);

  const indicatorQuery = useComputeIndicatorsEndpointApiDataIndicatorsSymbolGet(
    selectedSymbol,
    {
      interval: effectiveInterval,
      ...(run?.start_date ? { start: run.start_date } : {}),
      ...(run?.end_date ? { end: run.end_date } : {}),
      indicators: indicatorsJson,
    },
    {
      query: {
        enabled: !!selectedSymbol && !!effectiveInterval && indicatorConfigs.length > 0,
      },
    },
  );

  const indicatorSeries = useMemo(() => {
    const resp = indicatorQuery.data;
    if (!resp || resp.status !== 200) return [];
    return resp.data.series;
  }, [indicatorQuery.data]);

  // Count oscillator panes for dynamic height
  const oscillatorPaneCount = useMemo(() => {
    const types = new Set<string>();
    for (const s of indicatorSeries) {
      if (s.pane === "oscillator" && !hiddenIndicators.has(`${s.indicator_type}-${s.output_name}`)) {
        types.add(s.indicator_type);
      }
    }
    return types.size;
  }, [indicatorSeries, hiddenIndicators]);

  const chartHeight = 450 + oscillatorPaneCount * 150;

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
        if (times[mid]! < ts) lo = mid + 1;
        else hi = mid;
      }
      if (lo > 0 && ts - times[lo - 1]! < times[lo]! - ts) {
        return times[lo - 1] as UTCTimestamp;
      }
      return times[lo] as UTCTimestamp;
    };
  }, [sortedCandleTimes]);

  // Map trade id -> snapped entry/exit times
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
    const result: SeriesMarker<Time>[] = [];
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

  // Create/recreate chart when candles become available
  // biome-ignore lint/correctness/useExhaustiveDependencies: recreate on data availability + indicator pane count
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
      indicatorSeriesRefs.current = [];
    }

    const chart = createChart(container, getChartOptions(theme === "system" ? "dark" : theme, chartHeight));
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

    // Add indicator series
    const indRefs: ISeriesApi<SeriesType>[] = [];

    // Group oscillators by indicator_type to assign panes
    const oscillatorTypes: string[] = [];
    for (const s of indicatorSeries) {
      if (
        s.pane === "oscillator" &&
        !oscillatorTypes.includes(s.indicator_type)
      ) {
        oscillatorTypes.push(s.indicator_type);
      }
    }

    // Create panes for each oscillator type
    const oscillatorPaneIndexMap = new Map<string, number>();
    for (const oscType of oscillatorTypes) {
      const pane = chart.addPane();
      oscillatorPaneIndexMap.set(oscType, pane.paneIndex());
    }

    for (const s of indicatorSeries) {
      const key = `${s.indicator_type}-${s.output_name}`;
      const hidden = hiddenIndicators.has(key);

      const color = getSeriesColor(s);
      const isHistogram = s.output_name === "histogram" && s.indicator_type === "MACD";
      const paneIndex = s.pane === "oscillator" ? oscillatorPaneIndexMap.get(s.indicator_type) : undefined;

      const data = s.data
        .filter((p) => p.value != null)
        .map((p) => ({
          time: Math.floor(p.time / 1000) as UTCTimestamp,
          value: p.value as number,
        }));

      if (isHistogram) {
        const histData = data.map((d) => ({
          ...d,
          color: d.value >= 0 ? "#26a69a" : "#ef5350",
        }));
        const series = chart.addSeries(HistogramSeries, {
          title: s.display_label,
          priceScaleId: `osc-${s.indicator_type}`,
          visible: !hidden,
        }, paneIndex);
        series.setData(histData);
        indRefs.push({ key, series });
      } else {
        const lineWidth = s.output_name === "middle" ? 1 : 2;
        const lineStyle = s.output_name === "middle" ? 2 : 0; // dashed for middle bands
        const opts: Record<string, unknown> = {
          color,
          lineWidth,
          lineStyle,
          title: s.display_label,
          lastValueVisible: false,
          priceLineVisible: false,
          visible: !hidden,
        };
        if (s.pane === "oscillator") {
          opts.priceScaleId = `osc-${s.indicator_type}`;
        }
        const series = chart.addSeries(LineSeries, opts, paneIndex);
        series.setData(data);
        indRefs.push({ key, series });
      }
    }

    indicatorSeriesRefs.current = indRefs;

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
      indicatorSeriesRefs.current = [];
    };
  }, [hasCandles, theme, indicatorSeries, chartHeight]);

  // Toggle indicator visibility without rebuilding chart
  useEffect(() => {
    for (const { key, series } of indicatorSeriesRefs.current) {
      series.applyOptions({ visible: !hiddenIndicators.has(key) });
    }
  }, [hiddenIndicators]);

  // Set candle + volume data and initial range
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

    chart.applyOptions({ width: container.clientWidth });

    candleSeries.setData(candles);
    volumeSeries.setData(volumeData);

    chart.timeScale().fitContent();
  }, [candles, volumeData]);

  // Update markers separately
  useEffect(() => {
    markersRef.current?.setMarkers(markers);
  }, [markers]);

  const toggleIndicator = (key: string) => {
    setHiddenIndicators((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  // Build unique indicator chips for legend
  const indicatorChips = useMemo(() => {
    const seen = new Set<string>();
    const chips: { key: string; label: string; color: string }[] = [];
    for (const s of indicatorSeries) {
      const key = `${s.indicator_type}-${s.output_name}`;
      if (seen.has(key)) continue;
      seen.add(key);
      chips.push({
        key,
        label: s.display_label,
        color: getSeriesColor(s),
      });
    }
    return chips;
  }, [indicatorSeries]);

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
      {/* Indicator legend/toggle chips */}
      {indicatorChips.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-1.5">
          {indicatorChips.map((chip) => {
            const isHidden = hiddenIndicators.has(chip.key);
            return (
              <button
                key={chip.key}
                type="button"
                onClick={() => toggleIndicator(chip.key)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium transition-all",
                  isHidden
                    ? "border-border/50 text-muted-foreground/50 line-through"
                    : "border-border text-foreground",
                )}
              >
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{
                    backgroundColor: isHidden ? "#555" : chip.color,
                  }}
                />
                {chip.label}
              </button>
            );
          })}
        </div>
      )}
      {isLoading && (
        <div
          className="flex items-center justify-center text-muted-foreground"
          style={{ height: chartHeight }}
        >
          Loading chart data...
        </div>
      )}
      <div
        ref={containerRef}
        className={isLoading || candles.length === 0 ? "hidden" : ""}
        style={{ height: chartHeight }}
      />
      {!isLoading && candles.length === 0 && selectedSymbol && (
        <div
          className="flex items-center justify-center text-muted-foreground"
          style={{ height: 450 }}
        >
          No candle data available for {selectedSymbol}
        </div>
      )}
    </div>
  );
}
