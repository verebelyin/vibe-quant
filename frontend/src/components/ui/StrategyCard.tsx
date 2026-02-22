import { cn } from "@/lib/utils";

interface StrategyCardProps {
  name: string;
  description?: string;
  strategyType?: string;
  version?: number;
  isActive?: boolean;
  timeframe?: string;
  symbols?: string[];
  indicatorCount?: number;
  updatedAt?: string;
  onClick?: () => void;
  className?: string;
}

interface TypeMeta {
  color: string;
  label: string;
  symbol: string;
}

const TYPE_META: Record<string, TypeMeta> = {
  momentum:       { color: "#06b6d4", label: "MOMENTUM",    symbol: "◆" },
  mean_reversion: { color: "#8b5cf6", label: "MEAN REV",    symbol: "◆" },
  breakout:       { color: "#f59e0b", label: "BREAKOUT",    symbol: "◆" },
  trend_following:{ color: "#10b981", label: "TREND",       symbol: "◆" },
  arbitrage:      { color: "#3b82f6", label: "ARBITRAGE",   symbol: "◆" },
  volatility:     { color: "#f43f5e", label: "VOLATILITY",  symbol: "◆" },
  dsl:            { color: "#06b6d4", label: "DSL",         symbol: "◆" },
  discovery:      { color: "#a78bfa", label: "DISCOVERY",   symbol: "◆" },
  strategy:       { color: "#06b6d4", label: "STRATEGY",    symbol: "◆" },
};

const DEFAULT_COLOR = "#22d3ee";

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 2) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function StrategyCard({
  name,
  description,
  strategyType,
  version,
  isActive,
  timeframe,
  symbols = [],
  indicatorCount,
  updatedAt,
  onClick,
  className,
}: StrategyCardProps) {
  const typeKey = strategyType?.toLowerCase() ?? "";
  const meta = TYPE_META[typeKey];
  const color = meta?.color ?? DEFAULT_COLOR;
  const label = meta?.label ?? strategyType?.replace(/_/g, " ").toUpperCase() ?? "STRATEGY";

  const symbolLabel =
    symbols.length === 0
      ? null
      : symbols.length <= 2
        ? symbols.map((s) => s.replace(/-PERP$/, "").replace(/\/USDT$/, "")).join(" · ")
        : `${symbols.slice(0, 2).map((s) => s.replace(/-PERP$/, "").replace(/\/USDT$/, "")).join(" · ")} +${symbols.length - 2}`;

  return (
    <div
      style={{ "--type-color": color } as React.CSSProperties}
      className={cn(
        "group relative flex h-[164px] flex-col overflow-hidden rounded-xl",
        "border border-white/[0.07]",
        "transition-all duration-200 ease-out",
        onClick && "cursor-pointer hover:border-white/[0.14] hover:-translate-y-[2px] hover:shadow-2xl hover:shadow-black/50",
        className,
      )}
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onClick();
              }
            }
          : undefined
      }
    >
      {/* Base background */}
      <div className="absolute inset-0 bg-card" />

      {/* Radial type-color wash — always present, subtle */}
      <div
        style={{
          background: `radial-gradient(ellipse at 0% -20%, ${color}18 0%, transparent 65%)`,
        }}
        className="absolute inset-0 pointer-events-none"
      />

      {/* Hover: intensify the wash */}
      <div
        style={{
          background: `radial-gradient(ellipse at 0% -20%, ${color}28 0%, transparent 65%)`,
        }}
        className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200"
      />

      {/* Bottom accent line — gradient from type color, left to right */}
      <div
        style={{
          background: `linear-gradient(to right, ${color}90, ${color}20, transparent)`,
        }}
        className="absolute bottom-0 left-0 right-0 h-[2px] pointer-events-none"
      />

      {/* Content */}
      <div className="relative flex flex-1 flex-col px-4 py-3.5 z-10">

        {/* Row 1: type label + live indicator + version */}
        <div className="flex items-center justify-between gap-2 mb-2.5">
          <div className="flex items-center gap-1.5">
            <span
              style={{ color }}
              className="text-[9px] font-black tracking-[0.2em] uppercase"
            >
              {label}
            </span>
            {isActive && (
              <span className="inline-flex items-center gap-0.5 rounded-sm px-1 py-[1px] text-[7px] font-bold tracking-widest uppercase"
                style={{ color, border: `1px solid ${color}40`, background: `${color}14` }}
              >
                <span className="relative flex h-1 w-1 shrink-0">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ backgroundColor: color }} />
                  <span className="relative inline-flex rounded-full h-1 w-1" style={{ backgroundColor: color }} />
                </span>
                live
              </span>
            )}
          </div>
          {version !== undefined && (
            <span className="font-mono text-[10px] text-white/25 tabular-nums shrink-0">
              v{version}
            </span>
          )}
        </div>

        {/* Row 2: strategy name — the focal point */}
        <h3 className="text-[13px] font-bold leading-snug text-white/88 line-clamp-1 tracking-tight">
          {name}
        </h3>

        {/* Row 3: meta pills */}
        {(timeframe || symbolLabel || (indicatorCount !== undefined && indicatorCount > 0)) && (
          <div className="flex flex-wrap items-center gap-1 mt-2">
            {timeframe && (
              <span
                className="font-mono text-[9px] px-1.5 py-0.5 rounded-sm border border-white/[0.08] text-white/40"
                style={{ backgroundColor: `${color}0a` }}
              >
                {timeframe}
              </span>
            )}
            {symbolLabel && (
              <span
                className="font-mono text-[9px] px-1.5 py-0.5 rounded-sm border border-white/[0.08] text-white/40"
                style={{ backgroundColor: `${color}0a` }}
              >
                {symbolLabel}
              </span>
            )}
            {indicatorCount !== undefined && indicatorCount > 0 && (
              <span
                className="font-mono text-[9px] px-1.5 py-0.5 rounded-sm border border-white/[0.08] text-white/40"
                style={{ backgroundColor: `${color}0a` }}
              >
                {indicatorCount} ind{indicatorCount !== 1 ? "s" : ""}
              </span>
            )}
          </div>
        )}

        {/* Footer: description + updated time */}
        <div className="mt-auto pt-1.5">
          {description && (
            <p className="text-[10px] leading-snug text-white/38 line-clamp-1 mb-0.5">
              {description}
            </p>
          )}
          {updatedAt && (
            <p className="text-[9px] font-mono text-white/20 tabular-nums">
              {relativeTime(updatedAt)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
