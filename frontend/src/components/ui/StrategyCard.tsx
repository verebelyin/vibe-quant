import { cn } from "@/lib/utils";

export interface StrategyCardProps {
  name: string;
  description?: string | undefined;
  strategyType?: string | undefined;
  version?: number | undefined;
  isActive?: boolean | undefined;
  timeframe?: string | undefined;
  symbols?: string[] | undefined;
  indicatorCount?: number | undefined;
  updatedAt?: string | undefined;
  index?: number | undefined;
  onClick?: (() => void) | undefined;
  className?: string | undefined;
}

export const TYPE_META: Record<string, { color: string; label: string }> = {
  momentum:        { color: "#0ff",    label: "MOMENTUM"   },
  mean_reversion:  { color: "#a78bfa", label: "MEAN·REV"   },
  breakout:        { color: "#fbbf24", label: "BREAKOUT"   },
  trend_following: { color: "#34d399", label: "TREND"      },
  arbitrage:       { color: "#60a5fa", label: "ARB"        },
  volatility:      { color: "#fb7185", label: "VOL"        },
  dsl:             { color: "#22d3ee", label: "DSL"        },
  discovery:       { color: "#c4b5fd", label: "DISCOVERY"  },
  strategy:        { color: "#22d3ee", label: "STRATEGY"   },
};

const DEFAULT: typeof TYPE_META[string] = { color: "#22d3ee", label: "STRATEGY" };

function relativeTime(iso: string): string {
  const d = Date.now() - new Date(iso).getTime();
  const m = Math.floor(d / 60000);
  if (m < 2) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const days = Math.floor(h / 24);
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
  index = 0,
  onClick,
  className,
}: StrategyCardProps) {
  const key = strategyType?.toLowerCase() ?? "";
  const { color, label } = TYPE_META[key] ?? DEFAULT;

  const symbolStr =
    symbols.length === 0 ? null
    : symbols.length <= 2
      ? symbols.map(s => s.replace(/-PERP$/, "").replace(/\/USDT$/, "")).join(" · ")
      : `${symbols.slice(0, 2).map(s => s.replace(/-PERP$/, "").replace(/\/USDT$/, "")).join(" · ")} +${symbols.length - 2}`;

  return (
    <div
      className={cn(
        // Stagger entrance
        "animate-in fade-in-0 slide-in-from-bottom-2 fill-mode-both",
        // Base shape — sharp corners for terminal aesthetic
        "group relative flex h-[172px] flex-col overflow-hidden rounded-sm",
        // Border: thin all-around + thicker left accent
        "border border-white/[0.07] border-l-[2px]",
        "transition-all duration-150 ease-out",
        onClick && "cursor-pointer hover:-translate-y-[2px]",
        className,
      )}
      style={{
        borderLeftColor: color,
        animationDelay: `${index * 45}ms`,
        animationDuration: "320ms",
      }}
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); } } : undefined}
    >
      {/* ── Base bg ─────────────────────────────────────────── */}
      <div className="absolute inset-0 bg-[oklch(0.16_0.01_260)]" />

      {/* ── CRT scan-line texture — the terminal signature ──── */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage:
            "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.013) 2px, rgba(255,255,255,0.013) 3px)",
        }}
      />

      {/* ── Left-edge glow: bleeds in on hover ───────────────── */}
      <div
        className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200"
        style={{
          background: `linear-gradient(to right, ${color}20 0%, transparent 55%)`,
        }}
      />

      {/* ── Top-edge accent line ─────────────────────────────── */}
      <div
        className="absolute top-0 left-0 right-0 h-px pointer-events-none"
        style={{ background: `linear-gradient(to right, ${color}50, transparent 70%)` }}
      />

      {/* ── Hover border brightening ─────────────────────────── */}
      <div
        className="absolute inset-0 rounded-[inherit] pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-150"
        style={{ boxShadow: `0 0 0 1px rgba(255,255,255,0.09)` }}
      />

      {/* ── Content ──────────────────────────────────────────── */}
      <div className="relative z-10 flex flex-1 flex-col px-4 py-3.5">

        {/* Row 1 — type label · live · version */}
        <div className="flex items-center justify-between mb-2.5">
          <div className="flex items-center gap-1.5">
            <span
              className="font-mono text-[11px] font-black tracking-[0.18em] uppercase"
              style={{ color }}
            >
              {label}
            </span>
            {isActive && (
              <span
                className="inline-flex items-center gap-[3px] rounded-[2px] px-[6px] py-[2px] font-mono text-[9px] font-bold tracking-[0.12em] uppercase"
                style={{
                  color,
                  background: `${color}14`,
                  border: `1px solid ${color}30`,
                }}
              >
                <span className="relative flex h-[6px] w-[6px] shrink-0">
                  <span
                    className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-70"
                    style={{ backgroundColor: color }}
                  />
                  <span
                    className="relative inline-flex rounded-full h-[6px] w-[6px]"
                    style={{ backgroundColor: color }}
                  />
                </span>
                live
              </span>
            )}
          </div>
          {version !== undefined && (
            <span className="font-mono text-[11px] tabular-nums" style={{ color: "rgba(255,255,255,0.2)" }}>
              v{version}
            </span>
          )}
        </div>

        {/* Row 2 — strategy name */}
        <h3
          className="text-[15px] font-semibold leading-snug line-clamp-1 transition-colors duration-150"
          style={{ color: "rgba(255,255,255,0.88)" }}
        >
          {name}
        </h3>

        {/* Row 3 — meta chips */}
        {(timeframe || symbolStr || (indicatorCount !== undefined && indicatorCount > 0)) && (
          <div className="flex flex-wrap items-center gap-[5px] mt-[9px]">
            {timeframe && (
              <span
                className="font-mono text-[11px] px-1.5 py-0.5 rounded-[2px] border"
                style={{
                  color: `${color}99`,
                  borderColor: `${color}22`,
                  background: `${color}0a`,
                }}
              >
                {timeframe}
              </span>
            )}
            {symbolStr && (
              <span
                className="font-mono text-[11px] px-1.5 py-0.5 rounded-[2px] border"
                style={{
                  color: `${color}99`,
                  borderColor: `${color}22`,
                  background: `${color}0a`,
                }}
              >
                {symbolStr}
              </span>
            )}
            {indicatorCount !== undefined && indicatorCount > 0 && (
              <span
                className="font-mono text-[11px] px-1.5 py-0.5 rounded-[2px] border"
                style={{
                  color: `${color}99`,
                  borderColor: `${color}22`,
                  background: `${color}0a`,
                }}
              >
                {indicatorCount} ind{indicatorCount !== 1 ? "s" : ""}
              </span>
            )}
          </div>
        )}

        {/* Footer — description · timestamp */}
        <div className="mt-auto pt-1.5">
          {description && (
            <p className="font-mono text-[12px] leading-snug line-clamp-1 mb-0.5" style={{ color: "rgba(255,255,255,0.38)" }}>
              {description}
            </p>
          )}
          {updatedAt && (
            <p className="font-mono text-[11px] tabular-nums" style={{ color: "rgba(255,255,255,0.22)" }}>
              {relativeTime(updatedAt)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
