import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface StrategyCardProps {
  name: string;
  description?: string;
  strategyType?: string;
  version?: number;
  onClick?: () => void;
  className?: string;
}

const typeAccent: Record<string, string> = {
  momentum: "from-cyan-500/20 to-blue-500/20",
  "mean-reversion": "from-violet-500/20 to-purple-500/20",
  "mean_reversion": "from-violet-500/20 to-purple-500/20",
  breakout: "from-amber-500/20 to-orange-500/20",
  trend: "from-emerald-500/20 to-teal-500/20",
  scalping: "from-rose-500/20 to-pink-500/20",
};

const typeBadge: Record<string, string> = {
  momentum: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
  "mean-reversion": "bg-violet-500/10 text-violet-400 border-violet-500/20",
  "mean_reversion": "bg-violet-500/10 text-violet-400 border-violet-500/20",
  breakout: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  trend: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  scalping: "bg-rose-500/10 text-rose-400 border-rose-500/20",
};

export function StrategyCard({
  name,
  description,
  strategyType,
  version,
  onClick,
  className = "",
}: StrategyCardProps) {
  const typeKey = strategyType?.toLowerCase() ?? "";
  const gradient = typeAccent[typeKey] ?? "from-primary/10 to-primary/5";
  const badgeClass = typeBadge[typeKey] ?? "";

  const content = (
    <>
      {/* Gradient accent strip */}
      <div
        className={cn(
          "h-0.5 w-full rounded-t-[inherit] bg-gradient-to-r",
          gradient,
        )}
      />
      <CardHeader className="pb-0">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-sm font-semibold leading-tight">{name}</CardTitle>
          {version !== undefined && (
            <span className="shrink-0 rounded bg-muted/50 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
              v{version}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {strategyType && (
          <Badge
            variant="secondary"
            className={cn("mt-1.5 border text-[10px] font-medium uppercase tracking-wider", badgeClass)}
          >
            {strategyType}
          </Badge>
        )}
        {description && (
          <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-muted-foreground/80">
            {description}
          </p>
        )}
      </CardContent>
    </>
  );

  if (onClick) {
    return (
      <Card
        className={cn(
          "cursor-pointer overflow-hidden transition-all hover:border-primary/20",
          className,
        )}
        onClick={onClick}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onClick();
          }
        }}
      >
        {content}
      </Card>
    );
  }

  return (
    <Card className={cn("overflow-hidden transition-colors", className)}>
      {content}
    </Card>
  );
}
