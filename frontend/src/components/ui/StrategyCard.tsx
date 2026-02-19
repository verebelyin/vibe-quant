interface StrategyCardProps {
  name: string;
  description?: string;
  strategyType?: string;
  version?: number;
  onClick?: () => void;
  className?: string;
}

export function StrategyCard({
  name,
  description,
  strategyType,
  version,
  onClick,
  className = "",
}: StrategyCardProps) {
  const cardStyle = {
    backgroundColor: "hsl(var(--card))",
    color: "hsl(var(--card-foreground))",
    borderColor: "hsl(var(--border))",
  };

  const inner = (
    <>
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-sm font-bold leading-tight">{name}</h3>
        {version !== undefined && (
          <span className="shrink-0 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
            v{version}
          </span>
        )}
      </div>
      {strategyType && (
        <span
          className="mt-1.5 inline-block rounded-full px-2 py-0.5 text-xs font-medium"
          style={{
            backgroundColor: "hsl(var(--accent))",
            color: "hsl(var(--accent-foreground))",
          }}
        >
          {strategyType}
        </span>
      )}
      {description && (
        <p className="mt-2 line-clamp-2 text-xs" style={{ color: "hsl(var(--muted-foreground))" }}>
          {description}
        </p>
      )}
    </>
  );

  if (onClick) {
    return (
      <button
        type="button"
        onClick={onClick}
        className={`rounded-lg border p-4 text-left transition-colors cursor-pointer hover:brightness-95 dark:hover:brightness-110 ${className}`}
        style={cardStyle}
      >
        {inner}
      </button>
    );
  }

  return (
    <div className={`rounded-lg border p-4 transition-colors ${className}`} style={cardStyle}>
      {inner}
    </div>
  );
}
