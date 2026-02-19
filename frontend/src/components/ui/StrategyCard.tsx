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

export function StrategyCard({
  name,
  description,
  strategyType,
  version,
  onClick,
  className = "",
}: StrategyCardProps) {
  const content = (
    <>
      <CardHeader className="pb-0">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-sm leading-tight">{name}</CardTitle>
          {version !== undefined && (
            <span className="shrink-0 text-xs text-muted-foreground">v{version}</span>
          )}
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {strategyType && (
          <span className="mt-1.5 inline-block rounded-full bg-accent px-2 py-0.5 text-xs font-medium text-accent-foreground">
            {strategyType}
          </span>
        )}
        {description && (
          <p className="mt-2 line-clamp-2 text-xs text-muted-foreground">{description}</p>
        )}
      </CardContent>
    </>
  );

  if (onClick) {
    return (
      <Card
        className={cn("cursor-pointer transition-colors hover:bg-accent/50", className)}
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

  return <Card className={cn("transition-colors", className)}>{content}</Card>;
}
