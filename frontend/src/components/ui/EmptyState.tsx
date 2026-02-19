import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  title: string;
  description?: string;
  action?: { label: string; onClick: () => void };
  className?: string;
}

export function EmptyState({ title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn("flex flex-col items-center justify-center py-16 text-center", className)}>
      {/* Decorative ring */}
      <div className="relative mb-5">
        <div className="size-14 rounded-full border border-dashed border-muted-foreground/15" />
        <div className="absolute inset-2 rounded-full border border-muted-foreground/10" />
        <div className="absolute inset-[18px] rounded-full bg-primary/8" />
      </div>
      <h3 className="text-sm font-semibold text-foreground/80">{title}</h3>
      {description && (
        <p className="mt-1.5 max-w-xs text-xs leading-relaxed text-muted-foreground/60">
          {description}
        </p>
      )}
      {action && (
        <Button size="sm" className="mt-5" onClick={action.onClick}>
          {action.label}
        </Button>
      )}
    </div>
  );
}
