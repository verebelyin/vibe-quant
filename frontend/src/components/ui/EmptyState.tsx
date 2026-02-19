interface EmptyStateProps {
  title: string;
  description?: string;
  action?: { label: string; onClick: () => void };
  className?: string;
}

export function EmptyState({ title, description, action, className = "" }: EmptyStateProps) {
  return (
    <div className={`flex flex-col items-center justify-center py-12 text-center ${className}`}>
      <h3 className="text-sm font-semibold" style={{ color: "hsl(var(--muted-foreground))" }}>
        {title}
      </h3>
      {description && (
        <p
          className="mt-1 max-w-sm text-xs"
          style={{ color: "hsl(var(--muted-foreground))", opacity: 0.75 }}
        >
          {description}
        </p>
      )}
      {action && (
        <button
          type="button"
          onClick={action.onClick}
          className="mt-4 rounded-md px-3 py-1.5 text-xs font-medium transition-colors hover:brightness-90 dark:hover:brightness-110"
          style={{
            backgroundColor: "hsl(var(--primary))",
            color: "hsl(var(--primary-foreground))",
          }}
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
