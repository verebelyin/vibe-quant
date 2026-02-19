interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg";
  className?: string;
}

const sizeMap = {
  sm: 16,
  md: 24,
  lg: 40,
} as const;

export function LoadingSpinner({ size = "md", className = "" }: LoadingSpinnerProps) {
  const px = sizeMap[size];

  return (
    <output
      aria-label="Loading"
      className={`inline-block animate-spin rounded-full border-2 border-current border-t-transparent ${className}`}
      style={{
        width: px,
        height: px,
        color: "hsl(var(--muted-foreground))",
      }}
    >
      <span className="sr-only">Loading</span>
    </output>
  );
}
