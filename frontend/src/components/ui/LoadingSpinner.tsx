import { cn } from "@/lib/utils";

interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg";
  className?: string;
}

const sizeMap = {
  sm: "h-4 w-4",
  md: "h-6 w-6",
  lg: "h-10 w-10",
} as const;

export function LoadingSpinner({ size = "md", className }: LoadingSpinnerProps) {
  return (
    <output
      aria-label="Loading"
      className={cn(
        "inline-block animate-spin rounded-full border-2 border-current border-t-transparent text-muted-foreground",
        sizeMap[size],
        className,
      )}
    >
      <span className="sr-only">Loading</span>
    </output>
  );
}
