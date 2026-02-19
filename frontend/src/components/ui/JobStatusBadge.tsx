interface JobStatusBadgeProps {
  status: string;
  className?: string;
}

const statusStyles: Record<string, string> = {
  queued: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  running: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300 animate-pulse",
  completed: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
  failed: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  cancelled: "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
};

const fallbackStyle = "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300";

export function JobStatusBadge({ status, className = "" }: JobStatusBadgeProps) {
  const normalized = status.toLowerCase();
  const style = statusStyles[normalized] ?? fallbackStyle;

  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${style} ${className}`}
    >
      {status}
    </span>
  );
}
