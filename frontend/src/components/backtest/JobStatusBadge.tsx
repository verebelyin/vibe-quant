interface JobStatusBadgeProps {
  status: string;
}

const STATUS_STYLES: Record<string, { bg: string; text: string }> = {
  queued: { bg: "bg-gray-200", text: "text-gray-700" },
  running: { bg: "bg-blue-100", text: "text-blue-700" },
  completed: { bg: "bg-green-100", text: "text-green-700" },
  failed: { bg: "bg-red-100", text: "text-red-700" },
  cancelled: { bg: "bg-yellow-100", text: "text-yellow-700" },
};

const DEFAULT_STYLE = { bg: "bg-gray-100", text: "text-gray-600" };

export function JobStatusBadge({ status }: JobStatusBadgeProps) {
  const style = STATUS_STYLES[status] ?? DEFAULT_STYLE;

  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${style.bg} ${style.text}`}
    >
      {status}
    </span>
  );
}
