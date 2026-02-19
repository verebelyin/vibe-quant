interface DateRangePickerProps {
  startDate: string;
  endDate: string;
  onStartChange: (date: string) => void;
  onEndChange: (date: string) => void;
  className?: string;
}

function toISODate(date: Date): string {
  return date.toISOString().split("T")[0];
}

function getPresetRange(preset: string): [string, string] {
  const end = new Date();
  const start = new Date();

  switch (preset) {
    case "30d":
      start.setDate(end.getDate() - 30);
      break;
    case "90d":
      start.setDate(end.getDate() - 90);
      break;
    case "1y":
      start.setFullYear(end.getFullYear() - 1);
      break;
    case "ytd":
      start.setMonth(0, 1);
      break;
  }

  return [toISODate(start), toISODate(end)];
}

const presets = [
  { label: "Last 30d", key: "30d" },
  { label: "Last 90d", key: "90d" },
  { label: "Last 1y", key: "1y" },
  { label: "YTD", key: "ytd" },
] as const;

export function DateRangePicker({
  startDate,
  endDate,
  onStartChange,
  onEndChange,
  className = "",
}: DateRangePickerProps) {
  const handlePreset = (key: string) => {
    const [start, end] = getPresetRange(key);
    onStartChange(start);
    onEndChange(end);
  };

  const inputStyle = {
    backgroundColor: "hsl(var(--input))",
    borderColor: "hsl(var(--border))",
    color: "hsl(var(--foreground))",
  };

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="flex items-end gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium" style={{ color: "hsl(var(--muted-foreground))" }}>
            Start
          </label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => onStartChange(e.target.value)}
            className="rounded-md border px-2.5 py-1.5 text-sm outline-none focus:ring-2"
            style={{ ...inputStyle, "--tw-ring-color": "hsl(var(--ring))" } as React.CSSProperties}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium" style={{ color: "hsl(var(--muted-foreground))" }}>
            End
          </label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => onEndChange(e.target.value)}
            className="rounded-md border px-2.5 py-1.5 text-sm outline-none focus:ring-2"
            style={{ ...inputStyle, "--tw-ring-color": "hsl(var(--ring))" } as React.CSSProperties}
          />
        </div>
      </div>
      <div className="flex gap-1.5">
        {presets.map((p) => (
          <button
            key={p.key}
            type="button"
            onClick={() => handlePreset(p.key)}
            className="rounded-md px-2 py-1 text-xs font-medium transition-colors hover:brightness-90 dark:hover:brightness-110"
            style={{
              backgroundColor: "hsl(var(--accent))",
              color: "hsl(var(--accent-foreground))",
            }}
          >
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
}
