import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

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
  className,
}: DateRangePickerProps) {
  const handlePreset = (key: string) => {
    const [start, end] = getPresetRange(key);
    onStartChange(start);
    onEndChange(end);
  };

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex items-end gap-3">
        <div className="flex flex-col gap-1">
          <Label htmlFor="date-start">Start</Label>
          <Input
            id="date-start"
            type="date"
            value={startDate}
            onChange={(e) => onStartChange(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="date-end">End</Label>
          <Input
            id="date-end"
            type="date"
            value={endDate}
            onChange={(e) => onEndChange(e.target.value)}
          />
        </div>
      </div>
      <div className="flex gap-1.5">
        {presets.map((p) => (
          <Button key={p.key} variant="secondary" size="sm" onClick={() => handlePreset(p.key)}>
            {p.label}
          </Button>
        ))}
      </div>
    </div>
  );
}
