import { format, parse } from "date-fns";
import { Calendar as CalendarIcon } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

interface DatePickerProps {
  /** ISO date string (YYYY-MM-DD) */
  value: string;
  onChange: (date: string) => void;
  placeholder?: string;
  className?: string;
  id?: string;
}

function toISODate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

export function DatePicker({
  value,
  onChange,
  placeholder = "Pick a date",
  className,
  id,
}: DatePickerProps) {
  const date = value ? parse(value, "yyyy-MM-dd", new Date()) : undefined;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          id={id}
          variant="outline"
          className={cn(
            "w-full justify-start text-left font-normal",
            !date && "text-muted-foreground",
            className,
          )}
        >
          <CalendarIcon className="mr-2 h-4 w-4" />
          {date ? format(date, "MMM d, yyyy") : <span>{placeholder}</span>}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <Calendar
          mode="single"
          selected={date}
          onSelect={(d) => {
            if (d) onChange(toISODate(d));
          }}
          initialFocus
        />
      </PopoverContent>
    </Popover>
  );
}
