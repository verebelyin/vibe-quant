import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { DAYS_OF_WEEK, type DslConfig, SESSIONS } from "./types";

interface TimeTabProps {
  config: DslConfig;
  onConfigChange: (config: DslConfig) => void;
}

export function TimeTab({ config, onConfigChange }: TimeTabProps) {
  const updateTime = (patch: Partial<DslConfig["time"]>) => {
    onConfigChange({ ...config, time: { ...config.time, ...patch } });
  };

  const toggleDay = (day: string) => {
    const current = config.time.trading_days ?? [];
    const next = current.includes(day) ? current.filter((d) => d !== day) : [...current, day];
    updateTime({ trading_days: next.length > 0 ? next : undefined });
  };

  const toggleSession = (session: string) => {
    const current = config.time.sessions ?? [];
    const next = current.includes(session)
      ? current.filter((s) => s !== session)
      : [...current, session];
    updateTime({ sessions: next.length > 0 ? next : undefined });
  };

  return (
    <div className="space-y-6">
      {/* Trading Hours */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Trading Hours (UTC)</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <Label className="text-xs">Start</Label>
            <Input
              type="time"
              value={config.time.trading_hours?.start ?? ""}
              onChange={(e) =>
                updateTime({
                  trading_hours: {
                    start: e.target.value,
                    end: config.time.trading_hours?.end ?? "23:59",
                  },
                })
              }
            />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">End</Label>
            <Input
              type="time"
              value={config.time.trading_hours?.end ?? ""}
              onChange={(e) =>
                updateTime({
                  trading_hours: {
                    start: config.time.trading_hours?.start ?? "00:00",
                    end: e.target.value,
                  },
                })
              }
            />
          </div>
        </div>
        {config.time.trading_hours && (
          <button
            type="button"
            className="text-xs text-muted-foreground underline hover:text-foreground"
            onClick={() => updateTime({ trading_hours: undefined })}
          >
            Clear hours filter
          </button>
        )}
      </div>

      {/* Trading Days */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Trading Days</h3>
        <div className="flex flex-wrap gap-4">
          {DAYS_OF_WEEK.map((day) => (
            <div key={day} className="flex items-center gap-2 text-sm">
              <Checkbox
                id={`day-${day}`}
                checked={(config.time.trading_days ?? []).includes(day)}
                onCheckedChange={() => toggleDay(day)}
              />
              <Label htmlFor={`day-${day}`} className="text-sm font-normal">
                {day.slice(0, 3)}
              </Label>
            </div>
          ))}
        </div>
      </div>

      {/* Sessions */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Session Filters</h3>
        <div className="flex flex-wrap gap-4">
          {SESSIONS.map((session) => (
            <div key={session} className="flex items-center gap-2 text-sm">
              <Checkbox
                id={`session-${session}`}
                checked={(config.time.sessions ?? []).includes(session)}
                onCheckedChange={() => toggleSession(session)}
              />
              <Label htmlFor={`session-${session}`} className="text-sm font-normal">
                {session}
              </Label>
            </div>
          ))}
        </div>
      </div>

      {/* Funding avoidance */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Funding Rate</h3>
        <div className="flex items-center gap-2">
          <Checkbox
            id="funding-avoidance"
            checked={config.time.funding_avoidance ?? false}
            onCheckedChange={(v) => updateTime({ funding_avoidance: v === true })}
          />
          <Label htmlFor="funding-avoidance" className="text-sm font-normal">
            Avoid trades within 15 min of funding windows (00:00, 08:00, 16:00 UTC)
          </Label>
        </div>
      </div>
    </div>
  );
}
