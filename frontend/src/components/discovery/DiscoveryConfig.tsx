import { useState } from "react";
import { toast } from "sonner";
import {
  useGetIndicatorPoolApiDiscoveryIndicatorPoolGet,
  useLaunchDiscoveryApiDiscoveryLaunchPost,
} from "@/api/generated/discovery/discovery";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function DiscoveryConfig() {
  // GA parameters
  const [population, setPopulation] = useState(50);
  const [generations, setGenerations] = useState(100);
  const [crossoverRate, setCrossoverRate] = useState(0.8);
  const [mutationRate, setMutationRate] = useState(0.1);
  const [eliteCount, setEliteCount] = useState(5);
  const [tournamentSize, setTournamentSize] = useState(3);

  // Target config
  const [symbols, setSymbols] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState("1h");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  // Indicator pool
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>([]);

  const indicatorPoolQuery = useGetIndicatorPoolApiDiscoveryIndicatorPoolGet();
  const launchMutation = useLaunchDiscoveryApiDiscoveryLaunchPost();

  const indicators: Array<{ name: string; [key: string]: unknown }> =
    indicatorPoolQuery.data?.status === 200
      ? (indicatorPoolQuery.data.data as Array<{ name: string; [key: string]: unknown }>)
      : [];

  const indicatorNames = indicators.map((i) => String(i.name ?? i));

  function handleToggleIndicator(name: string) {
    setSelectedIndicators((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name],
    );
  }

  function handleSelectAll() {
    setSelectedIndicators(
      selectedIndicators.length === indicatorNames.length ? [] : [...indicatorNames],
    );
  }

  function handleLaunch() {
    const symbolList = symbols
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (symbolList.length === 0) {
      toast.error("Enter at least one symbol");
      return;
    }

    launchMutation.mutate(
      {
        data: {
          population,
          generations,
          mutation_rate: mutationRate,
          symbols: symbolList,
          timeframes: [timeframe],
          indicator_pool: selectedIndicators.length > 0 ? selectedIndicators : null,
          ...(startDate && { start_date: startDate }),
          ...(endDate && { end_date: endDate }),
        } as Record<string, unknown>,
      },
      {
        onSuccess: (resp) => {
          if (resp.status === 201) {
            toast.success("Discovery launched", {
              description: `Run ID: ${resp.data.run_id}`,
            });
          }
        },
        onError: (err: unknown) => {
          const message = err instanceof Error ? err.message : "Launch failed";
          toast.error("Discovery launch failed", { description: message });
        },
      },
    );
  }

  return (
    <div className="space-y-6">
      {/* GA Parameters */}
      <div className="space-y-4 rounded-lg border border-border bg-card p-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
          GA Parameters
        </h3>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="population">Population Size</Label>
            <Input
              id="population"
              type="number"
              min={10}
              value={population}
              onChange={(e) => setPopulation(Number(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="generations">Generations</Label>
            <Input
              id="generations"
              type="number"
              min={1}
              value={generations}
              onChange={(e) => setGenerations(Number(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="crossover-rate">Crossover Rate</Label>
            <Input
              id="crossover-rate"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={crossoverRate}
              onChange={(e) => setCrossoverRate(Number(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="mutation-rate">Mutation Rate</Label>
            <Input
              id="mutation-rate"
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={mutationRate}
              onChange={(e) => setMutationRate(Number(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="elite-count">Elite Count</Label>
            <Input
              id="elite-count"
              type="number"
              min={0}
              value={eliteCount}
              onChange={(e) => setEliteCount(Number(e.target.value))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="tournament-size">Tournament Size</Label>
            <Input
              id="tournament-size"
              type="number"
              min={2}
              value={tournamentSize}
              onChange={(e) => setTournamentSize(Number(e.target.value))}
            />
          </div>
        </div>
      </div>

      {/* Indicator Pool */}
      <div className="space-y-4 rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
            Indicator Pool
          </h3>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              {selectedIndicators.length}/{indicatorNames.length}
            </Badge>
            <Button
              type="button"
              variant="link"
              size="xs"
              onClick={handleSelectAll}
              disabled={indicatorNames.length === 0}
            >
              {selectedIndicators.length === indicatorNames.length ? "Deselect All" : "Select All"}
            </Button>
          </div>
        </div>

        {indicatorPoolQuery.isLoading && (
          <p className="text-xs text-muted-foreground">Loading indicators...</p>
        )}
        {indicatorNames.length === 0 && !indicatorPoolQuery.isLoading && (
          <p className="text-xs italic text-muted-foreground">No indicators available.</p>
        )}

        <div className="max-h-48 overflow-y-auto rounded-md border border-border bg-input p-2 dark:bg-input/30">
          {indicatorNames.map((name) => (
            <div
              key={name}
              className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm text-foreground transition-colors hover:opacity-80"
            >
              <Checkbox
                id={`ind-${name}`}
                checked={selectedIndicators.includes(name)}
                onCheckedChange={() => handleToggleIndicator(name)}
              />
              <Label htmlFor={`ind-${name}`} className="cursor-pointer font-mono text-xs">
                {name}
              </Label>
            </div>
          ))}
        </div>
      </div>

      {/* Target Config */}
      <div className="space-y-4 rounded-lg border border-border bg-card p-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
          Target Config
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="symbols">Symbols (comma-separated)</Label>
            <Input
              id="symbols"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              placeholder="BTCUSDT, ETHUSDT"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="disc-timeframe">Timeframe</Label>
            <Select value={timeframe} onValueChange={setTimeframe}>
              <SelectTrigger id="disc-timeframe" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1m">1 minute</SelectItem>
                <SelectItem value="5m">5 minutes</SelectItem>
                <SelectItem value="15m">15 minutes</SelectItem>
                <SelectItem value="1h">1 hour</SelectItem>
                <SelectItem value="4h">4 hours</SelectItem>
                <SelectItem value="1d">1 day</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="disc-start-date">Start Date</Label>
            <Input
              id="disc-start-date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="disc-end-date">End Date</Label>
            <Input
              id="disc-end-date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
        </div>
      </div>

      {/* Launch */}
      <Button
        type="button"
        className="w-full py-3 font-semibold"
        disabled={launchMutation.isPending}
        onClick={handleLaunch}
      >
        {launchMutation.isPending ? "Launching..." : "Launch Discovery"}
      </Button>
    </div>
  );
}
