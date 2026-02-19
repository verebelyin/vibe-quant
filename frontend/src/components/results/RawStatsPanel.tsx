import { useState } from "react";
import { useGetRunSummaryApiResultsRunsRunIdGet } from "@/api/generated/results/results";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface RawStatsPanelProps {
  runId: number;
}

export function RawStatsPanel({ runId }: RawStatsPanelProps) {
  const [open, setOpen] = useState(false);
  const query = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const data = query.data?.data;

  if (query.isLoading || query.isError || !data) return null;

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Raw Stats
        </CardTitle>
        <Button variant="ghost" size="sm" onClick={() => setOpen(!open)}>
          {open ? "Collapse" : "Expand"}
        </Button>
      </CardHeader>
      {open && (
        <CardContent>
          <pre className="max-h-[500px] overflow-auto rounded-md bg-muted p-4 text-xs font-mono leading-relaxed text-foreground">
            {JSON.stringify(data, null, 2)}
          </pre>
        </CardContent>
      )}
    </Card>
  );
}
