import type { BacktestResultResponse } from "@/api/generated/models";
import { useGetRunSummaryApiResultsRunsRunIdGet } from "@/api/generated/results/results";
import { PassBadge } from "@/components/results/PassBadge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface CrossWindowPanelProps {
  runId: number;
}

interface WindowRow {
  offset: number | null;
  sharpe: number | null;
  return_pct: number | null;
  max_dd: number | null;
  trades: number | null;
  passed: boolean | null;
}

export function CrossWindowPanel({ runId }: CrossWindowPanelProps) {
  const query = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const data = query.data?.data as BacktestResultResponse | undefined;
  const rows = (data?.cross_window_results as unknown as WindowRow[] | null) ?? null;

  if (!rows || rows.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Cross-Window Validation
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Offset (mo)</TableHead>
              <TableHead className="text-right">Sharpe</TableHead>
              <TableHead className="text-right">Return %</TableHead>
              <TableHead className="text-right">Max DD</TableHead>
              <TableHead className="text-right">Trades</TableHead>
              <TableHead>Pass</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r, i) => (
              <TableRow key={`${r.offset ?? i}-${i}`}>
                <TableCell className="font-mono text-xs">
                  {r.offset != null ? `+${r.offset}` : "--"}
                </TableCell>
                <TableCell className="text-right font-mono text-xs">
                  {r.sharpe != null ? r.sharpe.toFixed(2) : "--"}
                </TableCell>
                <TableCell className="text-right font-mono text-xs">
                  {r.return_pct != null ? `${(r.return_pct * 100).toFixed(2)}%` : "--"}
                </TableCell>
                <TableCell className="text-right font-mono text-xs">
                  {r.max_dd != null ? `${(r.max_dd * 100).toFixed(2)}%` : "--"}
                </TableCell>
                <TableCell className="text-right font-mono text-xs">
                  {r.trades ?? "--"}
                </TableCell>
                <TableCell>
                  <PassBadge passed={r.passed} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
