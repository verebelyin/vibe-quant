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

interface RegimeValidationPanelProps {
  runId: number;
}

interface RegimeRow {
  regime?: string;
  sharpe?: number | null;
  sharpe_ratio?: number | null;
  max_dd?: number | null;
  max_drawdown?: number | null;
  passed?: boolean | null;
}

export function RegimeValidationPanel({ runId }: RegimeValidationPanelProps) {
  const query = useGetRunSummaryApiResultsRunsRunIdGet(runId);
  const data = query.data?.data as BacktestResultResponse | undefined;
  const rows = (data?.cross_regime_results as unknown as RegimeRow[] | null) ?? null;

  if (!rows || rows.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Cross-Regime Validation
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Regime</TableHead>
              <TableHead className="text-right">Sharpe</TableHead>
              <TableHead className="text-right">Max DD</TableHead>
              <TableHead>Pass</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r, i) => {
              const sharpe = r.sharpe ?? r.sharpe_ratio ?? null;
              const dd = r.max_dd ?? r.max_drawdown ?? null;
              return (
                <TableRow key={`${r.regime ?? i}-${i}`}>
                  <TableCell className="font-mono text-xs">{r.regime ?? "--"}</TableCell>
                  <TableCell className="text-right font-mono text-xs">
                    {sharpe != null ? sharpe.toFixed(2) : "--"}
                  </TableCell>
                  <TableCell className="text-right font-mono text-xs">
                    {dd != null ? `${(dd * 100).toFixed(2)}%` : "--"}
                  </TableCell>
                  <TableCell>
                    <PassBadge passed={r.passed} />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
