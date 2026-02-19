import { useMemo } from "react";
import type { PaperPositionResponse } from "@/api/generated/models";
import { useGetPositionsApiPaperPositionsGet } from "@/api/generated/paper/paper";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

export function PositionsTable() {
  const { data: posResp, isLoading } = useGetPositionsApiPaperPositionsGet({
    query: { refetchInterval: 5_000 },
  });

  const positions: PaperPositionResponse[] = useMemo(() => {
    if (!posResp) return [];
    if (posResp.status === 200) return posResp.data;
    return [];
  }, [posResp]);

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-semibold text-foreground">Open Positions</h2>

      {isLoading ? (
        <p className="text-xs text-muted-foreground">Loading positions...</p>
      ) : positions.length === 0 ? (
        <p className="text-xs text-muted-foreground">No open positions.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-xs">Symbol</TableHead>
              <TableHead className="text-xs">Side</TableHead>
              <TableHead className="text-xs">Entry Price</TableHead>
              <TableHead className="text-xs">Quantity</TableHead>
              <TableHead className="text-xs">Leverage</TableHead>
              <TableHead className="text-xs">Unrealized PnL</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {positions.map((pos) => (
              <TableRow key={`${pos.symbol}-${pos.direction}`}>
                <TableCell className="font-mono text-xs text-foreground">{pos.symbol}</TableCell>
                <TableCell className="text-xs text-foreground">
                  <span
                    className={cn(
                      "font-medium",
                      pos.direction.toLowerCase() === "long" ? "text-green-600" : "text-red-600",
                    )}
                  >
                    {pos.direction}
                  </span>
                </TableCell>
                <TableCell className="font-mono text-xs text-foreground">
                  {pos.entry_price.toFixed(2)}
                </TableCell>
                <TableCell className="font-mono text-xs text-foreground">{pos.quantity}</TableCell>
                <TableCell className="font-mono text-xs text-foreground">{pos.leverage}x</TableCell>
                <TableCell
                  className={cn(
                    "font-mono text-xs font-medium",
                    pos.unrealized_pnl >= 0 ? "text-green-600" : "text-red-600",
                  )}
                >
                  {pos.unrealized_pnl >= 0 ? "+" : ""}
                  {pos.unrealized_pnl.toFixed(2)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
