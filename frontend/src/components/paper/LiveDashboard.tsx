import { useCallback, useEffect, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PaperPositionResponse } from "@/api/generated/models";
import {
  useGetOrdersApiPaperOrdersGet,
  useGetPositionsApiPaperPositionsGet,
} from "@/api/generated/paper/paper";
import { queryClient } from "@/api/query-client";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useWebSocket, type WsMessage } from "@/hooks/useWebSocket";
import { cn } from "@/lib/utils";

interface EquityPoint {
  time: string;
  equity: number;
}

interface OrderRecord {
  id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  timestamp: string;
  status: string;
}

const MAX_EQUITY_POINTS = 100;

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

interface EquityTooltipPayload {
  value: number;
  payload: EquityPoint;
}

function EquityTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: EquityTooltipPayload[];
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0];
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-sm shadow-md">
      <p className="text-muted-foreground">{point.payload.time}</p>
      <p className="font-medium text-foreground">{formatCurrency(point.value)}</p>
    </div>
  );
}

export function LiveDashboard() {
  const ws = useWebSocket("trading");
  const [pnl, setPnl] = useState<number | null>(null);
  const [prevPnl, setPrevPnl] = useState<number | null>(null);
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [flashedSymbols, setFlashedSymbols] = useState<Set<string>>(new Set());
  const flashTimerRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  // Positions from API (auto-refreshed by WS invalidation)
  const { data: posResp } = useGetPositionsApiPaperPositionsGet({
    query: { refetchInterval: 5_000 },
  });
  const positions: PaperPositionResponse[] = posResp?.status === 200 ? posResp.data : [];

  // Orders from API
  const { data: ordersResp } = useGetOrdersApiPaperOrdersGet({
    query: { refetchInterval: 10_000 },
  });
  const rawOrders = ordersResp?.status === 200 ? ordersResp.data : [];
  const recentOrders: OrderRecord[] = rawOrders.slice(0, 10).map((o, idx) => ({
    id: String(o.id ?? o.order_id ?? idx),
    symbol: String(o.symbol ?? "--"),
    side: String(o.side ?? o.direction ?? "--"),
    quantity: Number(o.quantity ?? o.qty ?? 0),
    price: Number(o.price ?? 0),
    timestamp: String(o.timestamp ?? o.created_at ?? ""),
    status: String(o.status ?? "filled"),
  }));

  const handleMessage = useCallback(
    (msg: WsMessage) => {
      if (msg.type === "pnl_update") {
        const newPnl = Number(msg.total_pnl ?? msg.pnl ?? 0);
        setPrevPnl(pnl);
        setPnl(newPnl);

        // Add equity point
        const equity = Number(msg.equity ?? msg.account_equity ?? 0);
        if (equity > 0) {
          setEquityHistory((prev) => {
            const next = [...prev, { time: new Date().toISOString(), equity }];
            return next.length > MAX_EQUITY_POINTS ? next.slice(-MAX_EQUITY_POINTS) : next;
          });
        }

        queryClient.invalidateQueries({ queryKey: ["/api/paper/status"] });
      } else if (msg.type === "position_update") {
        const symbol = String(msg.symbol ?? "");
        if (symbol) {
          setFlashedSymbols((prev) => new Set(prev).add(symbol));
          // Clear existing timer for this symbol
          const existing = flashTimerRef.current.get(symbol);
          if (existing) clearTimeout(existing);
          const timer = setTimeout(() => {
            setFlashedSymbols((prev) => {
              const next = new Set(prev);
              next.delete(symbol);
              return next;
            });
            flashTimerRef.current.delete(symbol);
          }, 600);
          flashTimerRef.current.set(symbol, timer);
        }
        queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
      }
    },
    [pnl],
  );

  useEffect(() => {
    if (ws.lastMessage) {
      handleMessage(ws.lastMessage);
    }
  }, [ws.lastMessage, handleMessage]);

  // Cleanup flash timers on unmount
  useEffect(() => {
    const timers = flashTimerRef.current;
    return () => {
      for (const timer of timers.values()) clearTimeout(timer);
      timers.clear();
    };
  }, []);

  const pnlTrend = pnl != null && prevPnl != null ? (pnl >= prevPnl ? "up" : "down") : null;

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-foreground">
          Live Dashboard
        </h3>
        <Badge variant="outline" className="font-mono text-xs">
          {ws.status}
        </Badge>
      </div>

      {/* PnL display */}
      <div className="flex items-center gap-4">
        <div>
          <p className="text-xs text-muted-foreground">Total PnL</p>
          <p
            className={cn(
              "font-mono text-3xl font-bold tracking-tight",
              pnl == null && "text-muted-foreground",
              pnl != null && pnl >= 0 && "text-green-600",
              pnl != null && pnl < 0 && "text-red-600",
            )}
          >
            {pnl != null ? (pnl >= 0 ? "+" : "") + formatCurrency(pnl) : "--"}
          </p>
        </div>
        {pnlTrend && (
          <span className={cn("text-lg", pnlTrend === "up" ? "text-green-500" : "text-red-500")}>
            {pnlTrend === "up" ? "\u2191" : "\u2193"}
          </span>
        )}
      </div>

      {/* Equity mini chart */}
      <div className="h-40">
        {equityHistory.length < 2 ? (
          <div className="flex h-full items-center justify-center">
            <p className="text-xs text-muted-foreground">Collecting equity data...</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={equityHistory} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="liveEquityGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#22c55e" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#22c55e" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
              <XAxis
                dataKey="time"
                tickFormatter={(v: string) => formatTime(v)}
                stroke="hsl(var(--muted-foreground))"
                fontSize={10}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                tickFormatter={(v: number) => formatCurrency(v)}
                stroke="hsl(var(--muted-foreground))"
                fontSize={10}
                tickLine={false}
                axisLine={false}
                width={70}
              />
              <Tooltip content={<EquityTooltip />} />
              <Area
                type="monotone"
                dataKey="equity"
                stroke="#22c55e"
                strokeWidth={1.5}
                fill="url(#liveEquityGrad)"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Open positions with live flash */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-muted-foreground">Open Positions</h4>
        {positions.length === 0 ? (
          <p className="text-xs text-muted-foreground">No open positions.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Symbol</TableHead>
                <TableHead className="text-xs">Side</TableHead>
                <TableHead className="text-xs">Entry</TableHead>
                <TableHead className="text-xs">Qty</TableHead>
                <TableHead className="text-xs">Lev</TableHead>
                <TableHead className="text-xs">uPnL</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {positions.map((pos) => {
                const isFlashing = flashedSymbols.has(pos.symbol);
                return (
                  <TableRow
                    key={`${pos.symbol}-${pos.direction}`}
                    className={cn(
                      "transition-colors duration-300",
                      isFlashing && "bg-yellow-50 dark:bg-yellow-900/20",
                    )}
                  >
                    <TableCell className="font-mono text-xs text-foreground">
                      {pos.symbol}
                    </TableCell>
                    <TableCell className="text-xs">
                      <span
                        className={cn(
                          "font-medium",
                          pos.direction.toLowerCase() === "long"
                            ? "text-green-600"
                            : "text-red-600",
                        )}
                      >
                        {pos.direction}
                      </span>
                    </TableCell>
                    <TableCell className="font-mono text-xs text-foreground">
                      {pos.entry_price.toFixed(2)}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-foreground">
                      {pos.quantity}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-foreground">
                      {pos.leverage}x
                    </TableCell>
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
                );
              })}
            </TableBody>
          </Table>
        )}
      </div>

      {/* Recent orders */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-muted-foreground">Recent Orders</h4>
        {recentOrders.length === 0 ? (
          <p className="text-xs text-muted-foreground">No orders yet.</p>
        ) : (
          <div className="max-h-48 overflow-y-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Time</TableHead>
                  <TableHead className="text-xs">Symbol</TableHead>
                  <TableHead className="text-xs">Side</TableHead>
                  <TableHead className="text-xs">Qty</TableHead>
                  <TableHead className="text-xs">Price</TableHead>
                  <TableHead className="text-xs">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recentOrders.map((order) => (
                  <TableRow key={order.id}>
                    <TableCell className="font-mono text-xs text-foreground">
                      {order.timestamp ? formatTime(order.timestamp) : "--"}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-foreground">
                      {order.symbol}
                    </TableCell>
                    <TableCell className="text-xs">
                      <span
                        className={cn(
                          "font-medium",
                          order.side.toLowerCase() === "buy" ? "text-green-600" : "text-red-600",
                        )}
                      >
                        {order.side}
                      </span>
                    </TableCell>
                    <TableCell className="font-mono text-xs text-foreground">
                      {order.quantity}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-foreground">
                      {order.price.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-xs text-foreground">{order.status}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>
    </div>
  );
}
