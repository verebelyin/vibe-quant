import { useEffect } from "react";
import { queryClient } from "../api/query-client";
import { useWebSocket, type WsMessage } from "./useWebSocket";

function handleMessage(msg: WsMessage): void {
  if (msg.type === "position_update") {
    queryClient.invalidateQueries({ queryKey: ["/api/paper/positions"] });
  } else if (msg.type === "pnl_update") {
    queryClient.invalidateQueries({ queryKey: ["/api/paper/status"] });
  }
}

export function useTradingWS() {
  const ws = useWebSocket("trading");

  useEffect(() => {
    if (ws.lastMessage) {
      handleMessage(ws.lastMessage);
    }
  }, [ws.lastMessage]);

  return ws;
}
