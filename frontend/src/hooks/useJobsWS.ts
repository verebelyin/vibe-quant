import { useEffect } from "react";
import { queryClient } from "../api/query-client";
import { useWebSocket, type WsMessage } from "./useWebSocket";

function handleMessage(msg: WsMessage): void {
  if (msg.type === "job_status") {
    queryClient.invalidateQueries({ queryKey: ["/api/backtest/jobs"] });
  } else if (msg.type === "job_complete") {
    queryClient.invalidateQueries({ queryKey: ["/api/backtest/jobs"] });
    queryClient.invalidateQueries({ queryKey: ["/api/results/runs"] });
  }
}

export function useJobsWS() {
  const ws = useWebSocket("jobs");

  useEffect(() => {
    if (ws.lastMessage) {
      handleMessage(ws.lastMessage);
    }
  }, [ws.lastMessage]);

  return ws;
}
