import { useEffect } from "react";
import { queryClient } from "../api/query-client";
import { useWebSocket, type WsMessage } from "./useWebSocket";

function handleMessage(msg: WsMessage): void {
  if (msg.type === "discovery_update") {
    queryClient.invalidateQueries({ queryKey: ["/api/discovery/jobs"] });
  }
}

export function useDiscoveryWS() {
  const ws = useWebSocket("discovery");

  useEffect(() => {
    if (ws.lastMessage) {
      handleMessage(ws.lastMessage);
    }
  }, [ws.lastMessage]);

  return ws;
}
