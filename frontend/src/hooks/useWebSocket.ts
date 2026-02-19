import { useCallback, useEffect, useRef, useState } from "react";

export type WsStatus = "connecting" | "connected" | "disconnected";

export interface WsMessage {
  type: string;
  [key: string]: unknown;
}

interface UseWebSocketReturn {
  status: WsStatus;
  lastMessage: WsMessage | null;
  send: (data: unknown) => void;
}

const MAX_BACKOFF_MS = 30_000;

function getWsUrl(channel: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/${channel}`;
}

export function useWebSocket(channel: string): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmountedRef = useRef(false);
  const connectRef = useRef<() => void>(null);

  const [status, setStatus] = useState<WsStatus>("disconnected");
  const [lastMessage, setLastMessage] = useState<WsMessage | null>(null);

  const connect = useCallback(() => {
    if (unmountedRef.current) return;

    setStatus("connecting");
    const ws = new WebSocket(getWsUrl(channel));
    wsRef.current = ws;

    ws.onopen = () => {
      if (unmountedRef.current) {
        ws.close();
        return;
      }
      retryRef.current = 0;
      setStatus("connected");
    };

    ws.onmessage = (event: MessageEvent) => {
      let parsed: WsMessage;
      try {
        parsed = JSON.parse(event.data as string) as WsMessage;
      } catch {
        return;
      }

      // Respond to server heartbeat
      if (parsed.type === "ping") {
        ws.send(JSON.stringify({ type: "pong" }));
        return;
      }

      setLastMessage(parsed);
    };

    ws.onclose = () => {
      if (unmountedRef.current) return;
      setStatus("disconnected");
      // Schedule reconnect inline to avoid circular dependency
      const delay = Math.min(1000 * 2 ** retryRef.current, MAX_BACKOFF_MS);
      retryRef.current += 1;
      timerRef.current = setTimeout(() => connectRef.current?.(), delay);
    };

    ws.onerror = () => {
      // onclose fires after onerror, reconnect handled there
      ws.close();
    };
  }, [channel]);

  // Keep ref in sync so reconnect timer calls the latest connect
  connectRef.current = connect;

  const send = useCallback((data: unknown) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    unmountedRef.current = false;
    connect();

    return () => {
      unmountedRef.current = true;
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  return { status, lastMessage, send };
}
