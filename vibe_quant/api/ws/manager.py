"""WebSocket connection manager for vibe-quant API."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL: float = 30.0


class ConnectionManager:
    def __init__(self) -> None:
        self._channels: dict[str, set[WebSocket]] = {}
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        if channel not in self._channels:
            self._channels[channel] = set()
        self._channels[channel].add(websocket)
        logger.debug("ws connect channel=%s clients=%d", channel, len(self._channels[channel]))

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        conns = self._channels.get(channel)
        if conns is not None:
            conns.discard(websocket)
            if not conns:
                del self._channels[channel]
        logger.debug("ws disconnect channel=%s", channel)

    async def broadcast(self, channel: str, data: dict[str, object]) -> None:
        conns = self._channels.get(channel)
        if not conns:
            return
        payload = json.dumps(data)
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_text(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            conns.discard(ws)
            logger.debug("ws removed dead connection channel=%s", channel)
        if not conns:
            del self._channels[channel]

    async def send_personal(self, websocket: WebSocket, data: dict[str, object]) -> None:
        payload = json.dumps(data)
        await websocket.send_text(payload)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            for channel, conns in list(self._channels.items()):
                dead: list[WebSocket] = []
                for ws in conns:
                    try:
                        await ws.send_json({"type": "ping"})
                    except Exception:  # noqa: BLE001
                        dead.append(ws)
                for ws in dead:
                    conns.discard(ws)
                    logger.debug("ws heartbeat removed stale connection channel=%s", channel)
                if not conns:
                    del self._channels[channel]
