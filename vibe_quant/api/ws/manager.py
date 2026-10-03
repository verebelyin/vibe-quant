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
        """Stop the ping task. Never raises (runs in lifespan shutdown)."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._heartbeat_task
            self._heartbeat_task = None

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        self._channels.setdefault(channel, set()).add(websocket)
        logger.debug("ws connect channel=%s clients=%d", channel, len(self._channels[channel]))

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        conns = self._channels.get(channel)
        if conns is not None:
            conns.discard(websocket)
            if not conns:
                self._channels.pop(channel, None)
        logger.debug("ws disconnect channel=%s", channel)

    async def _send_all(self, channel: str, payload: str) -> None:
        """Send to every socket of ``channel``; drop the ones that fail.

        Iterates a snapshot: sends await, and connect/disconnect may mutate the
        live set meanwhile ("Set changed size during iteration" used to turn a
        successful job launch into a 500 — inviting a duplicate launch).
        """
        conns = self._channels.get(channel)
        if not conns:
            return
        dead: list[WebSocket] = []
        for ws in list(conns):
            try:
                await ws.send_text(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, channel)
            logger.debug("ws removed dead connection channel=%s", channel)

    async def broadcast(self, channel: str, data: dict[str, object]) -> None:
        """Best-effort fan-out; never raises into the calling endpoint."""
        try:
            await self._send_all(channel, json.dumps(data))
        except Exception:  # noqa: BLE001
            logger.exception("ws broadcast failed channel=%s", channel)

    async def send_personal(self, websocket: WebSocket, data: dict[str, object]) -> None:
        payload = json.dumps(data)
        await websocket.send_text(payload)

    async def _heartbeat_loop(self) -> None:
        ping = json.dumps({"type": "ping"})
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            for channel in list(self._channels):
                try:
                    await self._send_all(channel, ping)
                except Exception:  # noqa: BLE001 — keep pinging other channels/rounds
                    logger.exception("ws heartbeat failed channel=%s", channel)
