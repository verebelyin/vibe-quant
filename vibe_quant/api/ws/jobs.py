"""WebSocket endpoint for job status updates."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from vibe_quant.api.ws.manager import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter()

_CHANNEL = "jobs"


def _get_manager(websocket: WebSocket) -> ConnectionManager:
    return websocket.app.state.ws_manager  # type: ignore[no-any-return]


@router.websocket("/ws/jobs")
async def ws_jobs(websocket: WebSocket) -> None:
    manager = _get_manager(websocket)
    await manager.connect(websocket, _CHANNEL)
    try:
        while True:
            data = await websocket.receive_json()
            # Client can send filter messages (e.g. {"subscribe_run_id": "xxx"})
            # Currently just acknowledge; filtering is future work
            await manager.send_personal(websocket, {"type": "ack", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket, _CHANNEL)
    except Exception:  # noqa: BLE001
        manager.disconnect(websocket, _CHANNEL)
        logger.exception("ws /ws/jobs unexpected error")
