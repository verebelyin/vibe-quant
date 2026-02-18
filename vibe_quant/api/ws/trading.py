"""WebSocket endpoint for paper trading position/PnL updates."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from vibe_quant.api.ws.manager import ConnectionManager

logger = logging.getLogger(__name__)

router = APIRouter()

_CHANNEL = "trading"


def _get_manager(websocket: WebSocket) -> ConnectionManager:
    return websocket.app.state.ws_manager  # type: ignore[no-any-return]


@router.websocket("/ws/trading")
async def ws_trading(websocket: WebSocket) -> None:
    manager = _get_manager(websocket)
    await manager.connect(websocket, _CHANNEL)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.send_personal(websocket, {"type": "ack", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket, _CHANNEL)
    except Exception:  # noqa: BLE001
        manager.disconnect(websocket, _CHANNEL)
        logger.exception("ws /ws/trading unexpected error")
