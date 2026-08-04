"""WebSocket connection manager for the realtime inspection channel.

Design constraints (3E):
- Persistence > Realtime Delivery. Broadcast is fire-and-forget from the
  inspection flow and is never awaited by it; a broadcast failure can never
  roll back an already-committed inspection.
- Dead / slow clients are isolated: each client is sent individually inside
  a timeout; a failing client is dropped without affecting the others.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)

_SEND_TIMEOUT = 5.0


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def client_count(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast(self, event: dict) -> int:
        """Send one event to every connected client.

        Returns the number of clients that received it. Never raises: each
        client is isolated; failures are logged and the client dropped.
        """
        async with self._lock:
            targets = list(self._connections)
        delivered = 0
        for ws in targets:
            try:
                await asyncio.wait_for(ws.send_json(event), timeout=_SEND_TIMEOUT)
                delivered += 1
            except Exception:
                logger.warning("ws broadcast failed for a client, dropping it")
                self.disconnect(ws)
        return delivered

    async def shutdown(self) -> None:
        async with self._lock:
            targets = list(self._connections)
        for ws in targets:
            try:
                await ws.close()
            except Exception:
                pass
        self._connections.clear()


manager = ConnectionManager()


def schedule_broadcast(event: dict) -> None:
    """Fire-and-forget broadcast so inspection flow never blocks on WS."""

    async def _run() -> None:
        try:
            await manager.broadcast(event)
        except Exception:
            logger.exception("unexpected broadcast failure")

    try:
        asyncio.get_running_loop().create_task(_run())
    except RuntimeError:
        logger.warning("no running event loop, broadcast skipped")
