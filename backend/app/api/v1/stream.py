import asyncio
import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.core.security import decode_access_token
from app.services.model.stream_engine import StreamingEngine

router = APIRouter()


@router.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
):
    if not token:
        await websocket.close(code=4001)
        return

    payload = decode_access_token(token)
    if payload is None:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    await websocket.send_json({"type": "connected", "user_id": payload.get("sub")})

    engine = StreamingEngine()
    interrupt_event = asyncio.Event()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "interrupt":
                interrupt_event.set()
                await websocket.send_json({"done": True, "interrupted": True, "sources": []})
                interrupt_event.clear()

            elif msg_type == "text":
                interrupt_event.clear()
                query = msg.get("content", "")
                async for token_char in engine.generate(
                    query=query,
                    interrupt_event=interrupt_event,
                ):
                    if interrupt_event.is_set():
                        break
                    await websocket.send_json({"token": token_char, "done": False})
                await websocket.send_json({"done": True, "sources": []})

    except WebSocketDisconnect:
        pass
