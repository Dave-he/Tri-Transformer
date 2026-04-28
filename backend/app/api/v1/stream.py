import asyncio
import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.security import decode_access_token
from app.services.model.stream_engine import StreamingEngine

router = APIRouter()


async def _sse_generator(session_id: str, query: str, token: str):
    payload = decode_access_token(token)
    if payload is None:
        yield f"data: {json.dumps({'error': 'Unauthorized'})}\n\n"
        return

    engine = StreamingEngine()
    interrupt_event = asyncio.Event()
    async for token_char in engine.generate(query=query, interrupt_event=interrupt_event):
        yield f"data: {json.dumps({'token': token_char, 'done': False})}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"


@router.get("/sse/{session_id}")
async def sse_stream(
    session_id: str,
    token: str = Query(..., description="JWT token"),
    request: Request = None,
):
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    query = "SSE streaming test"
    return StreamingResponse(
        _sse_generator(session_id=session_id, query=query, token=token),
        media_type="text/event-stream",
    )


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
