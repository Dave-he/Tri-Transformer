import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from starlette.testclient import TestClient
from httpx import ASGITransport, AsyncClient


class TestStreamingEngine:
    @pytest.mark.asyncio
    async def test_stream_generates_tokens(self):
        from app.services.model.stream_engine import StreamingEngine
        engine = StreamingEngine()
        tokens = []
        async for tok in engine.generate("hello", max_tokens=5):
            tokens.append(tok)
        assert len(tokens) > 0
        assert all("token" in t for t in tokens)

    @pytest.mark.asyncio
    async def test_stream_interrupt(self):
        from app.services.model.stream_engine import StreamingEngine
        import asyncio
        engine = StreamingEngine()
        engine._interrupt_event = asyncio.Event()
        engine._interrupt_event.set()
        tokens = []
        async for tok in engine.generate("hello", max_tokens=100):
            tokens.append(tok)
        done_token = next((t for t in tokens if t.get("done")), None)
        assert done_token is not None
        assert done_token.get("interrupted") is True


class TestWebSocketEndpoint:
    @pytest.mark.asyncio
    async def test_websocket_valid_token(self, client, auth_headers):
        token = auth_headers["Authorization"].split(" ")[1]
        ws_url = f"/api/v1/model/stream?token={token}"
        with client.stream("GET", ws_url) as response:
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_websocket_invalid_token(self):
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            with pytest.raises(Exception):
                async with ac.websocket_connect("/api/v1/model/stream?token=invalid") as ws:
                    pass

    @pytest.mark.asyncio
    async def test_websocket_text_message_stream(self, client, auth_headers):
        token = auth_headers["Authorization"].split(" ")[1]
        with client.stream("GET", f"/api/v1/model/stream?token={token}") as response:
            assert response.status_code == 200
            with pytest.raises(Exception):
                response.send_text('{"type":"text","content":"hello"}')

    @pytest.mark.asyncio
    async def test_websocket_interrupt(self, client, auth_headers):
        token = auth_headers["Authorization"].split(" ")[1]
        with client.stream("GET", f"/api/v1/model/stream?token={token}") as response:
            assert response.status_code == 200
            with pytest.raises(Exception):
                response.send_text('{"type":"interrupt"}')

    @pytest.mark.asyncio
    async def test_websocket_close_clean(self, client, auth_headers):
        token = auth_headers["Authorization"].split(" ")[1]
        with client.stream("GET", f"/api/v1/model/stream?token={token}") as response:
            assert response.status_code == 200
            with pytest.raises(Exception):
                response.close()
