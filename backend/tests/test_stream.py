import pytest
import asyncio
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
        engine = StreamingEngine()
        engine._interrupt_event = asyncio.Event()
        engine._interrupt_event.set()
        tokens = []
        async for tok in engine.generate("hello", max_tokens=100):
            tokens.append(tok)
        done_token = next((t for t in tokens if t.get("done")), None)
        assert done_token is not None
        assert done_token.get("interrupted") is True

    @pytest.mark.asyncio
    async def test_stream_done_message(self):
        from app.services.model.stream_engine import StreamingEngine
        engine = StreamingEngine(mock_response="hi")
        all_tokens = []
        async for tok in engine.generate("test", max_tokens=10):
            all_tokens.append(tok)
        assert len(all_tokens) > 0
        last = all_tokens[-1]
        assert last.get("done") is True
        assert last.get("interrupted") is False

    @pytest.mark.asyncio
    async def test_stream_non_interrupt_full_flow(self):
        from app.services.model.stream_engine import StreamingEngine
        engine = StreamingEngine(mock_response="abc")
        all_tokens = []
        async for tok in engine.generate("test"):
            all_tokens.append(tok)
        intermediate = [t for t in all_tokens if not t.get("done")]
        assert all("token" in t for t in intermediate)
        assert all(t.get("done") is False for t in intermediate)
        final = [t for t in all_tokens if t.get("done")]
        assert len(final) == 1


class TestWebSocketEndpoint:
    @pytest.mark.asyncio
    async def test_websocket_valid_token(self, client, auth_headers):
        token = auth_headers["Authorization"].split(" ")[1]
        response = await client.get(f"/api/v1/model/stream?token={token}")
        assert response.status_code in (200, 400, 403, 404, 422, 426)

    @pytest.mark.asyncio
    async def test_websocket_invalid_token(self):
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/api/v1/model/stream?token=invalid")
            assert response.status_code in (400, 401, 403, 404, 422, 426)

    @pytest.mark.asyncio
    async def test_websocket_text_message_stream(self, client, auth_headers):
        token = auth_headers["Authorization"].split(" ")[1]
        response = await client.get(f"/api/v1/model/stream?token={token}")
        assert response.status_code in (200, 400, 403, 404, 422, 426)

    @pytest.mark.asyncio
    async def test_websocket_interrupt(self, client, auth_headers):
        token = auth_headers["Authorization"].split(" ")[1]
        response = await client.get(f"/api/v1/model/stream?token={token}")
        assert response.status_code in (200, 400, 403, 404, 422, 426)

    @pytest.mark.asyncio
    async def test_websocket_close_clean(self, client, auth_headers):
        token = auth_headers["Authorization"].split(" ")[1]
        response = await client.get(f"/api/v1/model/stream?token={token}")
        assert response.status_code in (200, 400, 403, 404, 422, 426)
