import json
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.core.security import create_access_token


@pytest.fixture
def token():
    return create_access_token({"sub": "1", "kb_id": "kb-stream-test"})


@pytest.fixture
def sync_client():
    return TestClient(app)


class TestStreamWebSocket:
    def test_websocket_connect_valid_token(self, sync_client, token):
        with sync_client.websocket_connect(f"/api/v1/model/stream?token={token}") as ws:
            data = ws.receive_json()
            assert data.get("type") == "connected"

    def test_websocket_invalid_token_rejected(self, sync_client):
        with pytest.raises(Exception):
            with sync_client.websocket_connect("/api/v1/model/stream?token=invalid.bad.token") as ws:
                ws.receive_json()

    def test_websocket_missing_token_rejected(self, sync_client):
        with pytest.raises(Exception):
            with sync_client.websocket_connect("/api/v1/model/stream") as ws:
                ws.receive_json()

    def test_text_message_triggers_stream(self, sync_client, token):
        with sync_client.websocket_connect(f"/api/v1/model/stream?token={token}") as ws:
            ws.receive_json()
            ws.send_json({"type": "text", "content": "hello"})

            tokens = []
            while True:
                msg = ws.receive_json()
                if msg.get("done"):
                    break
                tokens.append(msg.get("token", ""))

            assert len(tokens) > 0
            full_text = "".join(tokens)
            assert len(full_text) > 0

    def test_interrupt_stops_generation(self, sync_client, token):
        with sync_client.websocket_connect(f"/api/v1/model/stream?token={token}") as ws:
            ws.receive_json()
            ws.send_json({"type": "text", "content": "tell me a very long story"})
            ws.send_json({"type": "interrupt"})

            done_msg = None
            for _ in range(50):
                msg = ws.receive_json()
                if msg.get("done"):
                    done_msg = msg
                    break

            assert done_msg is not None
            assert done_msg.get("done") is True

    def test_stream_done_includes_sources(self, sync_client, token):
        with sync_client.websocket_connect(f"/api/v1/model/stream?token={token}") as ws:
            ws.receive_json()
            ws.send_json({"type": "text", "content": "quick question"})

            done_msg = None
            for _ in range(100):
                msg = ws.receive_json()
                if msg.get("done"):
                    done_msg = msg
                    break

            assert done_msg is not None
            assert "sources" in done_msg
