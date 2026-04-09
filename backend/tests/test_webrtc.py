import pytest


@pytest.mark.asyncio
async def test_webrtc_offer(client, auth_headers):
    response = await client.post(
        "/api/v1/webrtc/offer",
        headers=auth_headers,
        json={"sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\n", "type": "offer"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "answer"
    assert "sdp" in data
    assert isinstance(data["sdp"], str)


@pytest.mark.asyncio
async def test_webrtc_candidate(client, auth_headers):
    response = await client.post(
        "/api/v1/webrtc/candidate",
        headers=auth_headers,
        json={
            "candidate": "candidate:1 1 UDP 2130706431 192.168.1.1 54321 typ host",
            "sdpMid": "0",
            "sdpMLineIndex": 0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True


@pytest.mark.asyncio
async def test_webrtc_interrupt(client, auth_headers):
    response = await client.post(
        "/api/v1/webrtc/interrupt",
        headers=auth_headers,
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True


@pytest.mark.asyncio
async def test_webrtc_offer_no_auth(client):
    response = await client.post(
        "/api/v1/webrtc/offer",
        json={"sdp": "v=0\r\n", "type": "offer"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_webrtc_candidate_no_auth(client):
    response = await client.post(
        "/api/v1/webrtc/candidate",
        json={"candidate": "candidate:1 1 UDP 2130706431 127.0.0.1 9 typ host"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_webrtc_interrupt_no_auth(client):
    response = await client.post(
        "/api/v1/webrtc/interrupt",
        json={},
    )
    assert response.status_code == 401
