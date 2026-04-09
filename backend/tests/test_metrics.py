import pytest


@pytest.mark.asyncio
async def test_get_metrics(client, auth_headers):
    response = await client.get("/api/v1/metrics", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "current" in data
    assert "history" in data
    current = data["current"]
    assert "retrievalAccuracy" in current
    assert "bleuScore" in current
    assert "hallucinationRate" in current
    assert isinstance(data["history"], list)


@pytest.mark.asyncio
async def test_get_training_status(client, auth_headers):
    response = await client.get("/api/v1/training/status", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "phase" in data
    assert "progress" in data
    assert "eta" in data
    assert "message" in data


@pytest.mark.asyncio
async def test_metrics_no_auth(client):
    r1 = await client.get("/api/v1/metrics")
    assert r1.status_code == 401

    r2 = await client.get("/api/v1/training/status")
    assert r2.status_code == 401
