import pytest


@pytest.mark.asyncio
async def test_submit_train_job(client, auth_headers):
    response = await client.post(
        "/api/v1/train/jobs",
        headers=auth_headers,
        json={
            "job_type": "lora_finetune",
            "config": {
                "model_stage": 1,
                "learning_rate": 1e-4,
                "num_epochs": 3,
            },
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_submit_invalid_job_type(client, auth_headers):
    response = await client.post(
        "/api/v1/train/jobs",
        headers=auth_headers,
        json={"job_type": "invalid_type", "config": {}},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_job_status(client, auth_headers):
    submit_resp = await client.post(
        "/api/v1/train/jobs",
        headers=auth_headers,
        json={"job_type": "lora_finetune", "config": {"num_epochs": 1}},
    )
    job_id = submit_resp.json()["job_id"]
    response = await client.get(
        f"/api/v1/train/jobs/{job_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] in ["pending", "running", "completed", "failed"]


@pytest.mark.asyncio
async def test_cancel_job(client, auth_headers):
    submit_resp = await client.post(
        "/api/v1/train/jobs",
        headers=auth_headers,
        json={"job_type": "lora_finetune", "config": {"num_epochs": 100}},
    )
    job_id = submit_resp.json()["job_id"]
    response = await client.delete(
        f"/api/v1/train/jobs/{job_id}",
        headers=auth_headers,
    )
    assert response.status_code in [200, 204]


@pytest.mark.asyncio
async def test_get_nonexistent_job(client, auth_headers):
    response = await client.get(
        "/api/v1/train/jobs/nonexistent-job-id",
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_train_unauthorized(client):
    response = await client.post(
        "/api/v1/train/jobs",
        json={"job_type": "lora_finetune", "config": {}},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_jobs(client, auth_headers):
    await client.post(
        "/api/v1/train/jobs",
        headers=auth_headers,
        json={"job_type": "lora_finetune", "config": {"num_epochs": 1}},
    )
    response = await client.get("/api/v1/train/jobs", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_jobs_start(client, auth_headers):
    response = await client.post(
        "/api/v1/train/jobs/start",
        headers=auth_headers,
        json={
            "i_model_id": "qwen2-audio",
            "o_model_id": "llama3-8b",
            "learning_rate": 1e-4,
            "batch_size": 8,
            "max_steps": 100,
            "phase": 0,
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "jobId" in data
    assert isinstance(data["jobId"], str)


@pytest.mark.asyncio
async def test_jobs_progress(client, auth_headers):
    response = await client.get("/api/v1/train/jobs/progress", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["idle", "running", "completed", "failed", "pending"]


@pytest.mark.asyncio
async def test_jobs_models(client, auth_headers):
    response = await client.get("/api/v1/train/jobs/models", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0
    model = data["models"][0]
    assert "id" in model
    assert "name" in model
    assert "type" in model
