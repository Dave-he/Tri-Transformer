import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.train_job import TrainJob


class TrainService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def submit_job(self, user_id: int, job_type: str, config: dict) -> TrainJob:
        job = TrainJob(
            id=str(uuid.uuid4()),
            user_id=user_id,
            job_type=job_type,
            status="pending",
            config=json.dumps(config),
        )
        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)
        return job

    async def get_job(self, job_id: str, user_id: int) -> Optional[TrainJob]:
        result = await self.db.execute(
            select(TrainJob).where(
                TrainJob.id == job_id,
                TrainJob.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def cancel_job(self, job_id: str, user_id: int) -> Optional[TrainJob]:
        job = await self.get_job(job_id, user_id)
        if not job:
            return None
        if job.status in ("completed", "failed"):
            return job
        job.status = "cancelled"
        job.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(job)
        return job

    async def list_jobs(self, user_id: int) -> list[TrainJob]:
        result = await self.db.execute(
            select(TrainJob).where(TrainJob.user_id == user_id).order_by(TrainJob.created_at.desc())
        )
        return result.scalars().all()


GALORE_DEFAULTS = {
    "rank": 128,
    "update_proj_gap": 200,
    "scale": 0.25,
}


def validate_galore_config(config: dict) -> dict:
    if not config.get("use_galore", False):
        return {"use_galore": False}
    result = {"use_galore": True}
    for key, default in GALORE_DEFAULTS.items():
        result[key] = config.get(key, default)
    rank = result["rank"]
    if rank < 1:
        raise ValueError(f"Invalid GaLore rank: {rank}, must be >= 1")
    return result
