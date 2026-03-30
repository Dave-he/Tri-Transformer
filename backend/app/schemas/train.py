from datetime import datetime
from pydantic import BaseModel, field_validator

from app.models.train_job import VALID_JOB_TYPES


class TrainJobRequest(BaseModel):
    job_type: str
    config: dict = {}

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v):
        if v not in VALID_JOB_TYPES:
            raise ValueError(f"job_type must be one of {VALID_JOB_TYPES}")
        return v


class TrainJobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    config: dict
    created_at: datetime
    updated_at: datetime
