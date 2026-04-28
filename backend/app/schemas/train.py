from datetime import datetime
from typing import Optional
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


class TrainConfigPreset(BaseModel):
    name: str
    description: str
    learning_rate: float
    batch_size: int
    epochs: int
    lora_rank: Optional[int] = None
    lora_alpha: Optional[float] = None


TRAIN_CONFIG_PRESETS = [
    TrainConfigPreset(
        name="default",
        description="基础训练配置",
        learning_rate=1e-4,
        batch_size=8,
        epochs=3,
    ),
    TrainConfigPreset(
        name="lora_finetune",
        description="LoRA微调配置",
        learning_rate=5e-5,
        batch_size=4,
        epochs=5,
        lora_rank=8,
        lora_alpha=16.0,
    ),
    TrainConfigPreset(
        name="deepspeed_zero3",
        description="DeepSpeed ZeRO-3分布式训练",
        learning_rate=2e-5,
        batch_size=16,
        epochs=10,
        lora_rank=16,
        lora_alpha=32.0,
    ),
]
