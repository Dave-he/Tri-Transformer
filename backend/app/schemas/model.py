from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ModelStatusResponse(BaseModel):
    loaded: bool
    model_name: str
    load_time: Optional[datetime] = None


class ModelLoadRequest(BaseModel):
    model_id: str


class ModelLoadResponse(BaseModel):
    task_id: str
    status: str = "loading"


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    parameters_count: int
    model_type: str
