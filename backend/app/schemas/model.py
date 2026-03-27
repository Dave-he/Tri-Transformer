from pydantic import BaseModel
from typing import Optional


class InferenceRequest(BaseModel):
    query: str
    context: list[str] = []
    history: list[dict] = []


class InferenceResponse(BaseModel):
    text: str
    confidence: float
    model: Optional[str] = None
