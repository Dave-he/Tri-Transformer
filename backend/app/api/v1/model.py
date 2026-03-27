from typing import Optional, Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.models.user import User
from app.services.model.mock_inference import get_inference_service
from app.services.model.inference_service import InferenceError

router = APIRouter()


class InferenceRequest(BaseModel):
    query: str
    context: list[str] = []
    history: list[dict] = []


class InferenceResponse(BaseModel):
    text: str
    confidence: float
    model: Optional[str] = None


@router.post("/inference", response_model=InferenceResponse)
async def inference(
    payload: InferenceRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    service = get_inference_service()
    try:
        result = await service.infer(
            query=payload.query,
            context=payload.context,
            history=payload.history,
        )
        return InferenceResponse(**result)
    except InferenceError as e:
        raise HTTPException(status_code=503, detail=str(e))
