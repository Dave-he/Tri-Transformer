from typing import Optional, Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.model import ModelStatusResponse, ModelLoadRequest, ModelLoadResponse, ModelInfoResponse
from app.services.model.mock_inference import get_inference_service
from app.services.model.inference_service import InferenceError
from app.services.model.model_service import get_model_service

router = APIRouter()


class InferenceRequest(BaseModel):
    query: str
    context: list[str] = []
    history: list[dict] = []


class InferenceResponse(BaseModel):
    text: str
    confidence: float
    model: Optional[str] = None


@router.get("/status", response_model=ModelStatusResponse)
async def model_status():
    svc = get_model_service()
    return ModelStatusResponse(**svc.get_status())


@router.post("/load", response_model=ModelLoadResponse, status_code=202)
async def load_model(
    payload: ModelLoadRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    svc = get_model_service()
    result = await svc.load_model(payload.model_id)
    return ModelLoadResponse(**result)


@router.get("/info", response_model=ModelInfoResponse)
async def model_info():
    svc = get_model_service()
    return ModelInfoResponse(**svc.get_info())


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
