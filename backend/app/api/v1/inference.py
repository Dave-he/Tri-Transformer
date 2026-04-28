from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.models.user import User
from app.services.inference.adapter import (
    LlamaCppInferenceAdapter,
    get_inference_service,
    switch_inference_mode,
    VALID_MODES,
)

router = APIRouter()


class InferenceModeResponse(BaseModel):
    mode: str
    service_type: str
    available: bool


class SwitchModeRequest(BaseModel):
    mode: str


class SwitchModeResponse(BaseModel):
    old_mode: str
    new_mode: str
    service_available: bool


@router.get("/mode", response_model=InferenceModeResponse)
async def get_inference_mode(
    current_user: Annotated[User, Depends(get_current_user)],
):
    from app.core.config import settings
    mode = settings.inference_mode
    try:
        svc = get_inference_service()
        if isinstance(svc, LlamaCppInferenceAdapter):
            service_type = "llamacpp_gguf"
            available = svc.is_available
        else:
            from app.services.model.mock_inference import MockInferenceService
            if isinstance(svc, MockInferenceService):
                service_type = "mock"
            else:
                service_type = "pytorch_direct"
            available = True
    except Exception:
        service_type = mode
        available = False
    return InferenceModeResponse(mode=mode, service_type=service_type, available=available)


@router.post("/mode", response_model=SwitchModeResponse)
async def switch_mode(
    payload: SwitchModeRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    if payload.mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{payload.mode}'. Valid: {sorted(VALID_MODES)}",
        )
    try:
        result = switch_inference_mode(payload.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to initialize service: {e}")
    return SwitchModeResponse(**result)
