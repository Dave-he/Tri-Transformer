from typing import Annotated, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.models.user import User
from app.services.model.webrtc_service import WebRTCService

router = APIRouter()


class OfferRequest(BaseModel):
    sdp: str
    type: str


class OfferResponse(BaseModel):
    sdp: str
    type: str


class CandidateRequest(BaseModel):
    candidate: str
    sdpMid: Optional[str] = None
    sdpMLineIndex: Optional[int] = None


class OkResponse(BaseModel):
    ok: bool


@router.post("/offer", response_model=OfferResponse)
async def webrtc_offer(
    payload: OfferRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    svc = WebRTCService()
    result = svc.handle_offer(sdp=payload.sdp, sdp_type=payload.type)
    return OfferResponse(**result)


@router.post("/candidate", response_model=OkResponse)
async def webrtc_candidate(
    payload: CandidateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    svc = WebRTCService()
    result = svc.handle_candidate(
        candidate=payload.candidate,
        sdp_mid=payload.sdpMid,
        sdp_mline_index=payload.sdpMLineIndex,
    )
    return OkResponse(**result)


@router.post("/interrupt", response_model=OkResponse)
async def webrtc_interrupt(
    current_user: Annotated[User, Depends(get_current_user)],
):
    svc = WebRTCService()
    result = svc.handle_interrupt()
    return OkResponse(**result)
