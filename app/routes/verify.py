from fastapi import APIRouter

from app.models import VerifyRequest, VerifyResponse
from app.agents.graph import run_pipeline

router = APIRouter()


@router.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    result = run_pipeline(req.query)
    return VerifyResponse(**result)
