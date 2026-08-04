from fastapi import APIRouter

from app.models import AdversarialRequest, AdversarialResponse
from app.guardrails.moderation import check_moderation

router = APIRouter()


@router.post("/adversarial-test", response_model=AdversarialResponse)
def adversarial_test(req: AdversarialRequest):
    result = check_moderation(req.prompt)
    return AdversarialResponse(
        blocked=result["blocked"],
        reason=result["reason"],
        status=result["status"],
    )
