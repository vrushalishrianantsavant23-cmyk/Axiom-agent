from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class VerifyRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Claim or question to verify")


class VerifyResponse(BaseModel):
    status: str
    answer: str
    semantic_entropy_score: float = 0.0
    is_consistent: bool = True
    confidence: float = 0.0
    sources: List[str] = []
    is_neutral_assessment: bool = True
    perspectives_considered: List[str] = []
    reasoning_trace: Dict[str, Any] = {}
    query_id: Optional[str] = None


class AdversarialRequest(BaseModel):
    prompt: str


class AdversarialResponse(BaseModel):
    blocked: bool
    reason: str
    status: str
