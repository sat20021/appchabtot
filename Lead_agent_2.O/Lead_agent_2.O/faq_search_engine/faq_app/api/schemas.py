from pydantic import BaseModel
from typing import Optional

class AskRequest(BaseModel):
    question: str
    domain: str = "default"
    threshold: Optional[float] = None

class AskResponse(BaseModel):
    answer: str
    confidence: float
    source: str
    matched_question: Optional[str] = None
    domain: Optional[str] = None
