"""Pydantic request/response schemas for the API."""
from typing import Optional
from pydantic import BaseModel, Field


class MemoryIn(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    captured_at: Optional[str] = None  # YYYY-MM-DD, optional


class RecallIn(BaseModel):
    question: str = Field(min_length=1)
    k: int = 5
    use_llm: bool = False  # True: let local Gemma phrase the answer (slower)


class Citation(BaseModel):
    chunk_id: int
    title: str
    source: Optional[str] = None
    page: Optional[int] = None
    text: str
    score: float


class RecallOut(BaseModel):
    state: str  # grounded | weak_evidence | no_evidence
    answer: Optional[str] = None
    verified: bool = False
    citations: list[Citation] = []
