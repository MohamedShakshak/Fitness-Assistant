"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel
from typing import Optional


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[list[ChatMessage]] = None
    filters: Optional[dict] = None
    llm_provider: Optional[str] = None
    top_k: Optional[int] = None


class SourceInfo(BaseModel):
    name: str
    category: str = ""
    body_part: str = ""
    equipment: str = ""
    level: str = ""
    score: float = 0
    rerank_score: Optional[float] = None
    source_db: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    context_used: bool


class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = None
    top_k: Optional[int] = None


class SearchResult(BaseModel):
    id: int
    text: str
    score: float
    metadata: dict
    rerank_score: Optional[float] = None


class SearchResponse(BaseModel):
    results: list[dict]


class FiltersResponse(BaseModel):
    body_part: list[str]
    equipment: list[str]
    level: list[str]
    category: list[str]