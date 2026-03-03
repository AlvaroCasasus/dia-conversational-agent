"""
schemas.py – Pydantic request / response schemas.

Defines the data contracts for the API:
- UploadResponse: returned after a successful document upload.
- QueryRequest: the user question payload.
- QueryResponse: the LLM answer plus source chunks.
- HealthResponse: service status and dependency checks.
"""
from pydantic import BaseModel
from typing import Optional

# ── Pydantic schemas ───────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Payload sent by the frontend when the user asks a question."""
    question: str
    collection: Optional[str] = None  # optionally scope to a specific document collection


class SourceChunk(BaseModel):
    """A single retrieved chunk returned alongside the answer."""
    document_name: str
    page: Optional[int] = None
    chunk_text: str
    score: float


class QueryResponse(BaseModel):
    """Response returned for a /query request."""
    answer: str
    sources: list[SourceChunk]


class DocumentInfo(BaseModel):
    """Metadata about an uploaded document."""
    id: str
    filename: str
    status: str  # e.g. "processed", "processing", "failed"
    uploaded_at: str
    num_chunks: int


class UploadResponse(BaseModel):
    """Response returned after a successful upload."""
    id: str
    filename: str
    message: str


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""
    status: str
    services: dict[str, str]