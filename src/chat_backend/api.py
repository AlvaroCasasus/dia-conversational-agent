"""
api.py – Single-file FastAPI backend for the RAG system.

Defines all API endpoints consumed by the Gradio frontend:
  - POST /upload        Upload a document for ingestion
  - GET  /documents     List all uploaded documents
  - DELETE /documents/{id}  Remove a document
  - POST /query         Ask a question (RAG retrieval + generation)
  - GET  /health        Service health check
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
import uuid
import datetime
from models.schemas import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    DocumentInfo,
    HealthResponse,
)

# ── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Conversational Agent API",
    version="0.1.0",
    description="Backend API for document ingestion and retrieval-augmented generation.",
)


# ── In-memory dummy store (replaced by real services later) ────────────────

_documents: dict[str, DocumentInfo] = {}

# ── Endpoints ──────────────────────────────────────────────────────────────


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a file upload, store it in MinIO, parse with Docling,
    chunk the text, embed, and persist vectors in Chroma.

    Currently returns a dummy response.
    """
    doc_id = str(uuid.uuid4())
    _documents[doc_id] = DocumentInfo(
        id=doc_id,
        filename=file.filename or "unknown",
        status="processed",
        uploaded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        num_chunks=42,  # dummy
    )
    return UploadResponse(
        id=doc_id,
        filename=file.filename or "unknown",
        message="Document uploaded and processed successfully (dummy).",
    )


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """
    Return metadata for every document that has been uploaded.
    """
    return list(_documents.values())


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Remove a document and its embeddings from the system.
    """
    if doc_id not in _documents:
        raise HTTPException(status_code=404, detail="Document not found")
    del _documents[doc_id]
    return {"message": f"Document {doc_id} deleted."}


@app.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """
    Receive a natural-language question, retrieve relevant chunks
    from Chroma, and generate an answer via the LLM.

    Currently returns a dummy response.
    """
    dummy_sources = [
        SourceChunk(
            document_name="example.pdf",
            page=3,
            chunk_text="This is a dummy retrieved chunk that would normally come from the vector store.",
            score=0.92,
        ),
        SourceChunk(
            document_name="example.pdf",
            page=7,
            chunk_text="Another relevant passage retrieved during the similarity search.",
            score=0.85,
        ),
    ]
    return QueryResponse(
        answer=(
            f"This is a dummy answer to your question: '{req.question}'. "
            "In a real implementation, retrieved context would be injected into "
            "the LLM prompt and the model's response returned here."
        ),
        sources=dummy_sources,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Return the status of this service and connectivity to dependencies.
    """
    return HealthResponse(
        status="ok",
        services={
            "chroma": "ok (dummy)",
            "minio": "ok (dummy)",
            "docling": "ok (dummy)",
        },
    )
