"""
config.py – Centralised configuration.

Reads environment variables (or .env file) and exposes them as
a Pydantic Settings object. Expected variables include:

- CHROMA_HOST / CHROMA_PORT
- MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_SECRET_KEY / MINIO_BUCKET
- DOCLING_URL
- LLM_API_KEY / LLM_MODEL
- EMBEDDING_MODEL
"""
