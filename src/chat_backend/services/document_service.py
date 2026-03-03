"""
document_service.py – Document processing orchestration.

Coordinates the full ingestion pipeline:
1. Upload raw file to MinIO.
2. Send file to Docling for text extraction.
3. Chunk the extracted text.
4. Generate embeddings for each chunk.
5. Store embeddings + metadata in Chroma.
"""
