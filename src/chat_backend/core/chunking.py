"""
chunking.py – Text chunking strategies.

Implements different strategies for splitting extracted document
text into smaller chunks suitable for embedding:
- Fixed-size with overlap
- Recursive character splitting
- Semantic paragraph-aware splitting

Each strategy returns a list of Chunk objects with positional metadata.
"""
