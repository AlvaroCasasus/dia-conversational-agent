"""
app.py – Gradio frontend for the RAG Conversational Agent.

Provides two tabs:
  1. Chat   – Ask questions and receive RAG-powered answers with source citations.
  2. Upload – Upload documents for ingestion into the vector store.

Connects to the FastAPI backend at BACKEND_URL (default http://backend:8000).
"""

import os
import gradio as gr
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# ── Helper functions that call the backend API ──────────────────────────────


def upload_file(file):
    """Upload a document to the backend and return a status message."""
    if file is None:
        return "No file selected."
    try:
        with open(file.name, "rb") as f:
            response = httpx.post(
                f"{BACKEND_URL}/upload",
                files={"file": (os.path.basename(file.name), f)},
                timeout=60,
            )
        response.raise_for_status()
        data = response.json()
        return f"✅ Uploaded **{data['filename']}**\n\n- ID: `{data['id']}`\n- {data['message']}"
    except httpx.HTTPError as exc:
        return f"❌ Upload failed: {exc}"


def list_documents():
    """Fetch the list of uploaded documents from the backend."""
    try:
        response = httpx.get(f"{BACKEND_URL}/documents", timeout=10)
        response.raise_for_status()
        docs = response.json()
        if not docs:
            return "No documents uploaded yet."
        lines = ["| Filename | Status | Chunks | Uploaded |", "| --- | --- | --- | --- |"]
        for doc in docs:
            lines.append(
                f"| {doc['filename']} | {doc['status']} | {doc['num_chunks']} | {doc['uploaded_at'][:19]} |"
            )
        return "\n".join(lines)
    except httpx.HTTPError as exc:
        return f"❌ Could not fetch documents: {exc}"


def query_rag(question, history):
    """
    Send the user's question to the /query endpoint and return the
    answer formatted with source citations.

    Compatible with Gradio's ChatInterface (receives question + history).
    """
    if not question.strip():
        return ""
    try:
        response = httpx.post(
            f"{BACKEND_URL}/query",
            json={"question": question},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        answer = data["answer"]

        # Append source citations
        if data.get("sources"):
            answer += "\n\n---\n**Sources:**\n"
            for i, src in enumerate(data["sources"], 1):
                page_info = f", p.{src['page']}" if src.get("page") else ""
                answer += (
                    f"\n{i}. **{src['document_name']}**{page_info} "
                    f"(score: {src['score']:.2f})\n"
                    f"   > {src['chunk_text'][:200]}\n"
                )
        return answer
    except httpx.HTTPError as exc:
        return f"❌ Query failed: {exc}"


def check_health():
    """Call /health and return a formatted status string."""
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        lines = [f"**Overall:** {data['status']}\n"]
        for svc, status in data.get("services", {}).items():
            lines.append(f"- **{svc}:** {status}")
        return "\n".join(lines)
    except httpx.HTTPError as exc:
        return f"❌ Health check failed: {exc}"


# ── Gradio UI ───────────────────────────────────────────────────────────────

with gr.Blocks(title="RAG Conversational Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 RAG Conversational Agent")

    with gr.Tab("💬 Chat"):
        gr.Markdown("Ask a question and get an answer grounded in your uploaded documents.")
        chatbot = gr.ChatInterface(
            fn=query_rag,
            type="messages",
        )

    with gr.Tab("📄 Documents"):
        gr.Markdown("Upload documents for ingestion or view previously uploaded files.")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Select a document", file_types=[".pdf", ".docx", ".txt", ".md"])
                upload_btn = gr.Button("Upload", variant="primary")
                upload_status = gr.Markdown()
            with gr.Column(scale=2):
                doc_list = gr.Markdown("Click **Refresh** to load documents.")
                refresh_btn = gr.Button("🔄 Refresh document list")

        upload_btn.click(fn=upload_file, inputs=file_input, outputs=upload_status)
        refresh_btn.click(fn=list_documents, outputs=doc_list)

    with gr.Tab("🩺 Health"):
        gr.Markdown("Check connectivity to backend services.")
        health_output = gr.Markdown()
        health_btn = gr.Button("Check Health", variant="secondary")
        health_btn.click(fn=check_health, outputs=health_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=3000)
