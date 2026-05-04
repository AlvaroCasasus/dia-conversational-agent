"""
generate_base_dataset.py
------------------------
Phase 1 of the H6 experiment.
Generates questions and ground truths from ChromaDB chunks using the LLM.
Does NOT call the RAG backend — answers and contexts are left empty.
Run this once, then use experiment_h6.py to generate answers for each k.
"""

import json
import os
import random
import warnings
from typing import List, Literal

from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import chromadb

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ==========================================
# CONFIGURATION
# ==========================================

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "rag_dia"

LLM_CONFIG = {
    "model": "llama3.1:8b",
    "base_url": "http://100.83.251.20:5000/v1",
    "api_key": "not_required",
    "temperature": 0.7
}

OUTPUT_FILE = "datasets/base_dataset.json"
N_SAMPLES = 100  # number of QA pairs to generate

# ==========================================
# DATA SCHEMA
# ==========================================

class QAPair(BaseModel):
    model_config = ConfigDict(extra='ignore')

    sample_id: str = Field(description="Consecutive ID.")
    generation_method: Literal["llm_generated"] = Field(default="llm_generated")
    language: str = Field(description="ISO language code", pattern=r"^[a-z]{2}$")
    question: str = Field(description="Student query.")
    answer: str = Field(default="", description="Filled later by experiment_h6.py")
    ground_truth: str = Field(description="Ideal concise answer.")
    contexts: List[str] = Field(default=[], description="Filled later by experiment_h6.py")
    reference_contexts: List[str] = Field(description="Chunks used to generate ground truth.")
    source_document: str = Field(description="Filename of the source PDF.")
    chunk_id: str = Field(description="Unique identifier for the chunk.")
    #question_type: str = Field(description="factual, procedural, comparative, out_of_scope or ambiguous")
    question_type: Literal["factual", "summarization", "multi_hop", "out_of_scope", "ambiguous"]
    topic: Literal["plan_de_estudios", "matricula", "tfm", "profesorado", "otros"] = Field(description="Thematic area")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="Difficulty level")

# ==========================================
# HELPERS
# ==========================================

def get_db_chunks():
    """Fetches all chunks from ChromaDB."""
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = client.get_collection(COLLECTION_NAME)
        res = collection.get()
        if not res['ids']:
            print("Collection is empty.")
            return []
        return [{
            "id": res['ids'][i],
            "text": res['documents'][i],
            "metadata": res['metadatas'][i]
        } for i in range(len(res['ids']))]
    except Exception as e:
        print(f"Connection failed: {e}")
        return []

# ==========================================
# GENERATION
# ==========================================

def generate_base_dataset(n: int = N_SAMPLES) -> list:
    chunks = get_db_chunks()
    print(f"Total chunks in DB: {len(chunks)}")
    if not chunks:
        return []

    random.shuffle(chunks)
    selected_chunks = chunks[:n]

    llm = ChatOpenAI(**LLM_CONFIG)

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", """You are an academic evaluator for RAG systems. Your task is to generate high-quality QA pairs in Spanish.

    #     STRICT FORMAT RULES:
    #     1. language: Always use 'es'.
    #     2. generation_method: Always use 'llm_generated'.
    #     3. answer: ALWAYS leave this field as an empty string "". Do NOT fill it.
    #     4. contexts: ALWAYS leave this as an empty list []. Do NOT fill it.
    #     5. topic: Categorize into: plan_de_estudios, matricula, tfm, profesorado, or otros.
    #     6. difficulty:
    #        - 'easy': Answer is explicitly stated in a single sentence.
    #        - 'medium': Requires consulting multiple parts of the text or minor paraphrasing.
    #        - 'hard': Answer is implicit, requires synthesis of multiple sources.

    #     Taxonomy of question_type:
    #     - factual: Single-hop fact.
    #     - procedural: Step-by-step process.
    #     - comparative: Synthesis of information.
    #     - out_of_scope: Plausible but missing from text.
    #     - ambiguous: Vague query.

    #     CRITICAL RULES FOR ground_truth:
    #     - Write as a complete natural sentence in Spanish, minimum 15 words.
    #     - NEVER copy raw codes or table text from the context (e.g. '14, 3 = ...' or 'ASI Natura 103000361').
    #     - NEVER use 'yes', 'no', or single words as the answer.
    #     - For out_of_scope questions: ground_truth must be exactly the string 'out_of_scope'.
    #     - Good example: 'La asignatura se evalúa mediante dos prácticas grupales con peso del 30% cada una y un examen final del 40%.'
    #     - Bad example: '14, 3 = Presentation of second assignment' or 'yes' or 'ASI Natura 103000361'
    #     """),
    #     ("human", "Context: {chunk_text}\nMetadata: {metadata}\nType requested: '{q_type}'")
    # ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an academic evaluator for RAG systems. Your task is to generate high-quality QA pairs in Spanish.

        STRICT FORMAT RULES:
        1. language: Always use 'es'.
        2. generation_method: Always use 'llm_generated'.
        3. answer: ALWAYS leave this field as an empty string "". Do NOT fill it.
        4. contexts: ALWAYS leave this as an empty list []. Do NOT fill it.
        5. topic: Categorize into: plan_de_estudios, matricula, tfm, profesorado, or otros.
        6. difficulty:
           - 'easy': Answer is explicitly stated in a single sentence.
           - 'medium': Requires consulting multiple parts of the text or minor paraphrasing.
           - 'hard': Answer is implicit, requires synthesis of multiple sources.

        Taxonomy of question_type:
        - factual: Single-hop fact explicitly stated in the text.
        - summarization: Requires synthesizing or grouping multiple elements of information.
        - multi_hop: Requires combining information from different parts of the document to infer the answer.
        - out_of_scope: Plausible but missing from text.
        - ambiguous: Vague query.

        CRITICAL RULES FOR ground_truth:
        - Write as a complete natural sentence in Spanish, minimum 15 words.
        - NEVER copy raw codes or table text from the context (e.g. '14, 3 = ...' or 'ASI Natura 103000361').
        - NEVER use 'yes', 'no', or single words as the answer.
        - For out_of_scope questions: ground_truth must be exactly the string 'out_of_scope'.
        - Good example: 'La asignatura se evalúa mediante dos prácticas grupales con peso del 30% cada una y un examen final del 40%.'
        - Bad example: '14, 3 = Presentation of second assignment' or 'yes' or 'ASI Natura 103000361'
        """),
        ("human", "Context: {chunk_text}\nMetadata: {metadata}\nType requested: '{q_type}'")
    ])

    generator = prompt | llm.with_structured_output(QAPair)

    #q_types = ["factual", "procedural", "comparative", "out_of_scope", "ambiguous"]
    q_types = ["factual", "summarization", "multi_hop", "out_of_scope", "ambiguous"]
    probabilities = [0.4, 0.2, 0.2, 0.1, 0.1]

    dataset = []

    for i, chunk in enumerate(selected_chunks):
        q_type = random.choices(q_types, weights=probabilities, k=1)[0]
        print(f"[{i+1}/{len(selected_chunks)}] Generating {q_type}...")

        try:
            record = generator.invoke({
                "chunk_text": chunk['text'],
                "metadata": json.dumps(chunk['metadata']),
                "q_type": q_type
            })

            record.sample_id = str(i + 1)

            # Force empty regardless of what the LLM filled
            record.answer = ""
            record.contexts = []

            if q_type == "out_of_scope":
                record.source_document = "N/A"
                record.reference_contexts = []
                record.chunk_id = "N/A"
                record.ground_truth = "out_of_scope"  # force label
            else:
                record.source_document = chunk['metadata'].get('source', 'unknown')
                record.chunk_id = chunk['id']
                record.reference_contexts = [chunk['text']]

            # Store chunk metadata for Phase 2 (experiment_h6.py)
            item = record.model_dump()
            item['chunk_metadata'] = chunk['metadata']
            dataset.append(item)

        except Exception as e:
            print(f"  Error: {e}")

    return dataset


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 1: Generating base dataset (questions + ground truth)")
    print("=" * 60)

    data = generate_base_dataset(N_SAMPLES)

    if data:
        os.makedirs("datasets", exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\nDone! Base dataset saved to {OUTPUT_FILE} ({len(data)} items)")
    else:
        print("No data generated.")