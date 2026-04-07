import json
import re
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

# ==========================================
# 1. DATA STRUCTURES DEFINITION (PYDANTIC)
# ==========================================

class CourseGuideSchema(BaseModel):
    course_description: str = Field(
        description="Brief summary or general description of what the course is about and its main goal."
    )
    general_info: str = Field(
        description="Course name, credits (ECTS), semester, course type (optativa/obligatoria), academic year, language, etc."
    )
    teaching_staff: str = Field(
        description="Names, roles (coordinator/professor), emails, office room numbers, and tutoring hours."
    )
    prerequisites: str = Field(
        description="Previous knowledge, recommended prior courses, or mandatory prerequisites."
    )
    competencies_and_outcomes: str = Field(
        description="General and specific competencies to be acquired, and expected learning outcomes (RA)."
    )
    syllabus: str = Field(
        description="Breakdown of the course topics, modules, or units (Temario)."
    )
    schedule: str = Field(
        description="Chronological schedule (cronograma), week-by-week activities, or important dates for theory classes, labs, and evaluations."
    )
    evaluation_criteria: str = Field(
        description="Detailed criteria for continuous evaluation and final exams, including weights/percentages, minimum grades, and types of tests."
    )
    bibliography_and_resources: str = Field(
        description="Required and recommended books, software, websites, or other learning resources."
    )

class QAPair(BaseModel):
    question: str = Field(description="The generated question simulating a student's query (in Spanish).")
    ground_truth: str = Field(description="The ideal, factual answer based strictly on the course guide (in Spanish).")
    question_type: str = Field(description="Category: 'Factual', 'Summarization', 'Multi-hop Reasoning', or 'Unanswerable'.")
    student_profile: str = Field(description="Simulated profile: 'Freshman', 'Senior', 'Formal tone', 'Informal tone', etc.")

class QADataset(BaseModel):
    questions: List[QAPair]

# ==========================================
# 2. LLM CONFIGURATION
# ==========================================

llm_extractor = ChatOpenAI(
    model="qwen2.5:32b",
    base_url="http://100.115.179.39:5000/v1",
    api_key="not_required",
    temperature=0.1
)

llm_generator = ChatOpenAI(
    model="qwen2.5:32b",
    base_url="http://100.115.179.39:5000/v1",
    api_key="not_required",
    temperature=0.7
)

# ==========================================
# 3. PIPELINE FUNCTIONS
# ==========================================

def _parse_json_response(text: str) -> dict:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{text[:500]}")


def extract_schema(guide_text: str) -> CourseGuideSchema:
    """Extracts structured facts from the Markdown text."""
    print("-> Extracting schema from the course guide...")

    schema_fields = "\n".join([
        f'  "{name}": "<{field.description}>"'
        for name, field in CourseGuideSchema.model_fields.items()
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Eres un asistente académico experto. Lee la guía docente universitaria y extrae la información clave.
Ignora el relleno y conserva solo los hechos puros. Escribe tu output en español.
Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin comillas de código markdown.

El JSON debe tener exactamente estas claves:
{{{{
{schema_fields}
}}}}"""),
        ("human", "Aquí está el texto de la guía docente:\n\n{text}\n\nResponde solo con el JSON:")
    ])

    chain = prompt | llm_extractor
    result = chain.invoke({"text": guide_text})
    raw = result.content if hasattr(result, "content") else str(result)

    parsed = _parse_json_response(raw)
    return CourseGuideSchema(**parsed)


def generate_questions(schema: CourseGuideSchema, num_questions: int = 20) -> QADataset:
    """Generates diverse Q&A pairs based on the extracted schema."""
    print(f"-> Generating a diverse dataset of {num_questions} questions...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un simulador avanzado de estudiantes universitarios.
Basándote en los datos del curso proporcionados, genera {num_questions} pares de pregunta y respuesta (Q&A).

REQUISITO DE IDIOMA: Los campos 'question' y 'ground_truth' DEBEN generarse en español.

TAXONOMÍA (distribuye las preguntas entre estos tipos):
1. Factual: Preguntas directas sobre hechos concretos (fechas, nombres, despachos).
2. Summarization: Preguntas que piden explicar una sección completa (ej: cómo funciona la evaluación).
3. Multi-hop Reasoning: Preguntas que requieren cruzar varios datos (ej: "Si trabajo por las mañanas, ¿puedo asistir a las tutorías?").
4. Unanswerable: Preguntas plausibles cuya respuesta NO está en los datos. El ground_truth debe indicar explícitamente que la guía docente no proporciona esa información. MÁXIMO 3 de este tipo.

DIVERSIDAD (perfiles de estudiante):
Aplica diferentes estilos: 'Freshman' (confuso/perdido), 'Senior' (directo/técnico), 'Tono informal' (como un WhatsApp), 'Tono formal' (como un email al profesor).

IMPORTANTE: Responde ÚNICAMENTE con un objeto JSON válido con esta estructura, sin texto adicional:
{{
  "questions": [
    {{
      "question": "<pregunta en español>",
      "ground_truth": "<respuesta ideal en español>",
      "question_type": "<Factual|Summarization|Multi-hop Reasoning|Unanswerable>",
      "student_profile": "<Freshman|Senior|Tono informal|Tono formal>"
    }}
  ]
}}"""),
        ("human", "COURSE DATA:\n{schema_json}\n\nResponde solo con el JSON:")
    ])

    schema_json = schema.model_dump_json(indent=2)
    chain = prompt | llm_generator
    result = chain.invoke({"num_questions": num_questions, "schema_json": schema_json})
    raw = result.content if hasattr(result, "content") else str(result)

    parsed = _parse_json_response(raw)
    return QADataset(**parsed)


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    pdf_path = "/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Sistemas de Planificación.pdf"
    NUM_QUESTIONS = 20

    # A. Load PDF and export to Markdown via DoclingLoader (no verbalization)
    print(f"-> Loading PDF: {pdf_path}")
    try:
        loader = DoclingLoader(
            file_path=pdf_path,
            export_type=ExportType.MARKDOWN
        )
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Error loading PDF: {e}")
        full_text = ""

    print(f"\n--- MARKDOWN TEXT (first 1000 chars) ---")
    print(full_text[:1000])

    # B. Extract schema
    extracted_schema = extract_schema(full_text)
    print("\n--- EXTRACTED SCHEMA ---")
    print(extracted_schema.model_dump_json(indent=2))

    # C. Generate Q&A dataset
    final_dataset = generate_questions(extracted_schema, num_questions=NUM_QUESTIONS)

    # D. Save results
    output_file = "dataset_baseline.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_dataset.model_dump(), f, ensure_ascii=False, indent=4)

    print(f"\n Done! Generated {len(final_dataset.questions)} questions → '{output_file}'")