"""
Generation: turn the question + retrieved chunks into a structured, cited answer.

Output (validated by Pydantic):
    answer      str                          the answer, grounded in the context
    sources     list[str]                    document paths the answer used
    confidence  "low" | "medium" | "high"    how well the context supports the answer
"""
import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


# ---------------------------------------------------------------------------
# 1. THE OUTPUT SHAPE. The LLM is forced to return exactly these three fields.
# ---------------------------------------------------------------------------
class RAGAnswer(BaseModel):
    answer: str = Field(description="The answer to the question, using only the context.")
    sources: list[str] = Field(
        description="Source paths of the context blocks used, copied exactly, "
                    "e.g. product-docs/workflows/creating-workflows.md")
    confidence: Literal["low", "medium", "high"] = Field(
        description="high: context states the answer directly. "
                    "medium: answer needs combining or inferring from the context. "
                    "low: context is missing or only partly relevant.")


# ---------------------------------------------------------------------------
# 2. THE PROMPT. Rules first, then the context, then the question.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an internal support assistant for a Customer Success team.
You answer questions using ONLY the context blocks provided below.

Rules:
1. Use only facts stated in the context. Do not use outside knowledge or guess.
2. If the context does not contain the answer, say so plainly and set confidence to "low".
3. Include specific numbers, limits, error codes and steps exactly as written in the context.
4. Keep the answer concise: a few sentences, or a short numbered list for procedures.
5. In "sources", list the source path of every context block you used, copied exactly.
6. If sources disagree, say so and prefer the most specific one (a resolved ticket or runbook
   over general docs for edge cases)."""

HUMAN_PROMPT = """Context:
{context}

Question: {question}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])


def format_context(docs: list[Document]) -> str:
    """Number each chunk and label it with its source path, so the model can cite it."""
    return "\n\n---\n\n".join(
        f"[{i}] source: {d.metadata['source']}\n{d.page_content}" for i, d in enumerate(docs, start=1))


# ---------------------------------------------------------------------------
# 3. THE CHAIN (LCEL): prompt | model-with-structured-output
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_chain():
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=os.getenv("GENERATION_MODEL", "gpt-4o-mini"), temperature=0)
    return PROMPT | llm.with_structured_output(RAGAnswer)


def generate(question: str, docs: list[Document]) -> RAGAnswer:
    if not docs:  # nothing retrieved: don't let the model answer from memory
        return RAGAnswer(answer="I couldn't find anything in the knowledge base about this.",
                         sources=[], confidence="low")

    result: RAGAnswer = get_chain().invoke({"context": format_context(docs), "question": question})

    # Guardrail: keep only sources that were actually in the context (no invented citations),
    # and remove duplicates while keeping order.
    allowed = {d.metadata["source"] for d in docs}
    result.sources = [s for s in dict.fromkeys(result.sources) if s in allowed]
    return result