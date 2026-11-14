"""
End-to-end: question -> retrieve -> generate -> answer.

Try it:
    python -m src.pipeline "What's the maximum number of steps in a workflow?"
"""
import sys

from src.generate import generate
from src.retrieve import retrieve


def answer(question: str, top_k: int = 5, mode: str = "hybrid") -> dict:
    docs = retrieve(question, top_k=top_k, mode=mode)
    result = generate(question, docs)
    return {
        "question": question,
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        # The raw retrieved text is returned too: the evals need it to score faithfulness.
        "contexts": [d.page_content for d in docs],
        "retrieved_sources": [d.metadata["source"] for d in docs],
    }


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What's the maximum number of steps in a workflow?"
    out = answer(question)
    print(f"\nQ: {out['question']}\n")
    print(f"A: {out['answer']}\n")
    print(f"Confidence: {out['confidence']}")
    print("Sources:")
    for s in out["sources"]:
        print(f"  - {s}")