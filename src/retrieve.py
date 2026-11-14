"""
Retrieval: find the chunks most relevant to a question.

Two modes, so we can measure the improvement:
  - "vector": semantic search in Pinecone only (the baseline / "floor")
  - "hybrid": vector search + BM25 keyword search, merged with Reciprocal Rank Fusion

Try it:
    python -m src.retrieve "What's the maximum number of steps in a workflow?"
"""
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
import snowballstemmer
from rank_bm25 import BM25Okapi

load_dotenv()

CHUNKS_PATH = Path("data/chunks.json")  # written by src/ingest.py
CANDIDATES = 20  # how many results each method returns before merging

# Fusion settings. Chosen by measurement: see evals/compare_retrieval.py
RRF_K = 10           # smaller = top ranks matter more; 60 is the common default
VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 0.5

# Words too common to help keyword search ("how", "does", "the", ...)
STOPWORDS = set("""a an the is are was were be been do does did how what when where which who why
can could should would will i we you it its of to in on for with from at by and or our my your
this that these those there their me us much many about as if not no any""".split())
_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop stopwords, and stem ("plans" -> "plan").
    Stemming fixed a measured failure: "Pro plan cost" did not match "Plans and Pricing"."""
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
    return _stemmer.stemWords(words)


# ---------------------------------------------------------------------------
# 1. VECTOR SEARCH (Pinecone): good at meaning, e.g. "can't log in" ~ "SSO redirect loop"
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_vector_store():
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    return PineconeVectorStore(index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings)


def vector_search(query: str, k: int = CANDIDATES) -> list[Document]:
    return get_vector_store().similarity_search(query, k=k)


# ---------------------------------------------------------------------------
# 2. BM25 KEYWORD SEARCH (local): good at exact terms, e.g. error codes, "HX-1042", "429"
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_bm25() -> tuple[BM25Okapi, list[Document]]:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"{CHUNKS_PATH} not found. Run: python -m src.ingest --corpus ./corpus")
    raw = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in raw]
    return BM25Okapi([tokenize(d.page_content) for d in docs]), docs


def bm25_search(query: str, k: int = CANDIDATES) -> list[Document]:
    bm25, docs = get_bm25()
    scores = bm25.get_scores(tokenize(query))
    best = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in best if scores[i] > 0]


# ---------------------------------------------------------------------------
# 3. FUSION: Reciprocal Rank Fusion. Each chunk scores 1 / (60 + rank) in each
#    list it appears in (times that method's weight), and the scores are added. A chunk ranked well by BOTH
#    methods rises to the top. It uses ranks, not raw scores, because vector
#    similarity and BM25 scores are on completely different scales.
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(result_lists: list[list[Document]], top_k: int,
                           weights: list[float] | None = None, k: int = RRF_K) -> list[Document]:
    weights = weights or [1.0] * len(result_lists)
    scores: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for results, weight in zip(result_lists, weights):
        for rank, doc in enumerate(results, start=1):
            cid = doc.metadata["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            by_id.setdefault(cid, doc)
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [by_id[cid] for cid in ranked]


def retrieve(query: str, top_k: int = 5, mode: str = "hybrid") -> list[Document]:
    """Main entry point used by the pipeline and the evals."""
    if mode == "vector":
        return vector_search(query, k=top_k)
    if mode == "bm25":
        return bm25_search(query, k=top_k)
    if mode == "hybrid":
        return reciprocal_rank_fusion([vector_search(query), bm25_search(query)], top_k,
                                      weights=[VECTOR_WEIGHT, BM25_WEIGHT], k=RRF_K)
    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What's the maximum number of steps in a workflow?"
    for mode in ("vector", "hybrid"):
        print(f"\n=== {mode} ===")
        for i, doc in enumerate(retrieve(question, mode=mode), start=1):
            print(f"{i}. {doc.metadata['source']}  |  {doc.metadata.get('section', '')}")