"""Retrieval returns results for every test query."""
import json
from pathlib import Path

from langchain_core.documents import Document

from conftest import needs_api
from src.retrieve import bm25_search, reciprocal_rank_fusion, retrieve, tokenize

QUERIES = json.loads(Path("evals/test_set.json").read_text(encoding="utf-8"))["queries"]


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("How much does the Pro plan cost?") == tokenize("pro plans costs")


def test_rrf_rewards_documents_found_by_both_methods():
    a, b, c = (Document(page_content=x, metadata={"chunk_id": x}) for x in "abc")
    fused = reciprocal_rank_fusion([[a, b], [b, c]], top_k=3)
    assert fused[0].metadata["chunk_id"] == "b"


def test_bm25_returns_results_for_every_query():
    for q in QUERIES:
        assert len(bm25_search(q["query"])) >= 1, q["id"]


@needs_api
def test_hybrid_returns_results_for_every_query():
    for q in QUERIES:
        docs = retrieve(q["query"], top_k=5)
        assert len(docs) >= 1, q["id"]
        assert all("source" in d.metadata for d in docs)