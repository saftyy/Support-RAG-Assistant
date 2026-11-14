"""
Compare retrieval strategies on the 50 test queries, WITHOUT calling the LLM.

For each query we run vector search and BM25 once (top 20 each), then try
several ways of combining them. Scoring uses the test set's expected_sources:
  - hit@5     an expected document is in the top 5
  - recall@5  fraction of expected documents in the top 5
  - MRR@5     1 / rank of the first expected document (1.0 = always ranked first)

Costs only 50 query embeddings (a fraction of a cent).

Run:
    python -m evals.compare_retrieval
"""
import json
from pathlib import Path

from src.retrieve import CANDIDATES, bm25_search, reciprocal_rank_fusion, vector_search

TEST_SET = Path("evals/test_set.json")
RESULTS_DIR = Path("evals/results")
TOP_K = 5

# name -> function(vector_results, bm25_results) -> top-5 documents
STRATEGIES = {
    "vector only": lambda v, b: v[:TOP_K],
    "bm25 only": lambda v, b: b[:TOP_K],
    "hybrid RRF k=60, weights 1.0/1.0": lambda v, b: reciprocal_rank_fusion([v, b], TOP_K, [1.0, 1.0], 60),
    "hybrid RRF k=60, weights 1.0/0.5": lambda v, b: reciprocal_rank_fusion([v, b], TOP_K, [1.0, 0.5], 60),
    "hybrid RRF k=10, weights 1.0/1.0": lambda v, b: reciprocal_rank_fusion([v, b], TOP_K, [1.0, 1.0], 10),
    "hybrid RRF k=10, weights 1.0/0.5": lambda v, b: reciprocal_rank_fusion([v, b], TOP_K, [1.0, 0.5], 10),
}


def score(docs, expected: set[str]) -> dict:
    sources = [d.metadata["source"] for d in docs]
    first = next((rank for rank, s in enumerate(sources, start=1) if s in expected), None)
    return {"hit": float(first is not None),
            "recall": len(expected & set(sources)) / len(expected),
            "mrr": 1.0 / first if first else 0.0}


def main() -> None:
    queries = json.loads(TEST_SET.read_text(encoding="utf-8"))["queries"]
    totals = {name: {"hit": 0.0, "recall": 0.0, "mrr": 0.0} for name in STRATEGIES}
    misses = {name: [] for name in STRATEGIES}

    for i, q in enumerate(queries, start=1):
        v, b = vector_search(q["query"], k=CANDIDATES), bm25_search(q["query"], k=CANDIDATES)
        for name, strategy in STRATEGIES.items():
            s = score(strategy(v, b), set(q["expected_sources"]))
            for key in totals[name]:
                totals[name][key] += s[key]
            if not s["hit"]:
                misses[name].append(q["id"])
        print(f"[{i}/{len(queries)}] {q['id']}", end="\r")

    n = len(queries)
    rows = [{"strategy": name, **{k: round(val / n, 3) for k, val in t.items()}, "misses": misses[name]}
            for name, t in totals.items()]

    lines = ["# Retrieval comparison (top 5, no LLM)", "",
             "| Strategy | hit@5 | recall@5 | MRR@5 | Missed queries |", "|---|---|---|---|---|"]
    lines += [f"| {r['strategy']} | {r['hit']:.2f} | {r['recall']:.2f} | {r['mrr']:.2f} | "
              f"{', '.join(r['misses']) or '-'} |" for r in rows]
    table = "\n".join(lines)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (RESULTS_DIR / "retrieval_comparison.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()