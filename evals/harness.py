"""
Eval harness: runs the 50-query test set through the pipeline and scores it.

Scores (RAGAS, judged by an LLM, 0 to 1, higher is better):
  - faithfulness        Is every claim in the answer supported by the retrieved context?
  - answer_relevancy    Does the answer actually address the question?
  - context_precision   Are the relevant chunks ranked near the top of what we retrieved?

Plus two simple, free retrieval checks against the test set's expected_sources:
  - source_hit          At least one expected source document was retrieved (0 or 1)
  - source_recall       Fraction of expected source documents that were retrieved

Writes to evals/results/:
  - results_<mode>.json   machine-readable: config, averages, every query
  - summary_<mode>.md     human-readable: table + worst cases

Run:
    python -m evals.harness                      # all 50 queries, hybrid retrieval
    python -m evals.harness --limit 5            # quick, cheap smoke test
    python -m evals.harness --mode vector        # baseline, to compare before vs after
"""
import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline import answer

load_dotenv()

TEST_SET = Path("evals/test_set.json")
RESULTS_DIR = Path("evals/results")
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]


def load_test_set(limit: int | None = None) -> list[dict]:
    queries = json.loads(TEST_SET.read_text(encoding="utf-8"))["queries"]
    return queries[:limit] if limit else queries


# ---------------------------------------------------------------------------
# 1. RUN: send every test question through the real pipeline.
# ---------------------------------------------------------------------------
def run_pipeline(queries: list[dict], mode: str, top_k: int) -> list[dict]:
    rows = []
    for i, q in enumerate(queries, start=1):
        start = time.perf_counter()
        out = answer(q["query"], top_k=top_k, mode=mode)
        latency = time.perf_counter() - start

        expected = set(q["expected_sources"])
        retrieved = set(out["retrieved_sources"])
        rows.append({
            "id": q["id"],
            "difficulty": q.get("difficulty"),
            "query": q["query"],
            "ground_truth": q["ground_truth_answer"],
            "answer": out["answer"],
            "confidence": out["confidence"],
            "cited_sources": out["sources"],
            "retrieved_sources": out["retrieved_sources"],
            "expected_sources": q["expected_sources"],
            "contexts": out["contexts"],
            "source_hit": float(bool(expected & retrieved)),
            "source_recall": len(expected & retrieved) / len(expected) if expected else 0.0,
            "latency_s": round(latency, 2),
        })
        print(f"[{i}/{len(queries)}] {q['id']} hit={rows[-1]['source_hit']:.0f} "
              f"conf={out['confidence']} ({latency:.1f}s)")
    return rows


# ---------------------------------------------------------------------------
# 2. SCORE: RAGAS uses an LLM as a judge to grade each answer.
# ---------------------------------------------------------------------------
def score_with_ragas(rows: list[dict]) -> None:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy

    dataset = EvaluationDataset.from_list([{
        "user_input": r["query"],
        "response": r["answer"],
        "retrieved_contexts": r["contexts"],
        "reference": r["ground_truth"],
    } for r in rows])

    judge = LangchainLLMWrapper(ChatOpenAI(model=os.getenv("GENERATION_MODEL", "gpt-4o-mini"), temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")))

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference()],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=4, timeout=120),  # few workers = fewer rate-limit errors
    )
    df = result.to_pandas()
    # Column names differ slightly between RAGAS versions; map them to our names.
    columns = {"faithfulness": "faithfulness",
               "answer_relevancy": "answer_relevancy",
               "llm_context_precision_with_reference": "context_precision"}
    for row, (_, scores) in zip(rows, df.iterrows()):
        for ragas_name, our_name in columns.items():
            value = scores.get(ragas_name)
            row[our_name] = None if value is None or math.isnan(value) else round(float(value), 3)


# ---------------------------------------------------------------------------
# 3. REPORT: averages + JSON + markdown.
# ---------------------------------------------------------------------------
def average(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows: list[dict]) -> dict:
    keys = RAGAS_METRICS + ["source_hit", "source_recall", "latency_s"]
    overall = {k: average(rows, k) for k in keys}
    by_difficulty = {}
    for level in ("easy", "medium", "hard"):
        subset = [r for r in rows if r["difficulty"] == level]
        if subset:
            by_difficulty[level] = {"n": len(subset), **{k: average(subset, k) for k in keys}}
    return {"overall": overall, "by_difficulty": by_difficulty}


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def write_markdown(path: Path, config: dict, summary: dict, rows: list[dict]) -> None:
    o = summary["overall"]
    lines = [
        f"# Eval summary ({config['mode']} retrieval)",
        "",
        f"Run at {config['timestamp']} | {config['n_queries']} queries | top_k={config['top_k']} | "
        f"model={config['generation_model']}",
        "",
        "| Metric | Score | Target |",
        "|---|---|---|",
        f"| Faithfulness | {fmt(o['faithfulness'])} | >= 0.70 |",
        f"| Answer relevancy | {fmt(o['answer_relevancy'])} | |",
        f"| Context precision | {fmt(o['context_precision'])} | >= 0.60 |",
        f"| Source hit rate | {fmt(o['source_hit'])} | |",
        f"| Source recall | {fmt(o['source_recall'])} | |",
        f"| Avg latency (s) | {fmt(o['latency_s'])} | |",
        "",
        "## By difficulty",
        "",
        "| Difficulty | n | Faithfulness | Answer relevancy | Context precision | Source hit |",
        "|---|---|---|---|---|---|",
    ]
    for level, s in summary["by_difficulty"].items():
        lines.append(f"| {level} | {s['n']} | {fmt(s['faithfulness'])} | {fmt(s['answer_relevancy'])} | "
                     f"{fmt(s['context_precision'])} | {fmt(s['source_hit'])} |")

    def badness(r):  # lowest combined score first
        return sum(r.get(k) if r.get(k) is not None else 0 for k in RAGAS_METRICS) + r["source_hit"]

    lines += ["", "## Weakest 5 queries (start failure analysis here)", ""]
    for r in sorted(rows, key=badness)[:5]:
        lines += [
            f"### {r['id']} ({r['difficulty']}): {r['query']}",
            f"- Scores: faithfulness {fmt(r.get('faithfulness'))}, relevancy {fmt(r.get('answer_relevancy'))}, "
            f"context precision {fmt(r.get('context_precision'))}, source hit {r['source_hit']:.0f}",
            f"- Expected sources: {', '.join(r['expected_sources'])}",
            f"- Retrieved sources: {', '.join(dict.fromkeys(r['retrieved_sources']))}",
            f"- Ground truth: {r['ground_truth']}",
            f"- Our answer: {r['answer']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the eval set and write results.")
    parser.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N queries")
    parser.add_argument("--skip-ragas", action="store_true", help="Only compute the free retrieval checks")
    args = parser.parse_args()

    queries = load_test_set(args.limit)
    rows = run_pipeline(queries, mode=args.mode, top_k=args.top_k)
    if not args.skip_ragas:
        print("\nScoring with RAGAS (LLM judge)...")
        score_with_ragas(rows)

    config = {
        "mode": args.mode, "top_k": args.top_k, "n_queries": len(rows),
        "generation_model": os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    summary = summarize(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"results_{args.mode}.json"
    md_path = RESULTS_DIR / f"summary_{args.mode}.md"
    json_path.write_text(json.dumps({"config": config, "summary": summary, "queries": rows}, indent=2),
                         encoding="utf-8")
    write_markdown(md_path, config, summary, rows)

    o = summary["overall"]
    print(f"\nFaithfulness {fmt(o['faithfulness'])} | Answer relevancy {fmt(o['answer_relevancy'])} | "
          f"Context precision {fmt(o['context_precision'])} | Source hit {fmt(o['source_hit'])}")
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()