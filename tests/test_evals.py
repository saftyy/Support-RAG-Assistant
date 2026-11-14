"""The eval harness runs end-to-end (pipeline + RAGAS + report files) on a small sample."""
import json
import sys

from conftest import needs_api
from evals import harness


@needs_api
def test_harness_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)   # don't overwrite the real results
    monkeypatch.setattr(sys, "argv", ["harness", "--limit", "2"])
    harness.main()

    data = json.loads((tmp_path / "results_hybrid.json").read_text(encoding="utf-8"))
    assert len(data["queries"]) == 2
    for metric in ("faithfulness", "answer_relevancy", "context_precision"):
        assert metric in data["summary"]["overall"]
    assert (tmp_path / "summary_hybrid.md").exists()