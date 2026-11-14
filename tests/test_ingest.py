"""Ingestion runs without errors and produces sensible chunks. No API calls."""
import sys
from pathlib import Path

from src import ingest

CORPUS = Path("corpus")


def test_loads_every_format_and_skips_scanned_pdfs():
    docs, skipped = ingest.load_corpus(CORPUS)
    types = {d.metadata["doc_type"] for d in docs}
    assert types == {"markdown", "pdf", "html"}
    assert len(docs) == 95                      # 60 md + 20 text PDFs + 15 html
    assert len(skipped) == 5                    # the 5 scanned PDFs, skipped on purpose
    assert all(d.page_content.strip() for d in docs)


def test_sources_match_test_set_format():
    """Sources must look like 'product-docs/...' so evals can compare them to expected_sources."""
    docs, _ = ingest.load_corpus(CORPUS)
    sources = {d.metadata["source"] for d in docs}
    assert "product-docs/workflows/creating-workflows.md" in sources
    assert all("\\" not in s for s in sources)  # forward slashes on Windows too


def test_chunks_have_ids_and_reasonable_size():
    docs, _ = ingest.load_corpus(CORPUS)
    chunks = [c for d in docs for c in ingest.chunk_document(d)]
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))            # unique ids -> re-ingesting overwrites, never duplicates
    # chunk text = small "Source/Section" header + at most CHUNK_SIZE characters of content
    assert max(len(c.page_content) for c in chunks) < ingest.CHUNK_SIZE + 300
    assert all(c.page_content.startswith("Source: ") for c in chunks)


def test_ingest_cli_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ingest", "--corpus", str(CORPUS), "--dry-run"])
    ingest.main()
    out = capsys.readouterr().out
    assert "Loaded 95 documents" in out
    assert Path("data/chunks.json").exists()