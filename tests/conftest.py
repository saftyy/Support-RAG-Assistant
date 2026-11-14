"""Shared test setup: run every test from the project root so relative paths
(./corpus, data/chunks.json, evals/test_set.json) resolve the same way as the CLI commands."""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

# Tests marked with this are skipped when API keys are missing (e.g. on a fresh clone or in CI).
needs_api = pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("PINECONE_API_KEY")),
    reason="needs OPENAI_API_KEY and PINECONE_API_KEY in .env")