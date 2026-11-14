"""Generation produces valid structured output."""
from langchain_core.documents import Document

from conftest import needs_api
from src import generate as gen
from src.generate import RAGAnswer, generate
from src.retrieve import retrieve


def test_empty_context_returns_low_confidence_without_calling_llm():
    result = generate("anything", [])
    assert result.confidence == "low" and result.sources == []


def test_invented_sources_are_removed(monkeypatch):
    """The guardrail drops any cited source that was not actually retrieved."""
    class FakeChain:
        def invoke(self, _):
            return RAGAnswer(answer="x", confidence="high",
                             sources=["real.md", "made-up.md", "real.md"])
    monkeypatch.setattr(gen, "get_chain", lambda: FakeChain())
    docs = [Document(page_content="text", metadata={"source": "real.md"})]
    assert generate("q", docs).sources == ["real.md"]


@needs_api
def test_real_answer_is_valid_and_grounded():
    question = "What's the maximum number of steps allowed in a single workflow?"
    result = generate(question, retrieve(question))
    assert isinstance(result, RAGAnswer)
    assert result.confidence in {"low", "medium", "high"}
    assert "100" in result.answer
    assert "product-docs/workflows/creating-workflows.md" in result.sources