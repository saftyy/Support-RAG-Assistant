"""
Ingestion: load the corpus (Markdown, PDF, HTML) -> chunk -> embed -> upsert to Pinecone.

Run with:
    python -m src.ingest --corpus ./corpus            # full run (calls OpenAI + Pinecone)
    python -m src.ingest --corpus ./corpus --dry-run  # load + chunk only, free
"""
import argparse
import json
import os
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 100   # characters shared between neighbouring chunks
CHUNKS_PATH = Path("data/chunks.json")  # local copy of every chunk (BM25 needs it later)


# ---------------------------------------------------------------------------
# 1. LOADERS: one function per file type. Each returns a LangChain Document
#    with the text plus metadata. "source" is the path relative to the corpus,
#    e.g. "product-docs/workflows/creating-workflows.md", which is exactly the
#    format the test set uses in expected_sources.
# ---------------------------------------------------------------------------
def load_markdown(path: Path, corpus: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return Document(page_content=text, metadata={
        "source": path.relative_to(corpus).as_posix(), "doc_type": "markdown"})


def load_pdf(path: Path, corpus: Path) -> Document | None:
    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if not text:
        return None  # scanned image with no text layer; we skip OCR on purpose
    return Document(page_content=text, metadata={
        "source": path.relative_to(corpus).as_posix(), "doc_type": "pdf"})


def load_html(path: Path, corpus: Path) -> Document:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    text = soup.get_text("\n", strip=True)
    return Document(page_content=text, metadata={
        "source": path.relative_to(corpus).as_posix(), "doc_type": "html", "title": title})


def load_corpus(corpus: Path) -> tuple[list[Document], list[str]]:
    docs, skipped = [], []
    for path in sorted(corpus.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".md" and path.name != "README.md":   # corpus README is not company knowledge
            docs.append(load_markdown(path, corpus))
        elif suffix == ".pdf":
            doc = load_pdf(path, corpus)
            if doc is None:
                skipped.append(path.relative_to(corpus).as_posix())
            else:
                docs.append(doc)
        elif suffix in (".html", ".htm"):
            docs.append(load_html(path, corpus))
    return docs, skipped


# ---------------------------------------------------------------------------
# 2. CHUNKING: Markdown is split on its headings first (each "##" section is one
#    topic), then anything still too long is split by size. PDFs and HTML have
#    no reliable headings, so they are split by size only.
#    Every chunk gets a short header ("Source: ... / Section: ...") so a chunk
#    taken from the middle of a document still says what it is about.
# ---------------------------------------------------------------------------
size_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)


def chunk_document(doc: Document) -> list[Document]:
    source = doc.metadata["source"]
    if doc.metadata["doc_type"] == "markdown":
        sections = header_splitter.split_text(doc.page_content)
    else:
        sections = [Document(page_content=doc.page_content)]

    chunks = []
    for section in sections:
        heading = " > ".join(section.metadata[k] for k in ("h1", "h2", "h3") if k in section.metadata)
        heading = heading or doc.metadata.get("title", "")
        for piece in size_splitter.split_text(section.page_content):
            context = f"Source: {source}" + (f"\nSection: {heading}" if heading else "")
            chunks.append(Document(
                page_content=f"{context}\n\n{piece}",
                metadata={**doc.metadata, "section": heading,
                          "chunk_id": f"{source}::{len(chunks)}"}))
    return chunks


# ---------------------------------------------------------------------------
# 3. EMBED + INDEX: turn each chunk into a vector with OpenAI and store it in
#    Pinecone. Chunk IDs are fixed (source + position), so running ingest
#    twice overwrites the same vectors instead of creating duplicates.
# ---------------------------------------------------------------------------
def index_chunks(chunks: list[Document]) -> None:
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    index_name = os.environ["PINECONE_INDEX_NAME"]
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # size of text-embedding-3-small vectors
            metric="cosine",
            spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"),
                                region=os.getenv("PINECONE_REGION", "us-east-1")))

    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the corpus into Pinecone.")
    parser.add_argument("--corpus", default="./corpus", help="Path to the corpus folder")
    parser.add_argument("--dry-run", action="store_true", help="Load and chunk only; skip embedding")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    docs, skipped = load_corpus(corpus)
    chunks = [c for d in docs for c in chunk_document(d)]

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(
        [{"text": c.page_content, "metadata": c.metadata} for c in chunks], indent=2), encoding="utf-8")

    by_type = {}
    for d in docs:
        by_type[d.metadata["doc_type"]] = by_type.get(d.metadata["doc_type"], 0) + 1
    print(f"Loaded {len(docs)} documents: {by_type}")
    print(f"Skipped {len(skipped)} scanned PDFs (no text layer, OCR not enabled):")
    for s in skipped:
        print(f"  - {s}")
    print(f"Created {len(chunks)} chunks -> saved to {CHUNKS_PATH}")

    if args.dry_run:
        print("Dry run: skipping embedding and Pinecone upload.")
        return
    index_chunks(chunks)
    print(f"Indexed {len(chunks)} chunks into Pinecone index '{os.environ['PINECONE_INDEX_NAME']}'.")


if __name__ == "__main__":
    main()