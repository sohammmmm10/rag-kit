"""Example: Ingest files from a directory."""

from pathlib import Path

from rag_bridge_kit import RAGPipeline
from rag_bridge_kit.loaders import TextLoader, MarkdownLoader
from rag_bridge_kit.chunkers import RecursiveChunker

# --- Load text files ---
# Change this path to your own docs directory
DOCS_DIR = Path(__file__).parent / "sample_docs"

# Create sample docs for this example
DOCS_DIR.mkdir(exist_ok=True)
(DOCS_DIR / "intro.txt").write_text(
    "rag-kit is a plug-and-play RAG pipeline library for Python. "
    "It lets you load, chunk, embed, store, retrieve, and generate "
    "all in one clean API."
)
(DOCS_DIR / "install.txt").write_text(
    "Install rag-kit with pip: pip install rag-kit. "
    "For OpenAI support: pip install rag-kit[openai]. "
    "For PDF support: pip install rag-kit[pdf]."
)
(DOCS_DIR / "usage.txt").write_text(
    "Create a RAGPipeline, call ingest() to load your documents, "
    "then call query() to ask questions. The pipeline handles "
    "chunking, embedding, storage, retrieval, and generation."
)

print("=== Text File Ingestion ===\n")
pipeline = RAGPipeline(
    loader=TextLoader(DOCS_DIR, glob_pattern="*.txt"),
    chunker=RecursiveChunker(chunk_size=200, chunk_overlap=30),
)

stats = pipeline.ingest()
print(f"Documents loaded: {stats.documents_loaded}")
print(f"Chunks created:   {stats.chunks_created}")
print(f"Chunks stored:    {stats.chunks_stored}")
print(f"Duration:         {stats.duration_seconds:.3f}s\n")

result = pipeline.query("How do I install rag-kit?")
print(f"Q: How do I install rag-kit?")
print(f"A: {result.answer}\n")

# --- Load Markdown files ---
print("=== Markdown Ingestion (split by heading) ===\n")

MD_DIR = Path(__file__).parent / "sample_md"
MD_DIR.mkdir(exist_ok=True)
(MD_DIR / "guide.md").write_text(
    "# rag-kit Guide\n\n"
    "Introduction to rag-kit.\n\n"
    "## Installation\n\n"
    "Run pip install rag-kit.\n\n"
    "## Quick Start\n\n"
    "Create a pipeline and start querying.\n\n"
    "## Advanced Usage\n\n"
    "Customize chunkers, embedders, and stores."
)

md_pipeline = RAGPipeline(
    loader=MarkdownLoader(MD_DIR, split_by_heading=True),
    chunker=RecursiveChunker(chunk_size=200, chunk_overlap=20),
)

stats = md_pipeline.ingest()
print(f"Sections loaded: {stats.documents_loaded}")
print(f"Chunks stored:   {stats.chunks_stored}\n")

result = md_pipeline.query("How to install?")
print(f"Q: How to install?")
print(f"A: {result.answer}")
