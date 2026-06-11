# rag-kit

`rag-kit` is a plug-and-play **Retrieval Augmented Generation** pipeline library for Python.

Load, chunk, embed, store, retrieve, and generate — all in one clean API.

## Why rag-kit?

- **Zero config** — works out of the box with sensible defaults.
- **Modular** — swap any component (loader, chunker, embedder, store, generator).
- **Lightweight** — no heavy dependencies by default.
- **Production-ready** — batch embedding, error handling, type hints everywhere.
- **Extensible** — bring your own components by extending base classes.

## Install

```bash
pip install -e .
```

With OpenAI support:

```bash
pip install -e ".[openai]"
```

With PDF support:

```bash
pip install -e ".[pdf]"
```

With ChromaDB (persistent vector store):

```bash
pip install -e ".[chromadb]"
```

With local sentence-transformers (no API key needed):

```bash
pip install -e ".[sentence-transformers]"
```

Install everything:

```bash
pip install -e ".[all]"
```

For development:

```bash
pip install -e ".[dev,all]"
```

## Quick Start

```python
from rag_kit import RAGPipeline

pipeline = RAGPipeline()

# Ingest documents
pipeline.ingest_texts([
    "Python is a high-level programming language.",
    "Machine learning is a subset of AI.",
    "RAG combines retrieval with generation.",
])

# Query
result = pipeline.query("What is RAG?")
print(result.answer)
print(f"Chunks retrieved: {len(result.retrieved_chunks)}")
```

## Load from Files

```python
from rag_kit import RAGPipeline
from rag_kit.loaders import TextLoader

pipeline = RAGPipeline(loader=TextLoader("docs/"))
stats = pipeline.ingest()
print(f"Ingested {stats.documents_loaded} docs, {stats.chunks_stored} chunks")

result = pipeline.query("What is the refund policy?")
print(result.answer)
```

## Load PDFs

```python
from rag_kit import RAGPipeline
from rag_kit.loaders import PDFLoader

pipeline = RAGPipeline(loader=PDFLoader("reports/"))
pipeline.ingest()
result = pipeline.query("What were Q4 earnings?")
```

## Load CSVs

```python
from rag_kit import RAGPipeline
from rag_kit.loaders import CSVLoader

pipeline = RAGPipeline(
    loader=CSVLoader("faq.csv", content_columns=["question", "answer"])
)
pipeline.ingest()
result = pipeline.query("How do I reset my password?")
```

## Load Markdown (split by headings)

```python
from rag_kit import RAGPipeline
from rag_kit.loaders import MarkdownLoader

pipeline = RAGPipeline(
    loader=MarkdownLoader("docs/", split_by_heading=True, heading_level=2)
)
pipeline.ingest()
result = pipeline.query("How to install?")
```

## Choose Your Chunking Strategy

```python
from rag_kit import RAGPipeline
from rag_kit.chunkers import FixedChunker, SentenceChunker, RecursiveChunker

# Fixed-size character chunks
pipeline = RAGPipeline(chunker=FixedChunker(chunk_size=512, chunk_overlap=64))

# Sentence-based chunks
pipeline = RAGPipeline(chunker=SentenceChunker(max_chunk_size=512, sentence_overlap=1))

# Recursive splitting (like LangChain)
pipeline = RAGPipeline(chunker=RecursiveChunker(chunk_size=512, chunk_overlap=64))
```

## Use OpenAI Embeddings + Generation

```python
import os
from rag_kit import RAGPipeline
from rag_kit.embedders import OpenAIEmbedder
from rag_kit.generators import OpenAIGenerator

api_key = os.environ["OPENAI_API_KEY"]

pipeline = RAGPipeline(
    embedder=OpenAIEmbedder(api_key=api_key),
    generator=OpenAIGenerator(api_key=api_key, model="gpt-4o-mini"),
)

pipeline.ingest_texts(["Your documents here..."])
result = pipeline.query("Your question here?")
print(result.answer)
```

## Use Local Embeddings (SentenceTransformers)

```python
from rag_kit import RAGPipeline
from rag_kit.embedders import SentenceTransformerEmbedder

pipeline = RAGPipeline(
    embedder=SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2"),
)

pipeline.ingest_texts(["Your documents..."])
result = pipeline.query("Your question?")
```

## Persistent Storage with ChromaDB

```python
from rag_kit import RAGPipeline
from rag_kit.stores import ChromaStore

pipeline = RAGPipeline(
    store=ChromaStore(collection_name="my-docs", persist_directory="./chroma_db"),
)

# Data persists across restarts!
pipeline.ingest_texts(["Important document content..."])
```

## Retrieve Without Generating

```python
pipeline = RAGPipeline()
pipeline.ingest_texts(["Doc 1...", "Doc 2..."])

# Just get the relevant chunks
chunks = pipeline.retrieve("search query", top_k=3)
for chunk in chunks:
    print(f"Score: {chunk.score:.4f} | {chunk.content[:80]}...")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     RAGPipeline                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  INGEST:   Loader → Chunker → Embedder → Store          │
│                                                         │
│  QUERY:    Embedder → Store (search) → Generator         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  Loaders:    TextLoader, PDFLoader, CSVLoader,           │
│              MarkdownLoader                              │
│                                                         │
│  Chunkers:   FixedChunker, SentenceChunker,              │
│              RecursiveChunker                             │
│                                                         │
│  Embedders:  DefaultEmbedder, OpenAIEmbedder,            │
│              SentenceTransformerEmbedder                  │
│                                                         │
│  Stores:     MemoryStore, ChromaStore                     │
│                                                         │
│  Generators: DefaultGenerator, OpenAIGenerator            │
└─────────────────────────────────────────────────────────┘
```

## CLI

```bash
rag-kit info
rag-kit ingest ./docs --glob "*.txt"
rag-kit query ./docs -q "What is RAG?" --top-k 3
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RAGKIT_CHUNK_SIZE` | `512` | Default chunk size |
| `RAGKIT_CHUNK_OVERLAP` | `64` | Default chunk overlap |
| `RAGKIT_TOP_K` | `5` | Default number of results |
| `RAGKIT_SIMILARITY_THRESHOLD` | `0.0` | Minimum similarity score |
| `RAGKIT_EMBEDDING_BATCH_SIZE` | `64` | Batch size for embeddings |

## Run Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

## Publish to PyPI

```bash
python -m build
twine upload dist/*
```

## License

MIT
