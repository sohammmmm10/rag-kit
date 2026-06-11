"""rag-kit quickstart example — works with zero configuration."""

from rag_kit import RAGPipeline

# Create a pipeline with all defaults (no API keys needed!)
pipeline = RAGPipeline()

# Ingest some documents directly
pipeline.ingest_texts([
    "Python is a high-level programming language known for its readability. "
    "It supports multiple programming paradigms including procedural, "
    "object-oriented, and functional programming.",

    "Machine learning is a subset of artificial intelligence that enables "
    "systems to learn and improve from experience without being explicitly "
    "programmed. It focuses on developing algorithms that can access data "
    "and use it to learn for themselves.",

    "Retrieval Augmented Generation (RAG) is a technique that combines "
    "information retrieval with text generation. It retrieves relevant "
    "documents from a knowledge base and uses them as context for an LLM "
    "to generate more accurate, grounded responses.",

    "FastAPI is a modern Python web framework for building APIs. "
    "It is based on standard Python type hints, is very fast, and "
    "provides automatic interactive API documentation.",
])

print(f"Ingested! Total chunks in store: {pipeline.count()}\n")

# Query the pipeline
questions = [
    "What is RAG?",
    "Tell me about Python",
    "What is machine learning?",
]

for q in questions:
    print(f"Q: {q}")
    result = pipeline.query(q, top_k=2)
    print(f"A: {result.answer}")
    print(f"   Chunks retrieved: {len(result.retrieved_chunks)}")
    print()
