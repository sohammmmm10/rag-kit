"""Example: Full RAG pipeline with OpenAI embeddings + generation.

Requires:
    pip install rag-kit[openai]
    export OPENAI_API_KEY="sk-..."
"""

import os

from rag_kit import RAGPipeline
from rag_kit.chunkers import SentenceChunker
from rag_kit.embedders import OpenAIEmbedder
from rag_kit.generators import OpenAIGenerator
from rag_kit.stores import MemoryStore

# Setup OpenAI components
api_key = os.environ["OPENAI_API_KEY"]

pipeline = RAGPipeline(
    chunker=SentenceChunker(max_chunk_size=300, sentence_overlap=1),
    embedder=OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small"),
    store=MemoryStore(),
    generator=OpenAIGenerator(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.3,
    ),
)

# Ingest some knowledge
pipeline.ingest_texts([
    "The refund policy allows returns within 30 days of purchase. "
    "Items must be in original condition with tags attached. "
    "Digital products are non-refundable after download.",

    "Shipping takes 3-5 business days for domestic orders. "
    "International shipping takes 7-14 business days. "
    "Express shipping is available for an additional fee.",

    "Customer support is available Monday through Friday, 9 AM to 5 PM EST. "
    "You can reach us by email at support@example.com or by phone at 1-800-EXAMPLE.",

    "Our premium plan includes unlimited API calls, priority support, "
    "and access to all features. The basic plan is free and includes "
    "1000 API calls per month.",
])

print(f"Total chunks stored: {pipeline.count()}\n")

# Ask questions
questions = [
    "What is the refund policy?",
    "How long does shipping take?",
    "How do I contact support?",
    "What's included in the premium plan?",
]

for q in questions:
    result = pipeline.query(q, top_k=3)
    print(f"Q: {q}")
    print(f"A: {result.answer}")
    if result.tokens_input:
        print(f"   Tokens: {result.tokens_input} in / {result.tokens_output} out")
    print()
