"""Command-line interface for rag-bridge-kit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_ingest(args: argparse.Namespace) -> None:
    from .loaders import TextLoader
    from .pipeline import RAGPipeline

    loader = TextLoader(args.path, glob_pattern=args.glob or "*.txt")
    pipeline = RAGPipeline(loader=loader)
    stats = pipeline.ingest()

    print(f"Ingested {stats.documents_loaded} document(s)")
    print(f"  Chunks created:  {stats.chunks_created}")
    print(f"  Embeddings:      {stats.embeddings_generated}")
    print(f"  Stored:          {stats.chunks_stored}")
    print(f"  Duration:        {stats.duration_seconds:.3f}s")


def _cmd_query(args: argparse.Namespace) -> None:
    from .loaders import TextLoader
    from .pipeline import RAGPipeline

    loader = TextLoader(args.path, glob_pattern=args.glob or "*.txt")
    pipeline = RAGPipeline(loader=loader)
    print(f"Loading documents from {args.path}...")
    stats = pipeline.ingest()
    print(f"Ingested {stats.chunks_stored} chunks.\n")

    result = pipeline.query(args.question, top_k=args.top_k)
    print(result.answer)


def _cmd_info(args: argparse.Namespace) -> None:
    from . import __version__

    print(f"rag-bridge-kit v{__version__}")
    print()
    print("Components:")
    print("  Loaders:    TextLoader, PDFLoader, CSVLoader, MarkdownLoader")
    print("  Chunkers:   FixedChunker, SentenceChunker, RecursiveChunker")
    print("  Embedders:  DefaultEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder")
    print("  Stores:     MemoryStore, ChromaStore")
    print("  Retrievers: SimilarityRetriever")
    print("  Generators: DefaultGenerator, OpenAIGenerator")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rag-bridge-kit",
        description="rag-bridge-kit: Plug-and-play RAG pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # --- ingest ---
    ingest_parser = sub.add_parser("ingest", help="Ingest documents into the pipeline")
    ingest_parser.add_argument("path", help="Path to file or directory")
    ingest_parser.add_argument("--glob", default="*.txt", help="Glob pattern for files")

    # --- query ---
    query_parser = sub.add_parser("query", help="Ingest + query documents")
    query_parser.add_argument("path", help="Path to file or directory")
    query_parser.add_argument("-q", "--question", required=True, help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    query_parser.add_argument("--glob", default="*.txt", help="Glob pattern for files")

    # --- info ---
    sub.add_parser("info", help="Show rag-bridge-kit information")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "query":
        _cmd_query(args)
    elif args.command == "info":
        _cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
