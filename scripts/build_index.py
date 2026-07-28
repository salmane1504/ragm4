"""Offline script: build (or rebuild) the ChromaDB vector index.

Run this once after cloning the repo or whenever the corpus / chunking
settings change:

    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the `ragm4` package importable when this file is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.core import Settings  # noqa: E402

from ragm4.embeddings import build_embed_model  # noqa: E402
from ragm4.pipeline import build_or_load_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAGM4 vector index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the existing Chroma collection and rebuild from scratch.",
    )
    args = parser.parse_args()

    # The embedding model must be configured *before* indexing — LlamaIndex
    # picks it up from the global Settings object.
    print("[build_index] Loading embedding model …")
    Settings.embed_model = build_embed_model()

    # The LLM is not needed for indexing; disable it to avoid the default
    # OpenAI fallback.
    Settings.llm = None

    build_or_load_index(force_rebuild=args.rebuild)
    print("[build_index] Index ready.")


if __name__ == "__main__":
    main()
