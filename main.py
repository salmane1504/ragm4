"""Interactive RAG chat entry point.

Usage
-----
    uv run python main.py

The chat model is read from the `.env` file at the repository root:

    OLLAMA_MODEL=qwen2.5:14b

If that key is missing, the program stops and tells you to pull a model with
`ollama pull …` and to write its name in `.env`.

Starting this script also starts the Ollama daemon (when it isn't already
running) and loads the model into memory; typing `exit` unloads the model and
shuts the daemon down again, giving the RAM back.

The first launch will build the vector index if it doesn't exist yet.
Type your question at the `you >` prompt. Type `exit` (or press Ctrl-D) to quit.
"""

from __future__ import annotations

import sys

from llama_index.core import Settings

from ragm4.config import TOP_K, MissingOllamaModelError, require_ollama_model
from ragm4.embeddings import build_embed_model
from ragm4.llm import build_llm
from ragm4.ollama_service import OllamaService, OllamaServiceError
from ragm4.pipeline import build_or_load_index


def _banner(model: str) -> None:
    print("=" * 60)
    print(" RAGM4 — local RAG chat (Ollama · nomic-embed · Chroma)")
    print(f" Model: {model}")
    print("=" * 60)
    print(" Type your question, or 'exit' to quit.\n")


def _chat(model: str) -> None:
    """Run the REPL against an already-served model."""
    # ---- 1. Configure the LlamaIndex global settings --------------------
    print("[main] Loading embedding model …")
    Settings.embed_model = build_embed_model()
    Settings.llm = build_llm()

    # ---- 2. Build or reopen the persistent vector index -----------------
    index = build_or_load_index()

    # ---- 3. Wire a query engine on top of the index ---------------------
    query_engine = index.as_query_engine(similarity_top_k=TOP_K)

    # ---- 4. REPL --------------------------------------------------------
    _banner(model)
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            break

        response = query_engine.query(question)
        print(f"\nbot > {response}\n")


def main() -> None:
    try:
        # Fail fast with a clear message before anything expensive happens.
        model = require_ollama_model()
        # Serve the model for the lifetime of the session; the context manager
        # unloads it (and stops the daemon it started) on the way out.
        with OllamaService() as service:
            _chat(service.model)
    except (MissingOllamaModelError, OllamaServiceError) as err:
        print(f"\n[error] {err}\n", file=sys.stderr)
        raise SystemExit(1) from err

    print(f"Goodbye. ({model} unloaded)")


if __name__ == "__main__":
    main()
