"""Interactive RAG chat entry point.

Usage
-----
    uv run python main.py

The first launch will build the vector index if it doesn't exist yet.
Type your question at the `you >` prompt. Type `exit` (or press Ctrl-D) to quit.
"""

from __future__ import annotations

from llama_index.core import Settings

from ragm4.config import TOP_K
from ragm4.embeddings import build_embed_model
from ragm4.llm import MLXQwenLLM
from ragm4.pipeline import build_or_load_index


def _banner() -> None:
    print("=" * 60)
    print(" RAGM4 — local RAG chat (Qwen2.5 · nomic-embed · Chroma)")
    print("=" * 60)
    print(" Type your question, or 'exit' to quit.\n")


def main() -> None:
    # ---- 1. Configure the LlamaIndex global settings --------------------
    print("[main] Loading embedding model …")
    Settings.embed_model = build_embed_model()

    print("[main] Loading Qwen2.5 through mlx-lm (this can take a moment) …")
    Settings.llm = MLXQwenLLM()

    # ---- 2. Build or reopen the persistent vector index -----------------
    index = build_or_load_index()

    # ---- 3. Wire a query engine on top of the index ---------------------
    query_engine = index.as_query_engine(similarity_top_k=TOP_K)

    # ---- 4. REPL --------------------------------------------------------
    _banner()
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

    print("Goodbye.")


if __name__ == "__main__":
    main()
