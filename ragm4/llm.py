"""LlamaIndex LLM backed by a local Ollama daemon.

Inference is delegated to the Ollama server (`ollama serve`), so the weights,
the chat template and the Metal acceleration are all handled by Ollama.
Nothing leaves the machine. The model tag comes from the `.env` file.
"""

from __future__ import annotations

from llama_index.llms.ollama import Ollama

from .config import (
    CONTEXT_WINDOW,
    MAX_NEW_TOKENS,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_REQUEST_TIMEOUT,
    TEMPERATURE,
    require_ollama_model,
)

SYSTEM_PROMPT: str = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context. If the context does not contain the answer, say you "
    "don't know."
)


def build_llm() -> Ollama:
    """Return a LlamaIndex-compatible LLM served by the local Ollama daemon."""
    return Ollama(
        model=require_ollama_model(),
        base_url=OLLAMA_BASE_URL,
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
        system_prompt=SYSTEM_PROMPT,
        # Keep the weights resident between questions instead of reloading them.
        keep_alive=OLLAMA_KEEP_ALIVE,
        # `context_window` is forwarded to Ollama as `num_ctx`.
        context_window=CONTEXT_WINDOW,
        temperature=TEMPERATURE,
        # `num_predict` is Ollama's equivalent of `max_new_tokens`.
        additional_kwargs={"num_predict": MAX_NEW_TOKENS},
    )
