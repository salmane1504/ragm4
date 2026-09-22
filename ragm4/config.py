"""Central configuration.

Every path is resolved relative to the repository root so the project can be run
from any working directory without breaking.

The chat model is **not** hard-coded: it is read from the `.env` file at the
repository root (key `OLLAMA_MODEL`). If it is missing, `require_ollama_model()`
raises a `MissingOllamaModelError` explaining how to fix it.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths --------------------------------------------------------------------

# Repository root (…/RAGM4)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Environment file holding the user's settings (never committed).
ENV_FILE: Path = ROOT_DIR / ".env"

# Values already present in the real environment win over the file.
load_dotenv(ENV_FILE, override=False)

# Raw corpus (a single long .txt file lives inside /data)
DATA_DIR: Path = ROOT_DIR / "data"

# Locally-downloaded model checkpoints
EMBED_MODEL_DIR: Path = ROOT_DIR / "nomic_embed"

# Persistent ChromaDB location + collection name
CHROMA_DIR: Path = ROOT_DIR / "storage" / "chroma"
CHROMA_COLLECTION: str = "ragm4_corpus"

# --- Chunking -----------------------------------------------------------------

CHUNK_SIZE: int = 512      # tokens per chunk (approx.)
CHUNK_OVERLAP: int = 64    # sliding-window overlap

# --- Retrieval ----------------------------------------------------------------

TOP_K: int = 4             # number of chunks retrieved per query

# --- LLM (Ollama) -------------------------------------------------------------

# Model tag as shown by `ollama ls`, read from `.env`. There is no default:
# the user must pull a model and declare it explicitly.
# Examples: "qwen2.5:7b", "qwen2.5:14b", "mistral", "llama3.2", etc.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "").strip()
# Local Ollama daemon endpoint.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
# Seconds to wait for a generation (first call also pays the model-load cost).
OLLAMA_REQUEST_TIMEOUT: float = 300.0
# How long Ollama keeps the model resident in RAM between questions.
OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()
# Context window exposed to the model (qwen2.5 supports 32k).
CONTEXT_WINDOW: int = 32_768


class MissingOllamaModelError(RuntimeError):
    """Raised when no chat model is declared in the `.env` file."""


_MISSING_MODEL_MESSAGE = (
    "No Ollama model configured.\n"
    "\n"
    "RAGM4 needs a chat model that is already available locally. Fix it with:\n"
    "  1. Pull a model from Ollama, e.g.  ollama pull qwen2.5:14b\n"
    "  2. List what you have with       ollama ls\n"
    f"  3. Put the exact tag in {ENV_FILE}, e.g.:\n"
    "         OLLAMA_MODEL=qwen2.5:14b\n"
    "\n"
    "See .env.example for the expected format."
)


def require_ollama_model() -> str:
    """Return the configured model tag, or raise a user-facing error."""
    if not OLLAMA_MODEL:
        raise MissingOllamaModelError(_MISSING_MODEL_MESSAGE)
    return OLLAMA_MODEL

# --- Generation ---------------------------------------------------------------

MAX_NEW_TOKENS: int = 512
TEMPERATURE: float = 0.2   # low temp keeps answers grounded in the context
