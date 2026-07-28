"""Central configuration.

Every path is resolved relative to the repository root so the project can be run
from any working directory without breaking.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths --------------------------------------------------------------------

# Repository root (…/RAGM4)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Raw corpus (a single long .txt file lives inside /data)
DATA_DIR: Path = ROOT_DIR / "data"

# Locally-downloaded model checkpoints
LLM_MODEL_DIR: Path = ROOT_DIR / "qwen2.5"
EMBED_MODEL_DIR: Path = ROOT_DIR / "nomic_embed"

# Persistent ChromaDB location + collection name
CHROMA_DIR: Path = ROOT_DIR / "storage" / "chroma"
CHROMA_COLLECTION: str = "ragm4_corpus"

# --- Chunking -----------------------------------------------------------------

CHUNK_SIZE: int = 512      # tokens per chunk (approx.)
CHUNK_OVERLAP: int = 64    # sliding-window overlap

# --- Retrieval ----------------------------------------------------------------

TOP_K: int = 4             # number of chunks retrieved per query

# --- Generation ---------------------------------------------------------------

MAX_NEW_TOKENS: int = 512
TEMPERATURE: float = 0.2   # low temp keeps answers grounded in the context
