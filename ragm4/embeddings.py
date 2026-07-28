"""Nomic-embed-text loader.

The model was downloaded locally from HuggingFace inside `/nomic_embed`. It uses
custom modelling code (NomicBertModel) so `trust_remote_code=True` is required.

Nomic embeddings expect a *task prefix* on both documents and queries; using the
proper prefix is what unlocks retrieval-quality vectors:
    - "search_document: <text>"  for indexed passages
    - "search_query:    <text>"  for user questions
LlamaIndex's HuggingFaceEmbedding wires these via `text_instruction` /
`query_instruction`.
"""

from __future__ import annotations

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .config import EMBED_MODEL_DIR


def build_embed_model() -> HuggingFaceEmbedding:
    """Instantiate the local nomic-embed-text model as a LlamaIndex embedding."""
    return HuggingFaceEmbedding(
        model_name=str(EMBED_MODEL_DIR),
        trust_remote_code=True,
        # Prefix strings recommended by the Nomic team.
        text_instruction="search_document: ",
        query_instruction="search_query: ",
        # Keep batches small — the model runs on CPU/MPS locally.
        embed_batch_size=8,
    )
