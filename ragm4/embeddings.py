"""Nomic-embed-text loader — fully offline, no HuggingFace API calls.

The model weights and tokenizer live in `/nomic_embed`. We load them using
transformers directly (native NomicBert support in transformers >= 4.41) with
offline environment variables set so zero network requests are made.

Nomic embeddings expect a *task prefix* on both documents and queries:
    - "search_document: <text>"  for indexed passages
    - "search_query:    <text>"  for user questions
"""

from __future__ import annotations

import os

# Force fully offline mode — must be set before importing transformers.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from typing import Any, List

import torch
import torch.nn.functional as F
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from transformers import AutoModel, AutoTokenizer

from .config import EMBED_MODEL_DIR

_DEFAULT_BATCH_SIZE = 8


class LocalNomicEmbedding(BaseEmbedding):
    """LlamaIndex embedding that loads nomic-embed-text entirely offline."""

    embed_batch_size: int = Field(default=_DEFAULT_BATCH_SIZE)

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(EMBED_MODEL_DIR),
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            str(EMBED_MODEL_DIR),
            local_files_only=True,
            trust_remote_code=False,
        )
        self._model.eval()

    @classmethod
    def class_name(cls) -> str:
        return "LocalNomicEmbedding"

    def _mean_pool(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        return sum_embeddings / mask_expanded.sum(dim=1).clamp(min=1e-9)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self._model(**encoded)
            embeddings = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._encode(["search_document: " + text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        prefixed = ["search_document: " + t for t in texts]
        return self._encode(prefixed)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._encode(["search_query: " + query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)


def build_embed_model() -> LocalNomicEmbedding:
    """Instantiate the local nomic-embed-text model (fully offline)."""
    return LocalNomicEmbedding(model_name="nomic-embed-text-local")
