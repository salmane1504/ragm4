"""LlamaIndex `CustomLLM` backed by mlx-lm running Qwen2.5-1.5B locally.

We wrap `mlx_lm` directly (rather than pulling `llama-index-llms-mlx-lm`) so the
dependency surface stays small and the chat template used by Qwen2.5 is applied
explicitly.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from .config import LLM_MODEL_DIR, MAX_NEW_TOKENS, TEMPERATURE


# Qwen2.5 context window (see qwen2.5/config.json → max_position_embeddings).
_CONTEXT_WINDOW: int = 32_768


class MLXQwenLLM(CustomLLM):
    """A LlamaIndex-compatible LLM that runs Qwen2.5 through mlx-lm."""

    # These are exposed as pydantic fields so LlamaIndex can serialise them.
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = TEMPERATURE

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # `load` accepts a HF-format directory (config.json + tokenizer.* +
        # model.safetensors) and returns the MLX model plus its tokenizer.
        model, tokenizer = load(str(LLM_MODEL_DIR))
        # `object.__setattr__` bypasses the pydantic model — these are pure
        # runtime handles that must NOT be serialised.
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_tokenizer", tokenizer)

    # ---- LlamaIndex plumbing ------------------------------------------------

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=_CONTEXT_WINDOW,
            num_output=self.max_new_tokens,
            model_name="Qwen2.5-1.5B (mlx-lm)",
        )

    # ---- Prompt formatting --------------------------------------------------

    def _format_prompt(self, prompt: str) -> str:
        """Apply Qwen2.5's chat template around the raw prompt."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "using only the provided context. If the context does not "
                    "contain the answer, say you don't know."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ---- Generation ---------------------------------------------------------

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """One-shot completion used by LlamaIndex query engines."""
        text_prompt = prompt if formatted else self._format_prompt(prompt)
        sampler = make_sampler(temp=self.temperature)
        output = generate(
            self._model,
            self._tokenizer,
            prompt=text_prompt,
            max_tokens=self.max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        return CompletionResponse(text=output)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming variant — mlx-lm doesn't stream through this simple wrapper,
        so we emit the full answer as a single chunk. Kept for API compliance."""
        response = self.complete(prompt, formatted=formatted, **kwargs)
        yield response
