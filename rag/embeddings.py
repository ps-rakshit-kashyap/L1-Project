"""Embedding provider abstraction.

This module hides the difference between real Ollama embeddings and the local
deterministic fallback vector implementation. The rest of the project only
needs a simple embed interface, so it does not have to care whether Ollama is
running or not.
"""

from __future__ import annotations

import logging
from typing import Any

from config import settings


logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Return Ollama embeddings when available."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        # Default to the project model names, but allow callers to override them.
        self.model = model or settings.embedding_model
        self.base_url = base_url or settings.ollama_base_url
        self._client: Any = None
        self._available = False
        self._initialize()

    def _initialize(self) -> None:
        try:
            # langchain_ollama is optional so local dev can still fall back cleanly.
            from langchain_ollama import OllamaEmbeddings

            self._client = OllamaEmbeddings(model=self.model, base_url=self.base_url)
            self._available = True
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("Ollama embeddings unavailable: %s", exc)
            self._client = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Use real Ollama embeddings when available, otherwise deterministic fallback vectors.
        if self._available and self._client is not None:
            return self._client.embed_documents(texts)
        return [self._fallback_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # Query embeddings must match document embeddings in dimension and style.
        if self._available and self._client is not None:
            return self._client.embed_query(text)
        return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> list[float]:
        # Hash-like character accumulation gives a lightweight similarity signal offline.
        vector = [0.0] * 256
        for idx, char in enumerate(text.encode("utf-8", errors="ignore")):
            vector[idx % 256] += (char % 31) / 31.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]
