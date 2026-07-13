"""LLM interaction layer.

This module is the thin boundary between the app and the local Ollama model.
It tries to use the real model when available, but it also protects the rest
of the app from runtime failures by returning a deterministic fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from config import settings


logger = logging.getLogger(__name__)


class LLMService:
    """Ollama-backed LLM with graceful fallback."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        # Keep model selection centralized so the UI and background services agree.
        self.model = model or settings.default_model
        self.base_url = base_url or settings.ollama_base_url
        self._client: Any = None
        self._available = False
        self._initialize()

    def _initialize(self) -> None:
        try:
            # ChatOllama is optional; if import or startup fails, the app still
            # remains usable.
            from langchain_ollama import ChatOllama

            self._client = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=settings.llm_temperature,
                num_ctx=8192,
            )
            self._available = True
        except Exception as exc:  # pragma: no cover
            logger.warning("Ollama unavailable: %s", exc)
            self._client = None
            self._available = False

    def generate(self, prompt: str) -> str:
        # Ask Ollama first, but protect the app from empty or failed
        # completions.
        if self._available and self._client is not None:
            try:
                response = self._client.invoke(prompt)
                content = getattr(response, "content", str(response)).strip()
                if content:
                    return content
                logger.warning("Ollama returned an empty response; using fallback")
            except Exception as exc:  # pragma: no cover
                logger.warning("Ollama generation failed: %s", exc)
        return self._fallback(prompt)

    def _fallback(self, prompt: str) -> str:
        # The fallback makes it obvious that the result came from retrieved
        # context rather than from an active model call.
        return (
            "Ollama is not available locally. "
            "Fallback response generated from retrieved context only.\n\n"
            f"Prompt excerpt: {prompt[:700]}"
        )
