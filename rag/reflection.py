"""Reflection pass for context sufficiency checks.

Reflection is the small self-check that happens after retrieval and before the
LLM is asked to answer. It looks for obvious signs that the first pass did not
retrieve enough evidence and can trigger a broader search.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.schemas import ChunkRecord


@dataclass
class ReflectionResult:
    needs_more_context: bool
    missing_signals: list[str]


class ReflectionEngine:
    """Detect when more retrieval is needed."""

    def assess(self, question: str, chunks: list[ChunkRecord]) -> ReflectionResult:
        # Reflection is a simple sufficiency check before the model is asked to answer.
        signals: list[str] = []
        if len(chunks) < 2:
            signals.append("Too few retrieved chunks")
        lower = question.lower()
        # If the user asks about auth and we found no auth vocabulary, request more context.
        if any(word in lower for word in ["auth", "login", "token", "jwt"]) and not self._has_keyword(chunks, ["auth", "token", "jwt", "middleware"]):
            signals.append("Authentication context appears incomplete")
        # Security questions should trigger a second retrieval pass if obvious signals are missing.
        if any(word in lower for word in ["security", "vulnerability"]) and not self._has_keyword(chunks, ["password", "secret", "validate", "sanitize"]):
            signals.append("Security context appears incomplete")
        return ReflectionResult(needs_more_context=bool(signals), missing_signals=signals)

    def _has_keyword(self, chunks: list[ChunkRecord], keywords: list[str]) -> bool:
        # Search all retrieved text for evidence of the requested concept.
        blob = "\n".join(chunk.content.lower() for chunk in chunks)
        return any(keyword in blob for keyword in keywords)
