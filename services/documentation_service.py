"""Documentation generation service.

This module converts indexed repository chunks into a simple Markdown report.
It is intentionally lightweight: the goal is to give the user a quick project
overview, not to produce a polished long-form document generator.
"""

from __future__ import annotations

from models.schemas import ChunkRecord


class DocumentationService:
    """Generate structured repository documentation."""

    def generate(self, repository: str, chunks: list[ChunkRecord]) -> str:
        # Turn the indexed file list into a concise Markdown document.
        files = sorted({chunk.file_path for chunk in chunks})
        # The first section is just an inventory of what was indexed.
        summary = "\n".join(f"- {path}" for path in files[:50])
        return (
            f"# Repository Overview\n\n"
            f"Repository: {repository}\n\n"
            f"## Files\n{summary}\n\n"
            f"## Architecture Summary\n"
            f"This repository was analyzed locally using semantic retrieval and code-aware chunking.\n\n"
            f"## Suggested Improvements\n"
            f"- Add tests around critical business logic.\n"
            f"- Centralize configuration and secrets.\n"
            f"- Add explicit interface boundaries between modules.\n"
        )
