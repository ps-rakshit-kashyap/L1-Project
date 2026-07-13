"""Repository ingestion and indexing orchestration.

This service coordinates the two-step indexing pipeline: first the repository
is scanned for files, then those files are parsed and pushed into the vector
store. It is the bridge between filesystem discovery and semantic retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

from loaders.repository_loader import RepositoryLoader, RepositorySummary
from parsers.code_parser import CodeParser
from rag.vector_store import VectorStore


logger = logging.getLogger(__name__)


class RepositoryService:
    """Coordinate scanning, parsing, and indexing."""

    def __init__(self, loader: RepositoryLoader, parser: CodeParser, store: VectorStore) -> None:
        self.loader = loader
        self.parser = parser
        self.store = store

    def index_repository(self, repository_root: Path) -> RepositorySummary:
        # The loader gives us a file inventory and also tells us which files are
        # worth parsing.
        summary = self.loader.scan(repository_root)
        # Clear any previous chunks for this repository before writing fresh
        # data, otherwise stale and current content would mix together.
        self.store.reset_repository(summary.repository)
        chunks = []
        for file_record in summary.files:
            # Each file is read and parsed individually so failures stay local
            # to that file instead of breaking the whole repository.
            file_path = repository_root / file_record.file_path
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            parsed = self.parser.parse_file(summary.repository, file_path, content)
            chunks.extend(parsed.chunks)
        # Once all chunks are collected, store them in the vector database.
        self.store.add_chunks(chunks)
        return summary
